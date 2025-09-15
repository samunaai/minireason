# run_countdown_pretraining.py
"""
This script trains a Transformer-based neural network to solve the Countdown numbers game.

The process involves several key stages:
1.  **Data Generation:** It begins by creating a dataset of Countdown problems and their
    solutions. It randomly selects a set of numbers and a target number, then uses a
    depth-first search solver to find valid sequences of operations to reach the target.
2.  **Tokenization:** Problems and their corresponding solution programs are converted into
    sequences of numerical tokens that the model can process. This includes special tokens
    for representing the game's structure (e.g., IN:, TGT:, ->, SEP).
3.  **Model Architecture:** The core is a Decoder-Only Transformer model, a popular
    architecture for sequence generation tasks, similar to models like GPT.
4.  **Training:** The model is trained to predict the next token in a solution sequence,
    given the problem and the preceding tokens. It uses a standard training loop with an
    AdamW optimizer, gradient clipping, and an early stopping mechanism. This version
    includes a hybrid loss function to specifically improve arithmetic correctness.
5.  **Evaluation:** After training, the model's performance is evaluated on a held-out
    validation set. Metrics include exact program match, the validity of generated
    arithmetic operations, and the consistency of state transitions.
6.  **Inference:** The script includes a sampling function to generate solutions for new
    Countdown problems, with options for both greedy decoding and controlled stochasticity.
"""
import random, itertools, time, sys, bisect, re, os, logging
from collections import defaultdict, Counter
from typing import List, Optional, Tuple
from multiprocessing import Pool, cpu_count

# FIX: Set thread env vars *before* importing torch for maximum reliability.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.utils.checkpoint as checkpoint
from tqdm import tqdm

# Set matplotlib backend for headless servers to prevent crashes.
import matplotlib
os.environ.setdefault("MPLBACKEND","Agg"); matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    # Guard SDP toggles for older PyTorch versions
    for fn in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_math_sdp"):
        if hasattr(torch.backends.cuda, fn):
            getattr(torch.backends.cuda, fn)(fn != "enable_math_sdp")  # prefer Flash/MemEfficient
# =========================
# CONFIGURATION & HYPERPARAMETERS
# =========================
HPARAMS = {
    "num_sources": 6,
    "num_samples": 100000,
    "max_solutions": 1,  # use only one solution per problem for a clearer signal
    "use_cached_data": True,
    "small_deck": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10],
    "large_deck": [25, 50, 75, 100],
    "target_range": (100, 999),
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 4,
    "epochs": 1000,
    "patience": 20,
    "batch_size": 32,
    "lr": 3e-4,
    "weight_decay": 0.01,
    "train_split_ratio": 0.9,
    # Loss weights
    "lambda_transition_ce": 0.25, # CE only on the final post-state bracket span
    "lambda_final_expr": 0.25,    # CE on tokens after EXPR
    "lambda_last_equation": 0.5,  # last equation loss (a op b = c)
    "lambda_eos_margin": 0.5,     # weight for EOS margin auxiliary
    "use_gradient_checkpointing": False,
    "seed": 42,
    # --- Stochastic decision, deterministic execution (inference) ---
    "stochastic_steps": 1,        # apply stochasticity to the first N steps; 0 = fully greedy
    "temperature_ops": 0.7,       # temperature for decision tokens (first operand's first digit and operator)
    "top_p_ops": 0.9,             # nucleus for decision tokens; 1.0 disables
    "top_k_ops": 0,               # top-k cap for decision tokens; 0 disables
}
SAFE_MAX_LEN = int(os.environ.get("MAX_SEQ_LEN", "2048"))
DATA_GEN_ATTEMPT_MULTIPLIER = 800
DATA_GEN_MIN_ATTEMPTS = 8000
NUM_INFERENCE_EXAMPLES = 5

# =========================
# Countdown Solver & Data Formatting
# =========================
class CountdownSolver:
    def __init__(self):
        self.solutions = []

    def solve(self, numbers, target, max_depth=6, max_solutions=1):
        """Return up to max_solutions distinct solution step-lists."""
        self.solutions = []
        if target in numbers:
            self.solutions = [[]]
            return self.solutions

        labeled = tuple(sorted([(v, f"n{i+1}") for i, v in enumerate(numbers)]))
        memo = set()
        self._dfs(labeled, target, [], memo, max_depth, max_solutions)
        return self.solutions[:max_solutions]

    def _dfs(self, items, target, steps, memo, max_depth, max_solutions):
        # Base case: stop if we've found enough solutions
        if len(self.solutions) >= max_solutions: return
        
        # Memoization: avoid re-computing for the same set of numbers
        key = tuple(v for v, _ in items)
        if key in memo: return
        memo.add(key)
        
        # Base case: stop if path is too long or not enough numbers left
        if len(items) < 2 or len(steps) >= max_depth: return

        # --- Commutative operations (+, *) ---
        combs = list(itertools.combinations(range(len(items)), 2))
        random.shuffle(combs)
        for a_idx, b_idx in combs:
            (av, al), (bv, bl) = items[a_idx], items[b_idx]
            cand = [('+', av + bv)]
            if av > 1 and bv > 1: cand.append(('*', av * bv))
            for op, resv in cand:
                new_steps = steps + [(av, op, bv, resv)]
                rest = [item for i, item in enumerate(items) if i not in (a_idx, b_idx)]
                bisect.insort(rest, (resv, "t"))

                if resv == target and len(rest) == 1:
                    self.solutions.append(new_steps)
                    if len(self.solutions) >= max_solutions: return
                else:
                    self._dfs(tuple(rest), target, new_steps, memo, max_depth, max_solutions)

        # --- Non-commutative operations (-, /) ---
        perms = list(itertools.permutations(range(len(items)), 2))
        random.shuffle(perms)
        for a_idx, b_idx in perms:
            (av, al), (bv, bl) = items[a_idx], items[b_idx]
            cand = []
            if av > bv: cand.append(('-', av - bv))
            if bv >= 1 and av % bv == 0: cand.append(('/', av // bv))
            for op, resv in cand:
                new_steps = steps + [(av, op, bv, resv)]
                rest = [item for i, item in enumerate(items) if i not in (a_idx, b_idx)]
                bisect.insort(rest, (resv, "t"))

                if resv == target and len(rest) == 1:
                    self.solutions.append(new_steps)
                    if len(self.solutions) >= max_solutions: return
                else:
                    self._dfs(tuple(rest), target, new_steps, memo, max_depth, max_solutions)

def take_pair(pool: List[Tuple[int, str]], val: int) -> Tuple[int, str]:
    for i, (v, e) in enumerate(pool):
        if v == val:
            return pool.pop(i)
    # if this ever triggers, the recorded steps don't consume the stated numbers
    raise AssertionError(f"value {val} not found in pool while building EXPR")
        
def _build_parenthesized_expr(initial_numbers: List[int], solution_steps: List) -> str:
    """Construct a fully parenthesized infix expression from the step sequence."""
    # maintain numeric value alongside expression (multiset behavior)
    pool = [(x, str(x)) for x in initial_numbers]

    for a, op, b, c in solution_steps:
        (va, ea) = take_pair(pool, a)
        (vb, eb) = take_pair(pool, b)
        expr = f"({ea} {op} {eb})"
        if   op == '+': vc = va + vb
        elif op == '-': vc = va - vb
        elif op == '*': vc = va * vb
        else:
            # solver guarantees safe integer division; assert to catch any drift
            assert vb != 0 and va % vb == 0, "invalid division in steps"
            vc = va // vb
        pool.append((vc, expr))
    # final expression (fully parenthesized)
    return pool[-1][1] if pool else ""

def steps_to_program(initial_numbers: List[int], solution_steps: List, target: int) -> Optional[str]:
    if solution_steps is None: return None
    if not solution_steps: 
        # For problems where target is one of the source numbers: emit EXPR as just the target
        return f"EXPR {target}"
 
    out = []
    current_numbers = Counter(initial_numbers)
    for n1, op, n2, res in solution_steps:
        before_state_str = ', '.join(str(x) for x in sorted(current_numbers.elements()))
        out.append(f"[ {before_state_str} ] ->")
        out.append(f"{n1} {op} {n2} = {res} ->")
        current_numbers[n1] -= 1
        current_numbers[n2] -= 1
        for k in {n1, n2}:
            if current_numbers.get(k, 0) <= 0:
                current_numbers.pop(k, None)
        current_numbers[res] += 1
        after_state_str = ', '.join(str(x) for x in sorted(current_numbers.elements()))
        out.append(f"[ {after_state_str} ] ;")
    program_str = " ".join(out)
    program_str = re.sub(r"\s*;\s*$", "", program_str)
    
    # Append EXPR (parenthesized expression) summarizing the steps
    expr = _build_parenthesized_expr(initial_numbers, solution_steps)
    program_str += f" ; EXPR {expr}"
    return program_str

def _gen_chunk(args):
    chunk_samples, num_sources, seed, small_deck, large_deck, target_range, max_solutions = args
    random.seed(seed)
    solver = CountdownSolver()
    inputs, outputs = [], []
    attempts = rejects_solver = 0
    successful_attempts = 0
    max_attempts = max(DATA_GEN_MIN_ATTEMPTS, DATA_GEN_ATTEMPT_MULTIPLIER * chunk_samples)

    while len(inputs) < chunk_samples and attempts < max_attempts:
        attempts += 1
        min_large = max(0, num_sources - len(small_deck)); max_large = min(len(large_deck), num_sources)
        num_large = random.randint(min_large, max_large); num_small = num_sources - num_large
        large = random.sample(large_deck, k=num_large); small = random.sample(small_deck,  k=num_small)
        source_numbers = large + small; random.shuffle(source_numbers)
        target = random.randint(target_range[0], target_range[1])
        if len(source_numbers) > 1 and target in source_numbers: continue
        
        sols = solver.solve(source_numbers, target, max_depth=num_sources - 1, max_solutions=max_solutions)
        if not sols:
            rejects_solver += 1
            continue
        successful_attempts += 1
        
        in_list = ', '.join(map(str, source_numbers))
        for steps in sols:
            program = steps_to_program(source_numbers, steps, target)
            if program is None: continue
            inputs.append(f"IN: [ {in_list} ] TGT: {target}")
            outputs.append(program)

    return inputs, outputs, attempts, rejects_solver, successful_attempts

def generate_countdown_data(config, n_workers=None):
    num_samples, num_sources = config['num_samples'], config['num_sources']
    seed, small_deck, large_deck = config['seed'], config['small_deck'], config['large_deck']
    target_range, max_solutions = config['target_range'], config.get('max_solutions', 1)
    
    n_workers = min(n_workers or cpu_count(), 64, num_samples)
    granularity = 4
    target_tasks = min(n_workers * granularity, num_samples) if n_workers > 0 else num_samples
    if target_tasks == 0: return [],[]
    per_task = max(1, num_samples // target_tasks); extras = num_samples % target_tasks
    tasks = [(per_task + (1 if i < extras else 0), num_sources, seed + 1000 * i,
              small_deck, large_deck, target_range, max_solutions) for i in range(target_tasks)]
    
    logging.info(f"Starting parallel data generation for {num_samples:,} samples on {n_workers} workers ({len(tasks)} tasks)...")
    ins, outs, total_attempts, total_rej_s, total_success = [], [], 0, 0, 0
    with Pool(processes=n_workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        pbar = tqdm(pool.imap_unordered(_gen_chunk, tasks, chunksize=2), total=len(tasks), desc="Generating Sample Batches")
        for ins_chunk, outs_chunk, a, rs, succ in pbar:
            ins += ins_chunk; outs += outs_chunk; total_attempts += a; total_rej_s += rs; total_success += succ
    
    avg_solutions = len(ins) / max(1, total_attempts); acc_rate = 100.0 * total_success / max(1, total_attempts)
    logging.info(f"Generated {len(ins):,} solutions from {total_attempts:,} problem draws "
          f"(~{avg_solutions:.2f} solutions/draw, {acc_rate:.1f}% draws with ≥1 solution).")
    logging.info(f"  Total Rejects (solver): {total_rej_s:,}.")
    if len(ins) < num_samples: logging.warning(f"Warning: Underfilled. Generated {len(ins):,}/{num_samples:,} samples.")
    return ins[:num_samples], outs[:num_samples]

# =========================
# Tokenizer & Parsing
# =========================
class Tokenizer:
    def __init__(self):
        self.special = ['PAD', 'EOS', 'SEP', 'IN:', 'TGT:', '[', ']', '(', ')', ',', '=', ';', '->', '+', '-', '*', '/']
        self.cmds = ['EXPR']
        self.digits = [str(d) for d in range(10)]
        vocab = self.special + self.cmds + self.digits
        self.tok2id = {t: i for i, t in enumerate(vocab)}; self.id2tok = {i: t for i, t in enumerate(vocab)}
        self._pad=self.tok2id['PAD']; self._eos=self.tok2id['EOS']; self._sep=self.tok2id['SEP']
        self._tok_re = re.compile(r'(->|\[|\]|\(|\)|,|=|;|\+|\-|\*|/)')

    @property
    def pad_id(self): return self._pad
    @property
    def eos_id(self): return self._eos
    @property
    def sep_id(self): return self._sep
    @property
    def vocab_size(self): return len(self.tok2id)

    def encode(self, text, max_len):
        spaced_text = self._tok_re.sub(r' \1 ', text); tokens = spaced_text.split()
        flat_tokens = [];
        for tok in tokens:
            if tok.isdigit() and len(tok) > 1: flat_tokens.extend(list(tok))
            else: flat_tokens.append(tok)
        flat_tokens.append('EOS')
        ids = [self.tok2id.get(t) for t in flat_tokens]
        if any(i is None for i in ids): raise ValueError(f"Unknown token in text: '{text}' from tokens '{flat_tokens}'")
        if len(ids) < max_len: ids += [self._pad] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        tokens = [self.id2tok.get(i, '?') for i in ids if i not in (self._pad, self._eos)]
        result = []; i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.isdigit():
                num_str = token; i += 1
                while i < len(tokens) and tokens[i].isdigit(): num_str += tokens[i]; i += 1
                result.append(num_str)
            else:
                result.append(token); i += 1
        s = " ".join(result)
        s = re.sub(r"\s*,\s*", ", ", s)
        s = re.sub(r"\[\s*", "[ ", s); s = re.sub(r"\s*\]", " ]", s)
        s = re.sub(r"\(\s*", "(", s);  s = re.sub(r"\s*\)", ")", s)  # tidy paren spacing
        s = re.sub(r"\s*->\s*", " -> ", s)
        s = re.sub(r"\s*=\s*", " = ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

def parse_initial_numbers(prompt_text: str) -> Optional[List[int]]:
    match = re.search(r"IN: \[ (.*?) \]", prompt_text)
    if match:
        parts = re.split(r"[,\s]+", match.group(1).strip())
        return [int(n) for n in parts if n]
    return None

# =========================
# Evaluation, Model & Training
# =========================
inference_mode = getattr(torch, "inference_mode", torch.no_grad)

@inference_mode()
def evaluate_program_metrics(model, tok, val_loader, device, max_out, max_batches=4):
    """Evaluates the model on granular metrics without relying on a single "correct" answer."""
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    n_prog_em, n, n_verified_ok, step_ok, step_total, step_state_ok, step_state_total = 0, 0, 0, 0, 0, 0, 0

    it = iter(val_loader)
    for _ in range(max_batches):
        try: x_batch, _ = next(it)
        except StopIteration: break
        if x_batch.numel() == 0: continue
        for x in x_batch:
            ids = x.tolist()
            if tok.sep_id not in ids: continue
            sep_idx = ids.index(tok.sep_id)
            prompt_ids = [t for t in ids[:sep_idx+1] if t != tok.pad_id]
            tgt_text = tok.decode(ids[sep_idx+1:])
            prompt_text = tok.decode(prompt_ids).replace(' SEP', '').strip()
            gen_ids = sample_one(model, tok, prompt_ids, max_out, device)
            if not gen_ids: continue
            gen_text = tok.decode(gen_ids)
            n += 1
            # With EXPR, compare full normalized strings
            if norm(gen_text) == norm(tgt_text):
                n_prog_em += 1
            tgt_m = re.search(r"TGT:\s*(\d+)", prompt_text)
            if tgt_m:
                tgt_val = int(tgt_m.group(1))
                states = re.findall(r"\[\s*([^\]]*?)\s*\]", gen_text)
                if states:
                    last_nums = [int(z) for z in re.split(r"[,\s]+", states[-1].strip()) if z]
                    if len(last_nums) == 1 and last_nums[0] == tgt_val: n_verified_ok += 1
            initial_nums = parse_initial_numbers(prompt_text)
            if initial_nums:
                steps = re.findall(r"\[\s*([^\]]*?)\s*\]\s*->\s*(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)\s*->\s*\[\s*([^\]]*?)\s*\]", gen_text)
                current_state = Counter(initial_nums)
                for before_str, n1_str, op, n2_str, res_str, after_str in steps:
                    n1, n2, res = int(n1_str), int(n2_str), int(res_str)
                    is_arith_ok = (op == '+' and n1 + n2 == res) or (op == '-' and n1 > n2 and n1 - n2 == res) or \
                                  (op == '*' and n1 > 1 and n2 > 1 and n1 * n2 == res) or (op == '/' and n2 >= 1 and n1 % n2 == 0 and n1 // n2 == res)
                    step_total += 1
                    if is_arith_ok: step_ok += 1
                    step_state_total += 1
                    try:
                        stated_before = Counter(int(z) for z in re.split(r"[,\s]+", before_str.strip()) if z)
                        stated_after  = Counter(int(z) for z in re.split(r"[,\s]+", after_str.strip()) if z)
                        if stated_before != current_state or not is_arith_ok: continue
                        if current_state.get(n1, 0) == 0 or current_state.get(n2, 0) - (n1 == n2) < 0: continue
                        current_state[n1] -= 1;
                        if current_state[n1] == 0: del current_state[n1]
                        current_state[n2] -= 1;
                        if current_state[n2] == 0: del current_state[n2]
                        current_state[res] += 1
                        if current_state == stated_after: step_state_ok += 1
                    except Exception: pass
    if n == 0: raise RuntimeError("Evaluation failed: No valid samples were processed.")
    return dict(samples=n, program_exact_match=n_prog_em / n, verified_target_state_accuracy=n_verified_ok / n,
                op_valid_rate=(step_ok / step_total) if step_total else 0.0,
                op_state_consistent_rate=(step_state_ok / step_state_total) if step_state_total else 0.0)

def plot_training_results(history, filename="countdown_training_plot.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss'); ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'bo-', label='Training Accuracy'); ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'ro-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Training and Validation Accuracy'); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); plt.savefig(filename)
    logging.info(f"\nTraining plot saved to {filename}")

class Block(nn.Module):
    def __init__(self, d, h, max_len):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1))

    def forward(self, x, key_padding_mask):
        L = x.size(1)
        x1 = self.ln1(x)
        attn_output, _ = self.attn(x1, x1, x1, attn_mask=self.causal_mask[:L, :L], key_padding_mask=key_padding_mask)
        x = x + attn_output; x = x + self.mlp(self.ln2(x))
        return x

class DecoderOnly(nn.Module):
    def __init__(self, vocab, d, h, n_layers, max_len, pad_id, use_grad_ckpt):
        super().__init__()
        self.pad_id, self.use_gradient_checkpointing = pad_id, use_grad_ckpt
        self.tok = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h, max_len) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len
        self.register_buffer("pos_idx", torch.arange(max_len))

    def forward_features(self, x):
        B, L = x.shape
        if L > self.max_len: x = x[:, -self.max_len:]; L = self.max_len
        key_padding_mask = (x == self.pad_id)
        pos_emb = self.pos(self.pos_idx[:L].to(x.device))
        h = self.tok(x) + pos_emb
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        for blk in self.blocks:
            if self.use_gradient_checkpointing and self.training:
                # FIX 3: Safer gradient checkpoint call
                h = checkpoint.checkpoint(lambda a, b: blk(a, b), h, key_padding_mask, use_reentrant=False)
            else:
                h = blk(h, key_padding_mask)
        return self.ln_f(h)

    def forward(self, x):
        h = self.forward_features(x)
        return self.head(h)

def _last_step_spans(token_ids: torch.Tensor, tok) -> Optional[Tuple[int, int, int]]:
    """
    Return (eq_idx, post_state_l, post_state_r_exclusive) for the *last* step before EXPR (or EOS).
    We look for "... a op b = c -> [ after ]" and return indices in token space.
    """
    ids = token_ids.tolist()
    try:
        end = ids.index(tok.tok2id['EXPR'])
    except ValueError:
        end = len(ids)
    eq_id, arr_id, lbr, rbr = tok.tok2id['='], tok.tok2id['->'], tok.tok2id['['], tok.tok2id[']']
    # last '=' before end
    eq_positions = [i for i, t in enumerate(ids[:end]) if t == eq_id]
    if not eq_positions:
        return None
    eq_idx = eq_positions[-1]
    # find the "-> [ ... ]" that follows this '='
    try:
        arrow2 = ids.index(arr_id, eq_idx + 1)
        l = ids.index(lbr, arrow2 + 1)
        r = ids.index(rbr, l + 1) + 1  # exclusive
    except ValueError:
        return None
    return eq_idx, l, r

 
def compute_masked_loss(model, hidden_states, targets, tok):
    h_for_pred = hidden_states[:, :-1, :].contiguous()
    t_for_pred = targets[:, 1:].contiguous()
    B, Tm1, D = h_for_pred.shape
    idx = torch.arange(Tm1, device=targets.device).unsqueeze(0)
    RP = tok.tok2id[')']  # right-paren token id (used in multiple scopes)

    with torch.no_grad():
        sep_pos = (targets == tok.sep_id).int().argmax(dim=1)
        eos_pos = (targets == tok.eos_id).int().argmax(dim=1)
        # Base CE mask on (SEP, EOS)  — include first token after SEP
        base_ce_mask = (idx >= sep_pos.unsqueeze(1)) & (idx + 1 < eos_pos.unsqueeze(1)) \
                       & (t_for_pred != tok.pad_id) & (t_for_pred != tok.eos_id)
        ce_mask = base_ce_mask.clone()

        # ===== A) Remove CE on result digits after every "=" (before EXPR) =====
        eq_id = tok.tok2id['=']
        digit_ids = [tok.tok2id[str(d)] for d in range(10)]
        for b in range(B):
            ids_b = targets[b].tolist()
            try:
                end_limit = ids_b.index(tok.tok2id['EXPR'])
            except ValueError:
                end_limit = len(ids_b)
            j = 0
            while j < end_limit:
                if ids_b[j] == eq_id:
                    k = j + 1
                    while k < end_limit and ids_b[k] in digit_ids:
                        if k - 1 < ce_mask.size(1):
                            ce_mask[b, k - 1] = False  # do not train result digits with CE
                        k += 1
                    j = k
                else:
                    j += 1

        # ===== B) Stop CE on trailing EXPR tokens and upweight EOS right after final ')' =====
        ce_weights = torch.ones_like(ce_mask, dtype=h_for_pred.dtype, device=h_for_pred.device)
        for b in range(B):
            ids_b = targets[b].tolist()
            try:
                eos_idx = ids_b.index(tok.eos_id)
            except ValueError:
                continue
            rpars = [i for i,t in enumerate(ids_b[:eos_idx]) if t == RP]
            if not rpars:
                continue
            r = rpars[-1]
            # remove CE supervision for any tokens after last ')' and before EOS
            for j in range(r + 1, eos_idx):
                if j - 1 < ce_mask.size(1):
                    ce_mask[b, j - 1] = False
            # upweight CE exactly where EOS must be predicted (time r -> target r+1)
            if r < ce_weights.size(1):
                ce_weights[b, r] = 5.0  # tuneable
        # Build a mask for the final post-state bracket span ONLY (last transition)
        last_transition_mask = torch.zeros_like(base_ce_mask)
        for b in range(B):
            spans = _last_step_spans(targets[b], tok)
            if spans is None:
                continue
            _, post_l, post_r = spans
            l = max(0, post_l - 1)  # predict '[' at t = (l - 1)
            r = post_r
            # include the following ';' if present
            semi = tok.tok2id[';']
            if r < targets.size(1) and targets[b, r].item() == semi:
                r += 1
            # supervise t indices that predict tokens in [post_l, r_exclusive)
            # i.e., t ∈ [l, r-1)
            if r - 1 > l:
                last_transition_mask[b, l:r-1] = True

        # Base CE excludes the last transition span to avoid double counting
        ce_mask = ce_mask & (~last_transition_mask)

    selected_h = h_for_pred[ce_mask]; selected_t = t_for_pred[ce_mask]; selected_w = ce_weights[ce_mask]
    # debug: supervised token count
    # logging.debug(f"supervised_tokens={int(ce_mask.sum().item())}")
    if selected_t.numel() == 0:
        base_cross_entropy, ce_correct, ce_tokens = hidden_states.sum() * 0.0, 0, 0
    else:
        logits = model.head(selected_h)
        per_tok = F.cross_entropy(logits, selected_t, reduction='none')
        base_cross_entropy = (per_tok * selected_w).mean()
        ce_correct = (logits.argmax(-1) == selected_t).sum().item()
        ce_tokens = selected_t.numel()


    # === Last-transition loss (CE on final [ ... ] bracket span) ===
    post_h = h_for_pred[last_transition_mask]
    post_t = t_for_pred[last_transition_mask]
    if post_t.numel() == 0:
        last_transition_loss = hidden_states.sum() * 0.0
    else:
        last_transition_loss = F.cross_entropy(model.head(post_h), post_t)

    # ===== Final-expression loss (only tokens after EXPR, capped at final ')') =====
    with torch.no_grad():
        Tm1 = t_for_pred.size(1)
        idx = torch.arange(Tm1, device=targets.device).unsqueeze(0)
        has_expr = (targets == tok.tok2id['EXPR']).any(dim=1)
        expr_pos = (targets == tok.tok2id['EXPR']).int().argmax(dim=1)
        expr_pos = torch.where(has_expr, expr_pos, torch.full_like(expr_pos, Tm1))  # if missing, mask empty
        eos_pos  = (targets == tok.eos_id).int().argmax(dim=1)
        # cap supervision at the last ')' before EOS (RP defined above)
        last_r = torch.full_like(eos_pos, fill_value=Tm1 - 1)
        for b in range(B):
            ids_b = targets[b].tolist()
            try:
                e = ids_b.index(tok.eos_id)
            except ValueError:
                continue
            rpars = [i for i, t in enumerate(ids_b[:e]) if t == RP]
            if rpars:
                last_r[b] = rpars[-1]
        expr_mask = (idx >= expr_pos.unsqueeze(1)) & (idx <= last_r.unsqueeze(1)) \
                    & (t_for_pred != tok.pad_id) & (t_for_pred != tok.eos_id)
    sum_h = h_for_pred[expr_mask]
    sum_t = t_for_pred[expr_mask]
    if sum_t.numel() == 0:
        final_expression_loss = hidden_states.sum() * 0.0
    else:
        final_expression_loss = F.cross_entropy(model.head(sum_h), sum_t)

    # === Last-equation loss: compare model's digit logits to value(a op b) (only last equation) ===
    with torch.no_grad():
        digit_ids = [tok.tok2id[str(d)] for d in range(10)]
        eq_meta = []  # (b, pos_list, truth_value_float)
        for b in range(B):
            ids_b = targets[b].tolist()
            try:
                end_limit = ids_b.index(tok.tok2id['EXPR'])
            except ValueError:
                end_limit = len(ids_b)
            # find last '=' before EXPR
            eq_positions = [i for i, t in enumerate(ids_b[:end_limit]) if t == tok.tok2id['=']]
            if not eq_positions:
                continue
            eq = eq_positions[-1]
            # parse a,op,b,c from text for truth value
            text_b = tok.decode([t for t in ids_b[:end_limit] if t not in (tok.pad_id, tok.eos_id)])
            m_all = list(re.finditer(r"(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*([0-9]+)", text_b))
            if not m_all:
                continue
            a_v = int(m_all[-1].group(1)); op = m_all[-1].group(2); b_v = int(m_all[-1].group(3))
            if   op == '+': truth_val = a_v + b_v
            elif op == '-': truth_val = a_v - b_v
            elif op == '*': truth_val = a_v * b_v
            elif op == '/': truth_val = a_v // b_v if b_v != 0 else 0
            else:           truth_val = 0
            # collect t indices that predict the result digits right after '='
            j = eq + 1
            pos_list = []
            while j < len(ids_b) and ids_b[j] in digit_ids:
                if 0 <= (j - 1) < Tm1:
                    pos_list.append(j - 1)
                j += 1
            if pos_list:
                eq_meta.append((b, pos_list, float(truth_val)))
    if eq_meta:
        all_logits = model.head(h_for_pred)  # (B, Tm1, V)
        losses = []
        for b_idx, pos_list, truth_val in eq_meta:
            logits_digits = all_logits[b_idx, pos_list][:, digit_ids]  # (L,10)
            probs = torch.softmax(logits_digits, dim=-1)
            digits = torch.arange(10, device=targets.device, dtype=probs.dtype)
            exp_digits = (probs * digits).sum(dim=-1)  # (L,)
            Ld = len(pos_list)
            # stable, device-friendly powers of 10
            exps = torch.arange(Ld - 1, -1, -1, device=targets.device, dtype=probs.dtype)
            weights = torch.pow(torch.tensor(10.0, device=targets.device, dtype=probs.dtype), exps)
            exp_num = (exp_digits * weights).sum()
            losses.append(torch.abs(exp_num - torch.tensor(truth_val, device=targets.device, dtype=probs.dtype)))
        last_equation_loss = torch.stack(losses).mean()
    else:
        last_equation_loss = hidden_states.sum() * 0.0
    # Optional: EOS margin auxiliary (sharpen stopping). Keep weight small.
    m = 2.0
    eos_margin_losses = []
    with torch.no_grad():
        eos_targets = []
        for b in range(B):
            ids_b = targets[b].tolist()
            try:
                eos_idx = ids_b.index(tok.eos_id)
            except ValueError:
                continue
            rpars = [i for i,t in enumerate(ids_b[:eos_idx]) if t == RP]
            if not rpars: continue
            eos_targets.append((b, rpars[-1]))
    for b, r in eos_targets:
        if 0 <= r < h_for_pred.size(1):
            z = model.head(h_for_pred[b, r:r+1, :]).squeeze(0)  # (V,)
            eos_logit = z[tok.eos_id]
            z_ = z.clone(); z_[tok.eos_id] = -1e9
            mx = z_.max()
            eos_margin_losses.append(F.relu(m + mx - eos_logit))
    eos_margin_loss = torch.stack(eos_margin_losses).mean() if eos_margin_losses else hidden_states.sum()*0.0

    total_loss = (
        base_cross_entropy
        + HPARAMS.get("lambda_transition_ce", 0.0) * last_transition_loss
        + HPARAMS.get("lambda_final_expr", 0.0) * final_expression_loss
        + HPARAMS.get("lambda_last_equation", 0.0) * last_equation_loss
        + HPARAMS.get("lambda_eos_margin", 0.0) * eos_margin_loss

    )
    return total_loss, ce_correct, ce_tokens


def run_epoch(is_train, model, loader, device, tok, opt=None, scaler=None, amp_dtype=torch.float32, use_amp=False):
    model.train(is_train)
    batch_losses, total_correct, total_tokens = [], 0, 0
    model_to_call = model.module if hasattr(model, "module") else model

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp), torch.set_grad_enabled(is_train):
            hidden_states = model_to_call.forward_features(x)
            loss, correct, n_tokens = compute_masked_loss(model_to_call, hidden_states, y, tok)
        if n_tokens > 0 or is_train:
            if is_train:
                opt.zero_grad(set_to_none=True)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            batch_losses.append(loss.item())
            total_correct += correct; total_tokens += n_tokens

    acc = (total_correct / total_tokens) if total_tokens else 0.0
    avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
    return acc, avg_loss

@inference_mode()
def sample_one(model, tokenizer, prompt_ids, max_out, device, amp_dtype=torch.float32, use_amp=False):
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()
    x = torch.tensor([prompt_ids], device=device)
    gen = []
    stop_ids = {tokenizer.eos_id, tokenizer.pad_id}
    id2tok = tokenizer.id2tok
    stochastic_steps = int(HPARAMS.get("stochastic_steps", 0))
    temp_ops, top_p_ops, top_k_ops = float(HPARAMS.get("temperature_ops", 0.7)), float(HPARAMS.get("top_p_ops", 1.0)), int(HPARAMS.get("top_k_ops", 0))

    decoded = []; steps_done, seen_eq_this_step = 0, False
    awaiting_first_digit, operator_sample_pending = False, False
    in_expr = False

    def _multinomial_from_logits(logits_1d: torch.Tensor) -> int:
        probs = torch.softmax(logits_1d / max(1e-6, temp_ops), dim=-1)
        if top_k_ops > 0:
            vals, idxs = torch.topk(probs, k=min(top_k_ops, probs.size(-1)))
            mask = torch.zeros_like(probs).scatter_(-1, idxs, 1.0)
            probs = (probs * mask); probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-9)
        if top_p_ops < 1.0:
            sp, si = torch.sort(probs, descending=True)
            csum = torch.cumsum(sp, dim=-1); cutoff_idx = (csum <= top_p_ops).nonzero(as_tuple=False).flatten()
            cutoff_k = int(cutoff_idx[-1].item()) + 1 if cutoff_idx.numel() > 0 else 1
            keep = si[:cutoff_k]; mask = torch.zeros_like(probs); mask[keep] = 1.0
            probs = probs * mask; probs = probs / probs.sum().clamp_min(1e-9)
        return int(torch.multinomial(probs, num_samples=1).item())

    for _ in range(min(max_out, base_model.max_len - len(prompt_ids))):
        if x.size(1) >= base_model.max_len: break
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            logits = base_model(x)[:, -1, :].squeeze(0).float()

        logits[tokenizer.pad_id] = float("-inf")
        logits[tokenizer.sep_id] = float("-inf")

        # After EXPR, allow only digits, ops, parens (and EOS)
        if in_expr:
            keep = [tokenizer.tok2id[str(d)] for d in range(10)] + \
                   [tokenizer.tok2id[o] for o in ['+','-','*','/','(',')']] + [tokenizer.eos_id] 
            mask = torch.full_like(logits, float("-inf"))
            mask[keep] = 0.0
            logits = logits + mask
            # Ensure PAD is still banned inside EXPR mask
            logits[tokenizer.pad_id] = float("-inf")
        
        if steps_done < stochastic_steps and (awaiting_first_digit or operator_sample_pending):
            nxt = _multinomial_from_logits(logits)
        else:
            nxt = int(logits.argmax(-1).item())
        
        if nxt in stop_ids: break
        gen.append(nxt)
        tok = id2tok.get(nxt, '?')
        decoded.append(tok)

        if tok == 'EXPR': in_expr = True

        if steps_done < stochastic_steps and not awaiting_first_digit and not operator_sample_pending:
            if len(decoded) >= 1 and decoded[-1] == '->': awaiting_first_digit = True
        if awaiting_first_digit and tok.isdigit():
            awaiting_first_digit, operator_sample_pending = False, True
        elif operator_sample_pending and tok in ['+','-','*','/']:
            operator_sample_pending = False
        if tok == '=': seen_eq_this_step = True
        if tok == '->' and seen_eq_this_step and not awaiting_first_digit and not operator_sample_pending:
            steps_done += 1; seen_eq_this_step = False
        
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return gen

def token_count(tok, s: str) -> int:
    spaced = tok._tok_re.sub(r' \1 ', s); count = 1
    for t in spaced.split(): count += len(t) if t.isdigit() and len(t) > 1 else 1
    return count

def setup_ddp():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return local_rank, dist.get_rank(), dist.get_world_size()

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp: local_rank, global_rank, world_size = setup_ddp()
    else: local_rank, global_rank, world_size = 0, 0, 1

    try:
        random.seed(HPARAMS['seed']); torch.manual_seed(HPARAMS['seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(HPARAMS['seed'])
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        
        config = HPARAMS.copy()
        archive_dir = "exp_logs_archive"
        if global_rank == 0:
            os.makedirs(archive_dir, exist_ok=True) 
        base_filename = f"countdown_data_{config['num_samples']}_samples_{config['num_sources']}_K{config.get('max_solutions',1)}_expr_tokenized.pt"
        cache_filename = os.path.join(archive_dir, base_filename)
        if global_rank == 0 and (not config['use_cached_data'] or not os.path.exists(cache_filename)):
            logging.info("Starting data generation...")
            ins, outs = generate_countdown_data(config)
            if not ins: raise RuntimeError("Data generation failed to produce any samples.")
            logging.info(f"Saving generated data to: {cache_filename}")
            torch.save({'ins': ins, 'outs': outs}, cache_filename)
        if is_ddp: dist.barrier()

        logging.info(f"[RANK {global_rank}] Loading data from {cache_filename}")
        data = torch.load(cache_filename, map_location="cpu"); ins, outs = data['ins'], data['outs']
        if not ins: raise RuntimeError("Data file is empty or generation failed.")

        tok = Tokenizer(); config['vocab_size'] = tok.vocab_size
        seqs = [f"{inp} SEP {out}" for inp, out in zip(ins, outs)]

        lens = [token_count(tok, s) for s in seqs]
        max_len_seen = max(lens) if lens else 0
        if max_len_seen > SAFE_MAX_LEN:
            keep_indices = [i for i, L in enumerate(lens) if L <= SAFE_MAX_LEN]
            dropped_count = len(seqs) - len(keep_indices)
            if dropped_count > 0:
                seqs = [seqs[i] for i in keep_indices]; ins  = [ins[i]  for i in keep_indices]; outs = [outs[i] for i in keep_indices]
                if global_rank == 0: logging.warning(f"Dropped {dropped_count} sequences exceeding MAX_SEQ_LEN={SAFE_MAX_LEN} (max seen={max_len_seen}).")
        if not seqs: raise RuntimeError(f"All samples were filtered out by MAX_SEQ_LEN={SAFE_MAX_LEN}.")
        if is_ddp: dist.barrier()

        ml = max(token_count(tok, s) for s in seqs)
        mi = max(token_count(tok, f"{inp} SEP") for inp in ins); mo = ml - mi
        config['max_input_len'], config['max_output_len'] = mi, mo
        
        enc = [tok.encode(s, ml) for s in seqs]; X = torch.tensor(enc, dtype=torch.long)
        assert ((X == tok.sep_id).sum(dim=1) == 1).all().item(), "Each sequence must have exactly one SEP."
        assert X.shape[1] == ml, f"Encoded length {X.shape[1]} must equal configured max_len {ml}."

        ds = TensorDataset(X, X.clone())
        n_train = int(config['train_split_ratio'] * len(ds)); g = torch.Generator().manual_seed(config['seed'])
        tr, va = random_split(ds, [n_train, len(ds) - n_train], generator=g)

        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        
        use_amp   = (device.type == 'cuda')
        use_bf16  = use_amp and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        scaler    = torch.cuda.amp.GradScaler(enabled=(use_amp and not use_bf16))

        loader_num_workers = min(4, cpu_count())
        loader_args = {'batch_size': config['batch_size'], 'num_workers': loader_num_workers, 'pin_memory': device.type=='cuda'}
        train_sampler = DistributedSampler(tr, shuffle=True) if is_ddp else None
        train_loader = DataLoader(tr, sampler=train_sampler, shuffle=(not is_ddp), drop_last=True, persistent_workers=(loader_num_workers > 0), **loader_args)
        val_loader_eval = DataLoader(va, shuffle=False, **loader_args) if global_rank == 0 else None

        model = DecoderOnly(tok.vocab_size, config['d_model'], config['n_heads'], config['n_layers'], ml, tok.pad_id, use_grad_ckpt=config['use_gradient_checkpointing']).to(device)
        if is_ddp: model = nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()], gradient_as_bucket_view=True)
        elif os.environ.get("ALLOW_COMPILE", "0") == "1":
            try: model = torch.compile(model); logging.info("Model compiled successfully.")
            except Exception as e: logging.warning(f"Could not compile model: {e}")

        params = [p for p in model.parameters() if p.requires_grad]
        decay_params = [p for p in params if p.dim() >= 2]
        no_decay_params = [p for p in params if p.dim() < 2]
        optim_groups = [{'params': decay_params, 'weight_decay': config['weight_decay']}, {'params': no_decay_params, 'weight_decay': 0.0}]
        try:
            opt = torch.optim.AdamW(optim_groups, lr=config['lr'], fused=True)
        except TypeError:
            opt = torch.optim.AdamW(optim_groups, lr=config['lr']) 

        if global_rank == 0:
            width=62; mi_print = max(mi - 1, 0)
            log_msg = f"""
{'='*width}
{'Run Configuration (Countdown Pretraining)':^{width}}
{'='*width}
{f'PyTorch Version:':<32} {torch.__version__}
{f'DDP Enabled:':<32} {'Yes' if is_ddp else 'No'}
{f'Total Dataset Size:':<32} {len(ds):,} samples
{f'Vocabulary Size:':<32} {config['vocab_size']} tokens
{f'd_model / n_heads / n_layers:':<32} {config['d_model']} / {config['n_heads']} / {config['n_layers']}
{f'Total Parameters:':<32} {sum(p.numel() for p in model.parameters() if p.requires_grad):,} 
{f'Max Sequence Length:':<32} {ml} (prompt={mi_print}, solution={mo})
{f'Batch Size / LR:':<32} {config['batch_size']} / {config['lr']}
{f'λ last_transition:':<32} {config.get('lambda_transition_ce', 0.0)}
{f'λ final_expression:':<32} {config.get('lambda_final_expr', 0.0)}
{f'λ last_equation:':<32} {config.get('lambda_last_equation', 0.0)}
{f'λ eos_margin:':<32} {config.get('lambda_eos_margin', 0.0)}
{'='*width}"""
            logging.info(log_msg)

        logging.info(f"Starting training on rank {global_rank}...")
        history = defaultdict(list); best_val_loss, patience_counter = float('inf'), 0
        stop_tensor = torch.tensor(0, device=device)
        
        for ep in range(config['epochs']):
            if is_ddp: train_sampler.set_epoch(ep)
            tr_acc, tr_loss = run_epoch(True, model, train_loader, device, tok, opt, scaler, amp_dtype, use_amp)
            val_loss_tensor = torch.tensor(0.0, device=device)
            if global_rank == 0:
                va_acc, va_loss = run_epoch(False, model, val_loader_eval, device, tok, None, None, amp_dtype, use_amp)
                val_loss_tensor.fill_(va_loss)
                history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
                history['train_acc'].append(tr_acc); history['val_acc'].append(va_acc)
            if is_ddp: dist.broadcast(val_loss_tensor, src=0)
            current_val_loss = val_loss_tensor.item()
            if global_rank == 0:
                logging.info(f"Epoch {ep+1:04d} | Train loss {tr_loss:.4f} | Val loss {current_val_loss:.4f}")
                if current_val_loss < best_val_loss:
                    best_val_loss, patience_counter = current_val_loss, 0
                    torch.save({"state": (model.module.state_dict() if is_ddp else model.state_dict()), "config": config}, "best_model.pt")
                    logging.info(f"  -> New best val loss. Checkpoint saved.")
                else:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        logging.info("Stopping early."); stop_tensor.fill_(1)
            if is_ddp: dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item(): break
        if is_ddp: dist.barrier()

        if global_rank == 0:
            plot_training_results(history)
            logging.info("\nLoading best model for final evaluation...")
            ckpt = torch.load("best_model.pt", map_location=device)
            final_model = DecoderOnly(tok.vocab_size, config['d_model'], config['n_heads'], config['n_layers'], ml, tok.pad_id, False).to(device)
            final_model.load_state_dict(ckpt['state'])
            metrics = evaluate_program_metrics(final_model, tok, val_loader_eval, device, config['max_output_len'])
            logging.info(f"\n--- Final Evaluation Metrics ---\n  Program Exact Match:         {metrics['program_exact_match']*100:.2f}%\n  Verified Target-State Acc.:  {metrics['verified_target_state_accuracy']*100:.2f}%\n  Op Validity Rate:            {metrics['op_valid_rate']*100:.2f}%\n  Op State-Consistency Rate:   {metrics['op_state_consistent_rate']*100:.2f}%\n----------------------------------\non {metrics['samples']} validation samples.")
            logging.info("\n--- Inference Examples ---")
            for i in range(min(NUM_INFERENCE_EXAMPLES, len(va))):
                x, _ = va[i]; ids = x.tolist(); sep_idx = ids.index(tok.sep_id)
                prompt_ids = [t for t in ids[:sep_idx+1] if t != tok.pad_id]
                gen_ids = sample_one(final_model, tok, prompt_ids, config['max_output_len'], device, amp_dtype, use_amp)
                logging.info(f"\n--- Example #{i+1} ---\nProblem:      {tok.decode(prompt_ids).replace(' SEP', '')}\nTrue Sol:     {tok.decode([t for t in ids[sep_idx+1:] if t != tok.pad_id])}\nGenerated Sol: {tok.decode(gen_ids)}")
    finally:
        if is_ddp: cleanup_ddp()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[logging.FileHandler("run_countdown_pretraining.log", mode='w'), logging.StreamHandler(sys.stdout)])
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    if int(os.environ.get("RANK", "0")) == 0:
        logging.info(f"\nTotal runtime: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")