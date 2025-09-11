# run_countdown_experiment.py
import math, random, itertools, time, sys, bisect, re, copy, os, contextlib
from collections import defaultdict, deque, Counter
from typing import Dict, List, Optional
from multiprocessing import Pool, cpu_count

# FIX: Set thread env vars *before* importing torch for maximum reliability.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.utils.checkpoint as checkpoint
import datetime
from tqdm import tqdm

# Set matplotlib backend for headless servers to prevent crashes.
import matplotlib
os.environ.setdefault("MPLBACKEND","Agg"); matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION & HYPERPARAMETERS
# =========================
HPARAMS = {
    "num_sources": 6,
    "num_samples": 100000,
    "max_num_vocab": 999,
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
    "use_gradient_checkpointing": False,
    "seed": 42,
}

# ... (Logger, CountdownSolver, data generation functions remain the same) ...
class Logger:
    """A simple utility to tee stdout to a log file."""
    def __init__(self, filename="outputs.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __enter__(self):
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.flush()
            self.log.close()
        finally:
            sys.stdout = self.terminal
class CountdownSolver:
    """Finds a step-by-step solution to a Countdown numbers puzzle using a DFS."""
    def __init__(self):
        self.solution = None

    def solve(self, numbers, target, max_depth=6):
        self.solution = None
        if target in numbers and len(numbers) == 1:
            self.solution = []
            return self.solution
            
        labeled = tuple(sorted([(v, f"n{i+1}") for i, v in enumerate(numbers)]))
        memo = set()
        self._dfs(labeled, target, [], memo, max_depth)
        return self.solution

    def _dfs(self, items, target, steps, memo, max_depth):
        if self.solution is not None: return
        key = tuple(v for v, _ in items)
        if key in memo: return
        memo.add(key)
        if len(items) < 2 or len(steps) >= max_depth: return

        idxs = range(len(items))
        for a_idx, b_idx in itertools.combinations(idxs, 2):
            (av, al), (bv, bl) = items[a_idx], items[b_idx]
            cand = [('+', av + bv)]
            if av > 1 and bv > 1: cand.append(('*', av * bv))
            for op, resv in cand:
                new_steps = steps + [(av, op, bv, resv)]
                rest = list(items); del rest[b_idx]; del rest[a_idx]
                bisect.insort(rest, (resv, "t"))
                if resv == target and len(rest) == 1:
                    self.solution = new_steps
                    return
                self._dfs(tuple(rest), target, new_steps, memo, max_depth)
                if self.solution is not None: return

        for a_idx, b_idx in itertools.permutations(idxs, 2):
            (av, al), (bv, bl) = items[a_idx], items[b_idx]
            cand = []
            if av > bv: cand.append(('-', av - bv))
            if bv >= 1 and av % bv == 0: cand.append(('/', av // bv))
            for op, resv in cand:
                new_steps = steps + [(av, op, bv, resv)]
                rest = list(items); i, j = sorted((a_idx, b_idx), reverse=True)
                del rest[i]; del rest[j]
                bisect.insort(rest, (resv, "t"))
                if resv == target and len(rest) == 1:
                    self.solution = new_steps
                    return
                self._dfs(tuple(rest), target, new_steps, memo, max_depth)
                if self.solution is not None: return

def steps_to_program(initial_numbers: List[int], solution_steps: List, target_value: int, max_num: int) -> Optional[str]:
    """Converts solver steps into the 'Stateful Infix' CoT format, filtering large numbers."""
    if not solution_steps:
        if len(initial_numbers) == 1 and initial_numbers[0] == target_value:
            return f"ANSWER {target_value}"
        return None

    out = []
    current_numbers = Counter(initial_numbers)
    
    for n1, op, n2, res in solution_steps:
        if res > max_num: return None
        
        before_state_str = ' '.join(str(x) for x in sorted(current_numbers.elements()))
        out.append(f"[ {before_state_str} ] ->")
        out.append(f"{n1} {op} {n2} = {res} ->")

        current_numbers[n1] -= 1
        current_numbers[n2] -= 1
        for k in {n1, n2}:
            if current_numbers.get(k, 0) <= 0:
                current_numbers.pop(k, None)
        current_numbers[res] += 1
        
        after_state_str = ' '.join(str(x) for x in sorted(current_numbers.elements()))
        out.append(f"[ {after_state_str} ] ;")

    program_str = " ".join(out)
    if program_str.endswith(';'): program_str = program_str[:-2]
    program_str += f" ; ANSWER {target_value}"
    return program_str

def _gen_chunk(args):
    """Helper function for a single worker process to generate a chunk of data."""
    chunk_samples, num_sources, seed, max_num_for_vocab, small_deck, large_deck, target_range = args
    assert num_sources <= (len(small_deck) + len(large_deck)), "num_sources exceeds the total deck size"
    random.seed(seed)
    solver = CountdownSolver()
    inputs, outputs, solved_steps = [], [], []
    attempts = rejects_solver = rejects_program = 0
    max_attempts = max(8000, 800 * chunk_samples)

    while len(inputs) < chunk_samples and attempts < max_attempts:
        attempts += 1
        
        min_large = max(0, num_sources - len(small_deck))
        max_large = min(len(large_deck), num_sources)
        num_large = random.randint(min_large, max_large)
        num_small = num_sources - num_large
        
        large = random.sample(large_deck, k=num_large)
        small = random.sample(small_deck,  k=num_small)
        source_numbers = large + small
        random.shuffle(source_numbers)
        
        target = random.randint(target_range[0], target_range[1])

        if len(source_numbers) > 1 and target in source_numbers:
            continue

        steps = solver.solve(source_numbers, target, max_depth=num_sources - 1)
        if steps is None: rejects_solver += 1; continue
        program = steps_to_program(source_numbers, steps, target, max_num=max_num_for_vocab)
        if program is None: rejects_program += 1; continue

        solved_steps.append(len(steps))
        in_list = ' '.join(map(str, source_numbers))
        inputs.append(f"IN: [ {in_list} ] TGT: {target}")
        outputs.append(program)

    return inputs, outputs, attempts, rejects_solver, rejects_program, solved_steps

def generate_countdown_data(config, n_workers=None):
    num_samples, num_sources = config['num_samples'], config['num_sources']
    seed, max_num_for_vocab = config['seed'], config['max_num_vocab']
    small_deck, large_deck = config['small_deck'], config['large_deck']
    target_range = config['target_range']
    
    n_workers = min(n_workers or cpu_count(), 64, num_samples)
    granularity = 4
    target_tasks = min(n_workers * granularity, num_samples) if n_workers > 0 else num_samples
    if target_tasks == 0: return [],[]
    per_task = max(1, num_samples // target_tasks)
    extras = num_samples % target_tasks
    tasks = [(per_task + (1 if i < extras else 0), num_sources, seed + 1000 * i, 
              max_num_for_vocab, small_deck, large_deck, target_range) 
              for i in range(target_tasks)]
    
    print(f"Starting parallel data generation for {num_samples:,} samples on {n_workers} workers ({len(tasks)} tasks)...")
    ins, outs, total_attempts, total_rej_s, total_rej_p, steps_all = [], [], 0, 0, 0, []
    with Pool(processes=n_workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        pbar = tqdm(pool.imap_unordered(_gen_chunk, tasks, chunksize=2), total=len(tasks), desc="Generating Sample Batches")
        for ins_chunk, outs_chunk, a, rs, rp, st in pbar:
            ins += ins_chunk; outs += outs_chunk; total_attempts += a; total_rej_s += rs; total_rej_p += rp; steps_all += st
    
    acc_rate = 100.0 * len(ins) / max(1, total_attempts)
    avg_steps = sum(steps_all) / max(1, len(steps_all))
    print(f"Generated {len(ins):,} samples from {total_attempts:,} total attempts ({acc_rate:.1f}% acceptance).")
    print(f"  Total Rejects: solver={total_rej_s:,}, program(filter)={total_rej_p:,}. Avg steps: {avg_steps:.2f}.")
    if len(ins) < num_samples: print(f"Warning: Underfilled. Generated {len(ins):,}/{num_samples:,} samples.")
    return ins[:num_samples], outs[:num_samples]
class Tokenizer:
    """A simple, task-specific tokenizer for the new CoT format."""
    def __init__(self, num_sources, max_num=999):
        self.special = ['PAD', 'EOS', 'SEP', 'IN:', 'TGT:', '[', ']', '=', ';', '->', '+', '-', '*', '/']
        self.cmds = ['ANSWER']
        self.nums = [str(n) for n in range(max_num + 1)]
        vocab = self.special + self.cmds + self.nums
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.id2tok = {i: t for i, t in enumerate(vocab)}
        self._pad=self.tok2id['PAD']; self._eos=self.tok2id['EOS']; self._sep=self.tok2id['SEP']
        self._tok_re = re.compile(r'(->)|(\[|\]|=|;|\+|\-|\*|/)')

    @property
    def pad_id(self): return self._pad
    @property
    def eos_id(self): return self._eos
    @property
    def sep_id(self): return self._sep
    @property
    def vocab_size(self): return len(self.tok2id)

    def encode(self, text, max_len):
        txt = self._tok_re.sub(r' \1\2 ', text)
        toks = txt.split() + ['EOS']
        ids = []
        for tok in toks:
            if tok not in self.tok2id:
                raise ValueError(f"Unknown token during encoding: '{tok}'")
            ids.append(self.tok2id[tok])
        if len(ids) < max_len: ids += [self._pad] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        out = [self.id2tok.get(i, '?') for i in ids if i not in (self._pad, self._eos)]
        return " ".join(out)
def parse_initial_numbers(prompt_text: str) -> Optional[List[int]]:
    match = re.search(r"IN: \[ (.*?) \]", prompt_text)
    if match: return [int(n) for n in match.group(1).split()]
    return None

def get_declared_answer(cot_text: str) -> Optional[int]:
    """Parses a CoT string to extract the final number after the LAST 'ANSWER' token."""
    matches = re.findall(r"ANSWER\s+(\d+)", cot_text)
    return int(matches[-1]) if matches else None

def verify_stateful_cot(initial_numbers: List[int], target: int, cot_text: str) -> bool:
    """Parses and verifies each step of a stateful CoT for legality and correctness."""
    try:
        current_numbers = Counter(initial_numbers)
        steps = re.findall(r"\[\s*(.*?)\s*\]\s*->\s*(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)\s*->\s*\[\s*(.*?)\s*\]", cot_text)
        
        if not steps:
            declared_answer = get_declared_answer(cot_text)
            return (declared_answer is not None and declared_answer == target and 
                    len(initial_numbers) == 1 and initial_numbers[0] == target)

        for before_str, n1_str, op, n2_str, res_str, after_str in steps:
            stated_before = Counter(int(n) for n in before_str.split())
            if stated_before != current_numbers: return False

            n1, n2, res = int(n1_str), int(n2_str), int(res_str)
            ok = False
            if op == '+': ok = (n1 + n2 == res)
            elif op == '*': ok = (n1 > 1 and n2 > 1 and n1 * n2 == res)
            elif op == '-': ok = (n1 > n2 and n1 - n2 == res)
            elif op == '/': ok = (n2 >= 1 and n1 % n2 == 0 and n1 // n2 == res)
            if not ok: return False
            
            current_numbers[n1] -= 1; current_numbers[n2] -= 1
            if current_numbers[n1] == 0: del current_numbers[n1]
            if current_numbers[n2] == 0: del current_numbers[n2]
            current_numbers[res] += 1
            stated_after = Counter(int(n) for n in after_str.split())
            if stated_after != current_numbers: return False
        
        if len(current_numbers) != 1: return False
        final_result = next(iter(current_numbers))
        declared_answer = get_declared_answer(cot_text)
        return (declared_answer is not None and
                final_result == declared_answer and final_result == target)
    except Exception:
        return False
inference_mode = getattr(torch, "inference_mode", torch.no_grad)
@inference_mode()
def evaluate_program_metrics(model, tok, val_loader, device, max_out, max_batches=4):
    """Evaluates the model on three tiers of accuracy."""
    model_to_eval = model.module if isinstance(model, nn.DataParallel) else model
    model_to_eval.eval()
    
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    
    n_prog_em, n_ans_declared_ok, n_ans_verified_ok, n = 0, 0, 0, 0
    it = iter(val_loader)
    for _ in range(max_batches):
        try: x_batch, _ = next(it)
        except StopIteration: break
        for x in x_batch:
            ids = x.tolist();
            if tok.sep_id not in ids: continue
            
            sep_idx = ids.index(tok.sep_id)
            prompt_ids = [t for t in ids[:sep_idx+1] if t != tok.pad_id]
            tgt_text = tok.decode(ids[sep_idx+1:])
            
            prompt_text = tok.decode(prompt_ids).replace(' SEP', '').strip()
            m_tgt = re.search(r"TGT:\s+(\d+)", prompt_text)
            if not m_tgt: continue
            tgt_val = int(m_tgt.group(1))

            gen_ids = sample_one(model_to_eval, tok, prompt_ids, max_out, device)
            if not gen_ids: continue
            gen_text = tok.decode(gen_ids)
            
            n += 1
            if norm(gen_text) == norm(tgt_text): n_prog_em += 1
            
            declared_answer = get_declared_answer(gen_text)
            if declared_answer is not None and declared_answer == tgt_val: n_ans_declared_ok += 1
            
            initial_nums = parse_initial_numbers(prompt_text)
            if initial_nums and verify_stateful_cot(initial_nums, tgt_val, gen_text): n_ans_verified_ok += 1
                
    if n == 0: raise RuntimeError("Evaluation failed: No valid samples were processed.")
                
    return dict(
        samples=n,
        program_exact_match=n_prog_em / n,
        declared_answer_accuracy=n_ans_declared_ok / n,
        verified_answer_accuracy=n_ans_verified_ok / n,
    )

def plot_training_results(history, filename="countdown_training_plot.png"):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_ylabel('Loss'); ax1.set_title('Training and Validation Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'bo-', label='Training Accuracy')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'ro-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Training and Validation Accuracy'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nTraining plot saved to {filename}")

# =========================
# Model & Training
# =========================
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
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderOnly(nn.Module):
    def __init__(self, vocab, d, h, n_layers, max_len, pad_id, use_gradient_checkpointing):
        super().__init__()
        self.pad_id = pad_id
        self.use_gradient_checkpointing = use_gradient_checkpointing
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
                # Wrap the block call in a lambda to handle non-tensor args robustly
                h = checkpoint.checkpoint(
                    lambda h_block, mask: blk(h_block, mask),
                    h, key_padding_mask, use_reentrant=False
                )
            else:
                h = blk(h, key_padding_mask)
        
        return self.ln_f(h)

    def forward(self, x):
        h = self.forward_features(x)
        return self.head(h)

def compute_masked_loss(model_head, hidden_states, targets, sep_id, pad_id, eos_id):
    h_for_pred = hidden_states[:, :-1, :].contiguous()
    t_for_pred = targets[:, 1:].contiguous()
    with torch.no_grad():
        sep_pos = (targets == sep_id).int().argmax(dim=1)
        idx = torch.arange(t_for_pred.size(1), device=targets.device).unsqueeze(0)
        mask = (idx + 1 > sep_pos.unsqueeze(1)) & (t_for_pred != pad_id) & (t_for_pred != eos_id)
    selected_h = h_for_pred[mask]
    selected_t = t_for_pred[mask]
    total_supervised = selected_t.size(0)
    if total_supervised == 0:
        # Return a graph-connected zero scalar loss if no supervised tokens
        return hidden_states.sum() * 0.0, 0, 0
    logits = model_head(selected_h)
    loss = F.cross_entropy(logits, selected_t)
    preds = logits.argmax(dim=-1)
    correct = (preds == selected_t).sum().item()
    return loss, correct, total_supervised

def train_one_epoch(model, opt, loader, device, tok, scaler):
    model.train()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    model_to_call = model.module if isinstance(model, nn.DataParallel) else model
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type=='cuda'):
            hidden_states = model_to_call.forward_features(x)
            loss, correct, n_tokens = compute_masked_loss(
                model_to_call.head, hidden_states, y, tok.sep_id, tok.pad_id, tok.eos_id
            )
        if n_tokens > 0:
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if scaler.is_enabled(): scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total_loss += loss.item() * n_tokens
            total_correct += correct
            total_tokens += n_tokens
    acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    avg_loss = total_loss / max(1, total_tokens)
    return acc, avg_loss

@inference_mode()
def validate(model, loader, device, tok):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    model_to_call = model.module if isinstance(model, nn.DataParallel) else model
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=device.type=='cuda'):
            hidden_states = model_to_call.forward_features(x)
            loss, correct, n_tokens = compute_masked_loss(
                model_to_call.head, hidden_states, y, tok.sep_id, tok.pad_id, tok.eos_id
            )
        if n_tokens > 0:
            total_loss += loss.item() * n_tokens
            total_correct += correct
            total_tokens += n_tokens
    acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    avg_loss = total_loss / max(1, total_tokens)
    return acc, avg_loss

# ... (sample_one and compute_lengths remain the same) ...
@inference_mode()
def sample_one(model, tokenizer, prompt_ids, max_out, device):
    model.eval()
    x = torch.tensor([prompt_ids], device=device)
    gen = []
    stop_ids = {tokenizer.eos_id, tokenizer.pad_id}
    ans_id = tokenizer.tok2id['ANSWER']
    saw_answer = False
    have_digit = False

    for _ in range(min(max_out, model.max_len - len(prompt_ids))):
        if x.size(1) >= model.max_len: break
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=(device.type=='cuda')):
            # The original `forward` method is still used here for simple inference
            logits = model(x)[:, -1, :]
        nxt = logits.argmax(-1).item()
        if nxt in stop_ids: break

        tok_str = tokenizer.id2tok.get(nxt, "")
        gen.append(nxt)

        if nxt == ans_id:
            saw_answer = True
        elif saw_answer:
            if tok_str.isdigit():
                have_digit = True
            elif have_digit:
                gen.pop()
                break
        
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return gen
def compute_lengths(num_sources:int)->tuple[int,int,int]:
    """Calculate max input/output lengths based on problem size to avoid truncation."""
    pre_sep = 5 * num_sources + 10
    post_sep = 28 * (num_sources - 1) + 12
    total = pre_sep + post_sep
    return pre_sep, post_sep, total
# =========================
# Main execution block
# =========================
def main():
    random.seed(HPARAMS['seed']); torch.manual_seed(HPARAMS['seed'])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(HPARAMS['seed'])
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    config = HPARAMS.copy()
    mi, mo, ml = compute_lengths(config['num_sources'])
    config['max_input_len'], config['max_output_len'], max_len = mi, mo, ml

    # ... (Data loading/generation logic remains the same) ...
    t0 = time.time()
    ins, outs = [], []
    if config['use_cached_data']:
        cache_filename = f"countdown_data_{config['num_samples']}_samples_{config['num_sources']}_sources_stateful_max{config['max_num_vocab']}.pt"
        if os.path.exists(cache_filename):
            print(f"\n{'='*60}\n  FOUND CACHE: Loading dataset from '{cache_filename}'\n  ({config['num_samples']:,} samples, {config['num_sources']} sources)\n{'='*60}\n")
            data = torch.load(cache_filename); ins, outs = data['ins'], data['outs']
            try:
                max_seen = max(int(t) for s in (ins+outs) for t in re.findall(r"\d+", s) if t.isdigit())
                if max_seen > config['max_num_vocab']:
                    raise RuntimeError(f"Cache numbers (up to {max_seen}) exceed current vocab limit ({config['max_num_vocab']})")
            except (ValueError, RuntimeError) as e:
                print(f"Error validating cache file: {e}. Regenerating data.")
                ins, outs = [], []
        
        if not ins:
            print(f"Generating new dataset (cache miss or invalid)...")
            ins, outs = generate_countdown_data(config)
            print(f"Saving newly generated dataset to '{cache_filename}' for future runs.")
            torch.save({'ins': ins, 'outs': outs}, cache_filename)
    else:
        print("Caching is disabled. Generating fresh dataset for this run only.")
        ins, outs = generate_countdown_data(config)
    
    if len(ins) < config['num_samples']:
        print(f"\nWarning: Data generation underfilled. Proceeding with {len(ins):,}/{config['num_samples']:,} samples.\n")
        config['num_samples'] = len(ins)
    if not ins: raise RuntimeError("Data generation produced zero valid samples. Aborting.")
    tok = Tokenizer(num_sources=config['num_sources'], max_num=config['max_num_vocab'])
    config['vocab_size'] = tok.vocab_size
    
    seqs = [f"{inp} SEP {out}" for inp, out in zip(ins, outs)]
    enc = [tok.encode(s, max_len) for s in seqs]
    X = torch.tensor(enc, dtype=torch.long)
    
    lens = [len([t for t in e if t != tok.pad_id]) for e in enc]
    print(f"Max actual encoded length: {max(lens)} (configured max_len: {max_len})")
    assert max(lens) <= max_len, "A sequence exceeded max_len, increase it in compute_lengths()"

    assert ((X == tok.sep_id).sum(dim=1) == 1).all().item(), "Each sequence must have exactly one SEP token."
    assert (X == tok.eos_id).any(dim=1).all().item(), "EOS token missing; sequence may be truncated."

    ds = TensorDataset(X, X.clone())
    n_train = int(0.9 * len(ds)); g = torch.Generator().manual_seed(config['seed'])
    tr, va = random_split(ds, [n_train, len(ds) - n_train], generator=g)
    if len(tr) == 0 or len(va) == 0: raise RuntimeError("Empty train or validation set. Increase num_samples.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_args = {'batch_size': config['batch_size'], 'num_workers': min(2, cpu_count())}
    if device.type == 'cuda':
        loader_args.update({'persistent_workers': len(tr) > 5000, 'pin_memory': True})
        try:
            DataLoader(dataset=[], pin_memory_device='cuda')
            loader_args['pin_memory_device'] = 'cuda'
        except TypeError:
            pass
    
    train_loader = DataLoader(tr, shuffle=True, drop_last=True, generator=g, **loader_args)
    val_loader = DataLoader(va, shuffle=False, **loader_args)

    model = DecoderOnly(
        tok.vocab_size, config['d_model'], config['n_heads'],
        config['n_layers'], max_len, tok.pad_id,
        use_gradient_checkpointing=config['use_gradient_checkpointing']
    ).to(device)
    
    n_gpus = torch.cuda.device_count()
    if n_gpus > 1: print(f"Using {n_gpus} GPUs via DataParallel."); model = nn.DataParallel(model)
    if n_gpus <= 1 and os.environ.get("ALLOW_COMPILE", "0") == "1":
        try: model = torch.compile(model); print("Model compiled successfully (PyTorch 2.0+).")
        except Exception: print("Could not compile model (PyTorch < 2.0 or other issue).")

    # ... (Optimizer setup remains the same) ...
    decay_params, no_decay_params, seen = [], [], set()
    for name, p in model.named_parameters():
        pid = id(p)
        if pid in seen: continue
        seen.add(pid)
        if p.dim() < 2 or "bias" in name or "ln" in name.lower(): no_decay_params.append(p)
        else: decay_params.append(p)
    
    optim_groups = [{'params': decay_params, 'weight_decay': config['weight_decay']}, {'params': no_decay_params, 'weight_decay': 0.0}]
    opt = torch.optim.AdamW(optim_groups, lr=config['lr'])
    scaler = torch.cuda.amp.GradScaler(enabled=False)


    print(f"\n{'='*50}\n{'Run Configuration':^50}\n{'='*50}")
    print(f"{'PyTorch Version:':<25} {torch.__version__}\n{'Using GPUs:':<25} {n_gpus}\n{'-'*50}")
    print(f"{'Dataset Size:':<25} {config['num_samples']:,} samples\n{'Problem Type:':<25} {config['num_sources']}-number Countdown\n{'Vocabulary Size:':<25} {config['vocab_size']} tokens\n{'-'*50}")
    print(f"{'Model d_model/n_heads/n_layers:':<25} {config['d_model']}/{config['n_heads']}/{config['n_layers']}\n{'Model Parameters:':<25} {sum(p.numel() for p in model.parameters() if p.requires_grad):,} total\n{'Max Sequence Length:':<25} {max_len} (pre={mi}, post={mo})\n{'-'*50}")
    print(f"{'Epochs:':<25} {config['epochs']} (patience={config['patience']})\n{'Global Batch Size:':<25} {config['batch_size']}\n{'Learning Rate:':<25} {config['lr']}")
    print(f"{'Gradient Checkpointing:':<25} {'Enabled' if config['use_gradient_checkpointing'] else 'Disabled'}\n{'='*50}\n")
    
    sdp_context = (torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
                   if torch.cuda.is_available() else contextlib.nullcontext())
    
    with sdp_context:
        print("Training with SDP backend enabled..." if torch.cuda.is_available() else "Training...")
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
        best_val_loss = float('inf'); patience_counter = 0
        train_start_time = time.time()
        
        for ep in range(config['epochs']):
            tr_acc, tr_loss = train_one_epoch(model, opt, train_loader, device, tok, scaler)
            va_acc, va_loss = validate(model, val_loader, device, tok)
            
            history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
            history['train_acc'].append(tr_acc); history['val_acc'].append(va_acc)
            
            elapsed_time = time.time() - train_start_time
            time_per_epoch = elapsed_time / (ep + 1)
            eta_str = str(datetime.timedelta(seconds=int((config['epochs'] - (ep + 1)) * time_per_epoch)))

            print(f"Epoch {ep+1:04d} | Train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | Val loss {va_loss:.4f} acc {va_acc*100:.2f}% | ETA: {eta_str}")
            
            if va_loss < best_val_loss:
                best_val_loss = va_loss; patience_counter = 0
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                state_to_save = model_to_save._orig_mod.state_dict() if hasattr(model_to_save, '_orig_mod') else model_to_save.state_dict()
                torch.save({ "state": copy.deepcopy(state_to_save), "config": config }, "best_model.pt")
                print(f"  -> New best val loss: {best_val_loss:.4f}. Checkpoint saved.")
            else:
                patience_counter += 1
                if patience_counter >= config['patience']: print(f"Validation loss has not improved for {config['patience']} epochs. Stopping early."); break

        plot_training_results(history)

        print("\nLoading best model for final evaluation...")
        checkpoint = torch.load("best_model.pt", map_location=device)
        final_model = DecoderOnly(
            tok.vocab_size, config['d_model'], config['n_heads'],
            config['n_layers'], max_len, tok.pad_id,
            use_gradient_checkpointing=False
        ).to(device)
        final_model.load_state_dict(checkpoint['state'])
        if torch.cuda.device_count() > 1:
            final_model = nn.DataParallel(final_model)
        
        metrics = evaluate_program_metrics(final_model, tok, val_loader, device, config['max_output_len'], max_batches=16)
        print(f"\n--- Final Evaluation Metrics ---\n"
              f"  Program Exact Match:  {metrics['program_exact_match']*100:.2f}%\n"
              f"  Declared Answer Acc:  {metrics['declared_answer_accuracy']*100:.2f}%\n"
              f"  Verified Answer Acc:  {metrics['verified_answer_accuracy']*100:.2f}%\n"
              f"----------------------------------\n"
              f"on {metrics['samples']} validation samples.")

        print("\n--- Inference Examples ---")
        val_iter = iter(val_loader)
        x_batch, _ = next(val_iter)
        for i in range(min(10, x_batch.size(0))):
            print(f"\n--- Example #{i+1} ---")
            ids = x_batch[i].tolist()
            sep_idx = ids.index(tok.sep_id)
            prompt_ids = [t for t in ids[:sep_idx+1] if t != tok.pad_id]
            gen_ids = sample_one(final_model, tok, prompt_ids, config['max_output_len'], device)
            
            print("Problem:    ", tok.decode(prompt_ids).replace(' SEP', ''))
            print("True Sol:   ", tok.decode([t for t in ids[sep_idx+1:] if t != tok.pad_id]))
            print("Generated:  ", tok.decode(gen_ids))

    print(f"\nTotal runtime: {str(datetime.timedelta(seconds=int(time.time()-t0)))}")

if __name__ == "__main__":
    with Logger('run_countdown_experiment.log'):
        main()