
```python
# run_countdown_experiment.py
import math, random, itertools, time, sys, bisect, re, copy, os, contextlib
from collections import defaultdict, Counter
from typing import Dict, List, Optional
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

# =========================
# Logging Utility
# =========================
class Logger:
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

# =========================
# Countdown Solver & Data Formatting
# =========================
class CountdownSolver:
    def __init__(self):
        self.solution = None

    def solve(self, numbers, target, max_depth=6):
        self.solution = None
        if target in numbers:
            self.solution = []
            return self.solution
            
        labeled = tuple(sorted([(v, f"n{i+1}") for i, v in enumerate(numbers)]))
        memo = set()
        self._dfs(labeled, target, [], memo, max_depth)
        return self.solution

    def _dfs(self, items, target, steps, memo, max_depth):
        if self.solution is not None and len(steps) >= len(self.solution): return
        key = tuple(v for v, _ in items)
        if key in memo: return
        memo.add(key)
        if len(items) < 2 or len(steps) >= max_depth: return

        for a_idx, b_idx in itertools.combinations(range(len(items)), 2):
            (av, al), (bv, bl) = items[a_idx], items[b_idx]
            cand = [('+', av + bv)]
            if av > 1 and bv > 1: cand.append(('*', av * bv))
            for op, resv in cand:
                new_steps = steps + [(av, op, bv, resv)]
                rest = list(items); del rest[b_idx]; del rest[a_idx]
                bisect.insort(rest, (resv, "t"))
                if resv == target and len(rest) == 1:
                    if self.solution is None or len(new_steps) < len(self.solution):
                        self.solution = new_steps
                else:
                    self._dfs(tuple(rest), target, new_steps, memo, max_depth)

        for a_idx, b_idx in itertools.permutations(range(len(items)), 2):
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
                    if self.solution is None or len(new_steps) < len(self.solution):
                        self.solution = new_steps
                else:
                    self._dfs(tuple(rest), target, new_steps, memo, max_depth)

def steps_to_program(initial_numbers: List[int], solution_steps: List, max_num: int) -> Optional[str]:
    if solution_steps is None: return None
    if not solution_steps:
        return "ANSWER"

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
    program_str = re.sub(r"\s*;\s*$", "", program_str)
    program_str += " ; ANSWER"
    return program_str

def _gen_chunk(args):
    chunk_samples, num_sources, seed, max_num_for_vocab, small_deck, large_deck, target_range = args
    random.seed(seed)
    solver = CountdownSolver()
    inputs, outputs = [], []
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
        
        program = steps_to_program(source_numbers, steps, max_num=max_num_for_vocab)
        if program is None: rejects_program += 1; continue

        in_list = ' '.join(map(str, source_numbers))
        inputs.append(f"IN: [ {in_list} ] TGT: {target}")
        outputs.append(program)

    return inputs, outputs, attempts, rejects_solver, rejects_program

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
    ins, outs, total_attempts, total_rej_s, total_rej_p = [], [], 0, 0, 0
    with Pool(processes=n_workers, initializer=torch.set_num_threads, initargs=(1,)) as pool:
        pbar = tqdm(pool.imap_unordered(_gen_chunk, tasks, chunksize=2), total=len(tasks), desc="Generating Sample Batches")
        for ins_chunk, outs_chunk, a, rs, rp in pbar:
            ins += ins_chunk; outs += outs_chunk; total_attempts += a; total_rej_s += rs; total_rej_p += rp
    
    acc_rate = 100.0 * len(ins) / max(1, total_attempts)
    print(f"Generated {len(ins):,} samples from {total_attempts:,} total attempts ({acc_rate:.1f}% acceptance).")
    print(f"  Total Rejects: solver={total_rej_s:,}, program(filter)={total_rej_p:,}.")
    if len(ins) < num_samples: print(f"Warning: Underfilled. Generated {len(ins):,}/{num_samples:,} samples.")
    return ins[:num_samples], outs[:num_samples]

# =========================
# Tokenizer & Parsing
# =========================
class Tokenizer:
    def __init__(self, max_num=999):
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
        ids = [self.tok2id.get(tok) for tok in toks]
        if any(i is None for i in ids): raise ValueError(f"Unknown token in text: '{text}'")
        if len(ids) < max_len: ids += [self._pad] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        out = [self.id2tok.get(i, '?') for i in ids if i not in (self._pad, self._eos)]
        return " ".join(out)

def parse_initial_numbers(prompt_text: str) -> Optional[List[int]]:
    match = re.search(r"IN: \[ (.*?) \]", prompt_text)
    if match: return [int(n) for n in match.group(1).split()]
    return None

def verify_stateful_cot(initial_numbers: List[int], target: int, cot_text: str) -> bool:
    try:
        current_numbers = Counter(initial_numbers)
        if not cot_text.strip().endswith("ANSWER"): return False
        
        if "->" not in cot_text:
            return len(initial_numbers) == 1 and initial_numbers[0] == target

        steps = re.findall(r"\[\s*(.*?)\s*\]\s*->\s*(\d+)\s*([\+\-\*\/])\s*(\d+)\s*=\s*(\d+)\s*->\s*\[\s*(.*?)\s*\]", cot_text)
        
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
            
            current_numbers[n1] -= 1
            if current_numbers[n1] == 0: del current_numbers[n1]
            current_numbers[n2] -= 1
            if current_numbers.get(n2) == 0: del current_numbers[n2]
            current_numbers[res] += 1

            stated_after = Counter(int(n) for n in after_str.split())
            if stated_after != current_numbers: return False
        
        if len(current_numbers) != 1: return False
        final_result = next(iter(current_numbers))
        return final_result == target
    except Exception:
        return False

# =========================
# Evaluation, Model & Training
# =========================
inference_mode = getattr(torch, "inference_mode", torch.no_grad)

@inference_mode()
def evaluate_program_metrics(model, tok, val_loader, device, max_out, max_batches=4):
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()
    norm = lambda s: re.sub(r"\s+", " ", s).strip()
    
    n_prog_em, n_ans_verified_ok, n = 0, 0, 0
    it = iter(val_loader)
    for _ in range(max_batches):
        try: x_batch, _ = next(it)
        except StopIteration: break
        for x in x_batch:
            ids = x.tolist()
            if tok.sep_id not in ids: continue
            
            sep_idx = ids.index(tok.sep_id)
            prompt_ids = [t for t in ids[:sep_idx+1] if t != tok.pad_id]
            tgt_text = tok.decode(ids[sep_idx+1:])
            prompt_text = tok.decode(prompt_ids).replace(' SEP', '').strip()
            
            m_tgt = re.search(r"TGT:\s+(\d+)", prompt_text)
            if not m_tgt: continue
            tgt_val = int(m_tgt.group(1))

            gen_ids = sample_one(model, tok, prompt_ids, max_out, device)
            if not gen_ids: continue
            gen_text = tok.decode(gen_ids)
            
            n += 1
            if norm(gen_text) == norm(tgt_text): n_prog_em += 1
            
            initial_nums = parse_initial_numbers(prompt_text)
            if initial_nums and verify_stateful_cot(initial_nums, tgt_val, gen_text):
                n_ans_verified_ok += 1
                
    if n == 0: raise RuntimeError("Evaluation failed: No valid samples were processed.")
                
    return dict(
        samples=n,
        program_exact_match=n_prog_em / n,
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
    def __init__(self, vocab, d, h, n_layers, max_len, pad_id, use_grad_ckpt):
        super().__init__()
        self.pad_id = pad_id
        self.use_gradient_checkpointing = use_grad_ckpt
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
                mask_device = key_padding_mask.to(h.device, non_blocking=True)
                h = checkpoint.checkpoint(
                    lambda h_block, mask: blk(h_block, mask),
                    h, mask_device, use_reentrant=False,
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
        return hidden_states.sum() * 0.0, 0, 0
    logits = model_head(selected_h)
    loss = F.cross_entropy(logits, selected_t)
    preds = logits.argmax(dim=-1)
    correct = (preds == selected_t).sum().item()
    return loss, correct, total_supervised

def train_one_epoch(model, opt, loader, device, tok, scaler):
    model.train()
    batch_losses, total_correct, total_tokens = [], 0, 0
    model_to_call = model.module if hasattr(model, "module") else model
    use_amp = device.type=='cuda' and torch.cuda.is_bf16_supported()
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_amp):
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
            batch_losses.append(loss.item())
            total_correct += correct
            total_tokens += n_tokens
    acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    avg_loss = sum(batch_losses) / max(1, len(batch_losses))
    return acc, avg_loss

@inference_mode()
def validate(model, loader, device, tok):
    model.eval()
    batch_losses, total_correct, total_tokens = [], 0, 0
    model_to_call = model.module if hasattr(model, "module") else model
    use_amp = device.type=='cuda' and torch.cuda.is_bf16_supported()
    
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_amp):
            hidden_states = model_to_call.forward_features(x)
            loss, correct, n_tokens = compute_masked_loss(
                model_to_call.head, hidden_states, y, tok.sep_id, tok.pad_id, tok.eos_id
            )
        if n_tokens > 0:
            batch_losses.append(loss.item())
            total_correct += correct
            total_tokens += n_tokens
    acc = (total_correct / total_tokens) if total_tokens > 0 else 0.0
    avg_loss = sum(batch_losses) / max(1, len(batch_losses))
    return acc, avg_loss

@inference_mode()
def sample_one(model, tokenizer, prompt_ids, max_out, device):
    base_model = model.module if hasattr(model, "module") else model
    base_model.eval()
    
    x = torch.tensor([prompt_ids], device=device)
    gen = []
    stop_ids = {tokenizer.eos_id, tokenizer.pad_id}
    ans_id = tokenizer.tok2id['ANSWER']
    use_amp = device.type=='cuda' and torch.cuda.is_bf16_supported()

    for _ in range(min(max_out, base_model.max_len - len(prompt_ids))):
        if x.size(1) >= base_model.max_len: break
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_amp):
            logits = base_model(x)[:, -1, :]
        nxt = logits.argmax(-1).item()
        if nxt in stop_ids: break
        gen.append(nxt)
        if nxt == ans_id:
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return gen

def compute_lengths(num_sources:int)->tuple[int,int,int]:
    pre_sep = 5 * num_sources + 10
    post_sep = 28 * (num_sources - 1) + 12
    total = pre_sep + post_sep
    return pre_sep, post_sep, total

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    return local_rank, global_rank, world_size

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    is_ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if is_ddp:
        local_rank, global_rank, world_size = setup_ddp()
    else:
        local_rank, global_rank, world_size = 0, 0, 1

    try:
        random.seed(HPARAMS['seed']); torch.manual_seed(HPARAMS['seed'])
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(HPARAMS['seed'])
        torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
        
        config = HPARAMS.copy()
        mi, mo, ml = compute_lengths(config['num_sources'])
        config['max_input_len'], config['max_output_len'] = mi, mo
        
        ins, outs = [], []
        if global_rank == 0:
            cache_filename = f"countdown_data_{config['num_samples']}_samples_{config['num_sources']}_sources_stateful_max{config['max_num_vocab']}_reasononly.pt"
            tmp_filename = cache_filename + ".tmp.pt"
            if os.path.exists(cache_filename) and config['use_cached_data']:
                print(f"[RANK 0] Loading data from cache: {cache_filename}")
                data = torch.load(cache_filename, map_location="cpu"); ins, outs = data['ins'], data['outs']
            else:
                print(f"[RANK 0] Generating new dataset...")
                ins, outs = generate_countdown_data(config)
                save_path = cache_filename if config['use_cached_data'] else tmp_filename
                print(f"[RANK 0] Saving data to: {save_path}")
                torch.save({'ins': ins, 'outs': outs}, save_path)

        if is_ddp: dist.barrier()

        if global_rank != 0:
            cache_filename = f"countdown_data_{config['num_samples']}_samples_{config['num_sources']}_sources_stateful_max{config['max_num_vocab']}_reasononly.pt"
            tmp_filename = cache_filename + ".tmp.pt"
            load_path = cache_filename if os.path.exists(cache_filename) and config['use_cached_data'] else tmp_filename
            data = torch.load(load_path, map_location="cpu"); ins, outs = data['ins'], data['outs']

        if is_ddp: dist.barrier()  # ensure all ranks finished loading the tmp/cache file
        if global_rank == 0 and not config['use_cached_data']:
            try: os.remove(tmp_filename)
            except OSError: pass

        if not ins: raise RuntimeError("Data generation failed or cache is empty.")
        
        tok = Tokenizer(max_num=config['max_num_vocab'])
        config['vocab_size'] = tok.vocab_size
        
        seqs = [f"{inp} SEP {out}" for inp, out in zip(ins, outs)]
        enc = [tok.encode(s, ml) for s in seqs]
        X = torch.tensor(enc, dtype=torch.long)
        
        assert ((X == tok.sep_id).sum(dim=1) == 1).all().item(), "Each sequence must have exactly one SEP."
        assert X.shape[1] == ml, f"Encoded length {X.shape[1]} must equal configured max_len {ml}."

        ds = TensorDataset(X, X.clone())
        n_train = int(0.9 * len(ds)); g = torch.Generator().manual_seed(config['seed'])
        tr, va = random_split(ds, [n_train, len(ds) - n_train], generator=g)

        device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
        loader_args = {'batch_size': config['batch_size'], 'num_workers': min(2, cpu_count()), 'pin_memory': device.type=='cuda'}
        
        train_sampler = DistributedSampler(tr, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True) if is_ddp else None
        train_loader = DataLoader(tr, sampler=train_sampler, shuffle=(train_sampler is None),
                                  drop_last=True, persistent_workers=True if loader_args["num_workers"] > 0 else False,
                                  **loader_args)
        
        if global_rank == 0:
            val_loader_eval = DataLoader(va, batch_size=config['batch_size'], shuffle=False, **loader_args)

        model = DecoderOnly(
            tok.vocab_size, config['d_model'], config['n_heads'], 
            config['n_layers'], ml, tok.pad_id, 
            use_grad_ckpt=config['use_gradient_checkpointing']
        ).to(device)
        
        if is_ddp:
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[torch.cuda.current_device()], gradient_as_bucket_view=True
            )        elif os.environ.get("ALLOW_COMPILE", "0") == "1":
            try: model = torch.compile(model); print("Model compiled successfully.")
            except Exception: print("Could not compile model.")

        decay_params, no_decay_params, seen = [], [], set()
        for name, p in model.named_parameters():
            if id(p) not in seen:
                if p.dim() < 2 or "bias" in name or "ln" in name.lower(): no_decay_params.append(p)
                else: decay_params.append(p)
                seen.add(id(p))
        
        optim_groups = [{'params': decay_params, 'weight_decay': config['weight_decay']}, {'params': no_decay_params, 'weight_decay': 0.0}]
        opt = torch.optim.AdamW(optim_groups, lr=config['lr'])
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        if global_rank == 0:
            width=62
            n_gpus = torch.cuda.device_count()
            print(f"\n{'='*width}\n{'Run Configuration':^{width}}\n{'='*width}")
            # ... (full printout here, as before) ...
            print(f"{'='*width}\n")

        if global_rank == 0: print(f"Starting training with {'DDP' if is_ddp else 'a single GPU'}...")
        history = defaultdict(list)
        best_val_loss = float('inf'); patience_counter = 0
        stop_tensor = torch.tensor(0, device=device)
        
        for ep in range(config['epochs']):
            if is_ddp: train_sampler.set_epoch(ep)
            tr_acc, tr_loss = train_one_epoch(model, opt, train_loader, device, tok, scaler)
            
            val_loss_tensor = torch.tensor(0.0, device=device)
            if global_rank == 0:
                va_acc, va_loss = validate(model, val_loader_eval, device, tok)
                val_loss_tensor.fill_(va_loss)
                history['train_loss'].append(tr_loss); history['val_loss'].append(va_loss)
                history['train_acc'].append(tr_acc); history['val_acc'].append(va_acc)
            
            if is_ddp: dist.broadcast(val_loss_tensor, src=0)
            current_val_loss = val_loss_tensor.item()
            
            if global_rank == 0:
                print(f"Epoch {ep+1:04d} | Train loss {tr_loss:.4f} | Val loss {current_val_loss:.4f}")
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss; patience_counter = 0
                    state_to_save = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                    torch.save({ "state": state_to_save, "config": config }, "best_model.pt")
                    print(f"  -> New best val loss. Checkpoint saved.")
                else:
                    patience_counter += 1
                    if patience_counter >= config['patience']:
                        print("Stopping early.")
                        stop_tensor.fill_(1)
            
            if is_ddp:
                dist.broadcast(stop_tensor, src=0)
            if stop_tensor.item():
                break
        
        if is_ddp: dist.barrier()

        if global_rank == 0:
            plot_training_results(history)
            print("\nLoading best model for final evaluation...")
            ckpt = torch.load("best_model.pt", map_location=device)
            final_model = DecoderOnly(tok.vocab_size, config['d_model'], config['n_heads'], config['n_layers'], ml, tok.pad_id, False).to(device)
            final_model.load_state_dict(ckpt['state'])
            
            metrics = evaluate_program_metrics(final_model, tok, val_loader_eval, device, config['max_output_len'])
            print(f"\n--- Final Evaluation Metrics ---\n"
                  f"  Program Exact Match:  {metrics['program_exact_match']*100:.2f}%\n"
                  f"  Verified Answer Acc:  {metrics['verified_answer_accuracy']*100:.2f}%\n"
                  f"----------------------------------\n"
                  f"on {metrics['samples']} validation samples.")

            print("\n--- Inference Examples ---")
            for i in range(min(5, len(va))):
                x, _ = va[i]
                ids = x.tolist()
                sep_idx = ids.index(tok.sep_id)
                prompt_ids = [t for t in ids[:sep_idx+1] if t != tok.pad_id]
                gen_ids = sample_one(final_model, tok, prompt_ids, config['max_output_len'], device)
                print("\n--- Example #{} ---".format(i+1))
                print("Problem:    ", tok.decode(prompt_ids).replace(' SEP', ''))
                print("True Sol:   ", tok.decode([t for t in ids[sep_idx+1:] if t != tok.pad_id]))
                print("Generated:  ", tok.decode(gen_ids))

    finally:
        if is_ddp:
            cleanup_ddp()

if __name__ == "__main__":
    main()