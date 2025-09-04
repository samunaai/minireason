# run_countdown_experiment.py
import math, random, itertools, time, sys, bisect, re, copy, os
from collections import defaultdict, deque
from typing import Dict
from multiprocessing import Pool, cpu_count

# Set thread env vars *before* importing torch for maximum reliability.
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import datetime
from tqdm import tqdm # For progress bars

# Set matplotlib backend for headless servers to prevent crashes.
import matplotlib
os.environ.setdefault("MPLBACKEND","Agg"); matplotlib.use(os.environ["MPLBACKEND"])
import matplotlib.pyplot as plt

# =========================
# Logging Utility
# =========================
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
        sys.stdout = self.terminal
        self.log.close()

# =========================
# Countdown Solver (Labeled & Optimized)
# =========================
class CountdownSolver:
    """Finds a step-by-step solution to a Countdown numbers puzzle using a DFS."""
    def __init__(self):
        self.solution = None

    def solve(self, numbers, target, max_depth=6):
        self.solution = None
        if target in numbers: return None
        labeled = tuple(sorted([(v, f"n{i+1}") for i, v in enumerate(numbers)]))
        memo = set()
        self._dfs(labeled, target, [], memo, max_depth, t_counter=1)
        return self.solution

    def _dfs(self, items, target, steps, memo, max_depth, t_counter):
        if self.solution is not None: return
        key = tuple(v for v, _ in items)
        if key in memo: return
        memo.add(key)
        if len(items) < 2 or len(steps) >= max_depth: return

        idxs = range(len(items))
        # Commutative ops: +, *
        for a_idx, b_idx in itertools.combinations(idxs, 2):
            (av, al), (bv, bl) = items[a_idx], items[b_idx]
            cand = [('+', av + bv)]
            if av > 1 and bv > 1: cand.append(('*', av * bv))
            for op, resv in cand:
                lr = f"t{t_counter}"
                new_steps = steps + [(al, op, bl, lr, resv)]
                if resv == target: self.solution = new_steps; return
                rest = list(items); del rest[b_idx]; del rest[a_idx]
                bisect.insort(rest, (resv, lr))
                self._dfs(tuple(rest), target, new_steps, memo, max_depth, t_counter + 1)
                if self.solution is not None: return

        # Non-commutative ops: -, /
        for a_idx, b_idx in itertools.permutations(idxs, 2):
            (av, al), (bv, bl) = items[a_idx], items[b_idx]
            cand = []
            if av > bv: cand.append(('-', av - bv))
            if bv > 1 and av % bv == 0: cand.append(('/', av // bv))
            for op, resv in cand:
                lr = f"t{t_counter}"
                new_steps = steps + [(al, op, bl, lr, resv)]
                if resv == target: self.solution = new_steps; return
                rest = list(items); i, j = sorted((a_idx, b_idx), reverse=True)
                del rest[i]; del rest[j]
                bisect.insort(rest, (resv, lr))
                self._dfs(tuple(rest), target, new_steps, memo, max_depth, t_counter + 1)
                if self.solution is not None: return

def steps_to_program(solution_steps, target_value):
    """Converts solver steps into a simple, linear program string."""
    op_map = {'+': 'ADD', '-': 'SUB', '*': 'MUL', '/': 'DIV'}
    out, target_label = [], None
    for la, op, lb, lr, resv in solution_steps:
        out.append(f"WRITE {lr} {op_map[op]} {la} {lb}")
        if resv == target_value:
            target_label = lr
    if target_label:
        out.append(f"ANSWER {target_label}")
        return " ".join(out)
    return None

def _gen_chunk(args):
    """Helper function for a single worker process to generate a chunk of data."""
    chunk_samples, num_sources, seed = args
    random.seed(seed)
    solver = CountdownSolver()
    inputs, outputs, solved_steps = [], [], []
    attempts = rejects_solver = rejects_program = 0
    max_attempts = max(4000, 400 * chunk_samples)

    while len(inputs) < chunk_samples and attempts < max_attempts:
        attempts += 1
        small = [random.randint(1, 10) for _ in range(num_sources - 2)]
        large = [random.choice([25, 50, 75, 100]) for _ in range(2)]
        source_numbers = small + large
        random.shuffle(source_numbers)
        target = random.randint(101, 999)

        steps = solver.solve(source_numbers, target, max_depth=num_sources - 1)
        if steps is None: rejects_solver += 1; continue
        program = steps_to_program(steps, target)
        if program is None: rejects_program += 1; continue

        solved_steps.append(len(steps))
        map_str = " ".join([f"n{i+1}={v}" for i, v in enumerate(source_numbers)])
        inputs.append(f"IN: {source_numbers} TGT: {target} MAP: {map_str}")
        outputs.append(program)

    return inputs, outputs, attempts, rejects_solver, rejects_program, solved_steps

def generate_countdown_data(num_samples, num_sources, seed=42, n_workers=None):
    """Generates a dataset in parallel with a progress bar."""
    n_workers = n_workers or min(cpu_count(), 32)
    
    per_worker = num_samples // n_workers
    extras = num_samples % n_workers
    tasks = []
    for i in range(n_workers):
        chunk_size = per_worker + (1 if i < extras else 0)
        tasks.append((chunk_size, num_sources, seed + 1000 * i))

    print(f"Starting parallel data generation for {num_samples:,} samples on {n_workers} workers...")
    
    ins, outs, total_attempts, total_rej_s, total_rej_p, steps_all = [], [], 0, 0, 0, []
    with Pool(processes=n_workers) as pool:
        # Use imap_unordered and tqdm for a real-time progress bar.
        pbar = tqdm(pool.imap_unordered(_gen_chunk, tasks), total=len(tasks), desc="Generating Chunks")
        for i, o, a, rs, rp, st in pbar:
            ins += i; outs += o; total_attempts += a; total_rej_s += rs; total_rej_p += rp; steps_all += st

    acc_rate = 100.0 * len(ins) / max(1, total_attempts)
    avg_steps = sum(steps_all) / max(1, len(steps_all))
    print(f"Generated {len(ins):,} samples from {total_attempts:,} total attempts ({acc_rate:.1f}% acceptance).")
    print(f"  Total Rejects: solver={total_rej_s:,}, program={total_rej_p:,}. Avg steps: {avg_steps:.2f}.")
    
    if len(ins) < num_samples:
        print(f"Warning: Underfilled. Generated {len(ins):,}/{num_samples:,} samples.")
    return ins[:num_samples], outs[:num_samples]


# =========================
# Tokenizer
# =========================
class Tokenizer:
    """A simple, task-specific tokenizer."""
    def __init__(self, num_sources, max_num=1000):
        self.special = ['PAD','EOS','SEP','IN:','TGT:','[',']',',','MAP:','=']
        self.ops = ['ADD', 'SUB', 'MUL', 'DIV']
        self.cmds = ['WRITE', 'ANSWER']
        self.nums = [str(n) for n in range(max_num + 1)]
        self.n_vars = [f"n{i}" for i in range(1, num_sources + 1)]
        self.t_vars = [f"t{i}" for i in range(1, num_sources)]
        vocab = self.special + self.ops + self.cmds + self.nums + self.n_vars + self.t_vars
        self.tok2id = {t: i for i, t in enumerate(vocab)}
        self.id2tok = {i: t for i, t in enumerate(vocab)}
        self._pad=self.tok2id['PAD']; self._eos=self.tok2id['EOS']; self._sep=self.tok2id['SEP']

    @property
    def pad_id(self): return self._pad
    @property
    def eos_id(self): return self._eos
    @property
    def sep_id(self): return self._sep
    @property
    def vocab_size(self): return len(self.tok2id)

    def encode(self, text, max_len):
        txt = text.replace('[',' [ ').replace(']',' ] ').replace(',',' , ').replace('=',' = ')
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

# =========================
# Execution & Plotting Helpers
# =========================
def parse_prompt_meta(prompt_text:str):
    """Extracts the target value and number mappings from a prompt string."""
    m_tgt = re.search(r"\bTGT:\s+(\d+)\b", prompt_text)
    tgt = int(m_tgt.group(1)) if m_tgt else None
    env = {}
    for k, v in re.findall(r"(n\d+)\s*=\s*(\d+)", prompt_text):
        env[k] = int(v)
    return tgt, env

def exec_program(prog_text:str, env:Dict[str,int]):
    """A simple interpreter to execute the generated program and return the final answer."""
    if not prog_text: return None
    toks = prog_text.strip().split()
    i = 0
    vals = dict(env)
    def val(tok):
        return vals.get(tok) if tok.startswith(('n','t')) else None
    
    while i < len(toks):
        cmd = toks[i]
        if cmd == 'WRITE' and i + 4 < len(toks):
            tvar, op, a, b = toks[i+1], toks[i+2], toks[i+3], toks[i+4]
            va, vb = val(a), val(b)
            if va is None or vb is None: return None
            if op == 'ADD': vals[tvar] = va + vb
            elif op == 'SUB': vals[tvar] = va - vb
            elif op == 'MUL': vals[tvar] = va * vb
            elif op == 'DIV':
                if vb == 0 or va % vb != 0: return None
                vals[tvar] = va // vb
            else: return None
            i += 5
        elif cmd == 'ANSWER' and i + 1 < len(toks):
            return val(toks[i+1])
        else:
            return None
    return None

inference_mode = getattr(torch, "inference_mode", torch.no_grad)

@inference_mode()
def evaluate_program_metrics(model, tok, val_loader, device, max_out, max_batches=4):
    """Evaluates the model on Program Exact Match and Answer Accuracy."""
    model_to_eval = model.module if isinstance(model, nn.DataParallel) else model
    model_to_eval.eval()
    
    n_prog_em, n_ans_ok, n = 0, 0, 0
    bad_gold_prints = 0
    it = iter(val_loader)
    for _ in range(max_batches):
        try:
            x_batch, _ = next(it)
        except StopIteration:
            break
        for x in x_batch:
            ids = x.tolist()
            if tok.sep_id not in ids:
                raise RuntimeError("SEP token missing from a validation sample, aborting evaluation.")
            
            sep_idx = ids.index(tok.sep_id)
            prompt_ids = [t for t in ids[:sep_idx+1] if t != tok.pad_id]
            tgt_text = tok.decode(ids[sep_idx+1:])
            
            gen_ids = sample_one(model_to_eval, tok, prompt_ids, max_out, device)
            if not gen_ids:
                continue
            gen_text = tok.decode(gen_ids)
            
            prompt_text = tok.decode(prompt_ids).replace(' SEP', '').strip()
            tgt_val, env = parse_prompt_meta(prompt_text)
            if tgt_val is None or not env: continue
            
            if exec_program(tgt_text, env) != tgt_val:
                if bad_gold_prints < 10:
                    print(f"\nWarning: Gold program failed validation for target {tgt_val}.\n  Prog: {tgt_text}")
                    bad_gold_prints += 1
                continue

            ans_gen = exec_program(gen_text, env)
            n += 1
            if gen_text.strip() == tgt_text.strip():
                n_prog_em += 1
            if ans_gen is not None and ans_gen == tgt_val:
                n_ans_ok += 1
                
    if n == 0:
        raise RuntimeError("Evaluation failed: No valid samples were processed. Check gold program validation.")
                
    return dict(
        samples=n,
        program_em=n_prog_em / n,
        answer_acc=n_ans_ok / n,
    )

def plot_training_results(history, filename="countdown_training_plot.png"):
    """Plots training and validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, [a * 100 for a in history['train_acc']], 'bo-', label='Training Accuracy')
    ax2.plot(epochs, [a * 100 for a in history['val_acc']], 'ro-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nTraining plot saved to {filename}")

# =========================
# Model & Training
# =========================
class Block(nn.Module):
    """A single transformer block."""
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))

    def forward(self, x, key_padding_mask):
        L = x.size(1)
        x1 = self.ln1(x)
        
        causal_mask = torch.triu(torch.ones((L, L), device=x.device, dtype=torch.bool), diagonal=1)
        
        attn_output = self.attn(x1, x1, x1, attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False)
        y = attn_output[0] if isinstance(attn_output, tuple) else attn_output
        
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x

class DecoderOnly(nn.Module):
    """A decoder-only transformer model with tied embeddings."""
    def __init__(self, vocab, d, h, n_layers, max_len, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab, d, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.tok.weight
        self.max_len = max_len

    def forward(self, x):
        B, L = x.shape
        if L > self.max_len:
            x = x[:, -self.max_len:]
            L = self.max_len
        key_padding_mask = (x == self.pad_id)
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        h = h.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        for blk in self.blocks:
            h = blk(h, key_padding_mask)
        return self.head(self.ln_f(h))

def masked_next_token_ce(logits, targets, sep_id, pad_id):
    """Computes cross-entropy loss only on the tokens after the SEP token."""
    B, L, V = logits.shape
    nxt_logits = logits[:, :-1, :].contiguous()
    nxt_tgts = targets[:, 1:].contiguous()
    with torch.no_grad():
        sep_pos = (targets == sep_id).int().argmax(dim=1)
        idx = torch.arange(L - 1, device=targets.device).unsqueeze(0)
        mask = (idx + 1 > sep_pos.unsqueeze(1)) & (nxt_tgts != pad_id)
    loss = F.cross_entropy(nxt_logits.reshape(-1, V), nxt_tgts.reshape(-1), reduction='none').view(B, L - 1)
    return (loss * mask).sum() / mask.sum().clamp_min(1)

def train_one_epoch(model, opt, loader, device, sep_id, pad_id):
    model.train()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = masked_next_token_ce(logits, y, sep_id, pad_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        tot_loss += loss.item()
        preds = logits[:, :-1, :].argmax(-1)
        tgt = y[:, 1:]
        with torch.no_grad():
            sep_pos = (y == sep_id).int().argmax(dim=1)
            idx = torch.arange(tgt.size(1), device=device).unsqueeze(0)
            mask = (idx + 1 > sep_pos.unsqueeze(1)) & (tgt != pad_id)
            tot_correct += (preds[mask] == tgt[mask]).sum().item()
            tot += mask.sum().item()
    acc = (tot_correct / tot) if tot > 0 else 0.0
    return acc, tot_loss / max(1, len(loader))

@inference_mode()
def validate(model, loader, device, sep_id, pad_id):
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = masked_next_token_ce(logits, y, sep_id, pad_id)
        
        tot_loss += loss.item()
        preds = logits[:, :-1, :].argmax(-1)
        tgt = y[:, 1:]
        sep_pos = (y == sep_id).int().argmax(dim=1)
        idx = torch.arange(tgt.size(1), device=device).unsqueeze(0)
        mask = (idx + 1 > sep_pos.unsqueeze(1)) & (tgt != pad_id)
        
        tot_correct += (preds[mask] == tgt[mask]).sum().item()
        tot += mask.sum().item()
    acc = (tot_correct / tot) if tot > 0 else 0.0
    return acc, tot_loss / max(1, len(loader))

@inference_mode()
def sample_one(model, tokenizer, prompt_ids, max_out, device):
    model.eval()
    x = torch.tensor([prompt_ids], device=device)
    gen = []
    
    ANSWER_ID = tokenizer.tok2id['ANSWER']
    valid_ids = torch.tensor(
        [tokenizer.tok2id[t] for t in (tokenizer.t_vars + tokenizer.n_vars)],
        device=device
    )
    
    if valid_ids.numel() == 0:
        return []

    for _ in range(min(max_out, model.max_len - len(prompt_ids))):
        if x.size(1) >= model.max_len: break
        
        logits = model(x)[:, -1, :]
        nxt = logits.argmax(-1).item()
        
        if nxt in (tokenizer.eos_id, tokenizer.pad_id): break
        gen.append(nxt)
        
        if nxt == ANSWER_ID:
            x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
            logits2 = model(x)[:, -1, :]
            mask = torch.full_like(logits2, float('-inf'))
            mask[:, valid_ids] = 0
            nxt2 = (logits2 + mask).argmax(-1).item()
            gen.append(nxt2)
            break
        
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)

    return gen

def compute_lengths(num_sources:int)->tuple[int,int,int]:
    """
    Calculate max input/output lengths based on problem size to avoid truncation.
    - Pre-SEP (prompt): Roughly 5*K+6 tokens. We use +10 for safety/readability.
    - Post-SEP (program): Roughly 5*(K-1)+2 tokens. We use +4 for safety.
    """
    pre_sep = 5 * num_sources + 10
    post_sep = 5 * (num_sources - 1) + 2 + 4
    total = pre_sep + post_sep
    return pre_sep, post_sep, total

# =========================
# Main execution block
# =========================
def main():
    # --- Config and Reproducibility ---
    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_num_threads(8)

    config = dict(
        num_sources=4, num_samples=100000, 
        d_model=128, n_heads=4, n_layers=4,
        lr=3e-4, weight_decay=0.01, 
        batch_size=1024, 
        epochs=1000,
        patience=20
    )
    
    mi, mo, ml = compute_lengths(config['num_sources'])
    config['max_input_len'], config['max_output_len'], max_len = mi, mo, ml

    t0 = time.time()
    # --- Data Prep ---
    ins, outs = generate_countdown_data(config['num_samples'], config['num_sources'], seed=42)
    
    if len(ins) < config['num_samples']:
        raise RuntimeError("Data generation failed to produce enough samples. Increase max_attempts or relax constraints.")

    tok = Tokenizer(num_sources=config['num_sources'])
    
    seqs = [f"{inp} SEP {out}" for inp, out in zip(ins, outs)]
    enc = [tok.encode(s, max_len) for s in seqs]
    X = torch.tensor(enc, dtype=torch.long)
    
    # --- Data Assertions ---
    assert ((X == tok.sep_id).sum(dim=1) == 1).all().item(), "Each sequence must have exactly one SEP."
    assert (X[:, :mi] == tok.sep_id).any(dim=1).all().item(), "A SEP token fell beyond max_input_len."
    assert all(tok.eos_id in row for row in X.tolist()), "EOS token missing; sequence may be truncated."

    ds = TensorDataset(X, X.clone())
    n_train = int(0.9 * len(ds))
    g = torch.Generator().manual_seed(42)
    tr, va = random_split(ds, [n_train, len(ds) - n_train], generator=g)

    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError("Empty train or validation set. Increase num_samples.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_args = {'batch_size': config['batch_size'], 'num_workers': 8, 'persistent_workers': True, 'pin_memory': True} if device.type=='cuda' else {'batch_size': config['batch_size']}
    
    train_loader = DataLoader(tr, shuffle=True, drop_last=True, generator=g, **loader_args)
    val_loader = DataLoader(va, shuffle=False, **loader_args)

    # --- Model and Optimizer ---
    model = DecoderOnly(tok.vocab_size, config['d_model'], config['n_heads'], config['n_layers'], max_len, tok.pad_id).to(device)
    
    n_gpus = torch.cuda.device_count()
    use_compile = True
    if n_gpus > 1:
        print(f"Using {n_gpus} GPUs via DataParallel.")
        model = nn.DataParallel(model)
        use_compile = False

    if use_compile:
        try:
            model = torch.compile(model)
            print("Model compiled successfully (PyTorch 2.0+).")
        except Exception:
            print("Could not compile model (PyTorch < 2.0 or other issue).")
            pass

    decay_params, no_decay_params, seen = [], [], set()
    for name, p in model.named_parameters():
        pid = id(p)
        if pid in seen: continue
        seen.add(pid)
        
        is_ln = ("ln" in name.lower())
        if p.dim() < 2 or "bias" in name or is_ln:
            no_decay_params.append(p)
        else:
            decay_params.append(p)
    
    decay_ids = {id(p) for p in decay_params}
    no_decay_ids = {id(p) for p in no_decay_params}
    assert decay_ids.isdisjoint(no_decay_ids), "Same parameter found in both decay and no_decay groups."

    optim_groups = [
        {'params': decay_params, 'weight_decay': config['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    opt = torch.optim.AdamW(optim_groups, lr=config['lr'])

    # --- Experimental Settings Printout ---
    print("\n" + "="*50)
    print(f"{'Run Configuration':^50}")
    print("="*50)
    print(f"{'Python Version:':<25} {sys.version.split()[0]}")
    print(f"{'PyTorch Version:':<25} {torch.__version__}")
    print(f"{'Using GPUs:':<25} {n_gpus}")
    print("-" * 50)
    print(f"{'Dataset Size:':<25} {config['num_samples']:,} samples")
    print(f"{'Problem Type:':<25} {config['num_sources']}-number Countdown")
    print(f"{'Vocabulary Size:':<25} {tok.vocab_size} tokens")
    print("-" * 50)
    print(f"{'Model Parameters:':<25} {sum(p.numel() for p in model.parameters() if p.requires_grad):,} total")
    print(f"{'Dimensions (d_model):':<25} {config['d_model']}")
    print(f"{'Attention Heads:':<25} {config['n_heads']}")
    print(f"{'Number of Layers:':<25} {config['n_layers']}")
    print(f"{'Max Sequence Length:':<25} {max_len} (pre={mi}, post={mo})")
    print("-" * 50)
    print(f"{'Epochs:':<25} {config['epochs']} (patience={config['patience']})")
    print(f"{'Global Batch Size:':<25} {config['batch_size']}")
    print(f"{'Batch Size per GPU:':<25} {config['batch_size'] // n_gpus if n_gpus > 0 else config['batch_size']}")
    print(f"{'Learning Rate:':<25} {config['lr']}")
    print(f"{'Weight Decay:':<25} {config['weight_decay']}")
    print("="*50 + "\n")

    # --- Training Loop ---
    print("Training...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_loss = float('inf')
    patience_counter = 0
    train_start_time = time.time()
    
    for ep in range(config['epochs']):
        tr_acc, tr_loss = train_one_epoch(model, opt, train_loader, device, tok.sep_id, tok.pad_id)
        va_acc, va_loss = validate(model, val_loader, device, tok.sep_id, tok.pad_id)
        
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        
        elapsed_time = time.time() - train_start_time
        epochs_left = config['epochs'] - (ep + 1)
        time_per_epoch = elapsed_time / (ep + 1)
        eta_seconds = epochs_left * time_per_epoch
        elapsed_str = str(datetime.timedelta(seconds=int(elapsed_time)))
        eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))

        print(f"Epoch {ep+1:04d} | Train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | Val loss {va_loss:.4f} acc {va_acc*100:.2f}% | Elapsed: {elapsed_str} | ETA: {eta_str}")
        
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            patience_counter = 0
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            state_to_save = model_to_save._orig_mod.state_dict() if hasattr(model_to_save, '_orig_mod') else model_to_save.state_dict()
            torch.save({ "state": copy.deepcopy(state_to_save), "config": config }, "best_model.pt")
            print(f"  -> New best val loss: {best_val_loss:.4f}. Checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Validation loss has not improved for {config['patience']} epochs. Stopping early.")
                break

    # --- Plotting ---
    plot_training_results(history)

    # --- Final Evaluation ---
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load("best_model.pt", map_location=device)
    
    final_model = DecoderOnly(tok.vocab_size, config['d_model'], config['n_heads'], config['n_layers'], max_len, tok.pad_id).to(device)
    final_model.load_state_dict(checkpoint['state'])
    
    metrics = evaluate_program_metrics(final_model, tok, val_loader, device, config['max_output_len'], max_batches=16)
    print(f"\nProgram EM: {metrics['program_em']*100:.2f}% | "
          f"Answer Acc: {metrics['answer_acc']*100:.2f}% "
          f"on {metrics['samples']} val samples.")

    # --- Inference Examples ---
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