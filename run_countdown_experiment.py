# countdown_cot_final_v11.py
import math, random, itertools, time, sys, bisect, re, copy
from collections import defaultdict, deque
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

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

def generate_countdown_data(num_samples, num_sources, seed=42):
    """Generates a dataset of Countdown problems and their program solutions."""
    random.seed(seed)
    solver = CountdownSolver()
    inputs, outputs = [], []
    max_attempts = max(2000, 200 * num_samples)
    attempts, rejects_solver, rejects_program = 0, 0, 0
    solved_steps = []
    
    print(f"Generating {num_samples} samples for {num_sources}-number problems (seed={seed})...")
    while len(inputs) < num_samples and attempts < max_attempts:
        attempts += 1
        small = [random.randint(1, 10) for _ in range(num_sources - 2)]
        large = [random.choice([25, 50, 75, 100]) for _ in range(2)]
        source_numbers = small + large
        random.shuffle(source_numbers)
        target = random.randint(101, 999)

        steps = solver.solve(source_numbers, target, max_depth=num_sources - 1)
        if steps is None:
            rejects_solver += 1
            continue
        
        program = steps_to_program(steps, target)
        if program is None:
            rejects_program += 1
            continue
        
        solved_steps.append(len(steps))
        map_str = " ".join([f"n{i+1}={v}" for i, v in enumerate(source_numbers)])
        inputs.append(f"IN: {source_numbers} TGT: {target} MAP: {map_str}")
        outputs.append(program)
        if len(inputs) > 0 and len(inputs) % 100 == 0:
            print(f"  ... {len(inputs)} / {num_samples} generated.")
    
    acc_rate = 100.0 * len(inputs) / max(1, attempts)
    avg_steps = sum(solved_steps) / max(1, len(solved_steps))
    print(f"Generated {len(inputs)} samples from {attempts} attempts "
          f"({acc_rate:.1f}% acceptance). Avg steps {avg_steps:.2f}. "
          f"Rejects: solver={rejects_solver}, program={rejects_program}.")
    return inputs, outputs

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
# Execution-Based Metrics Helpers
# =========================
def parse_prompt_meta(prompt_text:str):
    """Extracts the target value and number mappings from a prompt string."""
    m_tgt = re.search(r"\bTGT:\s+(\d+)\b", prompt_text)
    tgt = int(m_tgt.group(1)) if m_tgt else None
    env = {}
    m_map = re.search(r"\bMAP:\s+(.*)$", prompt_text)
    if m_map:
        for part in m_map.group(1).split():
            if '=' in part:
                k, v = part.split('=', 1)
                if k.startswith('n') and v.isdigit():
                    env[k] = int(v)
    return tgt, env

def exec_program(prog_text:str, env:dict[str,int]):
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

@torch.inference_mode()
def evaluate_program_metrics(model, tok, val_loader, device, max_out, max_batches=4):
    """Evaluates the model on Program Exact Match and Answer Accuracy."""
    model.eval()
    n_prog_em, n_ans_ok, n = 0, 0, 0
    it = iter(val_loader)
    for _ in range(max_batches):
        try:
            x_batch, _ = next(it)
        except StopIteration:
            break
        for x in x_batch:
            ids = x.tolist()
            if tok.sep_id not in ids: continue
            
            sep_idx = ids.index(tok.sep_id)
            prompt_ids = [t for t in ids[:sep_idx+1] if t != tok.pad_id]
            tgt_text = tok.decode(ids[sep_idx+1:])
            
            gen_ids = sample_one(model, tok, prompt_ids, max_out, device)
            gen_text = tok.decode(gen_ids)
            
            prompt_text = tok.decode(prompt_ids).replace(' SEP', '').strip()
            tgt_val, env = parse_prompt_meta(prompt_text)
            if tgt_val is None or not env: continue
            
            assert exec_program(tgt_text, env) == tgt_val, "Gold program failed execution!"

            ans_gen = exec_program(gen_text, env)
            n += 1
            if gen_text.strip() == tgt_text.strip():
                n_prog_em += 1
            if ans_gen is not None and ans_gen == tgt_val:
                n_ans_ok += 1
                
    return dict(
        samples=n,
        program_em=n_prog_em / n if n else 0.0,
        answer_acc=n_ans_ok / n if n else 0.0,
    )

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
        causal_mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        x1 = self.ln1(x)
        y, _ = self.attn(x1, x1, x1, attn_mask=causal_mask, key_padding_mask=key_padding_mask, need_weights=False)
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
        opt.zero_grad(set_to_none=True) # Perf improvement
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

@torch.inference_mode()
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

@torch.inference_mode()
def sample_one(model, tokenizer, prompt_ids, max_out, device):
    model.eval()
    x = torch.tensor([prompt_ids], device=device)
    gen = []
    for _ in range(min(max_out, model.max_len - len(prompt_ids))):
        if x.size(1) >= model.max_len: break
        
        logits = model(x)
        nxt = logits[:, -1, :].argmax(-1).item()
        
        if nxt == tokenizer.eos_id: break
        gen.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
        
        if len(gen) >= 2 and gen[-2] == tokenizer.tok2id.get('ANSWER'):
            break
    return gen

def compute_lengths(num_sources:int)->tuple[int,int,int]:
    """Calculate max input/output lengths based on problem size to avoid truncation."""
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

    config = dict(
        num_sources=4, num_samples=1000, d_model=128, n_heads=4, n_layers=4,
        lr=3e-4, weight_decay=0.01, batch_size=128, epochs=10,
    )
    
    mi, mo, ml = compute_lengths(config['num_sources'])
    config['max_input_len'], config['max_output_len'], max_len = mi, mo, ml

    t0 = time.time()
    # --- Data Prep ---
    ins, outs = generate_countdown_data(config['num_samples'], config['num_sources'], seed=42)
    if len(ins) < config['num_samples']:
        print("Warning: Low acceptance rate. Re-running data generation with new seed.")
        ins, outs = generate_countdown_data(config['num_samples'], config['num_sources'], seed=43)

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader_args = {'batch_size': config['batch_size'], 'num_workers': 2, 'persistent_workers': True, 'pin_memory': True} if device.type=='cuda' else {'batch_size': config['batch_size']}
    
    train_loader = DataLoader(tr, shuffle=True, drop_last=True, **loader_args)
    val_loader = DataLoader(va, shuffle=False, **loader_args)

    # --- Model and Optimizer ---
    model = DecoderOnly(tok.vocab_size, config['d_model'], config['n_heads'], config['n_layers'], max_len, tok.pad_id).to(device)
    
    ## FIX ##: A critical bug fix to avoid double-updating tied embedding weights.
    # We build a set of unique parameter objects to pass to the optimizer.
    decay_params, no_decay_params, seen_ids = [], [], set()
    for name, p in model.named_parameters():
        if p not in seen_ids:
            seen_ids.add(p)
            if p.dim() < 2 or "bias" in name or "ln" in name.lower():
                no_decay_params.append(p)
            else:
                decay_params.append(p)
    
    optim_groups = [
        {'params': decay_params, 'weight_decay': config['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]
    opt = torch.optim.AdamW(optim_groups, lr=config['lr'])

    print(f"\nModel: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
    print(f"Vocab: {tok.vocab_size} | Max len: {max_len} (pre={mi}, post={mo})")
    
    # --- Training Loop ---
    print("Training...")
    best_val_loss, best_model_state = float('inf'), None
    for ep in range(config['epochs']):
        tr_acc, tr_loss = train_one_epoch(model, opt, train_loader, device, tok.sep_id, tok.pad_id)
        va_acc, va_loss = validate(model, val_loader, device, tok.sep_id, tok.pad_id)
        print(f"Epoch {ep+1:02d} | Train loss {tr_loss:.4f} acc {tr_acc*100:.2f}% | Val loss {va_loss:.4f} acc {va_acc*100:.2f}%")
        
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  -> New best val loss: {best_val_loss:.4f}")

    # --- Final Evaluation ---
    if best_model_state:
        print("\nLoading best model for final evaluation...")
        model.load_state_dict(best_model_state)
    
    metrics = evaluate_program_metrics(model, tok, val_loader, device, config['max_output_len'], max_batches=8)
    print(f"\nProgram EM: {metrics['program_em']*100:.2f}% | "
          f"Answer Acc: {metrics['answer_acc']*100:.2f}% "
          f"on {metrics['samples']} val samples.")

    # --- Inference Example ---
    x_sample, _ = next(iter(val_loader))
    ids = x_sample[0].tolist()
    sep_idx = ids.index(tok.sep_id)
    prompt_ids = [t for t in ids[:sep_idx+1] if t != tok.pad_id]
    gen_ids = sample_one(model, tok, prompt_ids, config['max_output_len'], device)
    
    print("\n--- Inference Example ---")
    print("Problem:    ", tok.decode(prompt_ids).replace(' SEP', ''))
    print("True Sol:   ", tok.decode([t for t in ids[sep_idx+1:] if t != tok.pad_id]))
    print("Generated:  ", tok.decode(gen_ids))
    print(f"\nDone in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    with Logger('outputs_final_v11_bugfix.log'):
        main()