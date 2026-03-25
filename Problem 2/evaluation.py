from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn

# Training configuration

@dataclass
class TrainConfig:
    hidden_size: int = 128
    num_layers: int = 2
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 20
    embed_dim: int = 64
    dropout: float = 0.2
    #hyperparameters
    max_gen_len: int = 32
    num_generated: int = 300
    temperature: float = 0.7
    top_k: int = 5

# Count Trainable Parameters
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss = 0.0
    total_items = 0
    criterion = nn.CrossEntropyLoss()   # Cross entropy loss for next character prediction

    for x, lengths, y in loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        logits = model(x, lengths)
        loss = criterion(logits, y)  # Compute loss

        if train:
            # Reset gradients
            optimizer.zero_grad()
            # Backpropagation
            loss.backward()
            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() # Update weights

        total_loss += loss.item() * x.size(0)
        total_items += x.size(0)

    return total_loss / max(total_items, 1)

#Samples the next character index from model logits

@torch.no_grad()
def sample_next_char(logits, temperature=1.0, top_k=None):

    # Apply temperature scaling
    logits = logits / max(temperature, 1e-5)
    if top_k is not None and top_k > 0:  # keep only top-k most likely characters
        values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
        probs = torch.softmax(values, dim=-1)
        choice = torch.multinomial(probs, 1).item()
        return indices[choice].item()

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()


def _clean_name(s: str) -> str:
    return " ".join(s.split()).strip() #removing extra spaces


@torch.no_grad()
def generate_one(model, vocab, device, max_len=32, temperature=0.9, top_k=8):
    model.eval()
    prefix = [vocab.start_idx]

    for _ in range(max_len):
        # Convert prefix into tensor
        x = torch.tensor(prefix, dtype=torch.long, device=device).unsqueeze(0)
        lengths = torch.tensor([len(prefix)], dtype=torch.long, device=device)
        logits = model(x, lengths)[0] # Get model prediction for next character
        nxt = sample_next_char(logits, temperature=temperature, top_k=top_k)
        if nxt == vocab.end_idx: # Stop if end token predicted
            break
        prefix.append(nxt)   # Otherwise append predicted character

    return _clean_name(vocab.decode(prefix[1:])) # Decode generated indices (excluding start token)


@torch.no_grad()
def generate_many(model, vocab, device, n=300, max_len=32, temperature=0.9, top_k=8):
    out = []
    trials = 0
    
    #allow multiple tries if name invalid
    while len(out) < n and trials < n * 10:
        name = generate_one(model, vocab, device, max_len=max_len, temperature=temperature, top_k=top_k)
        trials += 1
        if not name:
            continue
        if 1 <= len(name) <= 30:
            out.append(name)
    return out

# Compute Novelty and Diversity Metrics
@torch.no_grad()
def compute_metrics(model, vocab, training_names: List[str], device, cfg: TrainConfig):
    generated = generate_many(
        model, vocab, device,
        n=cfg.num_generated,
        max_len=cfg.max_gen_len,
        temperature=cfg.temperature,
        top_k=cfg.top_k,
    )
    train_set = set(training_names)
    unique_count = len(set(generated))

    #how many generated names are not in training set
    novelty = 100.0 * sum(name not in train_set for name in generated) / max(len(generated), 1)
    
    #fraction of unique generated names
    diversity = unique_count / max(len(generated), 1)
    repeated_train = [name for name in generated if name in train_set][:10]
    samples = generated[:12]

    return {
        "generated": generated,
        "novelty_rate": novelty,
        "diversity": diversity,
        "samples": samples,
        "train_collisions": repeated_train,
    }


def train_and_evaluate(model, loaders, cfg: TrainConfig, vocab, names: List[str], device):
    train_loader, val_loader, test_loader = loaders
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    best_state = None
    best_val = float("inf")
    history = []

    for epoch in range(1, cfg.epochs + 1):
        tr_loss = run_epoch(model, train_loader, device, optimizer)
        va_loss = run_epoch(model, val_loader, device)
        history.append((epoch, tr_loss, va_loss))
        # Save best model based on validation loss
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"[{model.model_name}] Epoch {epoch:02d}/{cfg.epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f}")

    # Load best model after training
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final test evaluation
    test_loss = run_epoch(model, test_loader, device)
    # Compute name generation metrics
    metrics = compute_metrics(model, vocab, names, device, cfg)
    result = {
        "model": model.model_name,
        "architecture": model.architecture_description(),
        "parameters": count_parameters(model),
        "hidden_size": cfg.hidden_size,
        "layers": cfg.num_layers,
        "learning_rate": cfg.learning_rate,
        "embedding_dim": cfg.embed_dim,
        "dropout": cfg.dropout,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        **metrics,
    }
    return result, history

#Print model result

def print_result_block(result: Dict):
    print("\n" + "=" * 72)
    print(f"MODEL: {result['model']}")
    print("-" * 72)
    print("Architecture:")
    print(f"  {result['architecture']}")
    print(f"Trainable parameters: {result['parameters']:,}")
    print("Hyperparameters:")
    print(
        f"  hidden_size={result['hidden_size']}, layers={result['layers']}, "
        f"learning_rate={result['learning_rate']}, embedding_dim={result['embedding_dim']}, "
        f"dropout={result['dropout']}"
    )
    print("Quantitative evaluation:")
    print(f"  test_loss     = {result['test_loss']:.4f}")
    print(f"  novelty_rate  = {result['novelty_rate']:.2f}%")
    print(f"  diversity     = {result['diversity']:.4f}")
    print("Qualitative samples:")
    for s in result["samples"]:
        print(f"  - {s}")
    print("Failure-mode hints:")
    print("  - Repetition or near-duplicates")
    print("  - Awkward mixing of first/last names")
    print("  - Unusual spacing or truncated endings")
    if result["train_collisions"]:
        print("Seen-in-training examples among generated samples:")
        for s in result["train_collisions"][:5]:
            print(f"  - {s}")
