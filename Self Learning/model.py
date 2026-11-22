# pip install tokenizers
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 1. Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Initializing training on device: {device}\n")

# 2. Load text
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.'
file_path = os.path.join(script_dir, 'wiki_data.txt')

with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Loaded text file. Total characters: {len(text):,}")

# 3. Train a small BPE tokenizer on THIS dataset
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(
    vocab_size=2000,          # 1000-2000 is good for 170k chars
    min_frequency=2,
    special_tokens=["[UNK]"],
    show_progress=True,
)

tokenizer.train_from_iterator([text], trainer=trainer)
vocab_size = tokenizer.get_vocab_size()
print(f"Small BPE vocab size: {vocab_size}")

# 4. Encode
data_tokens = tokenizer.encode(text).ids
print(f"Total BPE tokens: {len(data_tokens):,}")

block_size = 32

# 80/20 split
split_idx = int(len(data_tokens) * 0.8)
train_tokens = data_tokens[:split_idx]
val_tokens = data_tokens[split_idx:]

def build_dataset(token_sequence):
    X, Y = [], []
    for i in range(len(token_sequence) - block_size):
        X.append(token_sequence[i:i + block_size])
        Y.append(token_sequence[i + block_size])
    return torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

print("Building train/validation tensors...")
X_train, Y_train = build_dataset(train_tokens)
X_val, Y_val = build_dataset(val_tokens)

train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

X_val, Y_val = X_val.to(device), Y_val.to(device)

# 5. Smaller model with tied embedding/output weights
class WikiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden_dim=128,
                 num_layers=1, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.emb_drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Weight tying: output layer uses embedding matrix
        if emb_dim == hidden_dim:
            self.fc.weight = self.embedding.weight

    def forward(self, x):
        emb = self.emb_drop(self.embedding(x))
        out, _ = self.lstm(emb)
        out = self.drop(out[:, -1, :])
        return self.fc(out)

model = WikiLSTM(
    vocab_size=vocab_size,
    emb_dim=128,
    hidden_dim=128,
    num_layers=1,
    dropout=0.3
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.02)

# 6. Training with early stopping
epochs = 100
patience = 5

best_val_loss = float('inf')
patience_counter = 0
best_state = None

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        logits = model(batch_X)
        loss = criterion(logits, batch_Y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)

    train_loss = running_loss / len(train_dataset)

    # Validate every epoch
    model.eval()
    val_running = 0.0

    with torch.no_grad():
        for i in range(0, len(X_val), 512):
            xb = X_val[i:i + 512]
            yb = Y_val[i:i + 512]

            logits_val = model(xb)
            loss_val = criterion(logits_val, yb)
            val_running += loss_val.item() * xb.size(0)

    val_loss = val_running / len(X_val)

    print(f"Epoch {epoch + 1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        print("   ↳ Validation improved. Saving checkpoint.")
    else:
        patience_counter += 1
        print(f"   ↳ No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("\nEarly stopping triggered.")
            break

# 7. Restore best and save
if best_state is not None:
    model.load_state_dict(best_state)

torch.save(model.state_dict(), 'wiki_lstm_small_bpe_weights.pth')
print("\nSaved as 'wiki_lstm_small_bpe_weights.pth'")