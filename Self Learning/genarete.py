import torch
import torch.nn as nn
import tiktoken

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 1. Setup the exact same BPE Tokenizer
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab
block_size = 32

# 2. Rebuild the exact model architecture
class WikiLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        return self.fc(out[:, -1, :])

# 3. Load the model and weights
hidden_dim = 128
model = WikiLSTM(vocab_size, hidden_dim).to(device)

# Point this to the .pth file you just saved
model.load_state_dict(torch.load('wiki_lstm_bpe_weights.pth'))
model.eval() 
print("BPE Model loaded successfully!\n")

# 4. Generate Text
# Provide a seed phrase to get the model started
seed_text = "tell me the way to fuck and kiss also i want a good realtionship advice "
current_context = enc.encode(seed_text)

# Ensure we only look at the last 32 tokens if the seed is too long
current_context = current_context[-block_size:]

print("Generating...\n")
print(seed_text, end="", flush=True)

# Temperature controls randomness. 0.7 keeps spelling accurate but allows creativity.
temperature = 0.7  

with torch.no_grad():
    for _ in range(150): # Number of tokens (words/subwords) to generate
        # Convert the current context into a tensor
        x_gen = torch.tensor([current_context], dtype=torch.long).to(device)
        
        # Forward pass and apply temperature
        logits = model(x_gen)
        logits = logits / temperature 
        probs = torch.softmax(logits, dim=1).squeeze()
        
        # Sample the next token ID
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        # Decode the token ID back to English and print it instantly
        print(enc.decode([next_token_id]), end="", flush=True)
        
        # Shift the context window forward
        current_context = current_context[1:] + [next_token_id]

print("\n")