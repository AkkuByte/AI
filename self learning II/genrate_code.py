import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# CONFIG — MUST MATCH TRAINING
# ============================================================

BLOCK_SIZE = 256
D_MODEL = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.10

MODEL_FILE = "nano_coder_best.pth"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# ============================================================
# TOKENIZER
# ============================================================

enc = tiktoken.get_encoding("gpt2")

VOCAB_SIZE = enc.n_vocab

print("Vocabulary:", VOCAB_SIZE)


# ============================================================
# ATTENTION
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.qkv = nn.Linear(
            D_MODEL,
            3 * D_MODEL,
            bias=False
        )

        self.proj = nn.Linear(
            D_MODEL,
            D_MODEL,
            bias=False
        )

    def forward(self, x):

        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        head_dim = C // N_HEAD

        q = q.view(
            B,
            T,
            N_HEAD,
            head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            N_HEAD,
            head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            N_HEAD,
            head_dim
        ).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True
        )

        out = out.transpose(
            1,
            2
        ).contiguous().view(
            B,
            T,
            C
        )

        return self.proj(out)


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(
            D_MODEL,
            4 * D_MODEL
        )

        self.fc2 = nn.Linear(
            4 * D_MODEL,
            D_MODEL
        )

        self.activation = nn.GELU(
            approximate="tanh"
        )

        self.dropout = nn.Dropout(
            DROPOUT
        )

    def forward(self, x):

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(
            D_MODEL
        )

        self.attn = CausalSelfAttention()

        self.ln2 = nn.LayerNorm(
            D_MODEL
        )

        self.mlp = MLP()

    def forward(self, x):

        x = x + self.attn(
            self.ln1(x)
        )

        x = x + self.mlp(
            self.ln2(x)
        )

        return x


# ============================================================
# MODEL
# ============================================================

class NanoCoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            D_MODEL
        )

        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            D_MODEL
        )

        self.blocks = nn.ModuleList(
            [
                TransformerBlock()
                for _ in range(N_LAYER)
            ]
        )

        self.ln_final = nn.LayerNorm(
            D_MODEL
        )

        self.lm_head = nn.Linear(
            D_MODEL,
            VOCAB_SIZE,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

    def forward(self, idx):

        B, T = idx.shape

        positions = torch.arange(
            T,
            device=idx.device
        )

        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=150,
        temperature=0.7,
        top_k=50
    ):

        self.eval()

        tokens = enc.encode(
            prompt,
            allowed_special="all"
        )

        idx = torch.tensor(
            [tokens],
            dtype=torch.long,
            device=device
        )

        for _ in range(max_new_tokens):

            idx_cond = idx[
                :, -BLOCK_SIZE:
            ]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type in ("mps", "cuda")
            ):

                logits = self(
                    idx_cond
                )

            logits = logits[
                :, -1, :
            ].float()

            logits /= max(
                temperature,
                1e-5
            )

            if top_k is not None:

                k = min(
                    top_k,
                    logits.size(-1)
                )

                values, _ = torch.topk(
                    logits,
                    k
                )

                threshold = values[
                    :, -1
                ].unsqueeze(-1)

                logits = torch.where(
                    logits < threshold,
                    torch.full_like(
                        logits,
                        float("-inf")
                    ),
                    logits
                )

            probs = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probs,
                1
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

            if next_token.item() == enc.eot_token:
                break

        return enc.decode(
            idx[0].tolist()
        )


# ============================================================
# LOAD MODEL
# ============================================================

print("Creating model...")

model = NanoCoder().to(device)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print()
print("Loading:", MODEL_FILE)

state = torch.load(
    MODEL_FILE,
    map_location="cpu"
)

model.load_state_dict(state)

model.eval()

model.to(device)

print("Model loaded successfully.")


# ============================================================
# GENERATION
# ============================================================

prompts = [
    "for i in rang"
]


for prompt in prompts:

    print()
    print("=" * 60)
    print("PROMPT:", repr(prompt))
    print("=" * 60)

    result = model.generate(
        prompt,
        max_new_tokens=250,
        temperature=0.4,
        top_k=50
    )

    print(result)