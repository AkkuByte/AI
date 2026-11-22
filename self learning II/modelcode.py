import os
import math
import time
import gc
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken


# ============================================================
# 0. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("=" * 60)
print("DEVICE")
print("=" * 60)
print(f"PyTorch : {torch.__version__}")
print(f"Device  : {device}")

if device.type == "mps":
    try:
        recommended = torch.mps.recommended_max_memory() / (1024 ** 3)
        print(f"MPS recommended memory: {recommended:.2f} GB")
    except Exception:
        pass

print()


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Model
BLOCK_SIZE = 256
D_MODEL = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.10

# Training
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4

MAX_STEPS = 2000
EVAL_INTERVAL = 50
EVAL_BATCHES = 10

LEARNING_RATE = 3e-4
MIN_LR = 3e-5

WARMUP_STEPS = 500
WEIGHT_DECAY = 0.10

BETA1 = 0.9
BETA2 = 0.95

GRAD_CLIP = 1.0

# Early stopping
EARLY_STOP_PATIENCE = 8

# Large-vocabulary output layer can consume lots of memory.
# Instead of creating logits for every token at once, process
# the language-model head in chunks.
LOGITS_CHUNK_SIZE = 2048

# FP16 gives a significant memory reduction on MPS.
USE_AMP = True

# Files
DATA_FILE = "github_train_data.txt"
TOKEN_CACHE = "github_train_data_gpt2.pt"
BEST_MODEL_FILE = "nano_coder_best.pth"

SEED = 1337


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if device.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 3. TOKENIZER + DATA
# ============================================================

print("=" * 60)
print("LOADING DATA")
print("=" * 60)

enc = tiktoken.get_encoding("gpt2")
VOCAB_SIZE = enc.n_vocab

if os.path.exists(TOKEN_CACHE):
    print(f"Loading token cache: {TOKEN_CACHE}")

    data = torch.load(
        TOKEN_CACHE,
        map_location="cpu"
    )

else:
    print(f"Reading: {DATA_FILE}")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    if len(text.strip()) == 0:
        raise ValueError("Training file is empty.")

    print("Tokenizing with GPT-2 BPE...")
    tokens = enc.encode(text, allowed_special="all")

    data = torch.tensor(tokens, dtype=torch.long)

    del tokens
    del text
    gc.collect()

    torch.save(data, TOKEN_CACHE)

    print(f"Saved token cache: {TOKEN_CACHE}")


if data.numel() <= BLOCK_SIZE + 1:
    raise ValueError(
        f"Dataset is too small. Need more than {BLOCK_SIZE + 1} tokens."
    )

split = int(0.90 * len(data))

train_data = data[:split].contiguous()
val_data = data[split:].contiguous()

print(f"Total tokens : {len(data):,}")
print(f"Train tokens : {len(train_data):,}")
print(f"Val tokens   : {len(val_data):,}")
print(f"Vocabulary   : {VOCAB_SIZE:,}")
print()


# ============================================================
# 4. FAST BATCH CREATION
# ============================================================

offsets = torch.arange(
    BLOCK_SIZE + 1,
    dtype=torch.long
)


def get_batch(split_name):

    source = train_data if split_name == "train" else val_data

    max_start = len(source) - BLOCK_SIZE - 1

    if max_start <= 0:
        raise ValueError(
            f"{split_name} dataset is too small for BLOCK_SIZE={BLOCK_SIZE}"
        )

    starts = torch.randint(
        low=0,
        high=max_start,
        size=(BATCH_SIZE,),
        device="cpu"
    )

    positions = (
        starts.unsqueeze(1)
        + offsets.unsqueeze(0)
    )

    batch = source[positions]

    # One CPU -> MPS transfer instead of two.
    batch = batch.to(device)

    x = batch[:, :-1]
    y = batch[:, 1:]

    return x, y


# ============================================================
# 5. MIXED PRECISION
# ============================================================

def autocast_context():

    if USE_AMP and device.type == "mps":
        return torch.autocast(
            device_type="mps",
            dtype=torch.float16
        )

    if USE_AMP and device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        )

    return nullcontext()


# ============================================================
# 6. TRANSFORMER COMPONENTS
# ============================================================

class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        if D_MODEL % N_HEAD != 0:
            raise ValueError(
                "D_MODEL must be divisible by N_HEAD."
            )

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

        self.dropout = nn.Dropout(DROPOUT)

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

        # PyTorch's optimized scaled dot-product attention.
        # This replaces the large explicit attention matrix + mask.
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=DROPOUT if self.training else 0.0,
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

        self.dropout = nn.Dropout(DROPOUT)

        self.activation = nn.GELU(
            approximate="tanh"
        )

    def forward(self, x):

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


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
# 7. MODEL
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

        # Weight tying.
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):

        if isinstance(module, nn.Linear):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02
            )

    def compute_loss(
        self,
        hidden,
        targets
    ):

        # Flatten [B,T,C] -> [B*T,C]
        hidden = hidden.reshape(
            -1,
            D_MODEL
        )

        targets = targets.reshape(
            -1
        )

        total_tokens = hidden.shape[0]

        total_loss = torch.zeros(
            (),
            device=hidden.device,
            dtype=torch.float32
        )

        # Process output layer in chunks.
        #
        # This prevents a gigantic:
        # [B*T, 50257]
        #
        # tensor from staying in memory.
        for start in range(
            0,
            total_tokens,
            LOGITS_CHUNK_SIZE
        ):

            end = min(
                start + LOGITS_CHUNK_SIZE,
                total_tokens
            )

            chunk_hidden = hidden[
                start:end
            ]

            chunk_targets = targets[
                start:end
            ]

            logits = self.lm_head(
                chunk_hidden
            )

            # CE in FP32 for numerical stability.
            chunk_loss = F.cross_entropy(
                logits.float(),
                chunk_targets
            )

            chunk_size = end - start

            total_loss = (
                total_loss
                + chunk_loss * chunk_size
            )

        return total_loss / total_tokens

    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        if T > BLOCK_SIZE:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"BLOCK_SIZE={BLOCK_SIZE}"
            )

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

        # During training, don't create the full
        # vocabulary logits tensor.
        if targets is not None:

            loss = self.compute_loss(
                x,
                targets
            )

            return None, loss

        # Only create logits when actually generating.
        logits = self.lm_head(x)

        return logits, None

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=200,
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
                :,
                -BLOCK_SIZE:
            ]

            with autocast_context():

                logits, _ = self(
                    idx_cond
                )

            logits = logits[
                :, -1, :
            ].float()

            # Temperature
            logits = logits / max(
                temperature,
                1e-5
            )

            # Top-k sampling
            if top_k is not None:

                k = min(
                    top_k,
                    logits.size(-1)
                )

                values, _ = torch.topk(
                    logits,
                    k
                )

                minimum = values[
                    :, -1
                ].unsqueeze(-1)

                logits = torch.where(
                    logits < minimum,
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
                num_samples=1
            )

            idx = torch.cat(
                [idx, next_token],
                dim=1
            )

            # GPT-2 EOS/EOT token.
            if next_token.item() == enc.eot_token:
                break

        return enc.decode(
            idx[0].tolist()
        )


# ============================================================
# 8. CREATE MODEL
# ============================================================

model = NanoCoder().to(device)

num_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print("=" * 60)
print("MODEL")
print("=" * 60)

print(
    f"Parameters : {num_parameters:,}"
)

print(
    f"Approx size: "
    f"{num_parameters * 4 / (1024 ** 2):.1f} MB FP32"
)

print()


# ============================================================
# 9. ADAMW PARAMETER GROUPS
# ============================================================

decay_params = []
no_decay_params = []

for parameter in model.parameters():

    if not parameter.requires_grad:
        continue

    if parameter.dim() >= 2:
        decay_params.append(parameter)
    else:
        no_decay_params.append(parameter)

optimizer = torch.optim.AdamW(
    [
        {
            "params": decay_params,
            "weight_decay": WEIGHT_DECAY
        },
        {
            "params": no_decay_params,
            "weight_decay": 0.0
        }
    ],
    lr=LEARNING_RATE,
    betas=(BETA1, BETA2),
    eps=1e-8
)


# ============================================================
# 10. LEARNING RATE SCHEDULE
# ============================================================

def get_lr(step):

    # Warmup
    if step < WARMUP_STEPS:

        return LEARNING_RATE * (
            step + 1
        ) / WARMUP_STEPS

    # Cosine decay
    progress = (
        step - WARMUP_STEPS
    ) / max(
        1,
        MAX_STEPS - WARMUP_STEPS
    )

    progress = min(
        max(progress, 0.0),
        1.0
    )

    return (
        MIN_LR
        + 0.5
        * (
            LEARNING_RATE
            - MIN_LR
        )
        * (
            1
            + math.cos(
                math.pi * progress
            )
        )
    )


def set_learning_rate(step):

    lr = get_lr(step)

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


# ============================================================
# 11. CHECKPOINT
# ============================================================

def save_model(path):

    # Save CPU tensors.
    # This avoids keeping MPS tensors tied to the checkpoint file.
    state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    torch.save(
        state,
        path
    )

    del state


# ============================================================
# 12. VALIDATION
# ============================================================

@torch.no_grad()
def estimate_val_loss():

    model.eval()

    losses = []

    for _ in range(EVAL_BATCHES):

        xb, yb = get_batch(
            "val"
        )

        with autocast_context():

            _, loss = model(
                xb,
                yb
            )

        losses.append(
            loss.item()
        )

    return sum(losses) / len(losses)


# ============================================================
# 13. TRAINING LOOP
# ============================================================

print("=" * 60)
print("TRAINING")
print("=" * 60)

print(
    f"Batch size       : {BATCH_SIZE}"
)

print(
    f"Gradient accum   : {GRAD_ACCUM_STEPS}"
)

print(
    f"Effective batch  : "
    f"{BATCH_SIZE * GRAD_ACCUM_STEPS}"
)

print(
    f"Context length   : {BLOCK_SIZE}"
)

print(
    f"Max steps        : {MAX_STEPS}"
)

print()


best_val_loss = float("inf")
bad_evals = 0

training_start = time.time()

for step in range(
    1,
    MAX_STEPS + 1
):

    model.train()

    optimizer.zero_grad(
        set_to_none=True
    )

    running_train_loss = 0.0

    # Gradient accumulation
    for micro_step in range(
        GRAD_ACCUM_STEPS
    ):

        xb, yb = get_batch(
            "train"
        )

        with autocast_context():

            _, loss = model(
                xb,
                yb
            )

        running_train_loss += (
            loss.item()
        )

        # Important:
        # divide before backward.
        loss = loss / GRAD_ACCUM_STEPS

        loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        GRAD_CLIP
    )

    # Update LR
    lr = set_learning_rate(
        step - 1
    )

    optimizer.step()

    avg_train_loss = (
        running_train_loss
        / GRAD_ACCUM_STEPS
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if step % EVAL_INTERVAL == 0:

        if device.type == "mps":

            torch.mps.synchronize()

        val_loss = estimate_val_loss()

        if device.type == "mps":

            torch.mps.synchronize()

        elapsed = (
            time.time()
            - training_start
        )

        tokens_seen = (
            step
            * BATCH_SIZE
            * BLOCK_SIZE
            * GRAD_ACCUM_STEPS
        )

        tokens_per_second = (
            tokens_seen
            / max(elapsed, 1e-6)
        )

        print(
            f"Step {step:6d}/{MAX_STEPS} | "
            f"Train {avg_train_loss:.4f} | "
            f"Val {val_loss:.4f} | "
            f"LR {lr:.2e} | "
            f"tok/s {tokens_per_second:,.0f}"
        )

        # MPS memory
        if device.type == "mps":

            try:

                memory = (
                    torch.mps.driver_allocated_memory()
                    / (1024 ** 3)
                )

                print(
                    f"                  "
                    f"MPS memory: {memory:.2f} GB"
                )

            except Exception:
                pass

        # Best checkpoint
        if val_loss < best_val_loss:

            best_val_loss = val_loss
            bad_evals = 0

            save_model(
                BEST_MODEL_FILE
            )

            print(
                f"                  "
                f"✓ Saved best model "
                f"(val={best_val_loss:.4f})"
            )

        else:

            bad_evals += 1

            print(
                f"                  "
                f"No improvement "
                f"({bad_evals}/{EARLY_STOP_PATIENCE})"
            )

            if bad_evals >= EARLY_STOP_PATIENCE:

                print()
                print(
                    "Early stopping triggered."
                )

                break


# ============================================================
# 14. LOAD BEST MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING BEST MODEL")
print("=" * 60)

best_state = torch.load(
    BEST_MODEL_FILE,
    map_location="cpu"
)

model.load_state_dict(
    best_state
)

model.to(device)
model.eval()

del best_state

if device.type == "mps":
    torch.mps.empty_cache()


# ============================================================
# 15. GENERATION TEST
# ============================================================

print()
print("=" * 60)
print("GENERATION TEST")
print("=" * 60)

prompts = [
    "def get_",
    "def fibonacci(",
    "class BinaryTree:",
    "import torch\n\nclass",
    "def binary_search("
]

for prompt in prompts:

    print()
    print("-" * 60)
    print(f"PROMPT: {prompt!r}")
    print("-" * 60)

    output = model.generate(
        prompt=prompt,
        max_new_tokens=150,
        temperature=0.65,
        top_k=50
    )

    print(output)


print()
print("=" * 60)
print("DONE")
print("=" * 60)