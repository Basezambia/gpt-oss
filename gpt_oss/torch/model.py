# ============================================================
# GPT-OSS TORCH — WORLD-FIRST / TOKEN-ON-DEMAND (HEXCORE)
# ============================================================

import json
import math
import os
from dataclasses import dataclass

import torch
from gpt_oss.torch.weights import Checkpoint

# ============================================================
# CONFIG (PERMISSIVE)
# ============================================================

@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    initial_context_length: int = 4096
    rope_theta: float = 150000.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

# ============================================================
# HEXCORE — WORLD STATE
# ============================================================

class HexCoreState(torch.nn.Module):
    def __init__(self, hidden_size, device=None):
        super().__init__()
        self.integrity = torch.nn.Parameter(torch.ones(1, device=device))
        self.ignorance = torch.nn.Parameter(torch.zeros(1, device=device))
        self.irreversibility = torch.nn.Parameter(torch.zeros(1, device=device))
        self.agency = torch.nn.Parameter(torch.ones(1, device=device))
        self.phase = torch.nn.Parameter(torch.zeros(hidden_size, device=device))

    def vector(self):
        return torch.cat(
            [self.integrity,
             self.ignorance,
             self.irreversibility,
             self.agency,
             self.phase],
            dim=0,
        )

    def viable(self):
        return self.agency.item() > 0.0

# ============================================================
# NORMALIZATION
# ============================================================

class RMSNorm(torch.nn.Module):
    def __init__(self, n, eps=1e-5, device=None):
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(n, device=device))

    def forward(self, x):
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t * t, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(x.dtype)

# ============================================================
# ROTARY EMBEDDING
# ============================================================

def _apply_rotary(x, cos, sin):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim, base, device):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.device = device

    def forward(self, q, k):
        n = q.shape[0]
        t = torch.arange(n, device=self.device)
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, device=self.device) / self.head_dim
        )
        inv = 1.0 / freq
        freqs = torch.einsum("i,j->ij", t, inv)
        cos, sin = freqs.cos(), freqs.sin()

        q = _apply_rotary(q.view(n, -1, self.head_dim), cos, sin).view_as(q)
        k = _apply_rotary(k.view(n, -1, self.head_dim), cos, sin).view_as(k)
        return q, k

# ============================================================
# ATTENTION
# ============================================================

def sdpa(q, k, v, scale):
    n, h, qm, d = q.shape
    k = k[:, :, None, :].expand(-1, -1, qm, -1)
    v = v[:, :, None, :].expand(-1, -1, qm, -1)
    mask = torch.triu(q.new_full((n, n), -float("inf")), diagonal=1)
    att = torch.einsum("qhmd,khmd->hmqk", q, k) * scale
    att += mask[None, None]
    w = torch.softmax(att, dim=-1)
    return torch.einsum("hmqk,khmd->qhmd", w, v).reshape(n, -1)

class AttentionBlock(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.norm = RMSNorm(cfg.hidden_size, device=device)
        self.qkv = torch.nn.Linear(
            cfg.hidden_size,
            cfg.head_dim * (cfg.num_attention_heads + 2 * cfg.num_key_value_heads),
            device=device,
            dtype=torch.bfloat16,
        )
        self.out = torch.nn.Linear(
            cfg.hidden_size,
            cfg.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.scale = 1 / math.sqrt(cfg.head_dim)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta, device)

    def forward(self, x):
        t = self.norm(x)
        qkv = self.qkv(t)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q, k = self.rope(q, k)
        return x + self.out(sdpa(q.unsqueeze(2), k, v, self.scale))

# ============================================================
# MLP
# ============================================================

class MLPBlock(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.norm = RMSNorm(cfg.hidden_size, device=device)
        self.fc1 = torch.nn.Linear(
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            device=device,
            dtype=torch.bfloat16,
        )
        self.fc2 = torch.nn.Linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x):
        a, b = self.fc1(self.norm(x)).chunk(2, dim=-1)
        return x + self.fc2(a * torch.sigmoid(a) * (b + 1))

# ============================================================
# TRANSFORMER BLOCK (WORLD-UPDATING)
# ============================================================

class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.attn = AttentionBlock(cfg, device)
        self.mlp = MLPBlock(cfg, device)
        self.hex_proj = torch.nn.Linear(
            cfg.hidden_size + 4,
            cfg.hidden_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x, hexcore: HexCoreState):
        x = self.attn(x)
        x = self.mlp(x)

        activity = x.mean(dim=(0, 1))
        hexcore.phase.data += torch.tanh(activity)
        hexcore.ignorance.data += torch.var(x)
        hexcore.irreversibility.data += torch.relu(torch.norm(activity) - 1.0)
        hexcore.integrity.data -= 0.01 * torch.norm(activity)
        hexcore.agency.data = hexcore.integrity - hexcore.irreversibility

        return x + self.hex_proj(hexcore.vector())

# ============================================================
# WORLD-FIRST MODEL
# ============================================================

class WorldModel(torch.nn.Module):
    def __init__(self, cfg, device):
        super().__init__()

        # IMPORTANT: names must match checkpoint
        self.embedding = torch.nn.Embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )

        self.hexcore = HexCoreState(cfg.hidden_size, device=device)

        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(cfg, device) for _ in range(cfg.num_hidden_layers)]
        )

        self.norm = RMSNorm(cfg.hidden_size, device=device)

        self.unembedding = torch.nn.Linear(
            cfg.hidden_size,
            cfg.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    # ---- WORLD PROBE ----
    def probe(self, embeddings):
        x = embeddings
        for blk in self.blocks:
            x = blk(x, self.hexcore)
            if not self.hexcore.viable():
                break
        return x

    # ---- TOKEN-FREE THINKING ----
    def world_step(self, steps=1):
        dummy = self.hexcore.phase.view(1, 1, -1)
        for _ in range(steps):
            self.probe(dummy)
            if not self.hexcore.viable():
                break

    # ---- TOKEN RENDERING ----
    def render_text(self, tokens: torch.Tensor, max_tokens=16):
        toks = tokens.tolist()
        for _ in range(max_tokens):
            emb = self.embedding(torch.tensor(toks, device=tokens.device))
            self.probe(emb.unsqueeze(0))
            if not self.hexcore.viable():
                break
            logits = self.unembedding(self.norm(emb[-1]))
            toks.append(torch.argmax(logits).item())
        return toks

    # ---- SAFE CHECKPOINT LOAD ----
    @staticmethod
    def from_checkpoint(path, device="cuda"):
        device = torch.device(device)
        with open(os.path.join(path, "config.json")) as f:
            cfg = ModelConfig(**json.load(f))

        model = WorldModel(cfg, device).eval()
        ckpt = Checkpoint(path, device)

        for name, param in model.named_parameters():
            if ckpt.has(name):
                param.data.copy_(ckpt.get(name))
            # HexCore + new params stay initialized

        return model
