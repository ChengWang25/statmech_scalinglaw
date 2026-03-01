"""Train GPT-2-like causal transformers from scratch on token ID datasets."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            )
            self.register_buffer("bias_mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias_mask[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.size()
        if T > self.config.block_size:
            raise ValueError("Sequence length exceeds block_size")

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), reduction="mean"
            )
        return logits, loss


def model_size_config(name: str) -> dict[str, int]:
    presets = {
        "small": {"n_layer": 6, "n_head": 4, "n_embd": 256},
        "medium": {"n_layer": 8, "n_head": 6, "n_embd": 384},
        "large": {"n_layer": 12, "n_head": 8, "n_embd": 512},
    }
    if name not in presets:
        raise ValueError(f"Unknown model preset {name}")
    return presets[name]


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    xb = torch.from_numpy(x.astype(np.int64)).to(device)
    yb = torch.from_numpy(y.astype(np.int64)).to(device)
    return xb, yb


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    batch_size: int,
    block_size: int,
    eval_iters: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, batch_size, block_size, device)
            with torch.autocast(
                device_type="cuda", enabled=use_amp, dtype=amp_dtype
            ):
                _, loss = model(xb, yb)
            losses.append(float(loss.item()))
        out[split] = float(np.mean(losses))
    model.train()
    return out


def get_lr(it: int, cfg: argparse.Namespace) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / max(1, cfg.warmup_iters)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tokens = np.load(args.train_tokens)
    val_tokens = np.load(args.val_tokens)
    if args.train_tokens_limit is not None and args.train_tokens_limit > 0:
        train_tokens = train_tokens[: args.train_tokens_limit]

    if len(train_tokens) <= args.block_size + 1:
        raise ValueError("train split too small for chosen block_size")
    if len(val_tokens) <= args.block_size + 1:
        raise ValueError("val split too small for chosen block_size")

    if args.vocab_size is None:
        vocab_size = int(max(train_tokens.max(), val_tokens.max()) + 1)
    else:
        vocab_size = args.vocab_size

    if args.model_preset:
        preset = model_size_config(args.model_preset)
        n_layer, n_head, n_embd = (
            preset["n_layer"],
            preset["n_head"],
            preset["n_embd"],
        )
    else:
        n_layer, n_head, n_embd = args.n_layer, args.n_head, args.n_embd

    gpt_cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=args.dropout,
        bias=args.bias,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_amp = device.type == "cuda" and args.amp

    model = GPT(gpt_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    best_val = float("inf")
    iter_num = 0
    t0 = time.time()

    log_path = out_dir / "train_log.jsonl"
    if log_path.exists() and args.overwrite:
        log_path.unlink()

    def save_ckpt(name: str) -> None:
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iter_num": iter_num,
            "best_val": best_val,
            "config": asdict(gpt_cfg),
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / name)

    for iter_num in range(1, args.max_iters + 1):
        lr = get_lr(iter_num, args) if args.decay_lr else args.learning_rate
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0

        for _micro in range(args.gradient_accumulation_steps):
            xb, yb = get_batch(
                train_tokens, args.batch_size, args.block_size, device
            )
            with torch.autocast(device_type="cuda", enabled=use_amp, dtype=amp_dtype):
                _, loss = model(xb, yb)
                loss = loss / args.gradient_accumulation_steps
            loss_accum += float(loss.item())
            scaler.scale(loss).backward()

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        if iter_num % args.eval_interval == 0 or iter_num == args.max_iters:
            losses = estimate_loss(
                model,
                train_tokens,
                val_tokens,
                args.batch_size,
                args.block_size,
                args.eval_iters,
                device,
                use_amp,
                amp_dtype,
            )
            elapsed = time.time() - t0
            rec = {
                "iter": iter_num,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "lr": lr,
                "elapsed_sec": elapsed,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
            print(json.dumps(rec))

            save_ckpt("ckpt_last.pt")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_ckpt("ckpt_best.pt")

    final = {
        "best_val_loss": float(best_val),
        "final_iter": int(iter_num),
        "device": str(device),
        "model_config": asdict(gpt_cfg),
        "num_train_tokens": int(len(train_tokens)),
        "num_val_tokens": int(len(val_tokens)),
        "args": vars(args),
    }
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    return final


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train GPT-2-like model from scratch")
    p.add_argument("--train-tokens", type=Path, required=True)
    p.add_argument("--val-tokens", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)

    p.add_argument("--model-preset", type=str, default="small", choices=["small", "medium", "large"])
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-embd", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=None)
    p.add_argument("--block-size", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--bias", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gradient-accumulation-steps", type=int, default=1)
    p.add_argument("--max-iters", type=int, default=2000)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=40)

    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-iters", type=int, default=200)
    p.add_argument("--lr-decay-iters", type=int, default=2000)
    p.add_argument("--decay-lr", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-tokens-limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    result = run_training(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
