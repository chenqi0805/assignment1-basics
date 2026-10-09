"""Training loop putting every component together (spec §5.3, problem
training_together).

CLI-configurable training of the Transformer LM on a memmap'd token stream:
AdamW with linear warmup + cosine decay, gradient clipping, periodic
validation evaluation, console + optional Weights & Biases logging, and
checkpoint/resume for preemptible runs.

Typical flow:
    python -m cs336_basics.tokenize_corpus --input <corpus> --output-dir <dir>
    python -m cs336_basics.train --train-data <dir>/train_tokens.npy \
        --val-data <dir>/val_tokens.npy --vocab-size 10000 --device cuda

The spec's §7.2 TinyStories reference hyperparameters are the defaults
(d_model 512, d_ff 1344, 4 layers, 16 heads, RoPE theta 10000); tune the
optimizer hyperparameters with the §7 learning-rate sweep.
"""

import argparse
import os
import time

import numpy as np
import torch

from .checkpoint import load_checkpoint, save_checkpoint
from .model import TransformerLM
from .nn_utils import cross_entropy, get_batch, gradient_clipping
from .optimizer import AdamW, get_lr_cosine_schedule


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", required=True, help="Token .npy file for training")
    parser.add_argument("--val-data", required=True, help="Token .npy file for validation")
    # Model hyperparameters (spec §7.2 TinyStories reference values).
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    # Optimizer hyperparameters (tune via the §7 learning-rate sweep).
    parser.add_argument("--max-learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # Run control.
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", default="", help="cpu | cuda | mps (empty: cuda if available, else cpu)"
    )
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=10, help="Validation batches per eval")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--checkpoint-path", default="checkpoint.pt")
    parser.add_argument("--resume", help="Checkpoint path to resume from")
    parser.add_argument(
        "--overfit-single-batch",
        action="store_true",
        help="Sample one minibatch and reuse it every step (spec's overfitting debug recipe)",
    )
    parser.add_argument("--wandb-project", help="Optional Weights & Biases project name")
    return parser


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.memmap,
    args: argparse.Namespace,
) -> float:
    """Mean validation cross-entropy over a fixed number of sampled batches."""
    model.eval()
    losses = []
    for _ in range(args.eval_iters):
        inputs, targets = get_batch(
            dataset, args.batch_size, args.context_length, args.device
        )
        logits = model(inputs)
        losses.append(cross_entropy(logits, targets).item())
    model.train()
    return float(np.mean(losses))


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if not args.device:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cuda":
        # Spec p.41 warning: TF32 kernels are silently broken on mps as of torch 2.6.
        torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # np.load with mmap_mode gives a np.memmap that honors the .npy dtype
    # header (plain np.memmap would reinterpret the file as raw uint8).
    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")
    print(
        f"Loaded {len(train_data):,} train / {len(val_data):,} val tokens; "
        f"total tokens = batch {args.batch_size} x steps {args.max_steps} x "
        f"context {args.context_length} = {args.batch_size * args.max_steps * args.context_length:,}"
    )

    device = torch.device(args.device)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters on {args.device}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )
    start_iter = 0
    if args.resume:
        start_iter = load_checkpoint(args.resume, model, optimizer)
        print(f"Resumed from {args.resume} at iteration {start_iter}")

    wandb_run = None
    if args.wandb_project:
        import wandb

        wandb_run = wandb.init(project=args.wandb_project, config=vars(args))

    def log(values: dict, step: int) -> None:
        print(" | ".join(f"{key}={value:.4f}" if isinstance(value, float) else
                         f"{key}={value}" for key, value in values.items()))
        if wandb_run is not None:
            wandb_run.log(values, step=step)

    fixed_batch = None
    if args.overfit_single_batch:
        fixed_batch = get_batch(train_data, args.batch_size, args.context_length, args.device)
        print("Overfit mode: reusing a single minibatch for all steps")

    model.train()
    os.makedirs(os.path.dirname(args.checkpoint_path) or ".", exist_ok=True)
    start_time = time.perf_counter()
    for step in range(start_iter, args.max_steps):
        lr = get_lr_cosine_schedule(
            step,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_steps,
            args.max_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = fixed_batch or get_batch(
            train_data, args.batch_size, args.context_length, args.device
        )
        step_start = time.perf_counter()
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradient_clipping(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step % args.log_interval == 0:
            tokens_per_sec = args.batch_size * args.context_length / (
                time.perf_counter() - step_start
            )
            log(
                {
                    "step": step,
                    "lr": lr,
                    "train_loss": loss.item(),
                    "tokens_per_sec": tokens_per_sec,
                    "elapsed_s": time.perf_counter() - start_time,
                },
                step=step,
            )
        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            val_loss = evaluate(model, val_data, args)
            log({"step": step, "val_loss": val_loss}, step=step)
        if step % args.checkpoint_interval == 0 or step == args.max_steps - 1:
            save_checkpoint(model, optimizer, step, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
