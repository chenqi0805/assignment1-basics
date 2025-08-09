#!/usr/bin/env python3
"""
Comprehensive training script for transformer language model.

This script provides a complete training loop with configurable hyperparameters,
memory-efficient data loading using np.memmap, checkpointing, and logging.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import logging

import numpy as np
import torch
import torch.nn.functional as F

# Import modules from cs336_basics
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import data_loading
from cs336_basics.checkpointing import save_checkpoint, load_checkpoint
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from cs336_basics.gradient_clipping import gradient_clipping

import wandb

def setup_logging(log_file: Optional[str] = None):
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([] if log_file is None else [logging.FileHandler(log_file)])
        ]
    )
    return logging.getLogger(__name__)


def load_memmap_dataset(data_path: str, dtype=np.int32) -> np.memmap:
    """Load dataset using memory mapping for efficiency."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # Assume data is stored as int32 tokens
    data = np.memmap(data_path, dtype=dtype, mode='r')
    return data


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return torch.exp(torch.tensor(loss)).item()


def evaluate_model(model: TransformerLM, valid_data: np.memmap, 
                  eval_steps: int, batch_size: int, context_length: int, 
                  device: str) -> tuple[float, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for _ in range(eval_steps):
            inputs, targets = data_loading(valid_data, batch_size, context_length, device)
            
            # Forward pass
            logits = model(inputs)
            
            # Reshape for loss computation
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            loss = cross_entropy(logits_flat, targets_flat)
            total_loss += loss.item()
    
    avg_loss = total_loss / eval_steps
    perplexity = compute_perplexity(avg_loss)
    
    model.train()
    return avg_loss, perplexity


def save_config(config: Dict[str, Any], save_path: str):
    """Save training configuration to JSON file."""
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train a transformer language model")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data (memory-mapped)")
    parser.add_argument("--valid_data", type=str, required=True,
                       help="Path to validation data (memory-mapped)")
    
    # Model hyperparameters
    parser.add_argument("--vocab_size", type=int, default=10000,
                       help="Vocabulary size")
    parser.add_argument("--num_layers", type=int, default=4,
                       help="Number of transformer layers")
    parser.add_argument("--context_length", type=int, default=256,
                       help="Context length")
    parser.add_argument("--d_model", type=int, default=512,
                       help="Model dimension")
    parser.add_argument("--num_heads", type=int, default=16,
                       help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=1344,
                       help="Feed-forward dimension")
    parser.add_argument("--theta", type=float, default=10000.0,
                       help="RoPE theta parameter")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4,
                       help="Peak learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=1e-5,
                       help="Minimum learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--beta1", type=float, default=0.9,
                       help="Adam beta1")
    parser.add_argument("--beta2", type=float, default=0.999,
                       help="Adam beta2")
    parser.add_argument("--eps", type=float, default=1e-8,
                       help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping")
    
    # Learning rate schedule
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps")
    parser.add_argument("--max_steps", type=int, default=5000,
                       help="Maximum number of training steps")
    
    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=100,
                       help="Log every N steps")
    parser.add_argument("--eval_interval", type=int, default=1000,
                       help="Evaluate every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                       help="Number of steps for evaluation")
    parser.add_argument("--save_interval", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Log file path")
    
    # Weights & Biases
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="transformer-lm",
                       help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                       help="Weights & Biases run name")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "mps",
                       help="Device to use for training")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file)
    logger.info("Starting training script")
    logger.info(f"Arguments: {args}")
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    save_config(config, os.path.join(args.checkpoint_dir, "config.json"))
    
    # Initialize Weights & Biases if requested
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=config
        )
        logger.info("Initialized Weights & Biases")
    
    # Load datasets with memory mapping
    logger.info("Loading datasets...")
    train_data = load_memmap_dataset(args.train_data)
    valid_data = load_memmap_dataset(args.valid_data)
    logger.info(f"Train data size: {len(train_data):,} tokens")
    logger.info(f"Valid data size: {len(valid_data):,} tokens")
    
    # Initialize model
    logger.info("Initializing model...")
    model = TransformerLM(
        theta=args.theta,
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        context_length=args.context_length,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=args.device
    ).to(args.device)

    model = torch.compile(model, backend="aot_eager")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        start_step = load_checkpoint(args.resume_from, model, optimizer)
        logger.info(f"Resumed from step {start_step}")
    
    # Training loop
    logger.info("Starting training loop...")
    model.train()
    
    for step in range(start_step, args.max_steps):
        start_time = time.time()
        
        # Update learning rate
        current_lr = learning_rate_schedule(
            t=step,
            alpha_max=args.learning_rate,
            alpha_min=args.min_learning_rate,
            t_w=args.warmup_steps,
            t_c=args.max_steps
        )
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Load batch
        try:
            inputs, targets = data_loading(train_data, args.batch_size, args.context_length, args.device)
        except ValueError as e:
            logger.error(f"Error loading data: {e}")
            break
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(inputs)
        
        # Compute loss
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        loss = cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        gradient_clipping(model.parameters(), args.max_grad_norm)
        
        # Optimizer step
        optimizer.step()
        
        step_time = time.time() - start_time
        
        # Logging
        if (step + 1) % args.log_interval == 0:
            perplexity = compute_perplexity(loss.item())
            tokens_per_sec = (args.batch_size * args.context_length) / step_time
            
            log_msg = (f"Step {step + 1:6d}/{args.max_steps} | "
                      f"Loss: {loss.item():.4f} | "
                      f"PPL: {perplexity:.2f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Tokens/s: {tokens_per_sec:.1f} | "
                      f"Time: {step_time:.2f}s")
            
            logger.info(log_msg)
            
            # Log to Weights & Biases
            if args.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/perplexity": perplexity,
                    "train/learning_rate": current_lr,
                    "train/tokens_per_second": tokens_per_sec,
                    "step": step + 1
                })
        
        # Evaluation
        if (step + 1) % args.eval_interval == 0:
            logger.info("Running evaluation...")
            eval_loss, eval_ppl = evaluate_model(
                model, valid_data, args.eval_steps, 
                args.batch_size, args.context_length, args.device
            )
            
            logger.info(f"Evaluation - Loss: {eval_loss:.4f}, Perplexity: {eval_ppl:.2f}")
            
            # Log to Weights & Biases
            if args.use_wandb:
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/perplexity": eval_ppl,
                    "step": step + 1
                })
        
        # Save checkpoint
        if (step + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_step_{step + 1}.pt")
            logger.info(f"Saving checkpoint to {checkpoint_path}")
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
    
    # Final evaluation
    logger.info("Running final evaluation...")
    final_eval_loss, final_eval_ppl = evaluate_model(
        model, valid_data, args.eval_steps, 
        args.batch_size, args.context_length, args.device
    )
    logger.info(f"Final evaluation - Loss: {final_eval_loss:.4f}, Perplexity: {final_eval_ppl:.2f}")
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.checkpoint_dir, "final_checkpoint.pt")
    logger.info(f"Saving final checkpoint to {final_checkpoint_path}")
    save_checkpoint(model, optimizer, args.max_steps, final_checkpoint_path)
    
    logger.info("Training completed!")
    
    # Finish Weights & Biases
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
