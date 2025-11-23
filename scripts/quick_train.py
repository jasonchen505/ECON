# scripts/quick_train.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.append(_PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))
from train import load_config, setup_experiment, run_training, setup_wandb
from utils.logging import get_logger

def main():
    parser = argparse.ArgumentParser(description="Quick train 100 episodes.")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "quick_train.yaml"),
        help="Path to YAML config (should set n_actions consistently)."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    
    logger = get_logger()
    logger.info("Setting up experiment environment...")
    runner, mac, learner, logger, device = setup_experiment(cfg)

   
    logger.info("Starting training...")
    setup_wandb(cfg, logger)
    run_training(cfg, runner, learner, logger, device)
    logger.info("Training finished.")

    ckpt_root = Path(getattr(getattr(cfg, "logging", object()), "checkpoint_path", "./models"))
    logger.info(f"Checkpoints saved under: {ckpt_root.resolve()}")

if __name__ == "__main__":
    main()
