# scripts/quick_test.py
# -*- coding: utf-8 -*-
import os
import sys
import argparse
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.append(_PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))

from train import load_config, update_config_with_args, setup_experiment, setup_wandb, run_test
from utils.logging import get_logger

def main():
    parser = argparse.ArgumentParser(description="Quick test ECON with saved checkpoints.")
    parser.add_argument("--config", type=str, default=os.path.join(_SCRIPT_DIR, "quick_train.yaml"),
                        help="Path to YAML config used in training (will override split to test).")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "test_models_small", "final"),
        help="Directory containing saved .th files."
    )
    parser.add_argument("--test_nepisode", type=int, default=100, help="Number of test episodes.")
    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    args = parser.parse_args()

    logger = get_logger()
    cfg = load_config(args.config)

    # override to test split
    if hasattr(cfg, "env_args"):
        cfg.env_args.dataset_split = "test"
        cfg.env_args.use_random_sampling = False
        cfg.env_args.loop_dataset = False

    if not hasattr(cfg, "wandb"):
        import types
        cfg.wandb = types.SimpleNamespace()
    cfg.wandb.use_wandb = bool(args.use_wandb)
    setattr(cfg, "test_nepisode", int(args.test_nepisode))

    runner, mac, learner, logger, device = setup_experiment(cfg)

    ckpt_dir = Path(args.checkpoint)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

    th_files = sorted([p.name for p in ckpt_dir.glob("*.th")])
    logger.info(f"Found checkpoint files: {th_files if th_files else '(none)'} in {ckpt_dir}")

    loaded = False
    try:
        if hasattr(learner, "load_models"):
            learner.load_models(str(ckpt_dir))
            logger.info("Loaded weights via learner.load_models()")
            loaded = True
    except Exception as e:
        logger.warning(f"learner.load_models failed: {e}")

    if not loaded:
        try:
            if hasattr(mac, "load_models"):
                mac.load_models(str(ckpt_dir))
                logger.info("Loaded weights via mac.load_models()")
                loaded = True
        except Exception as e:
            logger.warning(f"mac.load_models failed: {e}")

    setup_wandb(cfg, logger)

    logger.info("Running test...")
    run_test(cfg, runner, learner, logger, device)
    logger.info("Test finished.")

if __name__ == "__main__":
    main()
