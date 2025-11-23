

import os
import sys
import yaml
from types import SimpleNamespace
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
sys.path.append(_PROJECT_ROOT)
sys.path.append(os.path.join(_PROJECT_ROOT, "src"))

from train import load_config, setup_experiment
from runners import REGISTRY as r_REGISTRY
from utils.logging import get_logger

def _ns(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_ns(x) for x in d]
    return d

def _to_plain(obj):
    if isinstance(obj, SimpleNamespace):
        return {k: _to_plain(getattr(obj, k)) for k in vars(obj)}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(x) for x in obj]
    return obj

def _clone_config_for_split(cfg_ns: SimpleNamespace, split: str, ckpt_dir: str, log_dir: str):
    cfg = _to_plain(cfg_ns)
    cfg["env_args"]["dataset_split"] = split
    cfg["env_args"]["loop_dataset"] = False
    cfg["logging"]["checkpoint_path"] = ckpt_dir
    cfg["logging"]["log_path"] = log_dir
    return _ns(cfg)

def run_all(split_cfg: SimpleNamespace, learner=None, mac=None):
    logger = get_logger()
    runner = r_REGISTRY[split_cfg.runner](args=split_cfg, logger=logger)
    if mac is not None:
       
        runner.setup(mac=mac, scheme=mac.scheme if hasattr(mac, "scheme") else None,
                     preprocess=None, groups={"agents": split_cfg.n_agents})
    else:
   
        from controllers import REGISTRY as mac_REGISTRY
        mac = mac_REGISTRY[split_cfg.mac](runner._build_scheme(), runner._build_groups(), split_cfg)
        runner.setup(runner._build_scheme(), runner._build_groups(), None, mac)

    n = 0
    while True:
        try:
            runner.run(test_mode=(split_cfg.env_args.dataset_split != "train"))
            n += 1
            if n % 100 == 0:
                logger.info(f"[{split_cfg.env_args.dataset_split}] processed {n} episodes.")
        except StopIteration:
            logger.info(f"[{split_cfg.env_args.dataset_split}] finished: {n} episodes.")
            break
    return mac

def main():

    candidates = [
        os.path.join(_PROJECT_ROOT, "configs/full_train.yaml"),
        os.path.join(_PROJECT_ROOT, "scripts/quick_train.yaml"),
        os.path.join(_PROJECT_ROOT, "src/config/config.yaml"),
        os.path.join(_PROJECT_ROOT, "test_config.yaml"),
    ]
    cfg_path = None
    for p in candidates:
        if os.path.isfile(p):
            cfg_path = p
            break
    if cfg_path is None:
        raise FileNotFoundError("未找到可用的配置文件（尝试了 configs/full_train.yaml, scripts/quick_train.yaml, src/config/config.yaml, test_config.yaml）")

    cfg = load_config(cfg_path)

  
    train_cfg = _clone_config_for_split(cfg, "train", "./models_full/train", "./logs_full/train")
    runner, mac, learner, logger, device = setup_experiment(train_cfg)
   
    run_all(train_cfg, learner=learner, mac=mac)


    test_cfg = _clone_config_for_split(cfg, "test", "./models_full/test", "./logs_full/test")
 
    run_all(test_cfg, learner=None, mac=mac)

if __name__ == "__main__":
    main()