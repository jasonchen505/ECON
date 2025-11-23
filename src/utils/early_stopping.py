
from dataclasses import dataclass
from typing import Optional

@dataclass
class EarlyStoppingConfig:
    patience: int = 10            # number of eval steps with no improvement before stopping
    mode: str = "max"             # "max" (e.g., accuracy/reward) or "min" (e.g., loss)
    min_delta: float = 0.0        # minimum change to qualify as an improvement
    warmup: int = 0               # number of evals to skip early stopping checks

class EarlyStopping:
    def __init__(self, cfg: EarlyStoppingConfig):
        if cfg.mode not in ("max", "min"):
            raise ValueError("EarlyStoppingConfig.mode must be 'max' or 'min'")
        self.cfg = cfg
        self.best: Optional[float] = None
        self.counter: int = 0
        self._cmp = (lambda curr, best: curr > best + cfg.min_delta) if cfg.mode == "max"                         else (lambda curr, best: curr < best - cfg.min_delta)
        self._eval_calls = 0

    def step(self, metric_value: float):
        """
        Call at the end of each evaluation. Returns (should_stop, is_best).
        """
        self._eval_calls += 1

        # During warmup, just record best but do not count patience
        if self._eval_calls <= self.cfg.warmup:
            if self.best is None or self._cmp(metric_value, self.best):
                self.best = metric_value
            return False, self._eval_calls == self.cfg.warmup

        if self.best is None or self._cmp(metric_value, self.best):
            self.best = metric_value
            self.counter = 0
            return False, True  # improved
        else:
            self.counter += 1
            should_stop = self.counter >= self.cfg.patience
            return should_stop, False
