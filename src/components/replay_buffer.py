
from collections import deque
from typing import List, Optional
import random
import copy

class EpisodeReplayBuffer:
    """
    sample(k):  List[EpisodeBatch]
    """
    def __init__(self, buffer_size: int = 2000, seed: Optional[int] = None):
        self._buf = deque(maxlen=buffer_size)
        self._rng = random.Random(seed)

    def __len__(self):
        return len(self._buf)

    def insert_episode_batch(self, ep_batch):
        try:
            ep = ep_batch.to("cpu")
        except Exception:
            ep = copy.deepcopy(ep_batch)
        self._buf.append(ep)

    def can_sample(self, k: int) -> bool:
        return len(self._buf) >= max(1, k)

    def sample(self, k: int) -> List:
        k = min(k, len(self._buf))
        return self._rng.sample(list(self._buf), k)