
import logging
from typing import List
import torch
import torch.nn as nn
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


class OutputEncoder(nn.Module):

    def __init__(self, embedding_dim: int = 768, model_name: str = "BAAI/bge-large-en-v1.5", device=None):
        super().__init__()
        self.dim = embedding_dim
        self.device = device or torch.device("cpu")
        self.model_name = model_name

        self._st_model = None
        self._st_cls = None
        self.proj = None


        try:
            from sentence_transformers import SentenceTransformer
            self._st_cls = SentenceTransformer
            logger.info(f"OutputEncoder will try model: {self.model_name}")
        except Exception:
            self._st_cls = None
            logger.info("sentence-transformers not available; will use hash embedding fallback.")

    def _load_st_model(self):
      
        if self._st_model is None and self._st_cls is not None:
            try:
                self._st_model = self._st_cls(self.model_name, device=str(self.device))
                out_dim = self._st_model.get_sentence_embedding_dimension()
                if out_dim != self.dim:
                    self.proj = nn.Linear(out_dim, self.dim, bias=False).to(self.device)
                logger.info(f"OutputEncoder initialized with model: {self.model_name} (dim={out_dim}→{self.dim})")
            except Exception as e:
                logger.warning(f"Load sentence-transformers model failed ({self.model_name}): {e}; use fallback.")
                self._st_model = None

    def _hash_embed(self, texts: List[str]) -> torch.Tensor:

        vecs = np.zeros((len(texts), self.dim), dtype=np.float32)
        primes = [1000003, 1000033, 1000037, 1000039]
        for i, t in enumerate(texts):
            s = (t or "").encode("utf-8", errors="ignore")
            h1 = int(hashlib.md5(s).hexdigest(), 16)
            h2 = int(hashlib.sha1(s).hexdigest(), 16)
            h3 = int(hashlib.sha256(s).hexdigest(), 16)
            h4 = int(hashlib.blake2b(s, digest_size=16).hexdigest(), 16)
            hs = [h1, h2, h3, h4]
            for j, hv in enumerate(hs):
                pos = (hv // primes[j]) % self.dim
                vecs[i, pos] += 1.0
            n = np.linalg.norm(vecs[i]) + 1e-8
            vecs[i] /= n
        return torch.tensor(vecs, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def encode_output(self, text: str) -> torch.Tensor:

        return self.encode_outputs([text]).squeeze(0)

    @torch.no_grad()
    def encode_outputs(self, texts: List[str]) -> torch.Tensor:

        if not texts:
            return torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)

        self._load_st_model()

        if self._st_model is not None:
            try:
                embs = self._st_model.encode(
                    texts,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                    device=str(self.device)
                )
                if isinstance(embs, np.ndarray):
                    embs = torch.tensor(embs, dtype=torch.float32, device=self.device)
                if self.proj is not None:
                    embs = self.proj(embs)
                return embs
            except Exception as e:
                logger.warning(f"sentence-transformers encode failed; fallback to hash. Err={e}")

        return self._hash_embed(texts)
