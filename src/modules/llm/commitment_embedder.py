
import logging
from typing import List
import torch
import torch.nn as nn

from .llm_wrapper import LLMConfig
from modules.text_encoders.output_encoder import OutputEncoder

logger = logging.getLogger(__name__)


class CommitmentEmbedder(nn.Module):

    def __init__(self, args, llm_cfg: LLMConfig):
        super().__init__()
        self.dim = getattr(args, "commitment_embedding_dim", 768)
        self.device = getattr(args, "device", torch.device("cpu"))
        self.model_name = getattr(args, "commitment_embedding_model_name", "BAAI/bge-large-en-v1.5")

        self.encoder = OutputEncoder(
            embedding_dim=self.dim,
            model_name=self.model_name,
            device=self.device,
        )

    @torch.no_grad()
    def embed_commitments(self, texts: List[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)
        return self.encoder.encode_outputs(texts)