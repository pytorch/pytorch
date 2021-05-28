import torch.nn as nn
import torch.nn.functional as F


class DummyModel(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dense_input_size: int,
        dense_output_size: int,
        sparse: bool
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            num_embeddings, embedding_dim, sparse=sparse
        )
        self.dense = nn.Sequential(*[nn.Linear(dense_input_size, dense_output_size) for _ in range(10)])

    def forward(self, x):
        x = self.embedding(x)
        return F.softmax(self.dense(x), dim=1)
