import torch.nn as nn
import torch.nn.functional as F


class DummyModel(nn.Module):
    def __init__(
        self,
        num_embeddings=10,
        embedding_dim=10,
        dense_input_size=10,
        dense_output_size=10,
        sparse=True
    ):
        super().__init__()
        self.embedding = nn.EmbeddingBag(
            num_embeddings, embedding_dim, sparse=sparse
        )
        self.dense = nn.Sequential(*[nn.Linear(dense_input_size, dense_output_size) for _ in range(10)])

    def forward(self, x):
        x = self.embedding(x)
        return F.softmax(self.dense(x), dim=1)
