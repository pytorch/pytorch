
import torch.nn as nn


class EmbeddingNetwork1(nn.Module):
    def __init__(self, dim=5):
        super(EmbeddingNetwork1, self).__init__()
        self.emb = nn.Embedding(10, dim)
        self.lin1 = nn.Linear(dim, 1)
        self.seq = nn.Sequential(
            self.emb,
            self.lin1,
        )

    def forward(self, input):
        return self.seq(input)


class EmbeddingNetwork2(nn.Module):

    def __init__(self, in_space=10, dim=3):
        super(EmbeddingNetwork2, self).__init__()
        self.embedding = nn.Embedding(in_space, dim)
        self.seq = nn.Sequential(
            self.embedding,
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, indices):
        return self.seq(indices)
