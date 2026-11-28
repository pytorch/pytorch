"""Small recommendation and transformer workloads for Component Benchmark."""

import torch
from torch import nn


class FeatureInteraction(nn.Module):
    def forward(self, embeddings):
        return (embeddings.sum(dim=1).square() - embeddings.square().sum(dim=1)) / 2


class DeepFM(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embedding = nn.Embedding(1000, 16, device=device)
        self.dense = nn.Sequential(
            nn.Linear(8, 32, device=device), nn.ReLU(), nn.Linear(32, 16, device=device)
        )
        self.interaction = FeatureInteraction()
        self.output = nn.Linear(32, 1, device=device)

    def forward(self, dense_features, categorical_features):
        dense = self.dense(dense_features)
        interactions = self.interaction(self.embedding(categorical_features))
        return self.output(torch.cat((dense, interactions), dim=1)).sigmoid()


class Transformer(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.embedding = nn.Embedding(128, 32, device=device)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    32, 4, 64, dropout=0, batch_first=True, device=device
                )
                for _ in range(2)
            ]
        )
        self.output = nn.Linear(32, 128, device=device)

    def forward(self, tokens):
        values = self.embedding(tokens)
        for block in self.blocks:
            values = block(values)
        return self.output(values)


def make_workload(name, device):
    if name == "deepfm":
        model = DeepFM(device)
        inputs = (
            torch.randn(8, 8, device=device),
            torch.randint(1000, (8, 4), device=device),
        )
    else:
        model = Transformer(device)
        inputs = (torch.randint(128, (2, 8), device=device),)
    return model.eval(), inputs
