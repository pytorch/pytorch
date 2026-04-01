#!/usr/bin/env python3
"""Standalone CUDA training worker — completely independent from torch_vulkan.

Must be run with a Python environment that does NOT have torch_vulkan installed
as a package (pip install -e .), otherwise the entry_points registration will
auto-load PrivateUse1 and conflict with CUDA autograd.

Usage: python cuda_worker.py <config_json_path> <output_json_path>
"""
import gc
import gzip
import json
import math
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── MNIST Loading (standalone) ─────────────────────────────────

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
}
MNIST_DIR = Path(__file__).parent.parent / "data" / "mnist"


def _download_mnist():
    import urllib.request
    MNIST_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MNIST_URLS.items():
        path = MNIST_DIR / f"{name}.gz"
        if not path.exists():
            urllib.request.urlretrieve(url, path)


def load_mnist(max_samples=None):
    _download_mnist()
    with gzip.open(MNIST_DIR / "train_images.gz", "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        images = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).reshape(*dims).float() / 255.0
    with gzip.open(MNIST_DIR / "train_labels.gz", "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        labels = torch.frombuffer(bytearray(f.read()), dtype=torch.uint8).long()
    images = images.unsqueeze(1)
    if max_samples:
        images = images[:max_samples]
        labels = labels[:max_samples]
    return images, labels


# ── Model Definitions (copied from test_mnist_training.py) ─────

class MLPAllActivations(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.gn1 = nn.GroupNorm(8, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_relu = nn.Linear(8, 8)
        self.fc_sigmoid = nn.Linear(8, 8)
        self.fc_tanh = nn.Linear(8, 8)
        self.fc_gelu = nn.Linear(8, 8)
        self.fc_silu = nn.Linear(8, 8)
        self.fc_elu = nn.Linear(8, 8)
        self.fc_leaky = nn.Linear(8, 8)
        self.fc_mish = nn.Linear(8, 8)
        self.fc3 = nn.Linear(64, 32)
        self.prelu = nn.PReLU(32)
        self.fc_out = nn.Linear(32, 10)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.gn1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        chunks = x.chunk(8, dim=1)
        c0 = F.relu(self.fc_relu(chunks[0]))
        c1 = torch.sigmoid(self.fc_sigmoid(chunks[1]))
        c2 = torch.tanh(self.fc_tanh(chunks[2]))
        c3 = F.gelu(self.fc_gelu(chunks[3]))
        c4 = F.silu(self.fc_silu(chunks[4]))
        c5 = F.elu(self.fc_elu(chunks[5]))
        c6 = F.leaky_relu(self.fc_leaky(chunks[6]))
        c7 = F.mish(self.fc_mish(chunks[7]))
        x = torch.cat([c0, c1, c2, c3, c4, c5, c6, c7], dim=1)
        x = x + 0.01 * torch.sin(x)
        x = x * (1.0 + 0.01 * torch.cos(x))
        x = torch.clamp(x, -5.0, 5.0)
        x = self.fc3(x)
        x = self.prelu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x


class CNNBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        self.conv_dw = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.conv_pw = nn.Conv2d(64, 32, 1)
        self.conv_dilated = nn.Conv2d(32, 32, 3, padding=2, dilation=2)
        self.gn3 = nn.GroupNorm(8, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), value=0)
        x = x[:, :, 1:-1, 1:-1]
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.avg_pool2d(x, 2)
        x = F.relu(self.conv_dw(x))
        x = F.relu(self.conv_pw(x))
        x = F.relu(self.gn3(self.conv_dilated(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(4, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(4, ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = out + identity
        return F.relu(out)


class ResNetMini(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.GroupNorm(4, 16), nn.ReLU())
        self.res1 = ResBlock(16)
        self.res2 = ResBlock(16)
        self.down_conv = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.up_deconv = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)
        self.up_proj = nn.Conv2d(32, 16, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        d = self.down_conv(x)
        u1 = F.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        u1 = self.up_proj(u1)
        u2 = self.up_deconv(d)
        x = u1 + u2
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class TransformerClassifier(nn.Module):
    def __init__(self, patch_size=7, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.patch_size = patch_size
        num_patches = (28 // patch_size) ** 2
        patch_dim = patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, d_model) * 0.02)
        self.norm_pre = nn.LayerNorm(d_model)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn_qkv': nn.Linear(d_model, 3 * d_model),
                'attn_proj': nn.Linear(d_model, d_model),
                'norm2': nn.LayerNorm(d_model),
                'ff1': nn.Linear(d_model, d_model * 4),
                'ff2': nn.Linear(d_model * 4, d_model),
            }))
        self.norm_out = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10)
        self.nhead = nhead
        self.head_dim = d_model // nhead

    def forward(self, x):
        B = x.size(0)
        p = self.patch_size
        x = x.squeeze(1)
        patches = []
        for i in range(0, 28, p):
            for j in range(0, 28, p):
                patches.append(x[:, i:i+p, j:j+p].reshape(B, -1))
        x = torch.stack(patches, dim=1)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.norm_pre(x)
        for layer in self.layers:
            h = layer['norm1'](x)
            qkv = layer['attn_qkv'](h).reshape(B, -1, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(B, -1, self.nhead * self.head_dim)
            x = x + layer['attn_proj'](attn)
            h = layer['norm2'](x)
            h = F.gelu(layer['ff1'](h))
            x = x + layer['ff2'](h)
        x = self.norm_out(x)
        x = x[:, 0]
        x = self.head(x)
        return x


class AdvancedOpsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.hardswish(x)
        x = torch.clamp(x, -5, 5)
        x = x.flip(dims=[1])
        x = x.roll(shifts=1, dims=1)
        x = x.unsqueeze(2).expand(-1, -1, 2)
        x = x.mean(dim=2)
        x = self.fc2(x)
        branch1 = F.softplus(x[:, :32])
        branch2 = F.hardsigmoid(x[:, 32:])
        x = torch.cat([branch1, branch2], dim=1)
        mask = x > 0.5
        x = torch.where(mask, x, x * 0.2)
        x = self.fc3(x)
        return x


CLS_MAP = {
    "MLPAllActivations": MLPAllActivations,
    "CNNBasic": CNNBasic,
    "ResNetMini": ResNetMini,
    "TransformerClassifier": TransformerClassifier,
    "AdvancedOpsModel": AdvancedOpsModel,
}


def train_model(model, images, labels, device, dtype=torch.float32,
                epochs=3, batch_size=128, lr=0.01, momentum=0.0):
    loss_fn = nn.CrossEntropyLoss().to(device)
    images_dev = images.to(device)
    labels_dev = labels.to(device)
    model = model.to(dtype).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    epoch_losses, epoch_accs = [], []
    n = images_dev.size(0)

    for epoch in range(epochs):
        total_loss, correct, batches = 0.0, 0, 0
        for i in range(0, n, batch_size):
            x = images_dev[i:i+batch_size].to(dtype)
            y = labels_dev[i:i+batch_size]
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.float(), y)
            loss.backward()
            if dtype in (torch.float16, torch.bfloat16):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            preds = logits.detach().float().argmax(dim=1)
            correct += (preds == y).sum().item()
            batches += 1
        epoch_losses.append(total_loss / batches)
        epoch_accs.append(correct / n)
    return epoch_losses, epoch_accs


def main():
    config_path = sys.argv[1]
    output_path = sys.argv[2]

    with open(config_path) as f:
        config = json.load(f)

    dtype = getattr(torch, config["dtype"])
    images, labels = load_mnist(max_samples=config.get("max_samples", 1000))

    try:
        torch.manual_seed(42)
        model = CLS_MAP[config["cls_name"]](**config.get("kwargs", {}))
        losses, accs = train_model(
            model, images, labels, torch.device("cuda"),
            dtype=dtype, **config["train_kwargs"]
        )
        result = {
            "losses": [float(x) for x in losses],
            "accs": [float(x) for x in accs],
            "converged": losses[-1] < losses[0],
        }
    except Exception as e:
        import traceback
        result = {"error": str(e)[:300]}

    with open(output_path, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
