import time

import torch
import torch.nn as nn
import torchvision.models as models

from functorch import grad, make_functional, vmap
from opacus import PrivacyEngine
from opacus.utils.module_modification import convert_batchnorm_modules

device = "cuda"
batch_size = 128
torch.manual_seed(0)

model_functorch = convert_batchnorm_modules(models.resnet18(num_classes=10))
model_functorch = model_functorch.to(device)
criterion = nn.CrossEntropyLoss()

images = torch.randn(batch_size, 3, 32, 32, device=device)
targets = torch.randint(0, 10, (batch_size,), device=device)
func_model, weights = make_functional(model_functorch)


def compute_loss(weights, image, target):
    images = image.unsqueeze(0)
    targets = target.unsqueeze(0)
    output = func_model(weights, images)
    loss = criterion(output, targets)
    return loss


def functorch_per_sample_grad():
    compute_grad = grad(compute_loss)
    compute_per_sample_grad = vmap(compute_grad, (None, 0, 0))

    start = time.time()
    result = compute_per_sample_grad(weights, images, targets)
    torch.cuda.synchronize()
    end = time.time()

    return result, end - start  # end - start in seconds


torch.manual_seed(0)
model_opacus = convert_batchnorm_modules(models.resnet18(num_classes=10))
model_opacus = model_opacus.to(device)
criterion = nn.CrossEntropyLoss()
for p_f, p_o in zip(model_functorch.parameters(), model_opacus.parameters()):
    assert torch.allclose(p_f, p_o)  # Sanity check

privacy_engine = PrivacyEngine(
    model_opacus,
    sample_rate=0.01,
    alphas=[10, 100],
    noise_multiplier=1,
    max_grad_norm=10000.0,
)


def opacus_per_sample_grad():
    start = time.time()
    output = model_opacus(images)
    loss = criterion(output, targets)
    loss.backward()
    torch.cuda.synchronize()
    end = time.time()
    expected = [p.grad_sample for p in model_opacus.parameters()]
    for p in model_opacus.parameters():
        delattr(p, "grad_sample")
        p.grad = None
    return expected, end - start


for _ in range(5):
    _, seconds = functorch_per_sample_grad()
    print(seconds)

result, seconds = functorch_per_sample_grad()
print(seconds)

for _ in range(5):
    _, seconds = opacus_per_sample_grad()
    print(seconds)

expected, seconds = opacus_per_sample_grad()
print(seconds)

result = [r.detach() for r in result]
print(len(result))

# TODO: The following shows that the per-sample-grads computed are different.
# This concerns me a little; we should compare to a source of truth.
# for i, (r, e) in enumerate(list(zip(result, expected))[::-1]):
#     if torch.allclose(r, e, rtol=1e-5):
#         continue
#     print(-(i+1), ((r - e)/(e + 0.000001)).abs().max())
