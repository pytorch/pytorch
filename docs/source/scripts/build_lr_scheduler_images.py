from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    CyclicLR,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)


matplotlib.use("Agg")

LR_SCHEDULER_IMAGE_PATH = Path(__file__).parent / "lr_scheduler_images"

if not LR_SCHEDULER_IMAGE_PATH.exists():
    LR_SCHEDULER_IMAGE_PATH.mkdir()

model = torch.nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.05)

num_epochs = 100

scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=num_epochs // 5)
scheduler2 = ExponentialLR(optimizer, gamma=0.9)

schedulers = [
    (lambda opt: LambdaLR(opt, lr_lambda=lambda epoch: epoch // 30)),
    (lambda opt: MultiplicativeLR(opt, lr_lambda=lambda epoch: 0.95)),
    (lambda opt: StepLR(opt, step_size=30, gamma=0.1)),
    (lambda opt: MultiStepLR(opt, milestones=[30, 80], gamma=0.1)),
    (lambda opt: ConstantLR(opt, factor=0.5, total_iters=40)),
    (lambda opt: LinearLR(opt, start_factor=0.05, total_iters=40)),
    (lambda opt: ExponentialLR(opt, gamma=0.95)),
    (lambda opt: PolynomialLR(opt, total_iters=num_epochs / 2, power=0.9)),
    (lambda opt: CosineAnnealingLR(opt, T_max=num_epochs)),
    (lambda opt: CosineAnnealingWarmRestarts(opt, T_0=20)),
    (lambda opt: CyclicLR(opt, base_lr=0.01, max_lr=0.1, step_size_up=10)),
    (lambda opt: OneCycleLR(opt, max_lr=0.01, epochs=10, steps_per_epoch=10)),
    (lambda opt: ReduceLROnPlateau(opt, mode="min")),
    (lambda opt: ChainedScheduler([scheduler1, scheduler2])),
    (
        lambda opt: SequentialLR(
            opt, schedulers=[scheduler1, scheduler2], milestones=[num_epochs // 5]
        )
    ),
]


def plot_function(scheduler):
    plt.clf()
    plt.grid(color="k", alpha=0.2, linestyle="--")
    lrs = []
    optimizer.param_groups[0]["lr"] = 0.05
    scheduler = scheduler(optimizer)

    plot_path = LR_SCHEDULER_IMAGE_PATH / f"{scheduler.__class__.__name__}.png"
    if plot_path.exists():
        return

    for _ in range(num_epochs):
        lrs.append(optimizer.param_groups[0]["lr"])
        if isinstance(scheduler, ReduceLROnPlateau):
            val_loss = torch.randn(1).item()
            scheduler.step(val_loss)
        else:
            scheduler.step()

    plt.plot(range(num_epochs), lrs)
    plt.title(f"Learning Rate: {scheduler.__class__.__name__}")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.xlim([0, num_epochs])
    plt.savefig(plot_path)
    print(
        f"Saved learning rate scheduler image for {scheduler.__class__.__name__} at {plot_path}"
    )


for scheduler in schedulers:
    plot_function(scheduler)
