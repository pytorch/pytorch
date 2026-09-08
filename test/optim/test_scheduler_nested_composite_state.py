import torch


def test_nested_sequential_scheduler_initializes_last_lr():
    p = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([p], lr=0.1)
    inner = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=2),
            torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9),
        ],
        milestones=[2],
    )
    outer = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [inner, torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)],
        milestones=[3],
    )

    assert outer.get_last_lr() == inner.get_last_lr()
    assert len(outer.get_last_lr()) == 1
