import torch


def test_scheduler_base_lr_does_not_alias_tensor_initial_lr():
    param = torch.nn.Parameter(torch.tensor(1.0))
    lr = torch.tensor(0.1)
    optimizer = torch.optim.SGD([param], lr=lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=1)],
        milestones=[],
    )

    optimizer.param_groups[0]["lr"].add_(1.0)

    assert scheduler.base_lrs[0].item() == 0.1


def test_scheduler_fix_installation_is_idempotent():
    from torch.optim import _scheduler_fixes

    init = torch.optim.lr_scheduler.LRScheduler.__init__
    _scheduler_fixes.install()
    assert torch.optim.lr_scheduler.LRScheduler.__init__ is init


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
