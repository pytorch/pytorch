import torch


def foo(opt: torch.optim.Optimizer) -> None:
    opt.zero_grad()

opt_adagrad = torch.optim.Adagrad([torch.tensor(0.0)])
foo(opt_adagrad)

opt_adam = torch.optim.Adam([torch.tensor(0.0)], lr=1e-2, eps=1e-6)
foo(opt_adam)
