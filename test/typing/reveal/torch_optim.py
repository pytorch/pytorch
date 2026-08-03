import torch


def foo(opt: torch.optim.Optimizer) -> None:
    opt.zero_grad()


opt_adagrad = torch.optim.Adagrad([torch.tensor(0.0)])
reveal_type(opt_adagrad)  # E: {Adagrad}
foo(opt_adagrad)

opt_adam = torch.optim.Adam([torch.tensor(0.0)], lr=1e-2, eps=1e-6)
reveal_type(opt_adam)  # E: {Adam}
foo(opt_adam)
