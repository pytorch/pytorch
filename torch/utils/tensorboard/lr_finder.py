from functools import partial
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import math
import torch

from .writer import SummaryWriter

class Smoothener(object):
    """Helper class for creating a smooth moving average for a value (loss, etc) using `beta`.
    """
    def __init__(self, beta=0.98):
        self.beta, self.n, self.mov_avg = beta, 0, 0

    def smoothen(self, val):
        """Add `val` to calculate updated smoothed value.
        """
        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        return self.mov_avg / (1 - self.beta ** self.n)

class LRFinder(object):
    r"""The `LRFinder` class provides a means of quickly determining an optimal learning rate based on work that was
        initially described in the paper `Cyclical Learning Rates for Training Neural Networks`_.

    This implementation was adapted from the implementation from the fastai library: `fastai/fastai`_

    .. _Cyclical Learning Rates for Training Neural Networks:
        https://arxiv.org/abs/1506.01186
    .. _fastai/fastai:
        https://github.com/fastai/fastai/
    """
    def __init__(self,
                 model,
                 train_dl,
                 opt,
                 loss_func,
                 start_lr=1e-7,
                 end_lr=10.,
                 num_steps=100,
                 log_dir=None,
                 stop_early=True):
        """Args:
          model (nn.Module): The model to be trained
          train_dl (DataLoader): The training data
          opt (Optimizer): The optimizer used for training
          loss_func (func(nn.Module, torch.Tensor, torch.Tensor) -> torch.Tensor): A function that takes a model
              (i.e. the model provided) along with an x batch and a y batch (i.e. as from train_dl) and produces a
              scalar loss.
              Example:
                  >>> import torch.nn.functional as F
                  >>> def loss_func(model, xb, yb):
                  >>>     logits = model(xb)
                  >>>     return F.cross_entropy(logits, yb)

          start_lr (float): The learning rate at which to start the range test
              Default: 1e-7
          end_lr (float): The learning rate at which to end the range test
              Default: 10.
          num_steps (int): The number of total steps in the range test
              Default: 100
          log_dir (str): The logdir that Tensorboard is running on
              If no log_dir is provided then it will default to the ./runs/ directory
              Default: None
          stop_early (bool): Flag representing whether to stop early when the loss begins to diverge
              Default: True
        """

        self.model, self.train_dl, self.loss_func, self.num_steps = model, train_dl, loss_func, num_steps
        self.has_run, self.losses, self.lrs = False, [], []
        self.log_dir, self.num_runs = log_dir, 0
        self.opt, self.scheduler = self.__get_opt_and_scheduler(opt, start_lr, end_lr)
        self.stop_early = stop_early

    def __annealing_exp(self, pct, start, end):
        "Exponentially anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        return (end / start) ** pct

    def __get_opt_and_scheduler(self, opt, start_lr, end_lr):
        torch.save(opt.state_dict(), "opt_initial")
        for g in opt.param_groups:
            g['initial_lr'] = start_lr
            g['lr'] = start_lr

        anneal_func = partial(self.__annealing_exp, start=start_lr, end=end_lr)
        def schedule_fn(step):
            return anneal_func(step / float(self.num_steps))
        scheduler = LambdaLR(opt, schedule_fn)
        torch.save(opt.state_dict(), "opt_start_find")
        torch.save(scheduler.state_dict(), "scheduler_start_find")
        return opt, scheduler

    def __split_list(self, vals, skip_start, skip_end):
        return vals[skip_start:-skip_end] if skip_end > 0 else vals[skip_start:]

    def find(self):
        """Runs the LR range test
        """
        self.has_run = False
        torch.save(self.model.state_dict(), "tmp_model")
        self.model.train()
        self.opt.load_state_dict(torch.load("opt_start_find"))
        self.scheduler.load_state_dict(torch.load("scheduler_start_find"))

        # Train model and record loss vs. lr
        smoothener = Smoothener()
        self.losses = []
        self.lrs = []
        n_epochs = math.ceil(self.num_steps / len(self.train_dl))
        steps = 0
        for _ in range(n_epochs):
            for xb, yb in self.train_dl:
                steps += 1
                if steps > self.num_steps:
                    break

                for g in self.opt.param_groups:
                    lr = g.get('lr')
                    if lr is not None:
                        break
                loss = self.loss_func(self.model, xb, yb)
                loss.backward()
                self.opt.step()
                self.scheduler.step()
                self.opt.zero_grad()

                smooth_loss = smoothener.smoothen(loss)
                if steps == 1 or smooth_loss < best_loss:
                    best_loss = smooth_loss
                if self.stop_early and (smooth_loss > 4 * best_loss or torch.isnan(smooth_loss)):
                    stop = True
                    break
                else:
                    stop = False
                self.losses.append(smooth_loss.item())
                self.lrs.append(lr)
            if stop:
                break

        # Reload initial state of model and opt
        self.model.load_state_dict(torch.load("tmp_model"))
        self.model.eval()
        self.opt.load_state_dict(torch.load("opt_initial"))
        self.has_run = True

    def plot(self, skip_start=10, skip_end=5, push_to_tensorboard=True):
        """Plots the results of the LR range test.

        Args:
            skip_start (int): The number of points to skip from the start of the plot
            skip_end (int): The number of points to skip from the end of the plot
            push_to_tensorboard (bool): Flag representing whether to push the generated plot to Tensorboard
        """
        if not self.has_run:
            raise Exception("You need to run find() first!")
        lrs = self.__split_list(self.lrs, skip_start, skip_end)
        losses = self.__split_list(self.losses, skip_start, skip_end)
        fig, ax = plt.figure(), plt.gca()
        ax.plot(lrs, losses)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Learning Rate")
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        if push_to_tensorboard:
            writer = SummaryWriter(log_dir=self.log_dir)
            writer.add_figure("lr_finder/run-{}".format(self.num_runs), fig, self.num_runs)
            writer.close()
            self.num_runs += 1
