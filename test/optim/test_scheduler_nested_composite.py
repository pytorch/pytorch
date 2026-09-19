import torch
from torch.testing._internal.common_utils import TestCase


class TestNestedCompositeSchedulers(TestCase):
    def _lrs(self, scheduler):
        values = []
        for _ in range(5):
            values.append(scheduler.get_last_lr()[0])
            scheduler.optimizer.step()
            scheduler.step()
        return values

    def test_nested_sequential_matches_direct(self):
        def make(opt):
            return torch.optim.lr_scheduler.SequentialLR(
                opt,
                [
                    torch.optim.lr_scheduler.ConstantLR(opt, factor=0.5, total_iters=2),
                    torch.optim.lr_scheduler.ConstantLR(opt, factor=0.2, total_iters=10),
                ],
                milestones=[2],
            )

        opt1 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        direct = make(opt1)
        opt2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        nested = torch.optim.lr_scheduler.SequentialLR(opt2, [make(opt2)], milestones=[])

        self.assertEqual(self._lrs(direct), self._lrs(nested))

    def test_nested_chained_matches_direct(self):
        def make(opt):
            return torch.optim.lr_scheduler.ChainedScheduler(
                [
                    torch.optim.lr_scheduler.ConstantLR(opt, factor=0.5, total_iters=2),
                    torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9),
                ],
                optimizer=opt,
            )

        opt1 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        direct = make(opt1)
        opt2 = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        nested = torch.optim.lr_scheduler.SequentialLR(opt2, [make(opt2)], milestones=[])

        self.assertEqual(self._lrs(direct), self._lrs(nested))


if __name__ == "__main__":
    torch.testing._internal.common_utils.run_tests()
