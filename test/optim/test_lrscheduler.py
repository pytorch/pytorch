# Owner(s): ["module: optimizer", "module: LrScheduler" ]

import warnings
import math
import pickle
import weakref

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import (
    LambdaLR,
    MultiplicativeLR,
    SequentialLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    LRScheduler,
    CyclicLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ChainedScheduler,
    PolynomialLR,
    EPOCH_DEPRECATION_WARNING,
)
from torch.optim.swa_utils import SWALR
from torch.testing._internal.common_utils import (
    TestCase,
    load_tests,
    parametrize,
    instantiate_parametrized_tests,
    skipIfTorchDynamo
)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests


class TestLRScheduler(TestCase):
    class SchedulerTestNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, 1, 1)
            self.conv2 = torch.nn.Conv2d(1, 1, 1)

        def forward(self, x):
            return self.conv2(F.relu(self.conv1(x)))


    class LambdaLRTestObject:
        def __init__(self, value):
            self.value = value

        def __call__(self, epoch):
            return self.value * epoch

        def __eq__(self, other):
            if isinstance(other, self.__class__):
                return self.__dict__ == other.__dict__
            else:
                return False
    exact_dtype = True

    def setUp(self):
        super().setUp()
        self.net = self.SchedulerTestNet()
        self.opt = SGD(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": 0.5},
            ],
            lr=0.05,
        )

    def _check_warning_is_epoch_deprecation_warning(self, w, *, num_warnings: int = 1):
        """This function swallows the epoch deprecation warning which is produced when we
        call `scheduler.step(epoch)` with some not `None` value of `epoch`.
        this is deprecated, and this function will need to be removed/updated when
        the schedulers no longer accept the parameter at all.
        """
        self.assertEqual(len(w), num_warnings)
        for warning in w:
            self.assertEqual(len(warning.message.args), 1)
            self.assertEqual(warning.message.args[0], EPOCH_DEPRECATION_WARNING)

    def test_error_when_getlr_has_epoch(self):
        class MultiStepLR(torch.optim.lr_scheduler.LRScheduler):
            def __init__(self, optimizer, gamma, milestones, last_epoch=-1):
                self.init_lr = [group["lr"] for group in optimizer.param_groups]
                self.gamma = gamma
                self.milestones = milestones
                super().__init__(optimizer, last_epoch)

            def get_lr(self, step):
                global_step = self.last_epoch
                gamma_power = (
                    [0]
                    + [i + 1 for i, m in enumerate(self.milestones) if global_step >= m]
                )[-1]
                return [
                    init_lr * (self.gamma**gamma_power) for init_lr in self.init_lr
                ]

        optimizer = torch.optim.SGD([torch.rand(1)], lr=1)

        with self.assertRaises(TypeError):
            scheduler = MultiStepLR(optimizer, gamma=1, milestones=[10, 20])

    @skipIfTorchDynamo("Torchdynamo keeps references to optim in the guards and the stack of the graph break frames")
    def test_no_cyclic_references(self):
        import gc

        param = Parameter(torch.empty(10))
        optim = SGD([param], lr=0.5)
        scheduler = LambdaLR(optim, lambda epoch: 1.0)
        del scheduler

        self.assertTrue(
            len(gc.get_referrers(optim)) == 0,
            "Optimizer should contain no cyclic references",
        )

        gc.collect()
        del optim
        self.assertEqual(
            gc.collect(), 0, msg="Optimizer should be garbage-collected on __del__"
        )

    @skipIfTorchDynamo("Torchdynamo keeps references to optim in the guards and the stack of the graph break frames")
    def test_no_cyclic_references_in_step(self):
        import gc
        import weakref

        def run():
            param = torch.empty(10, requires_grad=True)
            optim = SGD(params=[param], lr=0.5)
            scheduler = LambdaLR(optim, lambda epoch: 1.0)
            param.sum().backward()
            optim.step()
            scheduler.step()

            return weakref.ref(scheduler)

        # To ensure that there are no reference cycles in scheduler,
        # we need to turn off the garbage collector. Since gc will
        # automatically collect unreachable objects.
        gc.disable()
        ref = run()

        assert ref() is None
        gc.enable()  # restore

    def test_old_pattern_warning(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern)

    def test_old_pattern_warning_with_arg(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern2():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    def test_old_pattern_warning_resuming(self):
        epochs = 35
        for i, group in enumerate(self.opt.param_groups):
            group["initial_lr"] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern)

    def test_old_pattern_warning_resuming_with_arg(self):
        epochs = 35
        for i, group in enumerate(self.opt.param_groups):
            group["initial_lr"] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        def old_pattern2():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    def test_old_pattern_warning_with_overridden_optim_step(self):
        epochs = 35
        for i, group in enumerate(self.opt.param_groups):
            group["initial_lr"] = 0.01

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3, last_epoch=10)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        def old_pattern2():
            for _ in range(epochs):
                scheduler.step()
                self.opt.step()

        self.assertWarnsRegex(UserWarning, r"how-to-adjust-learning-rate", old_pattern2)

    def test_new_pattern_no_warning(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_new_pattern_no_warning_with_arg(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            for _ in range(epochs):
                self.opt.step()
                scheduler.step()
            self.assertTrue(len(ws) == 0, "No warning should be raised")

    def test_new_pattern_no_warning_with_overridden_optim_step(self):
        epochs = 35
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")  # allow any warning to be raised
            scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
            self.assertTrue(len(ws) == 0, "No warning should be raised")

        # emulate use-case with optimizer.step overridden
        import types

        old_step = self.opt.step

        def new_step(o, *args, **kwargs):
            retval = old_step(*args, **kwargs)
            return retval

        self.opt.step = types.MethodType(new_step, self.opt)

        def new_pattern():
            for e in range(epochs):
                self.opt.step()
                scheduler.step()

        self.assertWarnsRegex(
            UserWarning, r"`optimizer.step\(\)` has been overridden", new_pattern
        )

    def _test_lr_is_constant_for_constant_epoch(self, scheduler):
        l = []

        for _ in range(10):
            scheduler.optimizer.step()
            with warnings.catch_warnings(record=True) as w:
                scheduler.step(2)
                self._check_warning_is_epoch_deprecation_warning(w)

            l.append(self.opt.param_groups[0]["lr"])
        self.assertEqual(min(l), max(l))

    def test_step_lr_is_constant_for_constant_epoch(self):
        scheduler = StepLR(self.opt, 2)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_exponential_lr_is_constant_for_constant_epoch(self):
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_constantlr_is_constant_for_constant_epoch(self):
        scheduler = ConstantLR(self.opt)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_linear_linearlr_is_constant_for_constant_epoch(self):
        scheduler = LinearLR(self.opt)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_polynomial_lr_is_constant_for_constant_epoch(self):
        scheduler = PolynomialLR(self.opt, power=0.9)
        self._test_lr_is_constant_for_constant_epoch(scheduler)

    def test_step_lr(self):
        # lr = 0.05     if epoch < 3
        # lr = 0.005    if 30 <= epoch < 6
        # lr = 0.0005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test(scheduler, targets, epochs)

    def test_get_last_lr_step_lr(self):
        from torch.nn import Parameter

        epochs = 10
        optimizer = torch.optim.SGD(
            [Parameter(torch.randn(2, 2, requires_grad=True))], 0.1
        )
        targets = [[0.1] * 3 + [0.01] * 3 + [0.001] * 3 + [0.0001]]
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_get_last_lr_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if 5 <= epoch < 9
        # lr = 0.00005   if 9 <= epoch
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 1
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_multi_step_lr(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test(scheduler, targets, epochs)

    def test_multi_step_lr_with_epoch(self):
        # lr = 0.05     if epoch < 2
        # lr = 0.005    if 2 <= epoch < 5
        # lr = 0.0005   if epoch < 9
        # lr = 0.00005   if epoch >= 9
        epochs = 10
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 3
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_with_epoch(scheduler, targets, epochs)

    def test_get_last_lr_constantlr(self):
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_get_last_lr_linearlr(self):
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 4
        end_factor = 3.0 / 5
        iters = 4
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05 * end_factor] * (
            epochs - iters
        )
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearLR(
            self.opt,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters,
        )
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_constantlr(self):
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        self._test(scheduler, targets, epochs)

    def test_linearlr(self):
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 2
        iters = 4
        interpolation = [
            start_factor + i * (1 - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test(scheduler, targets, epochs)

    def test_linearlr_start_factor_limits1(self):
        start_factor = 0.0
        iters = 4
        with self.assertRaises(ValueError):
            LinearLR(self.opt, start_factor=start_factor, total_iters=iters)

    def test_linearlr_start_factor_limits2(self):
        start_factor = 1.1
        iters = 4
        with self.assertRaises(ValueError):
            LinearLR(self.opt, start_factor=start_factor, total_iters=iters)

    def test_constantlr_with_epoch(self):
        # lr = 0.025     if epoch < 5
        # lr = 0.005    if 5 <= epoch
        epochs = 10
        single_targets = [0.025] * 5 + [0.05] * 5
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ConstantLR(self.opt, factor=1.0 / 2, total_iters=5)
        self._test_with_epoch(scheduler, targets, epochs)

    def test_linearlr_with_epoch(self):
        # lr = 0.025     if epoch == 0
        # lr = 0.03125   if epoch == 1
        # lr = 0.0375    if epoch == 2
        # lr = 0.04375   if epoch == 3
        # lr = 0.005     if 4 <= epoch
        epochs = 10
        start_factor = 1.0 / 2
        end_factor = 1.0
        iters = 4
        interpolation = [
            start_factor + i * (end_factor - start_factor) / iters for i in range(iters)
        ]
        single_targets = [x * 0.05 for x in interpolation] + [0.05] * (epochs - iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test_with_epoch(scheduler, targets, epochs)

    def test_exp_lr(self):
        epochs = 10
        single_targets = [0.05 * (0.9**x) for x in range(epochs)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test(scheduler, targets, epochs)

    def test_poly_lr(self):
        epochs = 10
        power = 0.9
        total_iters = 5
        single_targets = [
            (1.0 - x / total_iters) ** power * 0.05 for x in range(total_iters)
        ] + [0.0] * (epochs - total_iters)
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = PolynomialLR(self.opt, power=power, total_iters=total_iters)
        self._test(scheduler, targets, epochs)

    def test_cos_anneal_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        targets = [single_targets, [x * epochs for x in single_targets]]
        scheduler = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        self._test(scheduler, targets, epochs)

    def test_closed_form_step_lr(self):
        scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        closed_form_scheduler = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_linearlr(self):
        scheduler = LinearLR(
            self.opt, start_factor=1.0 / 3, end_factor=0.7, total_iters=4
        )
        closed_form_scheduler = LinearLR(
            self.opt, start_factor=1.0 / 3, end_factor=0.7, total_iters=4
        )
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_constantlr(self):
        scheduler = ConstantLR(self.opt, factor=1.0 / 3, total_iters=4)
        closed_form_scheduler = ConstantLR(self.opt, factor=1.0 / 3, total_iters=4)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_multi_step_lr(self):
        scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        closed_form_scheduler = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_exp_lr(self):
        scheduler = ExponentialLR(self.opt, gamma=0.9)
        closed_form_scheduler = ExponentialLR(self.opt, gamma=0.9)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_poly_lr(self):
        scheduler = PolynomialLR(self.opt, power=0.9)
        closed_form_scheduler = PolynomialLR(self.opt, power=0.9)
        self._test_against_closed_form(scheduler, closed_form_scheduler, 20)

    def test_closed_form_cos_anneal_lr(self):
        eta_min = 1e-10
        epochs = 20
        T_max = 5
        scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        closed_form_scheduler = CosineAnnealingLR(
            self.opt, T_max=T_max, eta_min=eta_min
        )
        self._test_against_closed_form(scheduler, closed_form_scheduler, epochs)

    def test_cos_anneal_lr_continue(self):
        eta_min = 0.1
        T_max = 5
        scheduler = CosineAnnealingLR(self.opt, T_max=T_max, eta_min=eta_min)
        self.opt.step()
        scheduler.step()
        original_lrs = scheduler._last_lr
        new_scheduler = CosineAnnealingLR(
            self.opt, T_max=T_max, eta_min=eta_min, last_epoch=0
        )
        new_lrs = new_scheduler._last_lr
        torch.testing.assert_close(original_lrs, new_lrs, rtol=1e-4, atol=1e-5)

    def test_reduce_lr_on_plateau1(self):
        epochs = 10
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 20]
        metrics = [10 - i * 0.0167 for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            threshold_mode="abs",
            mode="min",
            threshold=0.01,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau2(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2]
        metrics = [10 - i * 0.0165 for i in range(22)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau3(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [-0.8] * 2 + [-0.234] * 20
        scheduler = ReduceLROnPlateau(
            self.opt, mode="max", patience=5, cooldown=5, threshold_mode="abs"
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau4(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 20]
        metrics = [1.5 * (1.025**i) for i in range(20)]  # 1.025 > 1.1**0.25
        scheduler = ReduceLROnPlateau(
            self.opt, mode="max", patience=3, threshold_mode="rel", threshold=0.1
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau5(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [1.5 * (1.005**i) for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="max",
            threshold_mode="rel",
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau6(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 20]
        metrics = [1.5 * (0.85**i) for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt, mode="min", threshold_mode="rel", threshold=0.1
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau7(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.05] * (5 + 6) + [0.005] * 4]
        metrics = [1] * 7 + [0.6] + [0.5] * 12
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="min",
            threshold_mode="rel",
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_reduce_lr_on_plateau8(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        targets = [[0.5] * 6 + [0.4] * 14, [0.5] * 6 + [0.3] * 14]
        metrics = [1.5 * (1.005**i) for i in range(20)]
        scheduler = ReduceLROnPlateau(
            self.opt,
            mode="max",
            threshold_mode="rel",
            min_lr=[0.4, 0.3],
            threshold=0.1,
            patience=5,
            cooldown=5,
        )
        self._test_reduce_lr_on_plateau(scheduler, targets, metrics, epochs)

    def test_sequentiallr1(self):
        epochs = 19
        schedulers = [None] * 2
        targets = [
            [0.05, 0.04, 0.032]
            + [0.05 for x in range(4)]
            + [0.05 * 0.1 for x in range(4)]
            + [0.05 * 0.01 for x in range(4)]
            + [0.05 * 0.001 for x in range(4)]
        ]
        milestones = [3]
        schedulers[0] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=4)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr2(self):
        epochs = 13
        schedulers = [None] * 2
        targets = [[0.005, 0.005, 0.005] + [0.05 * 0.9**x for x in range(10)]]
        milestones = [3]
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr3(self):
        epochs = 12
        schedulers = [None] * 3
        targets = [
            [0.005, 0.005, 0.005]
            + [0.05, 0.04, 0.032]
            + [0.05, 0.05, 0.005, 0.005, 0.0005, 0.0005]
        ]
        milestones = [3, 6]
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=2)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        self._test(scheduler, targets, epochs)

    def test_sequentiallr4(self):
        optimizer = torch.optim.SGD([torch.tensor(0.5)], lr=0.1)
        prev_lr = optimizer.param_groups[0]["lr"]

        schedulers = [
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1),
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1),
        ]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers, milestones=[10]
        )

        new_lr = optimizer.param_groups[0]["lr"]

        # Ensure that multiple schedulers does not affect the initial learning rate
        self.assertEqual(prev_lr, new_lr)

    def test_get_last_lr_sequentiallr(self):
        epochs = 12
        milestones = [3, 6]
        schedulers = [None] * 3
        schedulers[0] = ConstantLR(self.opt, factor=0.1, total_iters=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.8)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=2)
        scheduler = SequentialLR(self.opt, schedulers=schedulers, milestones=milestones)
        constant_lr_target = [0.005] * 3
        exponential_lr_target = [0.05, 0.04, 0.032]
        step_lr_target = [0.05, 0.05, 0.005, 0.005, 0.0005, 0.0005]
        single_targets = constant_lr_target + exponential_lr_target + step_lr_target
        targets = [single_targets, [x * 10 for x in single_targets]]
        self._test_get_last_lr(scheduler, targets, epochs)

    def test_chained_lr2_get_last_lr_before_step(self):
        schedulers = [
            LinearLR(self.opt, start_factor=0.4, total_iters=3),
            MultiStepLR(self.opt, milestones=[4, 8, 10], gamma=0.1),
        ]
        scheduler = ChainedScheduler(schedulers)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr1(self):
        epochs = 10
        schedulers = [None] * 1
        targets = [[0.05] * 3 + [0.005] * 3 + [0.0005] * 3 + [0.00005] * 3]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr2(self):
        epochs = 10
        schedulers = [None] * 1
        targets = [[0.02, 0.03, 0.04] + [0.05] * 9]
        schedulers[0] = LinearLR(self.opt, start_factor=0.4, total_iters=3)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr3(self):
        epochs = 10
        schedulers = [None] * 2
        targets = [
            [0.02, 0.03, 0.04, 0.05] + [0.005] * 4 + [0.0005] * 3 + [0.00005] * 3
        ]
        schedulers[0] = LinearLR(self.opt, start_factor=0.4, total_iters=3)
        schedulers[1] = MultiStepLR(self.opt, milestones=[4, 8, 10], gamma=0.1)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr4(self):
        epochs = 9
        schedulers = [None] * 3
        targets = [
            [0.05 * 0.2 * 0.9**x for x in range(3)]
            + [0.05 * 0.2 * 0.9**3 * 0.1]
            + [0.05 * 0.9**x * 0.1 for x in range(4, 6)]
            + [0.05 * 0.9**x * 0.01 for x in range(6, 9)]
        ]
        schedulers[0] = ExponentialLR(self.opt, gamma=0.9)
        schedulers[1] = ConstantLR(self.opt, factor=0.2, total_iters=4)
        schedulers[2] = StepLR(self.opt, gamma=0.1, step_size=3)
        scheduler = ChainedScheduler(schedulers)
        self._test([scheduler], targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_chained_lr5(self):
        def poly_lr(lr: float):
            return [
                (lr * ((1.0 - x / total_iters) ** power)) for x in range(total_iters)
            ] + [0.0] * (epochs - total_iters)

        schedulers = [None] * 2
        epochs = 10
        power = 0.9
        total_iters = 5
        const_factor = 0.1
        single_targets = [x * const_factor for x in poly_lr(lr=0.05)]
        targets = [single_targets, [x * const_factor for x in poly_lr(0.5)]]
        schedulers[0] = PolynomialLR(self.opt, power=power, total_iters=total_iters)
        schedulers[1] = ConstantLR(self.opt, factor=const_factor)
        scheduler = ChainedScheduler(schedulers)
        self._test(scheduler, targets, epochs)
        self.assertEqual(scheduler.get_last_lr(), schedulers[-1].get_last_lr())

    def test_compound_step_and_multistep_lr(self):
        epochs = 10
        schedulers = [None] * 2
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        targets = [[0.05] * 2 + [0.005] * 1 + [5e-4] * 2 + [5e-5] + [5e-6] * 3 + [5e-8]]
        self._test(schedulers, targets, epochs)

    def test_compound_step_and_exp_lr(self):
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(3)]
        single_targets += [0.005 * (0.9**x) for x in range(3, 6)]
        single_targets += [0.0005 * (0.9**x) for x in range(6, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 12)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_exp_and_multistep_lr(self):
        epochs = 10
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(2)]
        single_targets += [0.005 * (0.9**x) for x in range(2, 5)]
        single_targets += [0.0005 * (0.9**x) for x in range(5, 9)]
        single_targets += [0.00005 * (0.9**x) for x in range(9, 11)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_exp_and_linearlr(self):
        epochs = 10
        iters = 4
        start_factor = 0.4
        end_factor = 0.9
        schedulers = [None] * 2
        single_targets = [0.05 * (0.9**x) for x in range(11)]
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (end_factor - start_factor)
        for i in range(iters, 11):
            single_targets[i] *= end_factor
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = LinearLR(
            self.opt,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=iters,
        )
        schedulers[1] = ExponentialLR(self.opt, gamma=0.9)
        self._test(schedulers, targets, epochs)

    def test_compound_step_and_constantlr(self):
        epochs = 10
        iters = 4
        factor = 0.4
        schedulers = [None] * 2
        single_targets = (
            [0.05 * 0.4] * 3
            + [0.005 * 0.4]
            + [0.005] * 2
            + [0.0005] * 3
            + [0.00005] * 3
        )
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = StepLR(self.opt, gamma=0.1, step_size=3)
        schedulers[1] = ConstantLR(self.opt, factor=0.4, total_iters=4)
        self._test(schedulers, targets, epochs)

    def test_compound_linearlr_and_multistep_lr(self):
        epochs = 10
        iters = 4
        start_factor = 0.4
        schedulers = [None] * 2
        single_targets = [0.05] * 2 + [0.005] * 3 + [0.0005] * 4 + [0.00005] * 2
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        schedulers[1] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_step_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        single_targets = [x * 0.1 ** (i // 3) for i, x in enumerate(single_targets)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_multistep_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        multipliers = [1] * 2 + [0.1] * 3 + [0.01] * 4 + [0.001]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9])
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_linearlr(self):
        epochs = 10
        iters = 4
        start_factor = 0.4
        eta_min = 1e-10
        schedulers = [None] * 2
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        for i in range(iters):
            single_targets[i] *= start_factor + i / iters * (1 - start_factor)
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers[0] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        schedulers[1] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        self._test(schedulers, targets, epochs)

    def test_compound_cosanneal_and_exp_lr(self):
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        multipliers = [0.1**i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets, [x * epochs for x in single_targets]]
        schedulers = [None] * 2
        schedulers[0] = CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min)
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        self._test(schedulers, targets, epochs)

    def test_compound_reduce_lr_on_plateau1(self):
        epochs = 10
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * 20
        multipliers = [0.1 ** (i // 3) for i in range(20)]
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0167 for i in range(20)]
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            threshold_mode="abs",
            mode="min",
            threshold=0.01,
            patience=5,
            cooldown=5,
        )
        schedulers[1] = StepLR(self.opt, gamma=0.1, step_size=3)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau2(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2
        multipliers = [1] * 3 + [0.1] * 5 + [0.01] * 4 + [0.001] * 10
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0165 for i in range(22)]
        schedulers = [None] * 2
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        schedulers[1] = MultiStepLR(self.opt, gamma=0.1, milestones=[3, 8, 12])
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau3(self):
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * (2 + 6) + [0.05] * (5 + 6) + [0.005] * 4
        multipliers = [0.1**i for i in range(epochs)]
        single_targets = [x * y for x, y in zip(multipliers, single_targets)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [-0.8] * 2 + [-0.234] * 20
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(
            self.opt, mode="max", patience=5, cooldown=5, threshold_mode="abs"
        )
        schedulers[1] = ExponentialLR(self.opt, gamma=0.1)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau4(self):
        epochs = 20
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.05
        epochs = 10
        eta_min = 1e-10
        single_targets = [
            eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * x / epochs)) / 2
            for x in range(epochs)
        ]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [1.5 * (1.025**i) for i in range(20)]  # 1.025 > 1.1**0.25
        schedulers = [None, None]
        schedulers[0] = ReduceLROnPlateau(
            self.opt, mode="max", patience=3, threshold_mode="rel", threshold=0.1
        )
        schedulers[1] = CosineAnnealingLR(self.opt, epochs, eta_min)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_compound_reduce_lr_on_plateau5(self):
        iters = 4
        start_factor = 0.4
        epochs = 22
        for param_group in self.opt.param_groups:
            param_group["lr"] = 0.5
        single_targets = [0.5] * 6 + [0.05] * 7 + [0.005] * 7 + [0.0005] * 2
        multipliers = [1] * 22
        for i in range(iters):
            multipliers[i] *= start_factor + i / iters * (1 - start_factor)
        single_targets = [x * y for x, y in zip(single_targets, multipliers)]
        targets = [single_targets]
        targets = targets[1:]  # test runs step before checking lr
        metrics = [10 - i * 0.0165 for i in range(22)]
        schedulers = [None] * 2
        schedulers[0] = ReduceLROnPlateau(
            self.opt,
            patience=5,
            cooldown=0,
            threshold_mode="abs",
            mode="min",
            threshold=0.1,
        )
        schedulers[1] = LinearLR(self.opt, start_factor=start_factor, total_iters=iters)
        self._test_reduce_lr_on_plateau(schedulers, targets, metrics, epochs)

    def test_cycle_lr_invalid_mode(self):
        with self.assertRaises(ValueError):
            scheduler = CyclicLR(self.opt, base_lr=0, max_lr=0, mode="CATS")

    def test_cycle_lr_triangular_mode_one_lr(self):
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        momentum_target = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular_mode_one_lr_no_momentum(self):
        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        momentum_target = [self.opt.defaults["momentum"]] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=False,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular2_mode_one_lr(self):
        lr_target = [
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            1.5,
            2.0,
            2.5,
            3.0,
            2.5,
            2.0,
            1.5,
            1,
            1.25,
            1.50,
            1.75,
            2.00,
            1.75,
        ]
        momentum_target = [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.5,
            4.0,
            3.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            4.75,
            4.5,
            4.25,
            4.0,
            4.25,
        ]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode="triangular2",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_exp_range_mode_one_lr(self):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        lr_target = [base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)]
        momentum_target = [max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=base_lr,
            max_momentum=max_lr,
            mode="exp_range",
            gamma=gamma,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular_mode(self):
        lr_target_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_target_2 = [x + 1 for x in lr_target_1]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [5, 4, 3, 2, 1, 2, 3, 4, 5, 4, 3]
        momentum_target_2 = [x + 1 for x in momentum_target_1]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicLR(
            self.opt,
            base_lr=[1, 2],
            max_lr=[5, 6],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[1, 2],
            max_momentum=[5, 6],
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_triangular2_mode(self):
        lr_target_1 = [
            1,
            2,
            3,
            4,
            5,
            4,
            3,
            2,
            1,
            1.5,
            2.0,
            2.5,
            3.0,
            2.5,
            2.0,
            1.5,
            1,
            1.25,
            1.50,
            1.75,
            2.00,
            1.75,
        ]
        lr_target_2 = [x + 2 for x in lr_target_1]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            4.5,
            4.0,
            3.5,
            3.0,
            3.5,
            4.0,
            4.5,
            5.0,
            4.75,
            4.5,
            4.25,
            4.0,
            4.25,
        ]
        momentum_target_2 = [x + 2 for x in momentum_target_1]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicLR(
            self.opt,
            base_lr=[1, 3],
            max_lr=[5, 7],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[1, 3],
            max_momentum=[5, 7],
            mode="triangular2",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_exp_range_mode(self):
        base_lr_1, max_lr_1 = 1, 5
        base_lr_2, max_lr_2 = 5, 12

        diff_lr_1 = max_lr_1 - base_lr_1
        diff_lr_2 = max_lr_2 - base_lr_2

        gamma = 0.9
        xs = [0, 0.25, 0.5, 0.75, 1, 0.75, 0.50, 0.25, 0, 0.25, 0.5, 0.75, 1]
        lr_target_1 = [base_lr_1 + x * diff_lr_1 * gamma**i for i, x in enumerate(xs)]
        lr_target_2 = [base_lr_2 + x * diff_lr_2 * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target_1, lr_target_2]
        momentum_target_1 = [
            max_lr_1 - x * diff_lr_1 * gamma**i for i, x in enumerate(xs)
        ]
        momentum_target_2 = [
            max_lr_2 - x * diff_lr_2 * gamma**i for i, x in enumerate(xs)
        ]
        momentum_targets = [momentum_target_1, momentum_target_2]
        scheduler = CyclicLR(
            self.opt,
            base_lr=[base_lr_1, base_lr_2],
            max_lr=[max_lr_1, max_lr_2],
            step_size_up=4,
            cycle_momentum=True,
            base_momentum=[base_lr_1, base_lr_2],
            max_momentum=[max_lr_1, max_lr_2],
            mode="exp_range",
            gamma=gamma,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target_1))

    def test_cycle_lr_triangular_mode_step_size_up_down(self):
        lr_target = [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            13.0 / 3,
            11.0 / 3,
            9.0 / 3,
            7.0 / 3,
            5.0 / 3,
            1.0,
        ]
        lr_targets = [lr_target, lr_target]
        momentum_target = [
            5.0,
            4.0,
            3.0,
            2.0,
            1.0,
            5.0 / 3,
            7.0 / 3,
            3.0,
            11.0 / 3,
            13.0 / 3,
            5.0,
        ]
        momentum_targets = [momentum_target, momentum_target]

        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=1,
            max_momentum=5,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_triangular2_mode_step_size_up_down(self):
        lr_base_target = [
            1.0,
            3.0,
            5.0,
            13.0 / 3,
            11.0 / 3,
            9.0 / 3,
            7.0 / 3,
            5.0 / 3,
            1.0,
            2.0,
            3.0,
            8.0 / 3,
            7.0 / 3,
            6.0 / 3,
            5.0 / 3,
            4.0 / 3,
            1.0,
            3.0 / 2,
            2.0,
            11.0 / 6,
            10.0 / 6,
            9.0 / 6,
            8.0 / 6,
            7.0 / 6,
        ]
        momentum_base_target = [
            5.0,
            3.0,
            1.0,
            5.0 / 3,
            7.0 / 3,
            3.0,
            11.0 / 3,
            13.0 / 3,
            5.0,
            4.0,
            3.0,
            10.0 / 3,
            11.0 / 3,
            4.0,
            13.0 / 3,
            14.0 / 3,
            5.0,
            4.5,
            4.0,
            25.0 / 6,
            13.0 / 3,
            4.5,
            14.0 / 3,
            29.0 / 6,
        ]
        deltas = [2 * i for i in range(0, 2)]
        base_lrs = [1 + delta for delta in deltas]
        max_lrs = [5 + delta for delta in deltas]
        lr_targets = [[x + delta for x in lr_base_target] for delta in deltas]
        momentum_targets = [
            [x + delta for x in momentum_base_target] for delta in deltas
        ]
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lrs,
            max_lr=max_lrs,
            step_size_up=2,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=base_lrs,
            max_momentum=max_lrs,
            mode="triangular2",
        )
        self._test_cycle_lr(
            scheduler, lr_targets, momentum_targets, len(lr_base_target)
        )

    def test_cycle_lr_exp_range_mode_step_size_up_down(self):
        base_lr, max_lr = 1, 5
        diff_lr = max_lr - base_lr
        gamma = 0.9
        xs = [
            0.0,
            0.5,
            1.0,
            5.0 / 6,
            4.0 / 6,
            3.0 / 6,
            2.0 / 6,
            1.0 / 6,
            0.0,
            0.5,
            1.0,
            5.0 / 6,
            4.0 / 6,
        ]
        lr_target = [base_lr + x * diff_lr * gamma**i for i, x in enumerate(xs)]
        lr_targets = [lr_target, lr_target]
        momentum_target = [max_lr - x * diff_lr * gamma**i for i, x in enumerate(xs)]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=2,
            step_size_down=6,
            cycle_momentum=True,
            base_momentum=base_lr,
            max_momentum=max_lr,
            mode="exp_range",
            gamma=gamma,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

    def test_cycle_lr_with_momentumless_optimizer(self):
        # Note [Temporarily set optimizer to Adam]
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # The TestLRScheduler object carries around an SGD optimizer to avoid having to
        # instantiate one for every test. This gets in the way for our very specific case
        # in which we need to use Adam (or really any optimizer that doesn't use momentum)
        # in order to test that the momentum bug in CyclicLR is fixed (the bug is described
        # in more detail in https://github.com/pytorch/pytorch/issues/19003 ).
        old_opt = self.opt
        self.opt = Adam(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": 0.5},
            ],
            lr=0.05,
        )

        lr_target = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lr_targets = [lr_target, lr_target]
        momentum_target = [None] * len(lr_target)
        momentum_targets = [momentum_target, momentum_target]
        scheduler = CyclicLR(
            self.opt,
            base_lr=1,
            max_lr=5,
            step_size_up=4,
            cycle_momentum=False,
            mode="triangular",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, len(lr_target))

        self.opt = old_opt  # set optimizer back to SGD

    def test_cycle_lr_cycle_momentum_fail_with_momentumless_optimizer(self):
        with self.assertRaises(ValueError):
            adam_opt = optim.Adam(self.net.parameters())
            scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=True)

    def test_cycle_lr_removed_after_out_of_scope(self):
        import gc
        import weakref

        gc.disable()

        def test():
            adam_opt = optim.Adam(self.net.parameters())
            scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False)
            return weakref.ref(scheduler)

        ref = test()
        assert ref() is None
        gc.enable()

    def test_cycle_lr_state_dict_picklable(self):
        adam_opt = optim.Adam(self.net.parameters())
        scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False)
        self.assertIsInstance(scheduler._scale_fn_ref, weakref.WeakMethod)
        state = scheduler.state_dict()
        self.assertNotIn("_scale_fn_ref", state)
        pickle.dumps(state)

    def test_cycle_lr_scale_fn_restored_from_state_dict(self):
        adam_opt = optim.Adam(self.net.parameters())

        # Case 1: Built-in mode
        scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, mode="triangular2")
        restored_scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False)
        restored_scheduler.load_state_dict(scheduler.state_dict())
        self.assertTrue(restored_scheduler.mode == scheduler.mode == "triangular2")
        self.assertIsNotNone(restored_scheduler._scale_fn_ref) and self.assertIsNotNone(scheduler._scale_fn_ref)
        self.assertIs(restored_scheduler._scale_fn_custom, None)
        self.assertIs(scheduler._scale_fn_custom, None)

        # Case 2: Custom `scale_fn`
        def scale_fn(_):
            return 0.5

        scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn)
        restored_scheduler = CyclicLR(adam_opt, base_lr=1, max_lr=5, cycle_momentum=False, scale_fn=scale_fn)
        restored_scheduler.load_state_dict(scheduler.state_dict())
        self.assertIs(scheduler._scale_fn_custom, scale_fn)
        self.assertIs(restored_scheduler._scale_fn_custom, scale_fn)

    def test_onecycle_lr_invalid_anneal_strategy(self):
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(
                self.opt, max_lr=1e-3, total_steps=10, anneal_strategy="CATS"
            )

    def test_onecycle_lr_invalid_pct_start(self):
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.opt, max_lr=1e-3, total_steps=10, pct_start=1.1)

    def test_onecycle_lr_cannot_calculate_total_steps(self):
        with self.assertRaises(ValueError):
            scheduler = OneCycleLR(self.opt, max_lr=1e-3)

    def test_onecycle_lr_linear_annealing(self):
        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="linear",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_linear_annealing_three_phases(self):
        lr_target = [1, 9, 17, 25, 17, 9, 1, 0.75, 0.5, 0.25]
        momentum_target = [22, 15, 8, 1, 8, 15, 22, 22, 22, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            div_factor=25,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="linear",
            pct_start=0.4,
            final_div_factor=4,
            three_phase=True,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_onecycle_lr_cosine_annealing(self):
        def annealing_cos(start, end, pct):
            cos_out = math.cos(math.pi * pct) + 1
            return end + (start - end) / 2.0 * cos_out

        lr_target = [
            1,
            13,
            25,
            annealing_cos(25, 0.5, 1 / 7.0),
            annealing_cos(25, 0.5, 2 / 7.0),
            annealing_cos(25, 0.5, 3 / 7.0),
            annealing_cos(25, 0.5, 4 / 7.0),
            annealing_cos(25, 0.5, 5 / 7.0),
            annealing_cos(25, 0.5, 6 / 7.0),
            0.5,
        ]
        momentum_target = [
            22,
            11.5,
            1,
            annealing_cos(1, 22, 1 / 7.0),
            annealing_cos(1, 22, 2 / 7.0),
            annealing_cos(1, 22, 3 / 7.0),
            annealing_cos(1, 22, 4 / 7.0),
            annealing_cos(1, 22, 5 / 7.0),
            annealing_cos(1, 22, 6 / 7.0),
            22,
        ]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10)

    def test_cycle_lr_with_adam(self):
        old_opt = self.opt
        self.opt = optim.Adam(
            [
                {"params": self.net.conv1.parameters()},
                {"params": self.net.conv2.parameters(), "lr": 0.5},
            ],
            lr=0.05,
        )

        lr_target = [1, 13, 25, 21.5, 18, 14.5, 11, 7.5, 4, 0.5]
        momentum_target = [22, 11.5, 1, 4, 7, 10, 13, 16, 19, 22]
        lr_targets = [lr_target, lr_target]
        momentum_targets = [momentum_target, momentum_target]
        scheduler = OneCycleLR(
            self.opt,
            max_lr=25,
            final_div_factor=2,
            base_momentum=1,
            max_momentum=22,
            total_steps=10,
            anneal_strategy="linear",
        )
        self._test_cycle_lr(scheduler, lr_targets, momentum_targets, 10, use_beta1=True)
        self.opt = old_opt  # set optimizer back to SGD

    def test_lambda_lr(self):
        epochs = 10
        self.opt.param_groups[0]["lr"] = 0.05
        self.opt.param_groups[1]["lr"] = 0.4
        targets = [
            [0.05 * (0.9**x) for x in range(epochs)],
            [0.4 * (0.8**x) for x in range(epochs)],
        ]
        scheduler = LambdaLR(
            self.opt, lr_lambda=[lambda x1: 0.9**x1, lambda x2: 0.8**x2]
        )
        self._test(scheduler, targets, epochs)

    def test_multiplicative_lr(self):
        epochs = 10
        self.opt.param_groups[0]["lr"] = 0.05
        self.opt.param_groups[1]["lr"] = 0.4
        targets = [
            [0.05 * (0.9**x) for x in range(epochs)],
            [0.4 * (0.8**x) for x in range(epochs)],
        ]
        scheduler = MultiplicativeLR(
            self.opt, lr_lambda=[lambda x1: 0.9, lambda x2: 0.8]
        )
        self._test(scheduler, targets, epochs)

    @parametrize("T_mult", [1, 2, 4])
    def test_CosineAnnealingWarmRestarts_lr1(self, T_mult):
        iters = 100
        eta_min = 1e-10
        T_i = 10
        T_cur = 0
        targets = [[0.05], [0.5]]
        scheduler = CosineAnnealingWarmRestarts(
            self.opt, T_0=T_i, T_mult=T_mult, eta_min=eta_min
        )
        for _ in range(1, iters, 1):
            T_cur += 1
            if T_cur >= T_i:
                T_cur = T_cur - T_i
                T_i = int(T_mult) * T_i
            targets[0] += [
                eta_min + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            ]
            targets[1] += [
                eta_min + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
            ]
        self._test(scheduler, targets, iters)

    def test_CosineAnnealingWarmRestarts_lr2(self):
        iters = 30
        eta_min = 1e-10
        T_mults = [1, 2, 4]
        for T_mult in T_mults:
            T_i = 10
            T_cur = 0
            targets = [[0.05], [0.5]]
            scheduler = CosineAnnealingWarmRestarts(
                self.opt, T_0=T_i, T_mult=T_mult, eta_min=eta_min
            )
            for _ in torch.arange(0.1, iters, 0.1):
                T_cur = round(T_cur + 0.1, 1)
                if T_cur >= T_i:
                    T_cur = T_cur - T_i
                    T_i = int(T_mult) * T_i
                targets[0] += [
                    eta_min
                    + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
                targets[1] += [
                    eta_min
                    + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
            self._test_CosineAnnealingWarmRestarts(scheduler, targets, iters)

    def test_CosineAnnealingWarmRestarts_lr3(self):
        epochs_for_T_mults = [
            [0, 1, 2, 3, 4, 5, 12, 27, 3, 4, 5, 6, 13],
            [0, 1, 2, 3, 4, 5, 25, 32, 33, 34, 80, 81, 3],
            [0, 0.1, 0.2, 0.3, 1.3, 2.3, 17.5, 18.5, 19.5, 29.5, 30.5, 31.5, 50],
        ]
        T_curs_for_T_mults = [
            [1, 2, 3, 4, 5, 2, 7, 3, 4, 5, 6, 3],
            [1, 2, 3, 4, 5, 15, 2, 3, 4, 10, 11, 3],
            [0.1, 0.2, 0.3, 1.3, 2.3, 7.5, 8.5, 9.5, 19.5, 20.5, 21.5, 10],
        ]
        T_is_for_T_mults = [
            [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10],
            [10, 10, 10, 10, 10, 20, 40, 40, 40, 80, 80, 10],
            [10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 90],
        ]
        eta_min = 1e-10
        T_mults = [1, 2, 3]
        for epochs, T_mult, T_curs, T_is in zip(
            epochs_for_T_mults, T_mults, T_curs_for_T_mults, T_is_for_T_mults
        ):
            targets = [[0.05], [0.5]]
            scheduler = CosineAnnealingWarmRestarts(
                self.opt, T_0=10, T_mult=T_mult, eta_min=eta_min
            )
            for T_cur, T_i in zip(T_curs, T_is):
                targets[0] += [
                    eta_min
                    + (0.05 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
                targets[1] += [
                    eta_min
                    + (0.5 - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
                ]
            self._test_interleaved_CosineAnnealingWarmRestarts(
                scheduler, targets, epochs
            )

    def test_swalr_no_anneal(self):
        epochs, swa_start, swa_lr = 10, 5, 0.01
        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        targets = [
            [lr] * (swa_start + 1) + [swa_lr] * (epochs - swa_start - 1)
            for lr in initial_lrs
        ]
        swa_scheduler = SWALR(self.opt, anneal_epochs=1, swa_lr=swa_lr)
        self._test_swalr(swa_scheduler, None, targets, swa_start, epochs)

    def test_swalr_cosine_anneal_after_multiplicative(self):
        # same swa_lr for different param_groups
        epochs, swa_start, swa_lr, anneal_epochs = 15, 5, 0.01, 5
        mult_factor = 0.9
        scheduler = MultiplicativeLR(self.opt, lr_lambda=lambda epoch: mult_factor)
        swa_scheduler = SWALR(self.opt, anneal_epochs=anneal_epochs, swa_lr=swa_lr)

        def anneal_coef(t):
            if t + 1 >= anneal_epochs:
                return 0.0
            return (1 + math.cos(math.pi * (t + 1) / anneal_epochs)) / 2

        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        targets_before_swa = [
            [lr * mult_factor**i for i in range(swa_start + 1)] for lr in initial_lrs
        ]
        swa_epochs = epochs - swa_start - 1
        targets = [
            lrs
            + [
                lrs[-1] * anneal_coef(t) + swa_lr * (1 - anneal_coef(t))
                for t in range(swa_epochs)
            ]
            for lrs in targets_before_swa
        ]

        self._test_swalr(swa_scheduler, scheduler, targets, swa_start, epochs)

    def test_swalr_linear_anneal_after_multiplicative(self):
        # separate swa_lr for different param_groups
        epochs, swa_start, swa_lrs, anneal_epochs = 15, 5, [0.01, 0.02], 4
        mult_factor = 0.9
        scheduler = MultiplicativeLR(self.opt, lr_lambda=lambda epoch: mult_factor)
        swa_scheduler = SWALR(
            self.opt,
            anneal_epochs=anneal_epochs,
            anneal_strategy="linear",
            swa_lr=swa_lrs,
        )

        def anneal_coef(t):
            if t + 1 >= anneal_epochs:
                return 0.0
            return 1 - (t + 1) / anneal_epochs

        initial_lrs = [group["lr"] for group in self.opt.param_groups]
        targets_before_swa = [
            [lr * mult_factor**i for i in range(swa_start + 1)] for lr in initial_lrs
        ]
        swa_epochs = epochs - swa_start - 1
        targets = [
            lrs
            + [
                lrs[-1] * anneal_coef(t) + swa_lr * (1 - anneal_coef(t))
                for t in range(swa_epochs)
            ]
            for lrs, swa_lr in zip(targets_before_swa, swa_lrs)
        ]

        self._test_swalr(swa_scheduler, scheduler, targets, swa_start, epochs)

    def _test_swalr(self, swa_scheduler, scheduler, targets, swa_start, epochs):
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )
            if epoch >= swa_start:
                self.opt.step()
                swa_scheduler.step()
            elif scheduler is not None:
                self.opt.step()
                scheduler.step()

    def test_swalr_hypers(self):
        # Test that SWALR raises errors for incorrect hyper-parameters
        with self.assertRaisesRegex(ValueError, "anneal_strategy must"):
            swa_scheduler = SWALR(self.opt, anneal_strategy="exponential", swa_lr=1.0)

        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            swa_scheduler = SWALR(self.opt, anneal_epochs=-1, swa_lr=1.0)
        with self.assertRaisesRegex(ValueError, "anneal_epochs must"):
            swa_scheduler = SWALR(self.opt, anneal_epochs=1.7, swa_lr=1.0)
        with self.assertRaisesRegex(ValueError, "swa_lr must"):
            swa_scheduler = SWALR(self.opt, swa_lr=[1.0, 0.1, 0.01])

    def test_step_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: StepLR(self.opt, gamma=0.1, step_size=3),
            lambda: StepLR(self.opt, gamma=0.01 / 2, step_size=1),
        )

    def test_multi_step_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: MultiStepLR(self.opt, gamma=0.1, milestones=[2, 5, 9]),
            lambda: MultiStepLR(self.opt, gamma=0.01, milestones=[1, 4, 6]),
        )

    def test_exp_step_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: ExponentialLR(self.opt, gamma=0.1),
            lambda: ExponentialLR(self.opt, gamma=0.01),
        )

    def test_cosine_lr_state_dict(self):
        epochs = 10
        eta_min = 1e-10
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingLR(self.opt, T_max=epochs, eta_min=eta_min),
            lambda: CosineAnnealingLR(self.opt, T_max=epochs // 2, eta_min=eta_min / 2),
            epochs=epochs,
        )

    def test_reduce_lr_on_plateau_state_dict(self):
        scheduler = ReduceLROnPlateau(self.opt, mode="min", factor=0.1, patience=2)
        for score in [1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 3.0, 2.0, 1.0]:
            scheduler.step(score)
        scheduler_copy = ReduceLROnPlateau(
            self.opt, mode="max", factor=0.5, patience=10
        )
        scheduler_copy.load_state_dict(scheduler.state_dict())
        for key in scheduler.__dict__.keys():
            if key not in {"optimizer", "is_better"}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    def test_lambda_lr_state_dict_fn(self):
        scheduler = LambdaLR(self.opt, lr_lambda=lambda x: x)
        state = scheduler.state_dict()
        self.assertIsNone(state["lr_lambdas"][0])

        scheduler_copy = LambdaLR(self.opt, lr_lambda=lambda x: x)
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__.keys():
            if key not in {"optimizer", "lr_lambdas"}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    def test_lambda_lr_state_dict_obj(self):
        scheduler = LambdaLR(self.opt, lr_lambda=self.LambdaLRTestObject(10))
        state = scheduler.state_dict()
        self.assertIsNotNone(state["lr_lambdas"][0])

        scheduler_copy = LambdaLR(self.opt, lr_lambda=self.LambdaLRTestObject(-1))
        scheduler_copy.load_state_dict(state)
        for key in scheduler.__dict__.keys():
            if key not in {"optimizer"}:
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])

    def test_CosineAnnealingWarmRestarts_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: CosineAnnealingWarmRestarts(self.opt, T_0=10, T_mult=2),
            lambda: CosineAnnealingWarmRestarts(self.opt, T_0=100),
        )

    def test_swa_lr_state_dict(self):
        self._check_scheduler_state_dict(
            lambda: SWALR(self.opt, anneal_epochs=3, swa_lr=0.5),
            lambda: SWALR(
                self.opt, anneal_epochs=10, anneal_strategy="linear", swa_lr=5.0
            ),
        )

    def _check_scheduler_state_dict(self, constr, constr2, epochs=10):
        scheduler = constr()
        for _ in range(epochs):
            scheduler.optimizer.step()
            scheduler.step()
        scheduler_copy = constr2()
        scheduler_copy.load_state_dict(scheduler.state_dict())
        for key in scheduler.__dict__.keys():
            if key != "optimizer":
                self.assertEqual(scheduler.__dict__[key], scheduler_copy.__dict__[key])
        self.assertEqual(scheduler.get_last_lr(), scheduler_copy.get_last_lr())

    def _test_get_last_lr(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        optimizers = {scheduler.optimizer for scheduler in schedulers}
        for epoch in range(epochs):
            result = [scheduler.get_last_lr() for scheduler in schedulers]
            [optimizer.step() for optimizer in optimizers]
            [scheduler.step() for scheduler in schedulers]
            target = [[t[epoch] for t in targets]] * len(schedulers)
            for t, r in zip(target, result):
                self.assertEqual(
                    target,
                    result,
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, t, r
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test_with_epoch(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        optimizers = {scheduler.optimizer for scheduler in schedulers}
        for epoch in range(epochs):
            [optimizer.step() for optimizer in optimizers]
            with warnings.catch_warnings(record=True) as w:
                [
                    scheduler.step(epoch) for scheduler in schedulers
                ]  # step before assert: skip initial lr
                self._check_warning_is_epoch_deprecation_warning(
                    w, num_warnings=len(schedulers)
                )
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test(self, schedulers, targets, epochs=10):
        if isinstance(schedulers, LRScheduler):
            schedulers = [schedulers]
        for epoch in range(epochs):
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )
            [scheduler.step() for scheduler in schedulers]

    def _test_CosineAnnealingWarmRestarts(self, scheduler, targets, epochs=10):
        for index, epoch in enumerate(torch.arange(0, epochs, 0.1)):
            epoch = round(epoch.item(), 1)
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[index],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[index], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test_interleaved_CosineAnnealingWarmRestarts(self, scheduler, targets, epochs):
        for index, epoch in enumerate(epochs):
            scheduler.step(epoch)
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[index],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[index], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test_against_closed_form(self, scheduler, closed_form_scheduler, epochs=10):
        self.setUp()
        targets = []
        for epoch in range(epochs):
            closed_form_scheduler.optimizer.step()
            with warnings.catch_warnings(record=True) as w:
                closed_form_scheduler.step(epoch)
                self._check_warning_is_epoch_deprecation_warning(w)
            targets.append([group["lr"] for group in self.opt.param_groups])
        self.setUp()
        for epoch in range(epochs):
            self.opt.step()
            scheduler.step()
            for i, param_group in enumerate(self.opt.param_groups):
                self.assertEqual(
                    targets[epoch][i],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, targets[epoch][i], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test_reduce_lr_on_plateau(
        self, schedulers, targets, metrics, epochs=10, verbose=False
    ):
        if isinstance(schedulers, (LRScheduler, ReduceLROnPlateau)):
            schedulers = [schedulers]
        for epoch in range(epochs):
            self.opt.step()
            for scheduler in schedulers:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(metrics[epoch])
                else:
                    scheduler.step()
            if verbose:
                print("epoch{}:\tlr={}".format(epoch, self.opt.param_groups[0]["lr"]))
            for param_group, target in zip(self.opt.param_groups, targets):
                self.assertEqual(
                    target[epoch],
                    param_group["lr"],
                    msg="LR is wrong in epoch {}: expected {}, got {}".format(
                        epoch, target[epoch], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )

    def _test_cycle_lr(
        self,
        scheduler,
        lr_targets,
        momentum_targets,
        batch_iterations,
        verbose=False,
        use_beta1=False,
    ):
        for batch_num in range(batch_iterations):
            if verbose:
                if "momentum" in self.opt.param_groups[0].keys():
                    print(
                        "batch{}:\tlr={},momentum={}".format(
                            batch_num,
                            self.opt.param_groups[0]["lr"],
                            self.opt.param_groups[0]["momentum"],
                        )
                    )
                elif use_beta1 and "betas" in self.opt.param_groups[0].keys():
                    print(
                        "batch{}:\tlr={},beta1={}".format(
                            batch_num,
                            self.opt.param_groups[0]["lr"],
                            self.opt.param_groups[0]["betas"][0],
                        )
                    )
                else:
                    print(
                        "batch{}:\tlr={}".format(
                            batch_num, self.opt.param_groups[0]["lr"]
                        )
                    )

            for param_group, lr_target, momentum_target in zip(
                self.opt.param_groups, lr_targets, momentum_targets
            ):
                self.assertEqual(
                    lr_target[batch_num],
                    param_group["lr"],
                    msg="LR is wrong in batch_num {}: expected {}, got {}".format(
                        batch_num, lr_target[batch_num], param_group["lr"]
                    ),
                    atol=1e-5,
                    rtol=0,
                )

                if use_beta1 and "betas" in param_group.keys():
                    self.assertEqual(
                        momentum_target[batch_num],
                        param_group["betas"][0],
                        msg="Beta1 is wrong in batch_num {}: expected {}, got {}".format(
                            batch_num,
                            momentum_target[batch_num],
                            param_group["betas"][0],
                        ),
                        atol=1e-5,
                        rtol=0,
                    )
                elif "momentum" in param_group.keys():
                    self.assertEqual(
                        momentum_target[batch_num],
                        param_group["momentum"],
                        msg="Momentum is wrong in batch_num {}: expected {}, got {}".format(
                            batch_num,
                            momentum_target[batch_num],
                            param_group["momentum"],
                        ),
                        atol=1e-5,
                        rtol=0,
                    )
            self.opt.step()
            scheduler.step()

    def test_cosine_then_cyclic(self):
        # https://github.com/pytorch/pytorch/issues/21965

        max_lr = 0.3
        base_lr = 0.1
        optim_lr = 0.5

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=optim_lr)
        lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=20, eta_min=0.1
        )
        lr_scheduler_2 = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=1, step_size_down=3
        )

        for i in range(40):
            optimizer.step()
            if i <= lr_scheduler_1.T_max:
                lr_scheduler_1.step()
            else:
                lr_scheduler_2.step()
            last_lr = optimizer.param_groups[0]["lr"]

        self.assertLessEqual(last_lr, max_lr)


instantiate_parametrized_tests(TestLRScheduler)


if __name__ == "__main__":
    print("These tests should be run through test/test_optim.py instead")
