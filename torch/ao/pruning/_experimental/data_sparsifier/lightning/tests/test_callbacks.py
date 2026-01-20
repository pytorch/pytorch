# mypy: allow-untyped-defs
import importlib
import math
import unittest
import warnings

import torch
import torch.nn as nn
from torch.ao.pruning._experimental.data_scheduler.base_data_scheduler import (
    BaseDataScheduler,
)
from torch.ao.pruning._experimental.data_sparsifier.base_data_sparsifier import (
    SUPPORTED_TYPES,
)
from torch.ao.pruning._experimental.data_sparsifier.data_norm_sparsifier import (
    DataNormSparsifier,
)
from torch.ao.pruning._experimental.data_sparsifier.lightning.callbacks._data_sparstity_utils import (
    _get_valid_name,
)
from torch.ao.pruning._experimental.data_sparsifier.lightning.callbacks.data_sparsity import (
    PostTrainingDataSparsity,
    TrainingAwareDataSparsity,
)
from torch.nn.utils.parametrize import is_parametrized
from torch.testing._internal.common_utils import run_tests, TestCase


class DummyModel(nn.Module):
    def __init__(self, iC: int, oC: list[int]):
        super().__init__()
        self.linears = nn.Sequential()
        i = iC
        for idx, c in enumerate(oC):
            self.linears.append(nn.Linear(i, c, bias=False))
            if idx < len(oC) - 1:
                self.linears.append(nn.ReLU())
            i = c


def _make_lightning_module(iC: int, oC: list[int]):
    import pytorch_lightning as pl  # type: ignore[import]

    class DummyLightningModule(pl.LightningModule):
        def __init__(self, ic: int, oC: list[int]):
            super().__init__()
            self.model = DummyModel(iC, oC)

        def forward(self):
            pass

    return DummyLightningModule(iC, oC)


class StepSLScheduler(BaseDataScheduler):
    """The sparsity param of each data group is multiplied by gamma every step_size epochs."""

    def __init__(
        self,
        data_sparsifier,
        schedule_param="sparsity_level",
        step_size=1,
        gamma=2,
        last_epoch=-1,
        verbose=False,
    ):
        self.gamma = gamma
        self.step_size = step_size
        super().__init__(data_sparsifier, schedule_param, last_epoch, verbose)

    def get_schedule_param(self):
        if not self._get_sp_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
                stacklevel=2,
            )
        data_groups = self.data_sparsifier.data_groups
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return {
                name: config[self.schedule_param]
                for name, config in data_groups.items()
            }

        return {
            name: config[self.schedule_param] * self.gamma
            for name, config in data_groups.items()
        }


class TestPostTrainingCallback(TestCase):
    def _check_on_fit_end(self, pl_module, callback, sparsifier_args):
        """Makes sure that each component of is working as expected while calling the
        post-training callback.
        Specifically, check the following -
            1. sparsifier config is the same as input config
            2. data sparsifier is correctly attached to the model
            3. sparsity is achieved after .step()
            4. non-sparsified values are the same as original values
        """
        callback.on_fit_end(42, pl_module)  # 42 is a dummy value

        # check sparsifier config
        for key, value in sparsifier_args.items():
            if callback.data_sparsifier.defaults[key] != value:
                raise AssertionError(
                    f"data_sparsifier.defaults[{key!r}] = {callback.data_sparsifier.defaults[key]}, expected {value}"
                )

        # assert that the model is correctly attached to the sparsifier
        for name, param in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            if type(param) not in SUPPORTED_TYPES:
                if valid_name in callback.data_sparsifier.state:
                    raise AssertionError(
                        f"valid_name {valid_name!r} should not be in data_sparsifier.state"
                    )
                if valid_name in callback.data_sparsifier.data_groups:
                    raise AssertionError(
                        f"valid_name {valid_name!r} should not be in data_sparsifier.data_groups"
                    )
                continue
            if valid_name not in callback.data_sparsifier.data_groups:
                raise AssertionError(
                    f"valid_name {valid_name!r} should be in data_sparsifier.data_groups"
                )
            if valid_name not in callback.data_sparsifier.state:
                raise AssertionError(
                    f"valid_name {valid_name!r} should be in data_sparsifier.state"
                )

            mask = callback.data_sparsifier.get_mask(name=valid_name)

            # assert that some level of sparsity is achieved
            sparsity = 1.0 - mask.float().mean()
            if not (sparsity > 0.0):
                raise AssertionError(f"expected sparsity > 0.0, got {sparsity}")

            # make sure that non-zero values in data after squash mask are equal to original values
            sparsified_data = callback.data_sparsifier.get_data(
                name=valid_name, return_original=False
            )
            if not torch.all(
                sparsified_data[sparsified_data != 0] == param[sparsified_data != 0]
            ):
                raise AssertionError(
                    "non-zero sparsified_data values do not match original param values"
                )

    @unittest.skipIf(
        not importlib.util.find_spec("pytorch_lightning"), "No pytorch_lightning"
    )
    def test_post_training_callback(self):
        sparsifier_args = {
            "sparsity_level": 0.5,
            "sparse_block_shape": (1, 4),
            "zeros_per_block": 4,
        }
        callback = PostTrainingDataSparsity(DataNormSparsifier, sparsifier_args)
        pl_module = _make_lightning_module(100, [128, 256, 16])

        self._check_on_fit_end(pl_module, callback, sparsifier_args)


class TestTrainingAwareCallback(TestCase):
    """Class to test in-training version of lightning callback
    Simulates model training and makes sure that each hook is doing what is expected
    """

    def _check_on_train_start(
        self, pl_module, callback, sparsifier_args, scheduler_args
    ):
        """Makes sure that the data_sparsifier and data_scheduler objects are being created
        correctly.
        Basically, confirms that the input args and sparsifier/scheduler args are in-line.
        """

        callback.on_train_start(42, pl_module)  # 42 is a dummy value

        # sparsifier and scheduler instantiated
        if not (
            callback.data_scheduler is not None and callback.data_sparsifier is not None
        ):
            raise AssertionError(
                f"data_scheduler is {callback.data_scheduler}, "
                f"data_sparsifier is {callback.data_sparsifier}, both should not be None"
            )

        # data sparsifier args are correct
        for key, value in sparsifier_args.items():
            if callback.data_sparsifier.defaults[key] != value:
                raise AssertionError(
                    f"data_sparsifier.defaults[{key!r}] = {callback.data_sparsifier.defaults[key]}, expected {value}"
                )

        # data scheduler args are correct
        for key, value in scheduler_args.items():
            actual = getattr(callback.data_scheduler, key)
            if actual != value:
                raise AssertionError(
                    f"data_scheduler.{key} = {actual}, expected {value}"
                )

    def _simulate_update_param_model(self, pl_module):
        """This function might not be needed as the model is being copied
        during train_epoch_end() but good to have if things change in the future
        """
        for _, param in pl_module.model.named_parameters():
            param.data = param + 1

    def _check_on_train_epoch_start(self, pl_module, callback):
        """Basically ensures that the sparsifier's state is correctly being restored.
        The state_dict() comparison is needed. Consider the flow -

        **Epoch: 1**
            1. on_train_epoch_start(): Nothing happens (for now)
            2. on_train_epoch_end():
                a) the model is copied into the data_sparsifier
                b) .step() is called
                c) internally, the state of each layer of the model inside
                   data sparsifier changes

        **Epoch: 2**
            1. on_train_epoch_start(): Assume nothing happens
            2. on_train_epoch_end():
                a) the model is copied into the data_sparsifier.
                   But wait! you need the config to attach layer
                   of the module to the sparsifier. If config is None,
                   the data_sparsifier uses the default config which we
                   do not want as the config of each layer changes after
                   .step()

        Hence, we need to dump and restore the state_dict() every time because we're
        copying the model after each epoch.
        Hence, it is essential to make sure that the sparsifier's state_dict() is being
        correctly dumped and restored.

        """
        # check if each component of state dict is being loaded correctly
        callback.on_train_epoch_start(42, pl_module)
        if callback.data_sparsifier_state_dict is None:
            return

        data_sparsifier_state_dict = callback.data_sparsifier.state_dict()

        # compare container objects
        container_obj1 = data_sparsifier_state_dict["_container"]
        container_obj2 = callback.data_sparsifier_state_dict["_container"]
        if len(container_obj1) != len(container_obj2):
            raise AssertionError(
                f"container lengths differ: {len(container_obj1)} vs {len(container_obj2)}"
            )
        for key, value in container_obj2.items():
            if key not in container_obj1:
                raise AssertionError(f"key {key!r} not in container_obj1")
            if not torch.all(value == container_obj1[key]):
                raise AssertionError(f"container values differ for key {key!r}")

        # compare state objects
        state_obj1 = data_sparsifier_state_dict["state"]
        state_obj2 = callback.data_sparsifier_state_dict["state"]
        if len(state_obj1) != len(state_obj2):
            raise AssertionError(
                f"state lengths differ: {len(state_obj1)} vs {len(state_obj2)}"
            )
        for key, value in state_obj2.items():
            if key not in state_obj1:
                raise AssertionError(f"key {key!r} not in state_obj1")
            if not ("mask" in value and "mask" in state_obj1[key]):
                raise AssertionError(f"'mask' not in value or state_obj1[{key!r}]")
            if not torch.all(value["mask"] == state_obj1[key]["mask"]):
                raise AssertionError(f"mask values differ for key {key!r}")

        # compare data_groups dict
        data_grp1 = data_sparsifier_state_dict["data_groups"]
        data_grp2 = callback.data_sparsifier_state_dict["data_groups"]
        if len(data_grp1) != len(data_grp2):
            raise AssertionError(
                f"data_groups lengths differ: {len(data_grp1)} vs {len(data_grp2)}"
            )
        for key, value in data_grp2.items():
            if key not in data_grp1:
                raise AssertionError(f"key {key!r} not in data_grp1")
            if value != data_grp1[key]:
                raise AssertionError(
                    f"data_groups[{key!r}] differ: {value} vs {data_grp1[key]}"
                )

    def _check_on_train_epoch_end(self, pl_module, callback):
        """Checks the following -
        1. sparsity is correctly being achieved after .step()
        2. scheduler and data_sparsifier sparsity levels are in-line
        """
        callback.on_train_epoch_end(42, pl_module)
        data_scheduler = callback.data_scheduler
        base_sl = data_scheduler.base_param

        for name, _ in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            mask = callback.data_sparsifier.get_mask(name=valid_name)

            # check sparsity levels
            sparsity = 1.0 - mask.float().mean()
            if not (sparsity > 0):
                raise AssertionError(f"expected sparsity > 0, got {sparsity}")

            last_sl = data_scheduler.get_last_param()
            last_epoch = data_scheduler.last_epoch

            # check sparsity levels of scheduler
            log_last_sl = math.log(last_sl[valid_name])
            log_actual_sl = math.log(
                base_sl[valid_name] * (data_scheduler.gamma**last_epoch)
            )
            if log_last_sl != log_actual_sl:
                raise AssertionError(
                    f"log_last_sl {log_last_sl} != log_actual_sl {log_actual_sl}"
                )

    def _check_on_train_end(self, pl_module, callback):
        """Confirms that the mask is squashed after the training ends
        This is achieved by making sure that each parameter in the internal container
        are not parametrized.
        """
        callback.on_train_end(42, pl_module)

        # check that the masks have been squashed
        for name, _ in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            if is_parametrized(callback.data_sparsifier._continer, valid_name):
                raise AssertionError(
                    f"expected {valid_name!r} to not be parametrized after squash"
                )

    @unittest.skipIf(
        not importlib.util.find_spec("pytorch_lightning"), "No pytorch_lightning"
    )
    def test_train_aware_callback(self):
        sparsifier_args = {
            "sparsity_level": 0.5,
            "sparse_block_shape": (1, 4),
            "zeros_per_block": 4,
        }
        scheduler_args = {"gamma": 2, "step_size": 1}

        callback = TrainingAwareDataSparsity(
            data_sparsifier_class=DataNormSparsifier,
            data_sparsifier_args=sparsifier_args,
            data_scheduler_class=StepSLScheduler,
            data_scheduler_args=scheduler_args,
        )

        pl_module = _make_lightning_module(100, [128, 256, 16])

        # simulate the training process and check all steps
        self._check_on_train_start(pl_module, callback, sparsifier_args, scheduler_args)

        num_epochs = 5
        for _ in range(num_epochs):
            self._check_on_train_epoch_start(pl_module, callback)
            self._simulate_update_param_model(pl_module)
            self._check_on_train_epoch_end(pl_module, callback)


if __name__ == "__main__":
    run_tests()
