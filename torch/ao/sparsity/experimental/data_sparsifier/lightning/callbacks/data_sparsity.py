from collections import defaultdict
from copy import deepcopy
import torch
from dataclasses import dataclass, field
from typing import Any, Optional, Dict
import pytorch_lightning as pl

from ._data_sparstity_utils import (
    _create_data_sparsifier,
    _attach_model_to_data_sparsifier,
    _log_sparsified_level,
    _create_data_scheduler,
    _get_valid_name
)


@dataclass
class _DataSparsity(pl.callbacks.Callback):
    data_sparsifier_type: Any
    data_sparsifier_args: Dict  # this the arguments for sparsifier_type (except data_list)
    data_scheduler_type: Optional[Any] = None  # Data sparsity scheduler, used only while in-training
    data_scheduler_args: Optional[Dict] = None  # this is the arguments for scheduler_type (except data_sparsifier)
    data_sparsifier: Any = field(init=False, default=None)
    sparsified: Optional[torch.nn.Module] = field(init=False, default=None)
    data_scheduler: Any = field(init=False, default=None)

    def __post_init__(self):
        pass


@dataclass
class PostTrainingDataSparsity(_DataSparsity):
    """Lightning callback that enables post-training sparsity.

    This callback aims to sparsify the model inside lightning module after training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_type (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this type is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            once the training completes.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

    Hooks implemented:
        on_fit_end()
            1. copies the model and attaches it to the sparsifier
            2. sparsier step() is called
            3. squashes the mask()
    """

    def on_fit_end(self, trainer, pl_module) -> None:
        self.sparsified = deepcopy(pl_module.model).eval()
        self.data_sparsifier = _create_data_sparsifier(self.data_sparsifier_args, self.data_sparsifier_type)

        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)

        self.data_sparsifier.step()

        self.data_sparsifier.squash_mask()  # currently squashes params for all mask

        _log_sparsified_level(self.sparsified, self.data_sparsifier)


class TrainingAwareDataSparsity(_DataSparsity):
    """Lightning callback that enables in-training sparsity.

    This callback aims to sparsify the model inside lightning module during training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_type (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this type is created when the
            training starts.
            Note: Objects should not be passed in here as they are created
            when the training starts.

        data_sparsifier_args (Dict)
            Dictionary of args to be passed to the data sparsifier.
            Note: data_list arg should be ignored

        data_scheduler_type (some implemented class of BaseDataScheduler)
            The data scheduler of this type is created when the training starts
            Note: Objects should not be passed in here as they are created
            when the training starts.

        data_scheduler_args(Dict)
            Dictionary of args to be passed to the data scheduler.
            **Note: data_sparsifier arg should be ignored as the recipe
            creates and pass sparsifier object into the class**

    Hooks implemented:
        on_train_start()
            Data sparsifier and scheduler objects are created.
            Pytorch model attached to the sparsifier

        on_train_epoch_start()
            Loads the state_dict of the data sparsifier

        on_train_epoch_end()
            1. Copies the model and attaches it to the sparsifier
            2. sparsifier step() and scheduler step()
            3. Dump state_dict of the current sparsifier

        on_train_end()
            squash mask
    """

    def __post_init__(self):
        assert self.data_scheduler_type is not None
        assert self.data_scheduler_args is not None
        self.data_sparsifier_state_dict = None

    def on_train_start(self, trainer, pl_module) -> None:
        # create sparsifier
        self.data_sparsifier = _create_data_sparsifier(self.data_sparsifier_args, self.data_sparsifier_type)
        self.sparsified = deepcopy(pl_module.model)

        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)  # just to populate the base_sl in the scheduler
        # create scheduler
        self.data_scheduler = _create_data_scheduler(self.data_sparsifier, self.data_scheduler_args, self.data_scheduler_type)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.data_sparsifier_state_dict is None:
            return  # probably first epoch

        # load the existing config for each data
        self.data_sparsifier.load_state_dict(self.data_sparsifier_state_dict)

    def __create_config_based_on_state(self, pl_module):
        config: Dict = defaultdict()
        if self.data_sparsifier_state_dict is None:
            return config
        for name, _ in pl_module.model.named_parameters():
            valid_name = _get_valid_name(name)
            config[valid_name] = self.data_sparsifier.data_groups[valid_name]

        return config

    def on_train_epoch_end(self, trainer, pl_module):
        self.sparsified = deepcopy(pl_module.model)
        config = self.__create_config_based_on_state(pl_module)

        # attach model to the data sparsifier
        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier, config=config)
        self.data_sparsifier.step()
        self.data_scheduler.step()

        self.data_sparsifier_state_dict = self.data_sparsifier.state_dict()

    def on_train_end(self, trainer, pl_module):
        self.data_sparsifier.squash_mask()
