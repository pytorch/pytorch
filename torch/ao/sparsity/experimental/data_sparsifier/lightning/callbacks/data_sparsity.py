from copy import deepcopy
import torch
from typing import Any, Optional
import pytorch_lightning as pl  # type: ignore

from ._data_sparstity_utils import _attach_model_to_data_sparsifier, _log_sparsified_level


class PostTrainingDataSparsity(pl.callbacks.Callback):
    """Lightning callback that enables post-training sparsity.

    This callback aims to sparsify the model inside lightning module after training.
    **Note that the model is copied and then sparsified, so the existing model is not modified**

    The sparsified model can be used for comparison and can be accessed using
        <callback_obj>.sparsified

    Args:
        data_sparsifier_class (some implemented class of BaseDataSparsifier)
            The data sparsifier object of this class is created when the
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
    def __init__(self, data_sparsifier_class, data_sparsifier_args):
        super().__init__()
        self.data_sparsifier_class = data_sparsifier_class
        self.data_sparsifier_args = data_sparsifier_args
        self.data_sparsifier: Any = None
        self.sparsified: Optional[torch.nn.Module] = None

    def on_fit_end(self, trainer, pl_module) -> None:
        self.sparsified = deepcopy(pl_module.model).eval()
        self.data_sparsifier = self.data_sparsifier_class(**self.data_sparsifier_args)

        _attach_model_to_data_sparsifier(self.sparsified, self.data_sparsifier)

        self.data_sparsifier.step()

        self.data_sparsifier.squash_mask()  # currently squashes params for all mask

        _log_sparsified_level(self.sparsified, self.data_sparsifier)
