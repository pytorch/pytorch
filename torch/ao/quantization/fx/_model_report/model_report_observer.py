import torch
from torch.ao.quantization.observer import ObserverBase


class ModelReportObserver(ObserverBase):
    r"""This observer is used to record additional information regarding keeping track
    of S = average_batch_activation_range/epoch_activation_range.

    The purpose of this information is to prepare a report to present to users on whether
    Dynamic or Static Quantization is more appropriate for their model given the general
    distributions of their data.

    * :attr:`num_batches_tracked` specifies number of batches passed through the observer

    * :attr:`average_batch_activation_range` defines average across the ranges of each batch passed through

    * :attr:`epoch_activation_min` defines the minimum value passed through the observer

    * :attr:`epoch_activation_max` defines the maximum value passed through the observer

    Note: this tool is meant for FX Graph Mode Quantization
    """

    def __init__(self):
        super().__init__(torch.qint8)
        self.num_batches_tracked = 0

        # keep track of the min and mix of the range for average batch and epoch as a whole
        self.average_batch_activation_range = torch.tensor(float(0))
        self.epoch_activation_min = torch.tensor(float("inf"))
        self.epoch_activation_max = torch.tensor(float("-inf"))

    def forward(self, x):
        x_copy = x.detach()  # avoid keeping autograd tape
        x_copy = x_copy.to(self.epoch_activation_min.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x_copy)

        # calculate new epoch range values
        epoch_min_val = torch.min(self.epoch_activation_min, min_val_cur)
        epoch_max_val = torch.max(self.epoch_activation_max, max_val_cur)

        self.epoch_activation_min.copy_(epoch_min_val)
        self.epoch_activation_max.copy_(epoch_max_val)

        # calculate the average batch activation range
        current_batch_range = max_val_cur - min_val_cur
        new_range = (
            self.average_batch_activation_range * self.num_batches_tracked
            + current_batch_range
        ) / (self.num_batches_tracked + 1)

        self.average_batch_activation_range = new_range
        self.num_batches_tracked += 1  # new batch was processed

        # return the passed in the value
        return x

    @torch.jit.export
    def get_batch_to_epoch_ratio(self):
        epoch_activation_range = self.epoch_activation_max - self.epoch_activation_min

        if epoch_activation_range == torch.tensor(float(0)):
            raise ValueError("Range for Epoch is 0")
        elif epoch_activation_range == torch.tensor(float("inf")):
            raise ValueError(
                "No data has been run through observer or infinity value present"
            )
        else:
            return self.average_batch_activation_range / epoch_activation_range

    @torch.jit.export
    def reset_batch_and_epoch_values(self):
        # set all the values back to their original defaults for a new epoch
        self.num_batches_tracked = 0
        self.average_batch_activation_range = torch.tensor(float(0))
        self.epoch_activation_min = torch.tensor(float("inf"))
        self.epoch_activation_max = torch.tensor(float("-inf"))

    @torch.jit.export
    def calculate_qparams(self):
        raise Exception(
            "calculate_qparams should not be called for ModelReportObserver"
        )
