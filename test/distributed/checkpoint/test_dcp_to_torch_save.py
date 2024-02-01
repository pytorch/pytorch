# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    ModelArgs,
    skip_if_lt_x_gpu,
    Transformer,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class TestDCPToTorchSave(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_dcp_to_torch_save(self) -> None:
        # Using a transformer model to simulate a 'complicated enough' state dict w/ nested modules
        model = Transformer(ModelArgs())
        dcp.save({"model": model}, checkpoint_id=self.temp_dir)

        torch_fn = self.temp_dir + ".pt"
        dcp_to_torch_save(self.temp_dir, torch_fn)

        loaded_sd = torch.load(torch_fn)
        self.assertEqual(loaded_sd, {"model": model.state_dict()})


if __name__ == "__main__":
    run_tests()
