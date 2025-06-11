# Owner(s): ["oncall: distributed checkpointing"]

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint import _HuggingFaceLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


CHECKPOINT_DIR = "checkpoint"


class MyTestModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_1 = torch.nn.Linear(5, 5)
        self.linear_2 = torch.nn.Linear(5, 1)
        self.emb = torch.nn.EmbeddingBag(5, 10)


class TestSingleRankSaveLoad(TestCase):
    @with_temp_dir
    def test_save(self) -> None:
        try:
            from safetensors.torch import load_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        dist_cp.save(
            state_dict=state_dict_to_save,
            storage_writer=dist_cp._HuggingFaceStorageWriter(path=CHECKPOINT_DIR),
        )

        state_dict_loaded = load_file(
            CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )
        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_loaded.keys())
        )
        for key in state_dict_to_save.keys():
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )

    @with_temp_dir
    def test_load(self) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        state_dict_to_load = MyTestModule().state_dict()
        save_file(
            state_dict_to_save, CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )

        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp._HuggingFaceStorageReader(path=CHECKPOINT_DIR),
        )

        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_to_load.keys())
        )
        for key in state_dict_to_save.keys():
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_to_load[key])
            )

    @with_temp_dir
    def test_load_into_empty_dict(self) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        save_file(
            state_dict_to_save, CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )

        state_dict_loaded = _load_state_dict_from_keys(
            storage_reader=dist_cp._HuggingFaceStorageReader(path=CHECKPOINT_DIR),
        )

        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_loaded.keys())
        )
        for key in state_dict_to_save.keys():
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_loaded[key])
            )

    @with_temp_dir
    def test_load_allowing_resize(self) -> None:
        try:
            from safetensors.torch import save_file
        except ImportError:
            print("safetensors not installed")
            return

        CHECKPOINT_DIR = self.temp_dir

        state_dict_to_save = MyTestModule().state_dict()
        save_file(
            state_dict_to_save, CHECKPOINT_DIR + "/model-00001-of-00001.safetensors"
        )

        state_dict_to_load = {}
        for key in state_dict_to_save.keys():
            state_dict_to_load[key] = torch.zeros(1)

        dist_cp.load(
            state_dict=state_dict_to_load,
            storage_reader=dist_cp._HuggingFaceStorageReader(path=CHECKPOINT_DIR),
            planner=_HuggingFaceLoadPlanner(allow_tensor_resize=True),
        )

        self.assertEqual(
            sorted(state_dict_to_save.keys()), sorted(state_dict_to_load.keys())
        )
        for key in state_dict_to_save.keys():
            self.assertTrue(
                torch.equal(state_dict_to_save[key], state_dict_to_load[key])
            )


if __name__ == "__main__":
    run_tests()
