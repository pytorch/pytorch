# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.utils.checkpoint import checkpoint

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


def get_cur_mem(rank, result, prefix):
    """Collect memory allocated values in a result dict in MB"""
    torch._C._cuda_clearCublasWorkspaces()
    result[prefix] = round(torch.cuda.memory_allocated() / 1024 / 1024)


class Model(nn.Module):
    def __init__(self, hidden_dim, with_fsdp=False, with_checkpoint=False):
        super().__init__()
        if with_fsdp:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3),
                FSDP(nn.BatchNorm2d(64)),
                nn.ReLU(inplace=True),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
            )
        if with_fsdp:
            self.blocks = nn.Sequential(
                nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),
                FSDP(nn.BatchNorm2d(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                FSDP(nn.BatchNorm2d(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                FSDP(nn.BatchNorm2d(hidden_dim)),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
            )
        else:
            self.blocks = nn.Sequential(
                nn.Conv2d(64, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
            )

        self.head = nn.Linear(hidden_dim, 10)
        self.with_checkpoint = with_checkpoint

    def forward(self, x):
        if self.with_checkpoint:
            return self.head(checkpoint(self.blocks, self.stem(x), use_reentrant=True))
        else:
            return self.head(self.blocks(self.stem(x)))


def create_model(with_fsdp, with_checkpoint, model_hidden_dim):
    torch.manual_seed(0)
    model = Model(model_hidden_dim, with_fsdp, with_checkpoint)
    if with_fsdp:
        model.stem = FSDP(model.stem)
        model.blocks = FSDP(model.blocks)
        model.head = FSDP(model.head)

    return model


class TestFSDPMemory(FSDPTest):
    @property
    def world_size(self):
        return 2

    def _dist_train(self, with_checkpoint, expected, model_hidden_dim, iterations):
        gpu_id = self.rank
        world_size = self.world_size

        batch = torch.randn(size=(2, 3, 224, 224)).cuda()

        model = create_model(
            with_fsdp=True,
            with_checkpoint=with_checkpoint,
            model_hidden_dim=model_hidden_dim,
        )
        model = model.cuda()
        model = FSDP(model)

        # We enable momentum so that after the first iteration, the optimizer state is added
        # to the total memory used.
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

        results = {}  # results of memory stats
        for iteration in range(iterations):
            get_cur_mem(gpu_id, results, f"iter {iteration}: start")

            out = model(batch)
            get_cur_mem(gpu_id, results, f"iter {iteration}: after fwd")

            out = sum(o.sum() for o in out[0])
            fake_loss = criterion(out, torch.tensor(0.0).cuda())
            get_cur_mem(gpu_id, results, f"iter {iteration}: after loss")

            fake_loss.backward()
            get_cur_mem(gpu_id, results, f"iter {iteration}: after bwd")

            optimizer.step()
            get_cur_mem(gpu_id, results, f"iter {iteration}: after step")

            # It is important to use `set_to_none` below, not optimizer.zero_grad() to reclaim memory.
            model.zero_grad(set_to_none=True)
            get_cur_mem(gpu_id, results, f"iter {iteration}: done")

        def cmp(results, expected):
            ret = ""
            self.assertEqual(results.keys(), expected.keys())
            for k, v in results.items():
                exp = expected[k]
                if abs(exp - v) > 1:  # allow 1MB rounding differences
                    ret += f"{k}: got {v}, expected {exp}\n"
            return ret

        output = cmp(results, expected)
        self.assertEqual(output, "")

    @skip_if_lt_x_gpu(2)
    @parametrize("ckpt", ["no_ckpt", "ckpt"])
    def test_fsdp_memory(self, ckpt):
        # hidden_dim 128: model size ~4MB
        model_hidden_dim = 128

        model = create_model(
            with_fsdp=False, with_checkpoint=False, model_hidden_dim=model_hidden_dim
        ).cuda()
        model_size_mb = round(torch.cuda.memory_allocated() / 1024 / 1024)
        del model

        sharded_model_size_mb = int(model_size_mb / self.world_size)

        # We have observed that sometimes after 3rd iteration, 4th one can fail (not on this
        # test but on much bigger scale tests). We run 4 iterations here just in case it happens.
        iterations = 4

        expected = {}

        for iteration in range(iterations):
            if iteration == 0:
                # sharded model size + 1MB temp memory
                expected[f"iter {iteration}: start"] = sharded_model_size_mb + 1
                # it is hard to calculate this memory size, get it from printed memory usage
                if ckpt == "ckpt":
                    expected[f"iter {iteration}: after fwd"] = 51
                    expected[f"iter {iteration}: after loss"] = 51
                else:
                    expected[f"iter {iteration}: after fwd"] = 340
                    expected[f"iter {iteration}: after loss"] = 340
                # sharded model size + sharded grad size + 1M temp memory
                expected[f"iter {iteration}: after bwd"] = 2 * sharded_model_size_mb + 1
            else:
                # after optimizer step in the first iteration, memory usage increased by
                # sharded_model_size_mb because of increased optimizer states memory usage
                expected[f"iter {iteration}: start"] = 2 * sharded_model_size_mb + 1
                if ckpt == "ckpt":
                    expected[f"iter {iteration}: after fwd"] = (
                        51 + sharded_model_size_mb
                    )
                    expected[f"iter {iteration}: after loss"] = (
                        51 + sharded_model_size_mb
                    )
                else:
                    expected[f"iter {iteration}: after fwd"] = (
                        340 + sharded_model_size_mb
                    )
                    expected[f"iter {iteration}: after loss"] = (
                        340 + sharded_model_size_mb
                    )
                expected[f"iter {iteration}: after bwd"] = 3 * sharded_model_size_mb + 1

            # sharded model size + sharded grad size + optimizer states + 1M temp memory
            expected[f"iter {iteration}: after step"] = 3 * sharded_model_size_mb + 1
            # grad memory is claimed after setting grad = None
            # sharded model size + optimizer states + 1M temp memory
            expected[f"iter {iteration}: done"] = 2 * sharded_model_size_mb + 1

        # Get the fsdp and checkpoint flags.
        with_ckpt = ckpt == "ckpt"

        self._dist_train(
            with_ckpt,
            expected,
            model_hidden_dim,
            iterations,
        )


instantiate_parametrized_tests(TestFSDPMemory)


if __name__ == "__main__":
    run_tests()
