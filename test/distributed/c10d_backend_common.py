# Owner(s): ["oncall: distributed"]

import os
import unittest
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_CUDA


@dataclass(frozen=True)
class BackendConfig:
    name: str
    device_type: str
    supports_coalescing: bool = False


C10D_BACKENDS = (
    BackendConfig("gloo", "cpu"),
    BackendConfig("nccl", "cuda", supports_coalescing=True),
    BackendConfig("nccl2", "cuda", supports_coalescing=True),
)

CUDA_BACKENDS = tuple(
    backend for backend in C10D_BACKENDS if backend.device_type == "cuda"
)


class C10dBackendTest:
    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        if self.device_type == "cuda":
            return torch.device(self.device_type, self.rank)
        return torch.device(self.device_type)

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _init_pg(self):
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=60),
        )


def instantiate_backend_tests(namespace, suite_name, base_class, backends):
    for backend in backends:
        backend_name = backend.name.replace("-", " ").title().replace(" ", "")
        class_name = f"{backend_name}{suite_name}Test"
        test_class = type(
            class_name,
            (base_class, MultiProcessTestCase),
            {
                "__module__": namespace["__name__"],
                "backend_name": backend.name,
                "device_type": backend.device_type,
                "supports_coalescing": backend.supports_coalescing,
            },
        )
        test_class = unittest.skipIf(
            not dist.is_backend_available(backend.name),
            f"{backend.name} backend is not available",
        )(test_class)
        if backend.device_type == "cuda":
            test_class = unittest.skipIf(
                not TEST_CUDA or torch.cuda.device_count() < 2,
                f"{backend.name} requires 2+ GPUs",
            )(test_class)
        namespace[class_name] = test_class
