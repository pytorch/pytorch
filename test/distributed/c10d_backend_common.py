# Owner(s): ["oncall: distributed"]

import os
import unittest
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import TEST_CUDA


STANDARD_DTYPES = (
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
)

FLOAT8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2fnuz,
)

COMPLEX_DTYPES = (
    torch.complex64,
    torch.complex128,
)


@dataclass(frozen=True)
class BackendConfig:
    name: str
    device_type: str
    supports_coalescing: bool = False
    supports_bitwise_reductions: bool = False
    supports_cuda_graph_barrier: bool = False
    supports_dropped_p2p_work: bool = False
    dtypes: tuple[torch.dtype, ...] = STANDARD_DTYPES
    float8_dtypes: tuple[torch.dtype, ...] = ()
    complex_dtypes: tuple[torch.dtype, ...] = COMPLEX_DTYPES


C10D_BACKENDS = (
    BackendConfig("gloo", "cpu", supports_bitwise_reductions=True),
    BackendConfig(
        "nccl",
        "cuda",
        supports_coalescing=True,
        supports_dropped_p2p_work=True,
        float8_dtypes=FLOAT8_DTYPES,
    ),
    BackendConfig(
        "nccl2",
        "cuda",
        supports_coalescing=True,
        supports_cuda_graph_barrier=True,
        supports_dropped_p2p_work=True,
        float8_dtypes=FLOAT8_DTYPES,
    ),
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
                "supports_bitwise_reductions": backend.supports_bitwise_reductions,
                "supports_cuda_graph_barrier": backend.supports_cuda_graph_barrier,
                "supports_dropped_p2p_work": backend.supports_dropped_p2p_work,
                "dtypes": backend.dtypes,
                "float8_dtypes": backend.float8_dtypes,
                "complex_dtypes": backend.complex_dtypes,
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
