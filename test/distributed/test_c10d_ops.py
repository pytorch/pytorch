# Owner(s): ["oncall: distributed"]

import tempfile
import unittest

import torch
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipUnless(torch.distributed.is_available(), "distributed required")
class TestC10dOps(TestCase):
    def test_c10d_existing_kernel_signatures(self):
        source = """
        #include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
        #include <torch/library.h>

        void check_registration() {
            torch::Library library(
                torch::Library::IMPL, "c10d", c10::DispatchKey::Meta,
                __FILE__, __LINE__);
            library.impl("allreduce_", [](
                at::TensorList tensors,
                const c10::intrusive_ptr<c10d::ProcessGroup>& group,
                const c10::intrusive_ptr<c10d::ReduceOp>& op,
                const std::optional<at::Tensor>& sparse_indices,
                bool async_op,
                int64_t timeout
            ) -> std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>> {
                return {tensors.vec(), nullptr};
            });
            library.impl("_allgather_base_", [](
                at::Tensor& output,
                at::Tensor& input,
                const c10::intrusive_ptr<c10d::ProcessGroup>& group,
                bool async_op,
                int64_t timeout
            ) -> std::tuple<at::Tensor, c10::intrusive_ptr<c10d::Work>> {
                return {output, nullptr};
            });
        }
        """
        with tempfile.TemporaryDirectory() as build_directory:
            module = torch.utils.cpp_extension.load_inline(
                name="c10d_existing_kernel_signatures",
                cpp_sources=source,
                functions=["check_registration"],
                build_directory=build_directory,
            )
            module.check_registration()


if __name__ == "__main__":
    run_tests()
