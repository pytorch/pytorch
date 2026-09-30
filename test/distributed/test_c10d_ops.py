# Owner(s): ["oncall: distributed"]

import tempfile
import unittest

import torch
import torch.utils.cpp_extension
from torch.testing._internal.common_utils import run_tests, TestCase


@unittest.skipUnless(torch.distributed.is_available(), "distributed required")
class TestC10dOps(TestCase):
    def test_config_schema_defaults(self):
        names = (
            "broadcast_",
            "allreduce_",
            "allreduce_coalesced_",
            "allgather_",
            "_allgather_base_",
            "allgather_coalesced_",
            "allgather_into_tensor_coalesced_",
            "reduce_scatter_",
            "_reduce_scatter_base_",
            "reduce_scatter_tensor_coalesced_",
            "reduce_",
            "gather_",
            "gather_into_tensor_",
            "alltoall_",
            "alltoall_base_",
        )
        for name in names:
            with self.subTest(op=name):
                op = getattr(torch.ops.c10d, name)
                self.assertEqual(op.overloads(), ["default"])
                config = op.default._schema.arguments[-1]
                self.assertEqual(config.name, "config")
                self.assertTrue(config.has_default_value())
                self.assertIsNone(config.default_value)
                self.assertEqual(str(config.type), "Optional[Any]")

    def test_meta_optional_config(self):
        group = torch.distributed.ProcessGroup(0, 1).boxed()
        reduce_op = torch.distributed.ReduceOp(torch.distributed.ReduceOp.SUM).boxed()
        tensor = torch.ones(2, device="meta")
        for kwargs in ({}, {"config": None}):
            outputs, _ = torch.ops.c10d.allreduce_(
                [tensor], group, reduce_op, None, False, **kwargs
            )
            self.assertEqual(outputs[0].shape, tensor.shape)
        with self.assertRaisesRegex(RuntimeError, "Raw c10d configuration calls"):
            torch.ops.c10d.allreduce_(
                [tensor], group, reduce_op, None, False, config="test"
            )

    def test_c10d_optional_config_kernel_signatures(self):
        source = """
        #include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
        #include <torch/library.h>

        class LegacyBackend : public c10d::Backend {
         public:
          LegacyBackend() : Backend(0, 1) {}
          const std::string getBackendName() const override { return "legacy"; }
          int calls = 0;
          c10::intrusive_ptr<c10d::Work> allreduce(
              std::vector<at::Tensor>& tensors,
              const c10d::AllreduceOptions& opts) override {
            ++calls;
            return nullptr;
          }
        };

        class ConfigBackend : public LegacyBackend {
         public:
          c10::intrusive_ptr<c10d::Work> allreduceConfig(
              std::vector<at::Tensor>& tensors,
              const c10d::AllreduceOptions& opts) override {
            return allreduce(tensors, opts);
          }
        };

        void check_registration() {
            std::vector<at::Tensor> tensors;
            c10d::AllreduceOptions opts;
            LegacyBackend legacy;
            legacy.allreduceConfig(tensors, opts);
            TORCH_CHECK(legacy.calls == 1);
            opts.config = c10::IValue(1);
            bool rejected = false;
            try {
                legacy.allreduceConfig(tensors, opts);
            } catch (const c10::Error&) {
                rejected = true;
            }
            TORCH_CHECK(rejected && legacy.calls == 1);
            ConfigBackend configured;
            configured.allreduceConfig(tensors, opts);
            TORCH_CHECK(configured.calls == 1);

            torch::Library library(
                torch::Library::IMPL, "c10d", c10::DispatchKey::Meta,
                __FILE__, __LINE__);
            library.impl("allreduce_", [](
                at::TensorList tensors,
                const c10::intrusive_ptr<c10d::ProcessGroup>& group,
                const c10::intrusive_ptr<c10d::ReduceOp>& op,
                const std::optional<at::Tensor>& sparse_indices,
                bool async_op,
                int64_t timeout,
                c10d::OptionalCollectiveConfig config
            ) -> std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<c10d::Work>> {
                return {tensors.vec(), nullptr};
            });
            library.impl("_allgather_base_", [](
                at::Tensor& output,
                at::Tensor& input,
                const c10::intrusive_ptr<c10d::ProcessGroup>& group,
                bool async_op,
                int64_t timeout,
                c10d::OptionalCollectiveConfig config
            ) -> std::tuple<at::Tensor, c10::intrusive_ptr<c10d::Work>> {
                return {output, nullptr};
            });
        }
        """
        with tempfile.TemporaryDirectory() as build_directory:
            module = torch.utils.cpp_extension.load_inline(
                name="c10d_optional_config_kernel_signatures",
                cpp_sources=source,
                functions=["check_registration"],
                build_directory=build_directory,
            )
            module.check_registration()


if __name__ == "__main__":
    run_tests()
