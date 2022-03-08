#pragma once

#include <torch/csrc/lazy/ts_backend/ts_node.h>

namespace torch_lazy_tensors {
namespace ir {
namespace ops {

// This IR was copied from code-generated output, but the entire _to_copy operator
// cannot be trivially code genereated since it is only desirable to capture IR for
// certain permutaions of _to_copy (e.g. dtype), and for the others it is difficult to even invoke
// the aten/eager fallback necessitating directly implementing the right to(device) behavior
class ToCopy : public torch::lazy::TsNode {
 public:
  ToCopy(const torch::lazy::Value& self, const c10::optional<at::ScalarType>& dtype, const c10::optional<at::Layout>& layout, const c10::optional<at::Device>& device, const c10::optional<bool>& pin_memory, const bool& non_blocking, const c10::optional<at::MemoryFormat>& memory_format, std::vector<torch::lazy::Shape>&& shapes)
      : torch::lazy::TsNode(torch::lazy::OpKind(at::aten::_to_copy),
              {self}, std::move(shapes),
              /* num_outputs */ 1,
              torch::lazy::MHash(dtype, layout, device, pin_memory, non_blocking, memory_format)),

        dtype(dtype),
        layout(layout),
        device(device),
        pin_memory(pin_memory),
        non_blocking(non_blocking),
        memory_format(memory_format) {}

  std::string ToString() const override {
    std::stringstream ss;
    ss << torch::lazy::TsNode::ToString();
    if (dtype.has_value()) {
        ss << ", dtype=" << dtype.value();
    } else {
        ss << ", dtype=null";
    }
    if (layout.has_value()) {
        ss << ", layout=" << layout.value();
    } else {
        ss << ", layout=null";
    }
    if (device.has_value()) {
        ss << ", device=" << device.value();
    } else {
        ss << ", device=null";
    }
    if (pin_memory.has_value()) {
        ss << ", pin_memory=" << pin_memory.value();
    } else {
        ss << ", pin_memory=null";
    }
    ss << ", non_blocking=" << non_blocking;
    if (memory_format.has_value()) {
        ss << ", memory_format=" << memory_format.value();
    } else {
        ss << ", memory_format=null";
    }
    return ss.str();
  }

  torch::lazy::TSOpVector Lower(std::shared_ptr<torch::jit::GraphFunction> function,
                   torch::lazy::TSLoweringContext* loctx) const override {
        std::vector<torch::jit::NamedValue> arguments;
    std::vector<torch::jit::NamedValue> kwarguments;
    arguments.reserve(1);
    kwarguments.reserve(6);
    size_t i = 0;
    arguments.emplace_back(loctx->GetOutputOp(operand(i++)));
    kwarguments.emplace_back("dtype", dtype);
    kwarguments.emplace_back("layout", layout);
    kwarguments.emplace_back("device", device);
    kwarguments.emplace_back("pin_memory", pin_memory);
    kwarguments.emplace_back("non_blocking", non_blocking);
    kwarguments.emplace_back("memory_format", memory_format);
    torch::lazy::TSOpVector _to_copy_out = torch::lazy::LowerTSBuiltin(function, op().op, arguments, kwarguments);
    CHECK_EQ(_to_copy_out.size(), 1);

    return _to_copy_out;

  }

  c10::optional<at::ScalarType> dtype;
  c10::optional<at::Layout> layout;
  c10::optional<at::Device> device;
  c10::optional<bool> pin_memory;
  bool non_blocking;
  c10::optional<at::MemoryFormat> memory_format;
};

} // namespace ops
} // namespace ir
} // namespace torch_lazy_tensors
