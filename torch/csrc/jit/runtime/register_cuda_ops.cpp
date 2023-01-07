// This file registers special JIT operators used to implement the PyTorch CUDA
// API in TorchScript.
#include <torch/csrc/api/include/torch/utils.h>
#include <torch/csrc/jit/cuda/cuda.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/runtime/custom_operator.h>
#include <torch/csrc/jit/runtime/operator.h>

namespace torch {
namespace jit {

namespace {

c10::AliasAnalysisKind aliasAnalysisFromSchema() {
  return c10::AliasAnalysisKind::FROM_SCHEMA;
}

void _device_synchronize(int64_t device_index) {
  // This is a helper API which synchronizes the device identified
  // by the device index. The device index of the device is passed as an
  // argument to this API.
  auto current_device_index = c10::cuda::current_device();
  // If the current_device and the device to synchronize are not
  // the same, set the device to the device_index of the device
  // to synchronize.
  if (current_device_index != device_index) {
    c10::cuda::set_device(device_index);
  }
  c10::cuda::device_synchronize();

  // Reset the device to current_device before synchronizing.
  if (current_device_index != device_index) {
    c10::cuda::set_device(current_device_index);
  }
}

RegisterOperators const reg({
    Operator(
        "cuda::current_stream.device(Device? device) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          auto device = pop(stack).toOptional<c10::Device>();
          c10::DeviceIndex device_index = device.has_value()
              ? device->index()
              : c10::cuda::current_device();
          auto s = c10::cuda::getCurrentCUDAStream(device_index);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::current_stream.int(int? val) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          auto idx = pop(stack).toOptional<int64_t>();
          c10::DeviceIndex device_index = idx.has_value()
              ? static_cast<c10::DeviceIndex>(idx.value())
              : c10::cuda::current_device();
          auto s = c10::cuda::getCurrentCUDAStream(device_index);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::default_stream.device(Device? device) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          auto device = pop(stack).toOptional<c10::Device>();
          c10::DeviceIndex device_index = device.has_value()
              ? device->index()
              : c10::cuda::current_device();
          auto s = c10::cuda::getDefaultCUDAStream(device_index);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::default_stream.int(int? val) -> __torch__.torch.classes.cuda.Stream",
        [](Stack& stack) {
          auto idx = pop(stack).toOptional<int64_t>();
          c10::DeviceIndex device_index = idx.has_value()
              ? static_cast<c10::DeviceIndex>(idx.value())
              : c10::cuda::current_device();
          auto s = c10::cuda::getDefaultCUDAStream(device_index);
          auto st = make_custom_class<torch::jit::CUDAStream>(s);
          push(stack, IValue(st));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::_current_device() -> int",
        [](Stack& stack) {
          auto v = c10::cuda::current_device();
          push(stack, static_cast<int>(v));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::_set_device(int64_t val) -> ()",
        [](Stack& stack) {
          int64_t idx = -1;
          pop(stack, idx);
          c10::cuda::set_device(static_cast<c10::DeviceIndex>(idx));
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::device_index(Device device) -> int",
        [](Stack& stack) {
          auto device = pop(stack);
          auto idx = device.toDevice().index();
          push(stack, idx);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::device_count() -> int",
        [](Stack& stack) { push(stack, at::cuda::device_count()); },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::set_stream(__torch__.torch.classes.cuda.Stream stream) -> ()",
        [](Stack& stack) {
          auto v = pop(stack);
          auto s = v.toCustomClass<torch::jit::CUDAStream>();
          auto stream_device_idx = static_cast<int64_t>(s->device_index());
          auto cur_device_idx =
              // NOLINTNEXTLINE(bugprone-signed-char-misuse)
              static_cast<int64_t>(c10::cuda::current_device());
          // If the stream is not on the current device, change the
          // device to the device of the stream.
          if (cur_device_idx != stream_device_idx) {
            c10::cuda::set_device(
                static_cast<c10::DeviceIndex>(stream_device_idx));
          }
          // To set the current CUDA stream using
          // c10::cuda::setCurrentCUDAStream, the jit::CUDAStream object needs
          // to be converted to c10::cuda::CUDAStream. Since the latter cannot
          // be returned from a class registered via TorchBind, this can only be
          // achieved by packing the c10::cuda::CUDAStream instance contained
          // inside the jit::CUDAStream object to a uint64_t representation, and
          // unpacking it inside this operator. The unpacked stream is then used
          // to set the current CUDA stream.
          auto packed = s->pack();
          auto unpacked = c10::cuda::CUDAStream::unpack(packed);
          c10::cuda::setCurrentCUDAStream(unpacked);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::synchronize() -> ()",
        [](Stack& stack) { c10::cuda::device_synchronize(); },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::synchronize.device(Device? device) -> ()",
        [](Stack& stack) {
          auto device = pop(stack).toOptional<c10::Device>();
          c10::DeviceIndex device_index = device.has_value()
              ? device->index()
              : c10::cuda::current_device();
          _device_synchronize(device_index);
        },
        aliasAnalysisFromSchema()),
    Operator(
        "cuda::synchronize.int(int? val) -> ()",
        [](Stack& stack) {
          auto idx = pop(stack).toOptional<int64_t>();
          c10::DeviceIndex device_index = idx.has_value()
              ? static_cast<c10::DeviceIndex>(idx.value())
              : c10::cuda::current_device();
          _device_synchronize(device_index);
        },
        aliasAnalysisFromSchema()),
});
} // namespace
} // namespace jit
} // namespace torch
