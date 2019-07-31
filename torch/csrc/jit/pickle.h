#include <torch/csrc/WindowsTorchApiMacro.h>
#include <ATen/core/ivalue.h>
#include <torch/csrc/jit/pickler.h>


namespace torch {
namespace jit {

/// Pickled values can be loaded in Python and C++:
///
/// \rst
/// .. code-block:: cpp
///
///   torch::IValue float_value(2.3);
///   std::string data = torch::jit::pickle({float_value});
///   std::ofstream out("data.pkl");
///   out << data;
///   out.flush();
///
///
///   std::ifstream in("data.pkl", std::ios::binary);
///   std::vector<torch::IValue> ivalues = torch::jit::unpickle(in);
///   std::cout << ivalues.at(0) << "\n";
///
/// .. code-block:: python
///
///   values = torch.load('data.pkl')
///   print(values)
///
/// \endrst
TORCH_API std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);

TORCH_API void pickle_stream(
    std::function<void(const char*, size_t)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);

TORCH_API std::vector<IValue> unpickle(
    std::function<void(const char*, size_t)> reader,
    std::function<bool()> bounds_chcker,
    std::vector<at::Tensor>* tensor_table = nullptr,
    ClassResolver class_resolver = nullptr);

TORCH_API std::vector<IValue> unpickle(
    const char* data,
    size_t size,
    std::vector<at::Tensor>* tensor_table = nullptr,
    ClassResolver class_resolver = nullptr);



// TORCH_API std::vector<IValue> unpickle(
//     std::istream& in,
//     std::vector<at::Tensor>* tensor_table = nullptr,
//     ClassResolver class_resolver = nullptr);

} // namespace jit
} // namespace torch
