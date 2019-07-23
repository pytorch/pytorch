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
///   torch::IValue tensor_value(at::ones(2, 2));
///
///   std::string data = torch::jit::Pickle({float_value, tensor_value});
///   std::ofstream out("data.pkl");
///   out << data;
///
///
///   std::ifstream in("data.pkl", ios::binary);
///   std::string data = std::string(
///       std::istreambuf_iterator<char>(in),
///       std::istreambuf_iterator<char>());
///   std::vector<torch::IValue> ivalues =
///       torch::jit::Unpickle(std::stringstream(data));
///   std::cout << ivalues.at(0) << "\n";
///   std::cout << ivalues.at(1) << "\n";
///
/// .. code-block:: python
///
///   values = pickle.load(open("data.pkl", "rb"))
///   print(values[0])
///   print(values[1])
/// \endrst
TORCH_API std::string Pickle(
    std::vector<IValue> ivalues,
    std::vector<at::Tensor>* tensor_table = nullptr);

TORCH_API std::vector<IValue> Unpickle(
    const char* data,
    size_t size,
    std::vector<at::Tensor>* tensor_table = nullptr,
    ClassResolver class_resolver = nullptr);

TORCH_API std::vector<IValue> Unpickle(
    const void* data,
    size_t size,
    std::vector<at::Tensor>* tensor_table = nullptr,
    ClassResolver class_resolver = nullptr);

TORCH_API std::vector<IValue> Unpickle(
    std::istream& in,
    std::vector<at::Tensor>* tensor_table = nullptr,
    ClassResolver class_resolver = nullptr);

} // namespace jit
} // namespace torch
