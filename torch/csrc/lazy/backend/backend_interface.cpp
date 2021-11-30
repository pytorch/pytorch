#include <torch/csrc/lazy/backend/backend_interface.h>

namespace torch {
namespace lazy {

std::atomic<const BackendImplInterface*> backend_impl_registry;

}  // namespace lazy
}  // namespace torch
