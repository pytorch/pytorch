#include "./schema.h"
#include "./operator_sets.h"

namespace {
using namespace ONNX_NAMESPACE;
class PyTorchSchemasRegisterer {
 public:
  PyTorchSchemasRegisterer() {
    OpSchemaRegistry::DomainToVersionRange::Instance().AddDomainToVersion(
        AI_ONNX_PYTORCH_DOMAIN,
        AI_ONNX_PYTORCH_DOMAIN_MIN_OPSET,
        AI_ONNX_PYTORCH_DOMAIN_MAX_OPSET);
    RegisterPyTorchOperatorSetSchema();
  }
};
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
static PyTorchSchemasRegisterer registerer{};
} // namespace
