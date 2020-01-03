#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/jit/fuser/common/ir.h>

#include <torch/csrc/jit/fuser/common/type.h>
#include <c10/util/Exception.h>

namespace torch{
namespace jit{
namespace fuser{
//Return new value of type that v1 and v2 promotes to
TORCH_API Val* promote_new(Val *v1, Val* v2);

TORCH_API Val* add(Val* v1, Val* v2);

}}}
