#include <ATen/core/jit_type.h>

namespace torch { namespace jit {

#define C10_USING(T) using ::c10::T;
  C10_FORALL_TYPES(C10_USING)
#undef C10_USING

#define C10_USING(T) using ::c10::T##Ptr;
  C10_FORALL_TYPES(C10_USING)
#undef C10_USING

using ::c10::Type;
using ::c10::TypePtr;
using ::c10::TypeEnv;
using ::c10::TypeMatchError;

using ::c10::getTypePtr;
using ::c10::TypeKind;

}} // namespace torch::jit
