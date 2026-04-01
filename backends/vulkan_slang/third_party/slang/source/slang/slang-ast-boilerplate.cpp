// slang-ast-boilerplate.cpp

#include "slang-ast-all.h"
#include "slang-ast-builder.h"
#include "slang-ast-forward-declarations.h"

namespace Slang
{
template<typename T>
struct Helper
{
    static void* create(ASTBuilder* builder) { return builder->createImpl<T>(); }

    static void destruct(void* obj) { ((T*)obj)->~T(); }
};

#if 0 // FIDDLE TEMPLATE:
%for _,T in ipairs(Slang.NodeBase.subclasses) do
const SyntaxClassInfo $T::kSyntaxClassInfo = {
    "$T",
    ASTNodeType::$T,
    $(#T.subclasses),
%  if T.isAbstract then
    nullptr, // create
    nullptr, // destruct
%  else
    &Helper<$T>::create,
    &Helper<$T>::destruct,
%  end
};
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 0
#include "slang-ast-boilerplate.cpp.fiddle"
#endif // FIDDLE END

static SyntaxClassInfo const* kAllSyntaxClasses[] = {
#if 0 // FIDDLE TEMPLATE:
%for _,T in ipairs(Slang.NodeBase.subclasses) do
    &$T::kSyntaxClassInfo,
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 1
#include "slang-ast-boilerplate.cpp.fiddle"
#endif // FIDDLE END
};

SyntaxClassBase::SyntaxClassBase(ASTNodeType tag)
{
    assert(int(tag) >= 0 && int(tag) < SLANG_COUNT_OF(kAllSyntaxClasses));
    _info = kAllSyntaxClasses[int(tag)];
}

} // namespace Slang
