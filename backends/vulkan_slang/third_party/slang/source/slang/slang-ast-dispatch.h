// slang-ast-dispatch.h
#pragma once

#include "slang-ast-forward-declarations.h"
#include "slang-syntax.h"

namespace Slang
{

template<typename Base, typename Result>
struct ASTNodeDispatcher
{
};

#if 0 // FIDDLE TEMPLATE:
%function generateDispatcher(BASE)
template<typename R>
struct ASTNodeDispatcher<$BASE, R>
{
    template<typename F>
    static R dispatch($BASE const* obj, F const& f)
    {
        switch (obj->getClass().getTag())
        {
        default:
            SLANG_UNEXPECTED("unhandled subclass in ASTNodeDispatcher::dispatch");

%  for _,T in ipairs(BASE.subclasses) do
%    if not T.isAbstract then
        case ASTNodeType::$T:
            return f(static_cast<$T*>(const_cast<$BASE*>(obj)));
%    end
%  end
        }
    }
};
%end
%generateDispatcher(Slang.TypeConstraintDecl)
%generateDispatcher(Slang.ArithmeticExpressionType)
%generateDispatcher(Slang.DeclRefBase)
%generateDispatcher(Slang.Val)
%generateDispatcher(Slang.Type)
%generateDispatcher(Slang.SubtypeWitness)
%generateDispatcher(Slang.IntVal)
%generateDispatcher(Slang.Modifier)
%generateDispatcher(Slang.DeclBase)
%generateDispatcher(Slang.Decl)
%generateDispatcher(Slang.Expr)
%generateDispatcher(Slang.Stmt)
%generateDispatcher(Slang.NodeBase)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 0
#include "slang-ast-dispatch.h.fiddle"
#endif // FIDDLE END

} // namespace Slang
