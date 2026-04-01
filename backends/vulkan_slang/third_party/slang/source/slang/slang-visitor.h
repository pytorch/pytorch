// slang-visitor.h
#ifndef SLANG_VISITOR_H_INCLUDED
#define SLANG_VISITOR_H_INCLUDED

// This file defines the basic "Visitor" pattern for doing dispatch
// over the various categories of syntax node.

#include "slang-ast-dispatch.h"
#include "slang-ast-forward-declarations.h"
#include "slang-syntax.h"

namespace Slang
{

// Dispatch

#if 0 // FIDDLE TEMPLATE:
%function SLANG_VISITOR_DISPATCH_RESULT_IMPL(baseType)
%  for _,T in ipairs(baseType.subclasses) do
%    if not T.isAbstract then
    Result _dispatchImpl($T* obj)
    {
        return ((Derived*)this)->visit$T(obj);
    }
%    end
%  end
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 0
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END

// Visitor with and without result

#if 0 // FIDDLE TEMPLATE:
%function SLANG_VISITOR_VISIT_RESULT_IMPL(baseType)
%  for _,T in ipairs(baseType.subclasses) do
    Result visit$T($T* obj)
    {
        return ((Derived*)this)->visit$(T.directSuperClass)(obj);
    }
%  end
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 1
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END

// Args

#if 0 // FIDDLE TEMPLATE:
%function SLANG_VISITOR_DISPATCH_ARG_IMPL(baseType)
%  for _, T in ipairs(baseType.subclasses) do
%    if not T.isAbstract then
virtual void _dispatchImpl($T* obj, Arg const& arg)
{
    ((Derived*)this)->visit$T(obj, arg);
}
%    end
%  end
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 2
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END

#if 0 // FIDDLE TEMPLATE:
%function SLANG_VISITOR_VISIT_ARG_IMPL(baseType)
% for _, T in ipairs(baseType.subclasses) do
void visit$T($T* obj, Arg const& arg)
{
    ((Derived*)this)->visit$(T.directSuperClass)(obj, arg);
}
%  end
%end
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 3
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END

//
// type Visitors
//

// Suppress VS2017 Unreachable code warning
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4702)
#endif

template<typename Derived, typename Result = void>
struct TypeVisitor
{
    Result dispatch(Type* type)
    {
        return ASTNodeDispatcher<Type, Result>::dispatch(
            type,
            [&](auto obj) { return _dispatchImpl(obj); });
    }

    Result dispatchType(Type* type)
    {
        return ASTNodeDispatcher<Type, Result>::dispatch(
            type,
            [&](auto obj) { return _dispatchImpl(obj); });
    }

#if 0 // FIDDLE TEMPLATE:
        % SLANG_VISITOR_DISPATCH_RESULT_IMPL(Slang.Type)
        % SLANG_VISITOR_VISIT_RESULT_IMPL(Slang.Type)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 4
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END
};

template<typename Derived, typename Arg>
struct TypeVisitorWithArg
{
    void dispatch(Type* type, Arg const& arg)
    {
        ASTNodeDispatcher<Type, void>::dispatch(type, [&](auto obj) { _dispatchImpl(obj, arg); });
    }

#if 0 // FIDDLE TEMPLATE:
    % SLANG_VISITOR_DISPATCH_ARG_IMPL(Slang.Type)
    % SLANG_VISITOR_VISIT_ARG_IMPL(Slang.Type)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 5
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END
};

//
// Expression Visitors
//

template<typename Derived, typename Result = void>
struct ExprVisitor
{
    Result dispatch(Expr* expr)
    {
        return ASTNodeDispatcher<Expr, Result>::dispatch(
            expr,
            [&](auto obj) { return _dispatchImpl(obj); });
    }

#if 0 // FIDDLE TEMPLATE:
    % SLANG_VISITOR_DISPATCH_RESULT_IMPL(Slang.Expr)
    % SLANG_VISITOR_VISIT_RESULT_IMPL(Slang.Expr)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 6
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END
};

template<typename Derived, typename Arg>
struct ExprVisitorWithArg
{
    void dispatch(Expr* expr, Arg const& arg)
    {
        ASTNodeDispatcher<Expr, void>::dispatch(expr, [&](auto obj) { _dispatchImpl(obj, arg); });
    }

#if 0 // FIDDLE TEMPLATE:
    % SLANG_VISITOR_DISPATCH_ARG_IMPL(Slang.Expr)
    % SLANG_VISITOR_VISIT_ARG_IMPL(Slang.Expr)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 7
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END
};

//
// Statement Visitors
//

template<typename Derived, typename Result = void>
struct StmtVisitor
{
    Result dispatch(Stmt* stmt)
    {
        return ASTNodeDispatcher<Stmt, Result>::dispatch(
            stmt,
            [&](auto obj) { return _dispatchImpl(obj); });
    }

#if 0 // FIDDLE TEMPLATE:
    % SLANG_VISITOR_DISPATCH_RESULT_IMPL(Slang.Stmt)
    % SLANG_VISITOR_VISIT_RESULT_IMPL(Slang.Stmt)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 8
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END
};

//
// Declaration Visitors
//

template<typename Derived, typename Result = void>
struct DeclVisitor
{
    Result dispatch(DeclBase* decl)
    {
        return ASTNodeDispatcher<DeclBase, Result>::dispatch(
            decl,
            [&](auto obj) { return _dispatchImpl(obj); });
    }

#if 0 // FIDDLE TEMPLATE:
    % SLANG_VISITOR_DISPATCH_RESULT_IMPL(Slang.DeclBase)
    % SLANG_VISITOR_VISIT_RESULT_IMPL(Slang.DeclBase)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 9
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END
};

template<typename Derived, typename Arg>
struct DeclVisitorWithArg
{
    void dispatch(DeclBase* decl, Arg const& arg)
    {
        ASTNodeDispatcher<Expr, void>::dispatch(decl, [&](auto obj) { _dispatchImpl(obj, arg); });
    }

#if 0 // FIDDLE TEMPLATE:
    % SLANG_VISITOR_DISPATCH_ARG_IMPL(Slang.DeclBase)
    % SLANG_VISITOR_VISIT_ARG_IMPL(Slang.DeclBase)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 10
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END
};


//
// Modifier Visitors
//

template<typename Derived, typename Result = void>
struct ModifierVisitor
{
    Result dispatch(Modifier* modifier)
    {
        return ASTNodeDispatcher<Modifier, Result>::dispatch(
            modifier,
            [&](auto obj) { return _dispatchImpl(obj); });
    }

#if 0 // FIDDLE TEMPLATE:
    % SLANG_VISITOR_DISPATCH_RESULT_IMPL(Slang.Modifier)
    % SLANG_VISITOR_VISIT_RESULT_IMPL(Slang.Modifier)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 11
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END
};

//
// Val Visitors
//

template<typename Derived, typename Result = void, typename TypeResult = void>
struct ValVisitor : TypeVisitor<Derived, TypeResult>
{
    Result dispatch(Val* val)
    {
        return ASTNodeDispatcher<Val, Result>::dispatch(
            val,
            [&](auto obj) { return _dispatchImpl(obj); });
    }

#if 0 // FIDDLE TEMPLATE:
    % SLANG_VISITOR_DISPATCH_RESULT_IMPL(Slang.Val)
    % SLANG_VISITOR_VISIT_RESULT_IMPL(Slang.Val)
#else // FIDDLE OUTPUT:
#define FIDDLE_GENERATED_OUTPUT_ID 12
#include "slang-visitor.h.fiddle"
#endif // FIDDLE END
};

// Re-activate VS2017 warning settings
#ifdef _MSC_VER
#pragma warning(pop)
#endif
} // namespace Slang

#endif
