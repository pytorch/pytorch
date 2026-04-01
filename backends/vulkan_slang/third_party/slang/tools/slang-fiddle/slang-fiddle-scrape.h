// slang-fiddle-scrape.h
#pragma once

#include "compiler-core/slang-lexer.h"
#include "slang-fiddle-diagnostics.h"

namespace fiddle
{
using namespace Slang;

class Val : public RefObject
{
public:
};

class Node : public Val
{
public:
};

// Grouping Tokens and Trivia

class TokenWithTriviaNode : public RefObject
{
public:
    TokenType getType() const { return token.type; }

    List<Token> leadingTrivia;
    Token token;
    List<Token> trailingTrivia;
    Count skipCount = 0;
};

struct TokenWithTrivia
{
public:
    TokenWithTrivia() {}

    TokenWithTrivia(RefPtr<TokenWithTriviaNode> node)
        : node(node)
    {
    }

    SourceLoc const& getLoc() const { return node->token.loc; }

    Token const& getToken() const { return node->token; }

    TokenType getType() const { return node ? node->getType() : TokenType::Unknown; }

    UnownedStringSlice getContent() const
    {
        return node ? node->token.getContent() : UnownedStringSlice();
    }

    Count getSkipCount() const { return node ? node->skipCount : 0; }

    void setType(TokenType type) const { node->token.type = type; }

    List<Token> const& getLeadingTrivia() const { return node->leadingTrivia; }
    List<Token> const& getTrailingTrivia() const { return node->trailingTrivia; }

    operator TokenWithTriviaNode*() { return node; }

private:
    RefPtr<TokenWithTriviaNode> node;
};


// Syntax

class Declarator : public Node
{
};

class NameDeclarator : public Declarator
{
public:
    NameDeclarator(TokenWithTrivia nameToken)
        : nameToken(nameToken)
    {
    }

    TokenWithTrivia nameToken;
};

class PtrDeclarator : public Declarator
{
public:
    PtrDeclarator(RefPtr<Declarator> base)
        : base(base)
    {
    }

    RefPtr<Declarator> base;
};

class Expr : public Node
{
public:
};

class ModifierNode : public Node
{
};

class AbstractModifier : public ModifierNode
{
};
class HiddenModifier : public ModifierNode
{
};

enum class Mode
{
    Fiddle,
    Cpp,
};

class Decl : public Node
{
public:
    template<typename T>
    T* findModifier()
    {
        for (auto m : modifiers)
        {
            if (auto found = as<T>(m))
                return found;
        }
        return nullptr;
    }

    List<RefPtr<ModifierNode>> modifiers;
    TokenWithTrivia nameToken;
    Mode mode = Mode::Cpp;
};

class ContainerDecl : public Decl
{
public:
    List<RefPtr<Decl>> members;
    Dictionary<String, RefPtr<Decl>> mapNameToMember;
};

class LogicalContainerDecl : public ContainerDecl
{
};

class LogicalNamespaceBase : public LogicalContainerDecl
{
};

class LogicalModule : public LogicalNamespaceBase
{
public:
};

class LogicalNamespace : public LogicalNamespaceBase
{
};

class PhysicalContainerDecl : public ContainerDecl
{
public:
    LogicalContainerDecl* logicalVersion = nullptr;
};

class SourceUnit : public PhysicalContainerDecl
{
public:
};

class PhysicalNamespaceDecl : public PhysicalContainerDecl
{
public:
};

class AggTypeDecl : public ContainerDecl
{
public:
    RefPtr<Expr> directBaseType;

    List<AggTypeDecl*> directSubTypeDecls;

    bool isSubTypeOf(char const* name);
};

class ClassDecl : public AggTypeDecl
{
};
class StructDecl : public AggTypeDecl
{
};

class VarDecl : public Decl
{
public:
    RefPtr<Expr> type;
    RefPtr<Expr> initExpr;
};

class FiddleMacroInvocation : public Decl
{
public:
    TokenWithTrivia fiddleToken; // the actual `FIDDLE` identifier

    RefPtr<Node> node; // the node whose generated content should get emitted...
};

class UncheckedExpr : public Expr
{
};

class CheckedExpr : public Expr
{
};

class NameExpr : public UncheckedExpr
{
public:
    NameExpr(TokenWithTrivia nameToken)
        : nameToken(nameToken)
    {
    }

    TokenWithTrivia nameToken;
};

class LiteralExpr : public UncheckedExpr
{
public:
    LiteralExpr(TokenWithTrivia token)
        : token(token)
    {
    }

    TokenWithTrivia token;
};

class MemberExpr : public UncheckedExpr
{
public:
    MemberExpr(RefPtr<Expr> base, TokenWithTrivia memberNameToken)
        : base(base), memberNameToken(memberNameToken)
    {
    }

    RefPtr<Expr> base;
    TokenWithTrivia memberNameToken;
};

class StaticMemberRef : public UncheckedExpr
{
public:
    StaticMemberRef(RefPtr<Expr> base, TokenWithTrivia memberNameToken)
        : base(base), memberNameToken(memberNameToken)
    {
    }

    RefPtr<Expr> base;
    TokenWithTrivia memberNameToken;
};

typedef Expr Arg;

class PtrType : public UncheckedExpr
{
public:
    PtrType(RefPtr<Expr> base)
        : base(base)
    {
    }

    RefPtr<Expr> base;
};

class SpecializeExpr : public UncheckedExpr
{
public:
    SpecializeExpr() {}

    RefPtr<Expr> base;
    List<RefPtr<Expr>> args;
};

class CallExpr : public UncheckedExpr
{
public:
    CallExpr(RefPtr<Expr> base, List<RefPtr<Arg>> args)
        : base(base), args(args)
    {
    }

    RefPtr<Expr> base;
    List<RefPtr<Arg>> args;
};

class DirectDeclRef : public CheckedExpr
{
public:
    DirectDeclRef(Decl* decl)
        : decl(decl)
    {
    }

    Decl* decl = nullptr;
};

//

Decl* findDecl_(ContainerDecl* outerDecl, UnownedStringSlice const& name);

template<typename T>
T* findDecl(ContainerDecl* outerDecl, UnownedStringSlice const& name)
{
    auto decl = findDecl_(outerDecl, name);
    if (!decl)
        return nullptr;

    auto asType = as<T>(decl);
    if (!asType)
    {
        // TODO: might need this case to be an error...
        return nullptr;
    }

    return asType;
}


RefPtr<SourceUnit> parseSourceUnit(
    SourceView* inputSourceView,
    LogicalModule* logicalModule,
    RootNamePool* rootNamePool,
    DiagnosticSink* sink,
    SourceManager* sourceManager,
    String outputFileName);

bool hasAnyFiddleInvocations(SourceUnit* sourceUnit);

void checkModule(LogicalModule* module, DiagnosticSink* sink);


void registerScrapedStuffWithScript(LogicalModule* logicalModule);

void emitSourceUnitMacros(
    SourceUnit* sourceUnit,
    StringBuilder& builder,
    DiagnosticSink* sink,
    SourceManager* sourceManager,
    LogicalModule* logicalModule);
} // namespace fiddle
