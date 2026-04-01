#ifndef SLANG_PARSER_H
#define SLANG_PARSER_H

#include "../compiler-core/slang-lexer.h"
#include "slang-compiler.h"
#include "slang-syntax.h"

namespace Slang
{
// Parse a source file into an existing translation unit
void parseSourceFile(
    ASTBuilder* astBuilder,
    TranslationUnitRequest* translationUnit,
    SourceLanguage sourceLanguage,
    TokenSpan const& tokens,
    DiagnosticSink* sink,
    Scope* outerScope,
    ContainerDecl* parentDecl);

Expr* parseTermFromSourceFile(
    ASTBuilder* astBuilder,
    TokenSpan const& tokens,
    DiagnosticSink* sink,
    Scope* outerScope,
    NamePool* namePool,
    SourceLanguage sourceLanguage);

struct SemanticsVisitor;

Stmt* parseUnparsedStmt(
    ASTBuilder* astBuilder,
    SemanticsVisitor* semantics,
    TranslationUnitRequest* translationUnit,
    SourceLanguage sourceLanguage,
    bool isInVariadicGenerics,
    TokenSpan const& tokens,
    DiagnosticSink* sink,
    Scope* currentScope,
    Scope* outerScope);

ModuleDecl* populateBaseLanguageModule(ASTBuilder* astBuilder, Scope* scope);

/// Information used to set up SyntaxDecl. Such decls
/// when correctly setup define a callback. For some of the callbacks it's necessary
/// for the `parseUserData` to be set the the associated classInfo
struct SyntaxParseInfo
{
    const char* keywordName;         ///< The keyword associated with this parse
    SyntaxParseCallback callback;    ///< The callback to apply to the parse
    SyntaxClass<NodeBase> classInfo; ///<
};

/// Get all of the predefined SyntaxParseInfos
ConstArrayView<SyntaxParseInfo> getSyntaxParseInfos();

/// Assumes the userInfo is the ReflectClassInfo
NodeBase* parseSimpleSyntax(Parser* parser, void* userData);

} // namespace Slang

#endif
