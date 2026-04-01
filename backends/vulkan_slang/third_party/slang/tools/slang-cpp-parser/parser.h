#pragma once

#include "compiler-core/slang-lexer.h"
#include "diagnostics.h"
#include "identifier-lookup.h"
#include "node-tree.h"
#include "node.h"

namespace CppParse
{
using namespace Slang;

class Parser
{
public:
    typedef uint32_t NodeTypeBitType;

    SlangResult expect(TokenType type, Token* outToken = nullptr);

    bool advanceIfMarker(Token* outToken = nullptr);
    bool advanceIfToken(TokenType type, Token* outToken = nullptr);
    bool advanceIfStyle(IdentifierStyle style, Token* outToken = nullptr);

    SlangResult pushAnonymousNamespace();
    SlangResult pushScope(ScopeNode* node);
    SlangResult consumeToClosingBrace(const Token* openBraceToken = nullptr);
    SlangResult popScope();

    /// Parse the contents of the source file
    SlangResult parse(SourceOrigin* sourceOrigin, const Options* options);

    void setKindEnabled(Node::Kind kind, bool isEnabled = true);
    bool isTypeEnabled(Node::Kind kind)
    {
        return (m_nodeTypeEnabled & (NodeTypeBitType(1) << int(kind))) != 0;
    }

    void setKindsEnabled(const Node::Kind* kinds, Index kindsCount, bool isEnabled = true);

    Parser(NodeTree* nodeTree, DiagnosticSink* sink);

protected:
    static Node::Kind _toNodeKind(IdentifierStyle style);

    bool _isMarker(const UnownedStringSlice& name);

    SlangResult _maybeConsumeScope();

    SlangResult _parsePreDeclare();
    SlangResult _parseTypeSet();

    SlangResult _maybeParseNode(Node::Kind kind);
    SlangResult _maybeParseContained(Node** outNode);

    SlangResult _parseTypeDef();
    SlangResult _parseEnum();
    SlangResult _parseMarker();
    SlangResult _parseSpecialMacro();

    SlangResult _maybeParseType(List<Token>& outToks, Token& outName);
    SlangResult _maybeParseType(UnownedStringSlice& outType, Token& outName);
    SlangResult _maybeParseType(Index& ioTemplateDepth, TokenReader::ParsingCursor& outCursor);

    SlangResult _parseExpression(List<Token>& outExprTokens);

    SlangResult _maybeParseTemplateArgs(Index& ioTemplateDepth);
    SlangResult _maybeParseTemplateArg(Index& ioTemplateDepth);

    /// Parse balanced - if a sink is set will report to that sink
    SlangResult _parseBalanced(DiagnosticSink* sink);

    bool _isCtor();

    /// Concatenate all tokens from start to the current position
    UnownedStringSlice _concatTokens(TokenReader::ParsingCursor start);
    UnownedStringSlice _concatTokens(const Token* toks, Index toksCount);

    UnownedStringSlice _concatType(
        TokenReader::ParsingCursor start,
        TokenReader::ParsingCursor nameCursor);

    void _getTypeTokens(
        TokenReader::ParsingCursor start,
        TokenReader::ParsingCursor nameCursor,
        List<Token>& outToks);

    /// Consume what looks like a template definition
    SlangResult _consumeTemplate();
    SlangResult _maybeConsume(IdentifierStyle style);

    SlangResult _consumeToSync();
    /// Consumes balanced parens. Will return an error if not matched. Assumes starts on opening (
    SlangResult _consumeBalancedParens();

    NodeTypeBitType m_nodeTypeEnabled;

    TokenList m_tokenList;
    TokenReader m_reader;

    List<ScopeNode*> m_scopeStack;

    ScopeNode* m_currentScope;    ///< The current scope being processed
    SourceOrigin* m_sourceOrigin; ///< The source origin that all tokens are in

    DiagnosticSink* m_sink; ///< Diagnostic sink

    NodeTree* m_nodeTree; ///< Shared state between parses. Nodes will be added to this

    const Options* m_options;
};

} // namespace CppParse
