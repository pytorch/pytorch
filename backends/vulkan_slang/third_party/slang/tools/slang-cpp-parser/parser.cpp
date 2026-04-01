#include "parser.h"

#include "compiler-core/slang-name-convention-util.h"
#include "core/slang-io.h"
#include "core/slang-string-util.h"
#include "identifier-lookup.h"
#include "options.h"

namespace CppParse
{
using namespace Slang;

// If fails then we need more bits to identify types
SLANG_COMPILE_TIME_ASSERT(int(Node::Kind::CountOf) <= 8 * sizeof(uint32_t));

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Parser !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Parser::Parser(NodeTree* nodeTree, DiagnosticSink* sink)
    : m_sink(sink), m_nodeTree(nodeTree), m_nodeTypeEnabled(0)
{
    // Enable types by default
    const Node::Kind defaultEnabled[] = {
        Node::Kind::ClassType,
        Node::Kind::StructType,
        Node::Kind::Namespace,
        Node::Kind::AnonymousNamespace,
        Node::Kind::Field,

        // These are disabled by default because AST uses macro magic to build up the types
        // Node::Type::TypeDef,
        // Node::Type::Enum,
        // Node::Type::EnumClass,

        Node::Kind::Callable,
    };
    setKindsEnabled(defaultEnabled, SLANG_COUNT_OF(defaultEnabled));
}

void Parser::setKindEnabled(Node::Kind kind, bool isEnabled)
{
    if (isEnabled)
    {
        m_nodeTypeEnabled |= (NodeTypeBitType(1) << int(kind));
    }
    else
    {
        m_nodeTypeEnabled &= ~(NodeTypeBitType(1) << int(kind));
    }
}

void Parser::setKindsEnabled(const Node::Kind* kinds, Index kindsCount, bool isEnabled)
{
    for (Index i = 0; i < kindsCount; ++i)
    {
        setKindEnabled(kinds[i], isEnabled);
    }
}

bool Parser::_isMarker(const UnownedStringSlice& name)
{
    return name.startsWith(m_options->m_markPrefix.getUnownedSlice()) &&
           name.endsWith(m_options->m_markSuffix.getUnownedSlice());
}

SlangResult Parser::expect(TokenType type, Token* outToken)
{
    if (m_reader.peekTokenType() != type)
    {
        m_sink->diagnose(m_reader.peekToken(), CPPDiagnostics::expectingToken, type);
        return SLANG_FAIL;
    }

    if (outToken)
    {
        *outToken = m_reader.advanceToken();
    }
    else
    {
        m_reader.advanceToken();
    }
    return SLANG_OK;
}

bool Parser::advanceIfToken(TokenType type, Token* outToken)
{
    if (m_reader.peekTokenType() == type)
    {
        Token token = m_reader.advanceToken();
        if (outToken)
        {
            *outToken = token;
        }
        return true;
    }
    return false;
}

bool Parser::advanceIfMarker(Token* outToken)
{
    const Token peekToken = m_reader.peekToken();
    if (peekToken.type == TokenType::Identifier && _isMarker(peekToken.getContent()))
    {
        m_reader.advanceToken();
        if (outToken)
        {
            *outToken = peekToken;
        }
        return true;
    }
    return false;
}

bool Parser::advanceIfStyle(IdentifierStyle style, Token* outToken)
{
    if (m_reader.peekTokenType() == TokenType::Identifier)
    {
        IdentifierStyle readStyle =
            m_nodeTree->m_identifierLookup->get(m_reader.peekToken().getContent());
        if (readStyle == style)
        {
            Token token = m_reader.advanceToken();
            if (outToken)
            {
                *outToken = token;
            }
            return true;
        }
    }
    return false;
}


SlangResult Parser::pushAnonymousNamespace()
{
    m_currentScope = m_currentScope->getAnonymousNamespace();

    if (m_sourceOrigin)
    {
        m_sourceOrigin->addNode(m_currentScope);
    }

    // Add the to the scope stack so can pop.
    m_scopeStack.add(m_currentScope);

    return SLANG_OK;
}

SlangResult Parser::pushScope(ScopeNode* scopeNode)
{
    // We can only have one 'special' scope.
    SLANG_ASSERT(scopeNode || m_scopeStack.getLast());

    // We keep to track.
    m_scopeStack.add(scopeNode);

    // If we pass nullptr, we don't update the current scope.
    if (scopeNode == nullptr)
    {
        return SLANG_OK;
    }

    if (m_sourceOrigin)
    {
        m_sourceOrigin->addNode(scopeNode);
    }

    if (scopeNode->m_name.hasContent())
    {
        // For anonymous namespace, we should look if we already have one and just reopen that.
        // Doing so will mean will find anonymous namespace clashes

        if (Node* foundNode = m_currentScope->findChild(scopeNode->m_name.getContent()))
        {
            if (scopeNode->isClassLike())
            {
                m_sink->diagnose(
                    m_reader.peekToken(),
                    CPPDiagnostics::typeAlreadyDeclared,
                    scopeNode->m_name.getContent());
                m_sink->diagnose(
                    foundNode->m_name,
                    CPPDiagnostics::seeDeclarationOf,
                    scopeNode->m_name.getContent());
                return SLANG_FAIL;
            }

            if (foundNode->m_kind == Node::Kind::Namespace)
            {
                if (foundNode->m_kind != scopeNode->m_kind)
                {
                    // Different types can't work
                    m_sink->diagnose(
                        m_reader.peekToken(),
                        CPPDiagnostics::typeAlreadyDeclared,
                        scopeNode->m_name.getContent());
                    return SLANG_FAIL;
                }

                ScopeNode* foundScopeNode = as<ScopeNode>(foundNode);
                SLANG_ASSERT(foundScopeNode);

                // Make sure the node is empty, as we are *not* going to add it, we are just going
                // to use the pre-existing namespace
                SLANG_ASSERT(scopeNode->m_children.getCount() == 0);

                // We can just use the pre-existing namespace
                m_currentScope = foundScopeNode;
                return SLANG_OK;
            }
        }
    }

    m_currentScope->addChild(scopeNode);
    m_currentScope = scopeNode;
    return SLANG_OK;
}

SlangResult Parser::popScope()
{
    if (m_scopeStack.getCount() <= 0)
    {
        m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::scopeNotClosed);
        return SLANG_FAIL;
    }

    ScopeNode* topScope = m_scopeStack.getLast();
    m_scopeStack.removeLast();

    // If the top is nullptr, we don't change the current scope
    if (topScope == nullptr)
    {
        return SLANG_OK;
    }

    m_currentScope = m_currentScope->m_parentScope;
    return SLANG_OK;
}

SlangResult Parser::_maybeConsumeScope()
{
    // Look for either ; or { to open scope
    while (true)
    {
        const TokenType type = m_reader.peekTokenType();
        if (type == TokenType::Semicolon)
        {
            m_reader.advanceToken();
            return SLANG_OK;
        }
        else if (type == TokenType::LBrace)
        {
            // m_reader.advanceToken();
            return consumeToClosingBrace();
        }
        else if (type == TokenType::EndOfFile)
        {
            return SLANG_OK;
        }

        m_reader.advanceToken();
    }
}

SlangResult Parser::consumeToClosingBrace(const Token* inOpenBraceToken)
{
    Token openToken;
    if (inOpenBraceToken)
    {
        openToken = *inOpenBraceToken;
    }
    else
    {
        openToken = m_reader.advanceToken();
    }
    SLANG_ASSERT(openToken.type == TokenType::LBrace);

    while (true)
    {
        switch (m_reader.peekTokenType())
        {
        case TokenType::EndOfFile:
            {
                m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::didntFindMatchingBrace);
                m_sink->diagnose(openToken, CPPDiagnostics::seeOpen);
                return SLANG_FAIL;
            }
        case TokenType::LBrace:
            {
                SLANG_RETURN_ON_FAIL(consumeToClosingBrace());
                break;
            }
        case TokenType::RBrace:
            {
                m_reader.advanceToken();
                return SLANG_OK;
            }
        default:
            {
                m_reader.advanceToken();
                break;
            }
        }
    }
}


SlangResult Parser::_parseEnum()
{
    // We are looking for
    // enum ([class name] | [name]) [: base] ( { | ; )

    Token enumToken;

    // consume enum
    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &enumToken));

    if (!m_currentScope->canContainTypes())
    {
        m_sink->diagnose(enumToken.loc, CPPDiagnostics::cannotDeclareTypeInScope);
        return SLANG_FAIL;
    }

    Node::Kind kind = Node::Kind::Enum;

    Token nameToken;
    if (advanceIfToken(TokenType::Identifier, &nameToken))
    {
        const IdentifierStyle style = m_nodeTree->m_identifierLookup->get(nameToken.getContent());

        if (style == IdentifierStyle::Class)
        {
            kind = Node::Kind::EnumClass;
            SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &nameToken));
        }
        else if (style == IdentifierStyle::None)
        {
            // It holds the name then
        }
        else
        {
            m_sink->diagnose(
                nameToken.loc,
                CPPDiagnostics::expectingIdentifier,
                nameToken.getContent());
            return SLANG_FAIL;
        }
    }

    RefPtr<EnumNode> node = new EnumNode(kind);
    node->m_name = nameToken;
    node->m_reflectionType = m_currentScope->getContainedReflectionType();

    if (advanceIfToken(TokenType::Colon))
    {
        // We may have tokens up to { or ;
        List<Token> backingTokens;

        while (true)
        {
            TokenType tokenType = m_reader.peekTokenType();
            if (tokenType == TokenType::Semicolon || tokenType == TokenType::LBrace ||
                tokenType == TokenType::EndOfFile)
            {
                break;
            }

            backingTokens.add(m_reader.advanceToken());
        }

        // TODO - Look up the backing type. It can only be an integral. We can assume it must be
        // defined before lookup for our uses here. If we can't find the type, we could assume it's
        // size is undefined

        if (backingTokens.getCount() > 0)
        {
            node->m_backingTokens.swapWith(backingTokens);
        }
    }

    pushScope(node);

    if (advanceIfToken(TokenType::Semicolon))
    {
        if (nameToken.type != TokenType::Invalid)
        {
            Node* node = m_currentScope->findChild(nameToken.getContent());
            if (node)
            {
                // Strictly speaking we should check the backing type etc, match, but for now ignore
                // and assume it's ok

                if (node->m_kind == kind)
                {
                    return SLANG_OK;
                }
                m_sink->diagnose(
                    nameToken.loc,
                    CPPDiagnostics::typeAlreadyDeclared,
                    nameToken.getContent());
                return SLANG_FAIL;
            }
            return popScope();
        }
    }

    SLANG_RETURN_ON_FAIL(expect(TokenType::LBrace));

    while (true)
    {
        TokenType tokenType = m_reader.peekTokenType();
        if (tokenType == TokenType::RBrace)
        {
            break;
        }

        RefPtr<EnumCaseNode> caseNode(new EnumCaseNode);

        // We could also check if the name is a valid identifier for name, for now just assume.
        SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &caseNode->m_name));

        if (node->findChild(caseNode->m_name.getContent()))
        {
            m_sink->diagnose(
                caseNode->m_name.loc,
                CPPDiagnostics::identifierAlreadyDefined,
                caseNode->m_name.getContent());
            return SLANG_FAIL;
        }

        caseNode->m_reflectionType = m_currentScope->getContainedReflectionType();

        // Add the value
        node->addChild(caseNode);

        if (advanceIfToken(TokenType::OpAssign))
        {
            List<Token> valueTokens;
            SLANG_RETURN_ON_FAIL(_parseExpression(valueTokens));

            if (valueTokens.getCount() > 0)
            {
                caseNode->m_valueTokens.swapWith(valueTokens);
            }
        }

        tokenType = m_reader.peekTokenType();
        if (tokenType == TokenType::Comma)
        {
            m_reader.advanceToken();
            continue;
        }

        break;
    }

    SLANG_RETURN_ON_FAIL(expect(TokenType::RBrace));
    SLANG_RETURN_ON_FAIL(expect(TokenType::Semicolon));

    return popScope();
}

SlangResult Parser::_consumeTemplate()
{
    // Skip the current 'template' token.
    m_reader.advanceToken();

    // Consume everything in <>
    SLANG_RETURN_ON_FAIL(expect(TokenType::OpLess));

    {
        Index arrowCount = 1;
        while (true)
        {
            auto tokenType = m_reader.peekTokenType();

            if (tokenType == TokenType::OpLess)
            {
                m_reader.advanceToken();
                arrowCount++;
            }
            else if (tokenType == TokenType::OpGreater)
            {
                m_reader.advanceToken();
                if (arrowCount == 1)
                {
                    break;
                }
                --arrowCount;
            }
            else if (tokenType == TokenType::OpRsh)
            {
                if (arrowCount < 2)
                {
                    m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::unexpectedTemplateClose);
                    return SLANG_FAIL;
                }
                m_reader.advanceToken();
                if (arrowCount == 2)
                {
                    break;
                }
                arrowCount -= 2;
            }
            else if (tokenType == TokenType::EndOfFile)
            {
                m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::unexpectedEndOfFile);
                return SLANG_FAIL;
            }
            else
            {
                m_reader.advanceToken();
            }
        }
    }

    // Search for { or ; to consume remaining
    while (true)
    {
        auto tokenType = m_reader.peekTokenType();

        switch (tokenType)
        {
        case TokenType::EndOfFile:
            {
                m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::unexpectedEndOfFile);
                return SLANG_FAIL;
            }
        case TokenType::Semicolon:
            {
                // Ends with semicolon if it's a template pre-declaration
                m_reader.advanceToken();
                return SLANG_OK;
            }
        case TokenType::LBrace:
            {
                // If ends with {, means could be body of a struct/class or a body of a
                // function/method. Consume it
                SLANG_RETURN_ON_FAIL(consumeToClosingBrace());
                // If we hit a ; just consume and ignore
                advanceIfToken(TokenType::Semicolon);
                return SLANG_OK;
            }
        default:
            {
                // Consume
                m_reader.advanceToken();
                break;
            }
        }
    }
}

SlangResult Parser::_maybeParseNode(Node::Kind kind)
{
    // We are looking for
    // struct/class identifier [: [public|private|protected] Identifier ] {
    // [public|private|proctected:]* marker ( identifier );

    if (kind == Node::Kind::Namespace)
    {
        // consume namespace
        SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier));

        Token name;
        if (advanceIfToken(TokenType::LBrace))
        {
            return pushAnonymousNamespace();
        }
        else if (advanceIfToken(TokenType::Identifier, &name))
        {
            if (advanceIfToken(TokenType::LBrace))
            {
                // Okay looks like we are opening a namespace
                RefPtr<ScopeNode> node(new ScopeNode(Node::Kind::Namespace));
                node->m_name = name;

                node->m_reflectionType = m_currentScope->getContainedReflectionType();
                // Push the node
                return pushScope(node);
            }
        }

        // Just ignore it then
        return SLANG_OK;
    }
    else if (Node::isKindEnumLike(kind))
    {
        return _parseEnum();
    }

    // Must be class | struct

    SLANG_ASSERT(kind == Node::Kind::ClassType || kind == Node::Kind::StructType);

    Token name;

    // consume class | struct
    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier));
    // Next is the class name
    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &name));


    if (m_reader.peekTokenType() == TokenType::Semicolon)
    {
        // pre declaration;
        return SLANG_OK;
    }

    RefPtr<ClassLikeNode> node(new ClassLikeNode(kind));
    node->m_name = name;

    // We default to the containing scope for reflection type.
    if (!m_options->m_requireMark)
    {
        node->m_reflectionType = m_currentScope->getContainedReflectionType();
    }
    else
    {
        // Defaults to not reflected
        SLANG_ASSERT(!node->isReflected());
    }

    if (advanceIfToken(TokenType::Colon))
    {
        // Could have public
        advanceIfStyle(IdentifierStyle::Access);

        if (!advanceIfToken(TokenType::Identifier, &node->m_super))
        {
            return SLANG_OK;
        }
    }

    // We only accept a single super class. Consume everything afterwards until we hit the { brace

    if (m_reader.peekTokenType() != TokenType::LBrace)
    {
        // Consume up until we see a brace else it's an error
        while (true)
        {
            const TokenType peekTokenType = m_reader.peekTokenType();
            if (peekTokenType == TokenType::EndOfFile)
            {
                // Expecting brace
                m_sink->diagnose(
                    m_reader.peekToken(),
                    CPPDiagnostics::expectingToken,
                    TokenType::LBrace);
                return SLANG_FAIL;
            }
            else if (peekTokenType == TokenType::LBrace)
            {
                break;
            }
            m_reader.advanceToken();
        }

        return pushScope(node);
    }

    const Token braceToken = m_reader.advanceToken();

    // Push the class scope
    return pushScope(node);
}

SlangResult Parser::_consumeToSync()
{
    while (true)
    {
        TokenType type = m_reader.peekTokenType();

        switch (type)
        {
        case TokenType::Semicolon:
            {
                m_reader.advanceToken();
                return SLANG_OK;
            }
        case TokenType::Pound:
        case TokenType::EndOfFile:
        case TokenType::LBrace:
        case TokenType::RBrace:
            {
                return SLANG_OK;
            }
        }

        m_reader.advanceToken();
    }
}

SlangResult Parser::_maybeParseTemplateArg(Index& ioTemplateDepth)
{
    switch (m_reader.peekTokenType())
    {
    case TokenType::Identifier:
        {
            TokenReader::ParsingCursor nameCursor;
            SLANG_RETURN_ON_FAIL(_maybeParseType(ioTemplateDepth, nameCursor));
            return SLANG_OK;
        }
    case TokenType::IntegerLiteral:
        {
            m_reader.advanceToken();
            return SLANG_OK;
        }
    default:
        break;
    }
    return SLANG_FAIL;
}

SlangResult Parser::_maybeParseTemplateArgs(Index& ioTemplateDepth)
{
    if (!advanceIfToken(TokenType::OpLess))
    {
        return SLANG_FAIL;
    }

    ioTemplateDepth++;

    while (true)
    {
        if (ioTemplateDepth == 0)
        {
            return SLANG_OK;
        }

        switch (m_reader.peekTokenType())
        {
        case TokenType::OpGreater:
            {
                if (ioTemplateDepth <= 0)
                {
                    m_sink->diagnose(m_reader.peekToken(), CPPDiagnostics::unexpectedTemplateClose);
                    return SLANG_FAIL;
                }
                ioTemplateDepth--;
                m_reader.advanceToken();
                return SLANG_OK;
            }
        case TokenType::OpRsh:
            {
                if (ioTemplateDepth <= 1)
                {
                    m_sink->diagnose(m_reader.peekToken(), CPPDiagnostics::unexpectedTemplateClose);
                    return SLANG_FAIL;
                }
                ioTemplateDepth -= 2;
                m_reader.advanceToken();
                return SLANG_OK;
            }
        default:
            {
                while (true)
                {
                    SLANG_RETURN_ON_FAIL(_maybeParseTemplateArg(ioTemplateDepth));

                    if (m_reader.peekTokenType() == TokenType::Comma)
                    {
                        m_reader.advanceToken();
                        // If there is a comma parse another arg
                        continue;
                    }
                    break;
                }
                break;
            }
        }
    }
}

SlangResult Parser::_maybeConsume(IdentifierStyle style)
{
    while (advanceIfStyle(style))
        ;
    return SLANG_OK;
}

// True if two of these token types of the same type placed immediately after one another
// produce a different token. Can be conservative, as if not strictly required
// it will just mean more spacing in the output
static bool _canRepeatTokenType(TokenType type)
{
    switch (type)
    {
    case TokenType::OpAdd:
    case TokenType::OpSub:
    case TokenType::OpAnd:
    case TokenType::OpOr:
    case TokenType::OpGreater:
    case TokenType::OpLess:
    case TokenType::Identifier:
    case TokenType::OpAssign:
    case TokenType::Colon:
        {
            return false;
        }
    default:
        break;
    }
    return true;
}

// Returns true if there needs to be a space between the previous token type, and the current token
// type for correct output. It is assumed that the token stream is appropriate.
// The implementation might need more sophistication, but this at least avoids Blah const *  ->
// Blahconst*
static bool _tokenConcatNeedsSpace(TokenType prev, TokenType cur)
{
    if ((cur == TokenType::OpAssign) || (prev == cur && !_canRepeatTokenType(cur)))
    {
        return true;
    }
    return false;
}

void Parser::_getTypeTokens(
    TokenReader::ParsingCursor start,
    TokenReader::ParsingCursor nameCursor,
    List<Token>& outToks)
{
    auto endCursor = m_reader.getCursor();
    m_reader.setCursor(start);

    while (!m_reader.isAtCursor(endCursor))
    {
        if (m_reader.getCursor() == nameCursor)
        {
            m_reader.advanceToken();
        }
        else
        {
            outToks.add(m_reader.advanceToken());
        }
    }
}

UnownedStringSlice Parser::_concatType(
    TokenReader::ParsingCursor start,
    TokenReader::ParsingCursor nameCursor)
{
    List<Token> toks;
    _getTypeTokens(start, nameCursor, toks);
    return _concatTokens(toks.getBuffer(), toks.getCount());
}

UnownedStringSlice Parser::_concatTokens(const Token* toks, Index toksCount)
{
    StringBuilder buf;

    TokenType prevTokenType = TokenType::Unknown;
    for (Index i = 0; i < toksCount; ++i)
    {
        const auto token = toks[i];

        // Check if we need a space between tokens
        if (_tokenConcatNeedsSpace(prevTokenType, token.type))
        {
            buf << " ";
        }

        buf << token.getContent();

        prevTokenType = token.type;
    }

    StringSlicePool* typePool = m_nodeTree->m_typePool;
    return typePool->getSlice(typePool->add(buf));
}

UnownedStringSlice Parser::_concatTokens(TokenReader::ParsingCursor start)
{
    auto endCursor = m_reader.getCursor();

    m_reader.setCursor(start);

    TokenType prevTokenType = TokenType::Unknown;

    StringBuilder buf;
    while (!m_reader.isAtCursor(endCursor))
    {
        const Token token = m_reader.advanceToken();
        // Check if we need a space between tokens
        if (_tokenConcatNeedsSpace(prevTokenType, token.type))
        {
            buf << " ";
        }
        buf << token.getContent();

        prevTokenType = token.type;
    }

    StringSlicePool* typePool = m_nodeTree->m_typePool;
    return typePool->getSlice(typePool->add(buf));
}

SlangResult Parser::_maybeParseType(
    Index& ioTemplateDepth,
    TokenReader::ParsingCursor& outNameCursor)
{
    outNameCursor = TokenReader::ParsingCursor();

    while (true)
    {
        if (m_reader.peekTokenType() == TokenType::Identifier)
        {
            const IdentifierStyle style =
                m_nodeTree->m_identifierLookup->get(m_reader.peekToken().getContent());

            if (style == IdentifierStyle::TypeModifier ||
                style == IdentifierStyle::IntegerModifier || style == IdentifierStyle::Class ||
                style == IdentifierStyle::Struct)
            {
                // These are ok keywords in this context
            }
            else if (hasFlag(style, IdentifierFlag::Keyword))
            {
                return SLANG_FAIL;
            }
        }

        _maybeConsume(IdentifierStyle::TypeModifier);

        if (advanceIfStyle(IdentifierStyle::IntegerModifier))
        {
            // Consume the integer typename (if there is one)
            const Token peekToken = m_reader.peekToken();
            if (peekToken.type == TokenType::Identifier)
            {
                const IdentifierStyle style =
                    m_nodeTree->m_identifierLookup->get(peekToken.getContent());
                if (style == IdentifierStyle::IntegerType)
                {
                    m_reader.advanceToken();
                }
            }
            break;
        }

        advanceIfToken(TokenType::Scope);
        while (true)
        {
            // if we have a struct/class prefix in front of a name just consume it.
            if (m_reader.peekTokenType() == TokenType::Identifier)
            {
                const IdentifierStyle style =
                    m_nodeTree->m_identifierLookup->get(m_reader.peekToken().getContent());
                if (style == IdentifierStyle::Class || style == IdentifierStyle::Struct)
                {
                    m_reader.advanceToken();
                }
            }

            Token identifierToken;
            if (!advanceIfToken(TokenType::Identifier, &identifierToken))
            {
                return SLANG_FAIL;
            }

            const IdentifierStyle style =
                m_nodeTree->m_identifierLookup->get(identifierToken.getContent());
            if (hasFlag(style, IdentifierFlag::Keyword))
            {
                return SLANG_FAIL;
            }

            if (advanceIfToken(TokenType::Scope))
            {
                continue;
            }
            break;
        }

        if (m_reader.peekTokenType() == TokenType::OpLess)
        {
            SLANG_RETURN_ON_FAIL(_maybeParseTemplateArgs(ioTemplateDepth));
        }

        if (m_reader.peekTokenType() == TokenType::Scope)
        {
            // Skip the scope and repeat
            m_reader.advanceToken();
            continue;
        }

        break;
    }

    // Strip all the consts etc modifiers
    _maybeConsume(IdentifierStyle::TypeModifier);

    // It's a reference and we are done
    if (advanceIfToken(TokenType::OpBitAnd))
    {
        return SLANG_OK;
    }

    while (true)
    {
        if (advanceIfToken(TokenType::OpMul))
        {
            // Strip all the consts
            _maybeConsume(IdentifierStyle::TypeModifier);
            continue;
        }
        break;
    }

    if (advanceIfToken(TokenType::LParent))
    {
        // TODO(JS):
        // Doesn't handle all the modifiers just (*SomeName)

        SLANG_RETURN_ON_FAIL(expect(TokenType::OpMul));
        outNameCursor = m_reader.getCursor();
        SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier));

        SLANG_RETURN_ON_FAIL(expect(TokenType::RParent));

        // We need to parse and add the params
        if (m_reader.peekTokenType() != TokenType::LParent)
        {
            m_sink->diagnose(
                m_reader.peekToken(),
                CPPDiagnostics::expectingToken,
                TokenType::LParent);
            return SLANG_FAIL;
        }

        // Consume the params
        SLANG_RETURN_ON_FAIL(_consumeBalancedParens());
    }
    else if (m_reader.peekTokenType() == TokenType::Identifier)
    {
        auto potentialNameCursor = m_reader.getCursor();
        m_reader.advanceToken();
        if (m_reader.peekTokenType() == TokenType::LBracket)
        {
            outNameCursor = potentialNameCursor;
            while (advanceIfToken(TokenType::LBracket))
            {
                List<Token> exprToks;
                SLANG_RETURN_ON_FAIL(_parseExpression(exprToks));
                SLANG_RETURN_ON_FAIL(expect(TokenType::RBracket));
            }
        }
        else
        {
            // Wasn't an array type..., so rewind
            m_reader.setCursor(potentialNameCursor);
        }
    }

    return SLANG_OK;
}

SlangResult Parser::_maybeParseType(List<Token>& outToks, Token& outName)
{
    // Set to unknown
    outName = Token();

    auto startCursor = m_reader.getCursor();

    TokenReader::ParsingCursor nameCursor;

    Index templateDepth = 0;
    SlangResult res = _maybeParseType(templateDepth, nameCursor);
    if (SLANG_FAILED(res) && m_sink->getErrorCount())
    {
        return res;
    }

    if (templateDepth != 0)
    {
        m_sink->diagnose(m_reader.peekToken(), CPPDiagnostics::unexpectedTemplateClose);
        return SLANG_FAIL;
    }

    auto endCursor = m_reader.getCursor();
    m_reader.setCursor(startCursor);

    if (nameCursor.isValid())
    {
        while (!m_reader.isAtCursor(endCursor))
        {
            if (m_reader.getCursor() == nameCursor)
            {
                outName = m_reader.advanceToken();
            }
            else
            {
                outToks.add(m_reader.advanceToken());
            }
        }
    }
    else
    {
        while (!m_reader.isAtCursor(endCursor))
        {
            outToks.add(m_reader.advanceToken());
        }
    }

    return SLANG_OK;
}

SlangResult Parser::_parseSpecialMacro()
{
    Token name;
    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &name));

    List<Token> params;

    if (m_reader.peekTokenType() == TokenType::LParent)
    {
        // Mark the start
        auto startCursor = m_reader.getCursor();

        // Consume the params
        SLANG_RETURN_ON_FAIL(_consumeBalancedParens());

        auto endCursor = m_reader.getCursor();
        m_reader.setCursor(startCursor);

        while (!m_reader.isAtCursor(endCursor))
        {
            params.add(m_reader.advanceToken());
        }
    }

    // Can do special handling here
    const UnownedStringSlice suffix = name.getContent().tail(m_options->m_markPrefix.getLength());

    if (suffix == "COM_INTERFACE")
    {
        // TODO(JS): It's a com interface. Extact the GUID
    }

    return SLANG_OK;
}

SlangResult Parser::_parseMarker()
{
    SLANG_ASSERT(
        m_reader.peekTokenType() == TokenType::Identifier &&
        _isMarker(m_reader.peekToken().getContent()) && m_currentScope->isClassLike());

    ClassLikeNode* node = as<ClassLikeNode>(m_currentScope);

    if (node->m_marker.type != TokenType::Unknown)
    {
        m_sink->diagnose(
            m_reader.peekToken(),
            CPPDiagnostics::classMarkerAlreadyFound,
            node->m_name.getContent());
        m_sink->diagnose(node->m_marker, CPPDiagnostics::previousLocation);
        return SLANG_FAIL;
    }

    // Set the marker token.
    node->m_marker = m_reader.advanceToken();

    // Looks like it's a marker
    UnownedStringSlice slice(node->m_marker.getContent());

    // Strip the prefix and suffix
    slice = UnownedStringSlice(
        slice.begin() + m_options->m_markPrefix.getLength(),
        slice.end() - m_options->m_markSuffix.getLength());

    // Strip ABSTRACT_ if it's there
    UnownedStringSlice abstractSlice("ABSTRACT_");
    if (slice.startsWith(abstractSlice))
    {
        slice = UnownedStringSlice(slice.begin() + abstractSlice.getLength(), slice.end());
    }

    // TODO: We could strip other stuff or have other heuristics there, but this is
    // probably okay for now

    // Set the typeSet
    node->m_typeSet = m_nodeTree->getOrAddTypeSet(slice);

    // Okay now looking for ( identifier)
    Token typeNameToken;

    SLANG_RETURN_ON_FAIL(expect(TokenType::LParent));
    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &typeNameToken));
    SLANG_RETURN_ON_FAIL(expect(TokenType::RParent));

    if (typeNameToken.getContent() != node->m_name.getContent())
    {
        m_sink->diagnose(
            typeNameToken,
            CPPDiagnostics::typeNameDoesntMatch,
            node->m_name.getContent());
        return SLANG_FAIL;
    }

    // If has the marker it is assumed reflected
    node->m_reflectionType = ReflectionType::Reflected;
    return SLANG_OK;
}

SlangResult Parser::_maybeParseType(UnownedStringSlice& outType, Token& outName)
{
    auto startCursor = m_reader.getCursor();

    Index templateDepth = 0;

    TokenReader::ParsingCursor nameCursor;

    SlangResult res = _maybeParseType(templateDepth, nameCursor);
    if (SLANG_FAILED(res) && m_sink->getErrorCount())
    {
        return res;
    }

    if (templateDepth != 0)
    {
        m_sink->diagnose(m_reader.peekToken(), CPPDiagnostics::unexpectedTemplateClose);
        return SLANG_FAIL;
    }

    if (nameCursor.isValid())
    {
        const auto cursor = m_reader.getCursor();
        m_reader.setCursor(nameCursor);
        outName = m_reader.peekToken();
        m_reader.setCursor(cursor);

        // Extract the contents
        List<Token> toks;
        _getTypeTokens(startCursor, nameCursor, toks);
        outType = _concatTokens(toks.getBuffer(), toks.getCount());
    }
    else
    {
        // We can build up the out type, from the tokens we found
        outType = _concatTokens(startCursor);
    }
    return SLANG_OK;
}

static bool _isBalancedOpen(TokenType tokenType)
{
    return tokenType == TokenType::LBrace || tokenType == TokenType::LParent ||
           tokenType == TokenType::LBracket;
}

static bool _isBalancedClose(TokenType tokenType)
{
    return tokenType == TokenType::RBrace || tokenType == TokenType::RParent ||
           tokenType == TokenType::RBracket;
}

static TokenType _getBalancedClose(TokenType tokenType)
{
    SLANG_ASSERT(_isBalancedOpen(tokenType));
    switch (tokenType)
    {
    case TokenType::LBrace:
        return TokenType::RBrace;
    case TokenType::LParent:
        return TokenType::RParent;
    case TokenType::LBracket:
        return TokenType::RBracket;
    default:
        return TokenType::Unknown;
    }
}

SlangResult Parser::_parseBalanced(DiagnosticSink* sink)
{
    const TokenType openTokenType = m_reader.peekTokenType();
    if (!_isBalancedOpen(openTokenType))
    {
        return SLANG_FAIL;
    }

    // Save the start token
    const Token startToken = m_reader.advanceToken();
    // Get the token type that would close the open
    const TokenType closeTokenType = _getBalancedClose(openTokenType);

    while (true)
    {
        const TokenType tokenType = m_reader.peekTokenType();

        // If we hit the closing token, we are done
        if (tokenType == closeTokenType)
        {
            m_reader.advanceToken();
            return SLANG_OK;
        }

        // If we hit a balanced open, recurse
        if (_isBalancedOpen(tokenType))
        {
            SLANG_RETURN_ON_FAIL(_parseBalanced(sink));
            continue;
        }

        // If we hit a close token that doesn't match, then the balancing has gone wrong
        if (_isBalancedClose(tokenType))
        {
            // Only diagnose if required
            if (sink)
            {
                sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::unexpectedUnbalancedToken);
                sink->diagnose(startToken, CPPDiagnostics::seeOpen);
            }
            return SLANG_FAIL;
        }

        // If we hit the end of the file and have not hit the closing token, then
        // somethings gone wrong
        if (tokenType == TokenType::EndOfFile)
        {
            if (sink)
            {
                sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::unexpectedEndOfFile);
                sink->diagnose(startToken, CPPDiagnostics::seeOpen);
            }

            return SLANG_FAIL;
        }

        // Skip the token
        m_reader.advanceToken();
    }
}

SlangResult Parser::_consumeBalancedParens()
{
    SLANG_ASSERT(m_reader.peekTokenType() == TokenType::LParent);

    Index parenCount = 0;

    while (true)
    {
        const TokenType tokenType = m_reader.peekTokenType();

        switch (tokenType)
        {
        case TokenType::LParent:
            {
                parenCount++;
                break;
            }
        case TokenType::RParent:
            {
                --parenCount;
                // If no more parens then we are done
                if (parenCount == 0)
                {
                    m_reader.advanceToken();
                    return SLANG_OK;
                }
                break;
            }
        case TokenType::EndOfFile:
            {
                // If we hit the end of the file, then not balanced
                return SLANG_FAIL;
            }
        default:
            break;
        }

        m_reader.advanceToken();
    }
}

SlangResult Parser::_parseExpression(List<Token>& outExprTokens)
{
    Index parenCount = 0;
    Index bracketCount = 0;

    // TODO(JS): NOTE! This doesn't handle an expression that contains a template params in
    // Something<Arg1, 3>, because without knowing what Something is, it's not known if < is a
    // comparison or or a 'template' bracket
    //
    // This can be worked around in the originating source by placing in parens

    while (true)
    {
        TokenType tokenType = m_reader.peekTokenType();

        switch (tokenType)
        {
        case TokenType::LParent:
            {
                parenCount++;
                break;
            }
        case TokenType::RParent:
            {
                // If no parens, and nothing else is open then we are done
                if (parenCount == 0)
                {
                    if (bracketCount)
                    {
                        m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::cannotParseExpression);
                        return SLANG_FAIL;
                    }

                    return SLANG_OK;
                }
                --parenCount;
                break;
            }
        case TokenType::LBracket:
            {
                bracketCount++;
                break;
            }
        case TokenType::RBracket:
            {
                // If no brackets are open we are done
                if (bracketCount == 0)
                {
                    if (parenCount)
                    {
                        m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::cannotParseExpression);
                        return SLANG_FAIL;
                    }
                    return SLANG_OK;
                }
                --bracketCount;
                break;
            }
        case TokenType::EndOfFile:
            {
                if ((bracketCount | parenCount) == 0)
                {
                    return SLANG_OK;
                }
                m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::cannotParseExpression);
                return SLANG_FAIL;
            }
        case TokenType::RBrace:
        case TokenType::Semicolon:
        case TokenType::Comma:
            {
                if ((bracketCount | parenCount) == 0)
                {
                    return SLANG_OK;
                }
                break;
            }

        default:
            break;
        }

        outExprTokens.add(m_reader.advanceToken());
    }
}

SlangResult Parser::_parseTypeDef()
{
    if (!m_currentScope->canContainTypes())
    {
        m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::cannotDeclareTypeInScope);
        return SLANG_FAIL;
    }

    // Consume the typedef
    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier));

    Token nameToken;
    // Parse the type
    List<Token> toks;
    SLANG_RETURN_ON_FAIL(_maybeParseType(toks, nameToken));

    // Followed by the name
    if (nameToken.type != TokenType::Identifier)
    {
        SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &nameToken));
    }

    if (Node::lookupNameInScope(m_currentScope, nameToken.getContent()))
    {
        m_sink->diagnose(
            nameToken.loc,
            CPPDiagnostics::identifierAlreadyDefined,
            nameToken.getContent());
        return SLANG_FAIL;
    }

    SLANG_RETURN_ON_FAIL(expect(TokenType::Semicolon));

    RefPtr<TypeDefNode> node = new TypeDefNode;
    node->m_name = nameToken;
    node->m_reflectionType = m_currentScope->getContainedReflectionType();

    // Set what aliases too
    node->m_targetTypeTokens.swapWith(toks);

    m_currentScope->addChild(node);

    return SLANG_OK;
}


bool Parser::_isCtor()
{
    bool isCtor = false;
    // It's a constructor
    if (m_currentScope->isClassLike() && m_reader.peekTokenType() == TokenType::Identifier &&
        m_reader.peekToken().getContent() == m_currentScope->m_name.getContent())
    {
        // We need to check it's followed immediately by ( to be sure it's a ctor

        auto cursor = m_reader.getCursor();
        m_reader.advanceToken();
        isCtor = (m_reader.peekTokenType() == TokenType::LParent);
        m_reader.setCursor(cursor);
    }

    return isCtor;
}

bool isAlphaNumeric(char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9');
}

SlangResult Parser::_maybeParseContained(Node** outNode)
{
    *outNode = nullptr;

    _maybeConsume(IdentifierStyle::CallableMisc);

    bool isStatic = false;
    bool isVirtual = false;

    while (m_reader.peekTokenType() == TokenType::Identifier)
    {
        const IdentifierStyle style =
            m_nodeTree->m_identifierLookup->get(m_reader.peekToken().getContent());

        // Check for virtualness
        if (style == IdentifierStyle::Virtual)
        {
            isVirtual = true;
            m_reader.advanceToken();
            continue;
        }

        // Check if static
        if (style == IdentifierStyle::Static)
        {
            isStatic = true;
            m_reader.advanceToken();
            continue;
        }

        break;
    }

    _maybeConsume(IdentifierStyle::CallableMisc);

    UnownedStringSlice typeName;
    Token nameToken;

    bool isConstructor = false;

    if (m_currentScope->isClassLike())
    {
        // If it's a dtor
        if (advanceIfToken(TokenType::OpBitNot, &nameToken))
        {
            // Dtor
            // For Dtor we don't hold the full name just the ~
            Token tok;
            SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &tok));

            if (tok.getContent() != m_currentScope->m_name.getContent())
            {
                m_sink->diagnose(
                    m_reader.peekLoc(),
                    CPPDiagnostics::destructorNameDoesntMatch,
                    m_currentScope->m_name.getContent());
                return SLANG_FAIL;
            }
        }
        else if (_isCtor())
        {
            nameToken = m_reader.advanceToken();
            isConstructor = true;
        }
    }

    // If don't have a name it's not a dtor or ctor, so see if it's a type
    if (nameToken.type == TokenType::Unknown)
    {
        if (SLANG_FAILED(_maybeParseType(typeName, nameToken)))
        {
            if (m_sink->getErrorCount())
            {
                return SLANG_FAIL;
            }

            _consumeToSync();
            return SLANG_OK;
        }
    }

    if (nameToken.type == TokenType::Unknown)
    {
        // Has a calling convention (must be a function/method)
        Token callingConventionToken;
        advanceIfStyle(IdentifierStyle::CallingConvention, &callingConventionToken);

        // Expecting a name
        if (!advanceIfToken(TokenType::Identifier, &nameToken))
        {
            _consumeToSync();
            return SLANG_OK;
        }
    }

    // Handles other scenarios, but here for catching operator overloading
    if (nameToken.type == TokenType::Identifier)
    {
        const auto style = m_nodeTree->m_identifierLookup->get(nameToken.getContent());
        if (style != IdentifierStyle::None)
        {
            _consumeToSync();
            return SLANG_OK;
        }
    }

    if (m_reader.peekTokenType() == TokenType::LParent)
    {
        if (!m_currentScope->canContainCallable())
        {
            SLANG_RETURN_ON_FAIL(_consumeBalancedParens());
            // Consume everything up to ; or {
            SLANG_RETURN_ON_FAIL(_consumeToSync());

            return SLANG_OK;
        }

        // Looks like it's a callable
        m_reader.advanceToken();

        List<CallableNode::Param> params;

        if (m_reader.peekTokenType() != TokenType::RParent)
        {
            while (true)
            {
                Token paramName;
                UnownedStringSlice type;
                SlangResult res = _maybeParseType(type, paramName);

                if (SLANG_FAILED(res))
                {
                    m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::expectingType);
                    return res;
                }

                if (paramName.type != TokenType::Identifier)
                {
                    if (m_reader.peekTokenType() == TokenType::Identifier)
                    {
                        paramName = m_reader.advanceToken();
                    }
                }

                // If we have a name check for default value
                if (paramName.type == TokenType::Identifier && advanceIfToken(TokenType::OpAssign))
                {
                    // Check if we have a default value
                    List<Token> exprTokens;
                    SLANG_RETURN_ON_FAIL(_parseExpression(exprTokens));
                }

                CallableNode::Param param;
                param.m_name = paramName;
                param.m_type = type;

                params.add(param);

                {
                    const auto peekType = m_reader.peekTokenType();
                    if (peekType == TokenType::RParent)
                    {
                        break;
                    }
                    if (peekType == TokenType::Comma)
                    {
                        m_reader.advanceToken();
                        continue;
                    }
                }

                m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::expectingToken, ", or ) or =");
                return SLANG_FAIL;
            }
        }

        // Skip )
        m_reader.advanceToken();

        // Parse suffix
        bool isPure = false;

        // const?
        _maybeConsume(IdentifierStyle::TypeModifier);

        if (isConstructor)
        {
            // Initializer list
            if (advanceIfToken(TokenType::Colon))
            {
                while (true)
                {
                    auto peekType = m_reader.peekTokenType();
                    if (peekType == TokenType::Semicolon || peekType == TokenType::LBrace ||
                        peekType == TokenType::EndOfFile)
                    {
                        break;
                    }
                    // Consume
                    m_reader.advanceToken();
                }
            }
        }

        // = 0 ? or = default
        if (advanceIfToken(TokenType::OpAssign))
        {
            if (m_reader.peekTokenType() == TokenType::IntegerLiteral)
            {
                Int value = -1;
                if (SLANG_SUCCEEDED(
                        StringUtil::parseInt(m_reader.peekToken().getContent(), value)) &&
                    value == 0)
                {
                    isPure = true;
                    m_reader.advanceToken();
                }
                else
                {
                    m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::expectingToken, "0");
                    return SLANG_FAIL;
                }
            }
            else if (advanceIfStyle(IdentifierStyle::Default))
            {
            }
            else
            {
                m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::cannotParseCallable);
                return SLANG_FAIL;
            }
        }

        if (m_reader.peekTokenType() == TokenType::Semicolon)
        {
            m_reader.advanceToken();
        }
        else if (m_reader.peekTokenType() == TokenType::LBrace)
        {
            SLANG_RETURN_ON_FAIL(consumeToClosingBrace());
        }
        else
        {
            m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::expectingToken, "; or {");
            return SLANG_FAIL;
        }

        RefPtr<CallableNode> callableNode = new CallableNode;

        callableNode->m_returnType = typeName;
        callableNode->m_name = nameToken;
        callableNode->m_reflectionType = m_currentScope->getContainedReflectionType();

        callableNode->m_isVirtual = isVirtual;
        callableNode->m_isPure = isPure;
        callableNode->m_isStatic = isStatic;

        callableNode->m_params.swapWith(params);

        Node* nodeWithName = m_currentScope->findChild(nameToken.getContent());

        if (nodeWithName)
        {
            CallableNode* initialOverload = as<CallableNode>(nodeWithName);
            if (!initialOverload)
            {
                m_sink->diagnose(m_reader.peekLoc(), CPPDiagnostics::cannotOverload);
                m_sink->diagnose(nodeWithName->getSourceLoc(), CPPDiagnostics::seeDeclarationOf);
                return SLANG_FAIL;
            }

            callableNode->m_nextOverload = initialOverload->m_nextOverload;
            initialOverload->m_nextOverload = initialOverload;

            m_currentScope->addChildIgnoringName(callableNode);
        }
        else
        {
            m_currentScope->addChild(callableNode);
        }

        *outNode = callableNode;
        return SLANG_OK;
    }
    else
    {
        // Looks like variable
        if (!m_currentScope->canContainFields() || nameToken.type != TokenType::Identifier)
        {
            _consumeToSync();
            return SLANG_OK;
        }

        // Check if has a default value
        if (advanceIfToken(TokenType::OpAssign))
        {
            List<Token> exprTokens;
            SLANG_RETURN_ON_FAIL(_parseExpression(exprTokens));
        }

        // Hit end of field/variable
        if (m_reader.peekTokenType() == TokenType::Semicolon)
        {
            RefPtr<FieldNode> fieldNode = new FieldNode;

            fieldNode->m_fieldType = typeName;
            fieldNode->m_name = nameToken;
            fieldNode->m_reflectionType = m_currentScope->getContainedReflectionType();
            fieldNode->m_isStatic = isStatic;
            if (fieldNode->m_reflectionType == ReflectionType::Reflected)
            {
                static const char* illegalTypes[] = {
                    "size_t",
                    "Int",
                    "UInt",
                    "Index",
                    "Count",
                    "UIndex",
                    "UCount",
                    "PtrInt",
                    "intptr_t",
                    "uintptr_t"};
                for (const auto& illegalType : illegalTypes)
                {
                    int index = typeName.indexOf(UnownedStringSlice(illegalType));
                    if (index != -1)
                    {
                        index += UnownedStringSlice(illegalType).getLength();
                        if (index >= typeName.getLength() || !isAlphaNumeric(typeName[index]))
                        {
                            // Cannot use this type in a field (as it's arch dependent
                            m_sink->diagnose(
                                nameToken,
                                CPPDiagnostics::cannoseUseArchDependentType,
                                illegalType);
                            return SLANG_FAIL;
                        }
                    }
                }
            }
            m_currentScope->addChild(fieldNode);

            *outNode = fieldNode;
            return SLANG_OK;
        }
    }

    _consumeToSync();
    return SLANG_OK;
}

/* static */ Node::Kind Parser::_toNodeKind(IdentifierStyle style)
{
    switch (style)
    {
    case IdentifierStyle::Class:
        return Node::Kind::ClassType;
    case IdentifierStyle::Struct:
        return Node::Kind::StructType;
    case IdentifierStyle::Namespace:
        return Node::Kind::Namespace;
    case IdentifierStyle::Enum:
        return Node::Kind::Enum;
    case IdentifierStyle::TypeDef:
        return Node::Kind::TypeDef;
    default:
        return Node::Kind::Invalid;
    }
}

static UnownedStringSlice _trimUnderscorePrefix(const UnownedStringSlice& slice)
{
    if (slice.getLength() && slice[0] == '_')
    {
        return UnownedStringSlice(slice.begin() + 1, slice.end());
    }
    else
    {
        return slice;
    }
}

SlangResult Parser::_parsePreDeclare()
{
    // Skip the declare type token
    m_reader.advanceToken();

    SLANG_RETURN_ON_FAIL(expect(TokenType::LParent));

    // Get the typeSet
    Token typeSetToken;
    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &typeSetToken));
    TypeSet* typeSet = m_nodeTree->getOrAddTypeSet(typeSetToken.getContent());

    SLANG_RETURN_ON_FAIL(expect(TokenType::Comma));

    // Get the type of type
    Node::Kind nodeKind;
    {
        Token typeToken;
        SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &typeToken));

        const IdentifierStyle style = m_nodeTree->m_identifierLookup->get(typeToken.getContent());

        if (style != IdentifierStyle::Struct && style != IdentifierStyle::Class)
        {
            m_sink->diagnose(
                typeToken,
                CPPDiagnostics::expectingTypeKeyword,
                typeToken.getContent());
            return SLANG_FAIL;
        }
        nodeKind = _toNodeKind(style);
    }

    Token name;
    Token super;

    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &name));

    if (advanceIfToken(TokenType::Colon))
    {
        SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &super));
    }

    SLANG_RETURN_ON_FAIL(expect(TokenType::RParent));

    switch (nodeKind)
    {
    case Node::Kind::ClassType:
    case Node::Kind::StructType:
        {
            RefPtr<ClassLikeNode> node(new ClassLikeNode(nodeKind));

            node->m_name = name;
            node->m_super = super;
            node->m_typeSet = typeSet;

            // Assume it is reflected
            node->m_reflectionType = ReflectionType::Reflected;

            SLANG_RETURN_ON_FAIL(pushScope(node));
            // Pop out of the node
            popScope();
            break;
        }
    default:
        {
            return SLANG_FAIL;
        }
    }


    return SLANG_OK;
}

SlangResult Parser::_parseTypeSet()
{
    // Skip the declare type token
    m_reader.advanceToken();

    SLANG_RETURN_ON_FAIL(expect(TokenType::LParent));

    Token typeSetToken;
    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &typeSetToken));

    TypeSet* typeSet = m_nodeTree->getOrAddTypeSet(typeSetToken.getContent());

    SLANG_RETURN_ON_FAIL(expect(TokenType::Comma));

    // Get the type of type
    Token typeToken;
    SLANG_RETURN_ON_FAIL(expect(TokenType::Identifier, &typeToken));

    SLANG_RETURN_ON_FAIL(expect(TokenType::RParent));

    // Set the typename
    typeSet->m_typeName = typeToken.getContent();

    return SLANG_OK;
}

SlangResult Parser::parse(SourceOrigin* sourceOrigin, const Options* options)
{
    SLANG_ASSERT(options);
    m_options = options;

    // Set the current origin
    m_sourceOrigin = sourceOrigin;

    SourceFile* sourceFile = sourceOrigin->m_sourceFile;

    SourceManager* manager = sourceFile->getSourceManager();

    SourceView* sourceView = manager->createSourceView(sourceFile, nullptr, SourceLoc::fromRaw(0));

    Lexer lexer;

    // Set up the scope stack
    m_scopeStack.clear();

    m_currentScope = m_nodeTree->m_rootNode;
    m_scopeStack.add(m_currentScope);

    if (!options->m_requireMark)
    {
        m_currentScope->m_reflectionOverride = ReflectionType::Reflected;
    }

    lexer.initialize(sourceView, m_sink, m_nodeTree->m_namePool, manager->getMemoryArena());
    m_tokenList = lexer.lexAllSemanticTokens();
    // See if there were any errors
    if (m_sink->getErrorCount())
    {
        return SLANG_FAIL;
    }

    m_reader = TokenReader(m_tokenList);

    while (true)
    {
        switch (m_reader.peekTokenType())
        {
        case TokenType::OpBitNot:
            {
                // Handle dtor
                if (m_currentScope->isClassLike())
                {
                    Node* containedNode = nullptr;
                    SLANG_RETURN_ON_FAIL(_maybeParseContained(&containedNode));
                }
                else
                {
                    // consume
                    m_reader.advanceToken();
                }
                break;
            }
        case TokenType::Identifier:
            {
                const IdentifierStyle style =
                    m_nodeTree->m_identifierLookup->get(m_reader.peekToken().getContent());

                switch (style)
                {
                case IdentifierStyle::Extern:
                    {
                        m_reader.advanceToken();

                        Token externType;
                        SLANG_RETURN_ON_FAIL(expect(TokenType::StringLiteral, &externType));

                        if (advanceIfToken(TokenType::LBrace))
                        {
                            // Push a 'special' scope (which is basically transparent)
                            pushScope(nullptr);
                        }
                        break;
                    }
                case IdentifierStyle::Template:
                    {
                        SLANG_RETURN_ON_FAIL(_consumeTemplate());
                        break;
                    }
                case IdentifierStyle::PreDeclare:
                    {
                        SLANG_RETURN_ON_FAIL(_parsePreDeclare());
                        break;
                    }
                case IdentifierStyle::TypeSet:
                    {
                        SLANG_RETURN_ON_FAIL(_parseTypeSet());
                        break;
                    }
                case IdentifierStyle::Reflected:
                    {
                        m_reader.advanceToken();
                        if (m_currentScope)
                        {
                            m_currentScope->m_reflectionOverride = ReflectionType::Reflected;
                        }
                        break;
                    }
                case IdentifierStyle::Unreflected:
                    {
                        m_reader.advanceToken();
                        if (m_currentScope)
                        {
                            m_currentScope->m_reflectionOverride = ReflectionType::NotReflected;
                        }
                        break;
                    }
                case IdentifierStyle::Access:
                    {
                        m_reader.advanceToken();
                        SLANG_RETURN_ON_FAIL(expect(TokenType::Colon));
                        break;
                    }
                case IdentifierStyle::TypeDef:
                    {
                        if (isTypeEnabled(Node::Kind::TypeDef))
                        {
                            SLANG_RETURN_ON_FAIL(_parseTypeDef());
                        }
                        else
                        {
                            m_reader.advanceToken();
                            SLANG_RETURN_ON_FAIL(_consumeToSync());
                        }
                        break;
                    }
                default:
                    {
                        IdentifierFlags flags = getFlags(style);

                        if (flags & IdentifierFlag::StartScope)
                        {
                            Node::Kind kind = _toNodeKind(style);
                            SLANG_ASSERT(kind != Node::Kind::Invalid);

                            if (isTypeEnabled(kind))
                            {
                                SLANG_RETURN_ON_FAIL(_maybeParseNode(kind));
                            }
                            else
                            {
                                SLANG_RETURN_ON_FAIL(_maybeConsumeScope());
                            }
                        }
                        else
                        {
                            UnownedStringSlice content = m_reader.peekToken().getContent();

                            // If it's a marker handle it
                            if (_isMarker(content))
                            {
                                if (!m_currentScope->isClassLike())
                                {
                                    m_sink->diagnose(
                                        m_reader.peekLoc(),
                                        CPPDiagnostics::classMarkerOutsideOfClass);
                                    return SLANG_FAIL;
                                }

                                SLANG_RETURN_ON_FAIL(_parseMarker());
                                break;
                            }

                            if (m_options->m_markPrefix.getLength() > 0 &&
                                content.startsWith(m_options->m_markPrefix.getUnownedSlice()))
                            {
                                SLANG_RETURN_ON_FAIL(_parseSpecialMacro());
                                break;
                            }


                            // Special case the node that's the root of the hierarchy (as far as
                            // reflection is concerned) This could be a field
                            if (m_currentScope->canContainFields() ||
                                m_currentScope->canContainCallable())
                            {
                                Node* containedNode = nullptr;
                                SLANG_RETURN_ON_FAIL(_maybeParseContained(&containedNode));
                            }
                            else
                            {
                                m_reader.advanceToken();
                            }
                        }
                        break;
                    }
                }
                break;
            }
        case TokenType::LBrace:
            {
                SLANG_RETURN_ON_FAIL(consumeToClosingBrace());
                break;
            }
        case TokenType::RBrace:
            {
                SLANG_RETURN_ON_FAIL(popScope());
                m_reader.advanceToken();
                break;
            }
        case TokenType::EndOfFile:
            {
                // Okay we need to confirm that we are in the root node, and with no open braces
                if (m_currentScope != m_nodeTree->getRootNode())
                {
                    m_sink->diagnose(m_reader.peekToken(), CPPDiagnostics::braceOpenAtEndOfFile);
                    return SLANG_FAIL;
                }
                if (m_sink->getErrorCount())
                    return SLANG_FAIL;
                return SLANG_OK;
            }
        case TokenType::Pound:
            {
                Token token = m_reader.peekToken();
                if (token.flags & TokenFlag::AtStartOfLine)
                {
                    // We are just going to ignore all of these for now....
                    m_reader.advanceToken();
                    for (;;)
                    {
                        auto t = m_reader.peekToken();
                        if (t.type == TokenType::EndOfFile || (t.flags & TokenFlag::AtStartOfLine))
                        {
                            break;
                        }
                        m_reader.advanceToken();
                    }
                    break;
                }
                // Skip it then
                m_reader.advanceToken();
                break;
            }
        default:
            {
                // Skip it then
                m_reader.advanceToken();
                break;
            }
        }
    }
}

} // namespace CppParse
