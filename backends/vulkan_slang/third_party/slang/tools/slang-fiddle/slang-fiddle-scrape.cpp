// slang-fiddle-scrape.cpp
#include "slang-fiddle-scrape.h"

#include "slang-fiddle-script.h"

namespace fiddle
{

// Parser

struct Parser
{
private:
    DiagnosticSink& _sink;
    List<TokenWithTrivia> _tokens;

    TokenWithTrivia const* _cursor = nullptr;
    TokenWithTrivia const* _end = nullptr;

    LogicalModule* _module = nullptr;

    ContainerDecl* _currentParentDecl = nullptr;

    struct WithParentDecl
    {
    public:
        WithParentDecl(Parser* outer, ContainerDecl* decl)
        {
            _outer = outer;
            _saved = outer->_currentParentDecl;

            outer->_currentParentDecl = decl;
        }

        ~WithParentDecl() { _outer->_currentParentDecl = _saved; }

    private:
        Parser* _outer;
        ContainerDecl* _saved;
    };

public:
    Parser(DiagnosticSink& sink, List<TokenWithTrivia> const& tokens, LogicalModule* module)
        : _sink(sink), _tokens(tokens), _module(module)
    {
        _cursor = tokens.begin();
        _end = tokens.end() - 1;
    }

    bool _isRecovering = false;

    TokenWithTrivia const& peek() { return *_cursor; }

    SourceLoc const& peekLoc() { return peek().getLoc(); }

    TokenType peekType() { return peek().getType(); }

    TokenWithTrivia read()
    {
        _isRecovering = false;
        if (peekType() != TokenType::EndOfFile)
            return *_cursor++;
        else
            return *_cursor;
    }

    TokenWithTrivia expect(TokenType expected)
    {
        if (peekType() == expected)
        {
            return read();
        }

        if (!_isRecovering)
        {
            _sink.diagnose(peekLoc(), fiddle::Diagnostics::unexpected, peekType(), expected);
        }
        else
        {
            // TODO: need to skip until we see what we expected...
            _sink.diagnose(SourceLoc(), fiddle::Diagnostics::internalError);
        }

        return TokenWithTrivia();
    }

    TokenWithTrivia expect(const char* expected)
    {
        if (peekType() == TokenType::Identifier)
        {
            if (peek().getContent() == expected)
            {
                return read();
            }
        }

        if (!_isRecovering)
        {
            _sink.diagnose(peekLoc(), fiddle::Diagnostics::unexpected, peekType(), expected);
        }
        else
        {
            // TODO: need to skip until we see what we expected...
            _sink.diagnose(SourceLoc(), fiddle::Diagnostics::internalError);
        }

        return TokenWithTrivia();
    }

    bool advanceIf(TokenType type)
    {
        if (peekType() == type)
        {
            read();
            return true;
        }

        return false;
    }

    bool advanceIf(char const* name)
    {
        if (peekType() == TokenType::Identifier)
        {
            if (peek().getContent() == name)
            {
                read();
                return true;
            }
        }

        return false;
    }

    RefPtr<Expr> parseCppSimpleExpr()
    {
        switch (peekType())
        {
        case TokenType::Identifier:
            {
                auto nameToken = expect(TokenType::Identifier);
                return new NameExpr(nameToken);
            }
            break;

        case TokenType::IntegerLiteral:
            {
                auto token = read();
                return new LiteralExpr(token);
            }
            break;

        case TokenType::LParent:
            {
                expect(TokenType::LParent);
                auto inner = parseCppExpr();
                expect(TokenType::RParent);

                // TODO: handle a cast, in the case that the lookahead
                // implies we should parse one...
                switch (peekType())
                {
                case TokenType::Identifier:
                case TokenType::LParent:
                    {
                        auto arg = parseCppExpr();
                        return inner;
                    }
                    break;

                default:
                    return inner;
                }
            }
            break;

        default:
            expect(TokenType::Identifier);
            _sink.diagnose(SourceLoc(), fiddle::Diagnostics::internalError);
            return nullptr;
        }
        return nullptr;
    }

    RefPtr<Expr> parseCppExpr()
    {
        auto base = parseCppSimpleExpr();
        for (;;)
        {
            switch (peekType())
            {
            default:
                return base;

            case TokenType::OpMul:
                {
                    expect(TokenType::OpMul);
                    switch (peekType())
                    {
                    default:
                        // treat as introducting a pointer type
                        return base;
                    }
                }
                break;

            case TokenType::Scope:
                {
                    expect(TokenType::Scope);
                    auto memberName = expect(TokenType::Identifier);
                    base = new StaticMemberRef(base, memberName);
                }
                break;
            case TokenType::LParent:
                {
                    // TODO: actually parse this!
                    readBalanced();
                }
                break;

            case TokenType::OpLess:
                {
                    auto specialize = RefPtr(new SpecializeExpr());
                    specialize->base = base;

                    // Okay, we have a template application here.
                    expect(TokenType::OpLess);
                    specialize->args = parseCppTemplateArgs();
                    parseGenericCloser();

                    base = specialize;
                }
                break;
            }
        }
    }

    RefPtr<Expr> parseCppSimpleTypeSpecififer()
    {

        switch (peekType())
        {
        case TokenType::Identifier:
            {
                auto nameToken = expect(TokenType::Identifier);
                return new NameExpr(nameToken);
            }
            break;

        default:
            expect(TokenType::Identifier);
            _sink.diagnose(SourceLoc(), fiddle::Diagnostics::internalError);
            return nullptr;
        }
    }

    List<RefPtr<Expr>> parseCppTemplateArgs()
    {
        List<RefPtr<Expr>> args;
        for (;;)
        {
            switch (peekType())
            {
            case TokenType::OpGeq:
            case TokenType::OpGreater:
            case TokenType::OpRsh:
            case TokenType::EndOfFile:
                return args;
            }

            auto arg = parseCppExpr();
            if (arg)
                args.add(arg);

            if (!advanceIf(TokenType::Comma))
                return args;
        }
    }

    void parseGenericCloser()
    {
        if (advanceIf(TokenType::OpGreater))
            return;

        if (peekType() == TokenType::OpRsh)
        {
            peek().setType(TokenType::OpGreater);

            return;
        }

        expect(TokenType::OpGreater);
    }

    RefPtr<Expr> parseCppTypeSpecifier()
    {
        auto result = parseCppSimpleTypeSpecififer();
        for (;;)
        {
            switch (peekType())
            {
            default:
                return result;

            case TokenType::Scope:
                {
                    expect(TokenType::Scope);
                    auto memberName = expect(TokenType::Identifier);
                    auto memberRef = RefPtr(new StaticMemberRef(result, memberName));
                    result = memberRef;
                }
                break;

            case TokenType::OpLess:
                {
                    auto specialize = RefPtr(new SpecializeExpr());
                    specialize->base = result;

                    // Okay, we have a template application here.
                    expect(TokenType::OpLess);
                    specialize->args = parseCppTemplateArgs();
                    parseGenericCloser();

                    result = specialize;
                }
                break;
            }
        }
    }

    struct UnwrappedDeclarator
    {
        RefPtr<Expr> type;
        TokenWithTrivia nameToken;
    };

    UnwrappedDeclarator unwrapDeclarator(RefPtr<Declarator> declarator, RefPtr<Expr> type)
    {
        if (!declarator)
        {
            UnwrappedDeclarator result;
            result.type = type;
            return result;
        }

        if (auto ptrDeclarator = as<PtrDeclarator>(declarator))
        {
            return unwrapDeclarator(ptrDeclarator->base, new PtrType(type));
        }
        else if (auto nameDeclarator = as<NameDeclarator>(declarator))
        {
            UnwrappedDeclarator result;
            result.type = type;
            result.nameToken = nameDeclarator->nameToken;
            return result;
        }
        else
        {
            _sink.diagnose(SourceLoc(), Diagnostics::unexpected, "declarator type", "known");
            return UnwrappedDeclarator();
        }
    }

    RefPtr<Expr> parseCppType()
    {
        auto typeSpecifier = parseCppTypeSpecifier();
        auto declarator = parseCppDeclarator();
        return unwrapDeclarator(declarator, typeSpecifier).type;
    }

    RefPtr<Expr> parseCppBase()
    {
        // TODO: allow `private` and `protected`
        // TODO: insert a default `public` keyword, if one is missing...
        advanceIf("public");
        return parseCppType();
    }

    void parseCppAggTypeDecl(RefPtr<AggTypeDecl> decl)
    {
        decl->mode = Mode::Cpp;

        // read the type name
        decl->nameToken = expect(TokenType::Identifier);

        // Read the bases clause.
        //
        // TODO: handle multiple bases...
        //
        if (advanceIf(TokenType::Colon))
        {
            decl->directBaseType = parseCppBase();
        }

        expect(TokenType::LBrace);
        addDecl(decl);
        WithParentDecl withParent(this, decl);

        // We expect any `FIDDLE()`-marked aggregate type
        // declaration to start with a `FIDDLE(...)` invocation,
        // so that there is a suitable insertion point for
        // the expansion step.
        //
        {
            auto saved = _cursor;
            bool found = peekFiddleEllipsisInvocation();
            _cursor = saved;
            if (!found)
            {
                _sink.diagnose(
                    peekLoc(),
                    fiddle::Diagnostics::expectedFiddleEllipsisInvocation,
                    decl->nameToken.getContent());
            }
        }

        parseCppDecls(decl);
        expect(TokenType::RBrace);
    }

    bool peekFiddleEllipsisInvocation()
    {
        if (!advanceIf("FIDDLE"))
            return false;

        if (!advanceIf(TokenType::LParent))
            return false;

        if (!advanceIf(TokenType::Ellipsis))
            return false;

        return true;
    }

    RefPtr<Declarator> parseCppSimpleDeclarator()
    {
        switch (peekType())
        {
        case TokenType::Identifier:
            {
                auto nameToken = expect(TokenType::Identifier);
                return RefPtr(new NameDeclarator(nameToken));
            }

        default:
            return nullptr;
        }
    }

    RefPtr<Declarator> parseCppPostfixDeclarator()
    {
        auto result = parseCppSimpleDeclarator();
        for (;;)
        {
            switch (peekType())
            {
            default:
                return result;

            case TokenType::LBracket:
                readBalanced();
                return result;
            }
        }
        return result;
    }

    RefPtr<Declarator> parseCppDeclarator()
    {
        advanceIf("const");

        if (advanceIf(TokenType::OpMul))
        {
            auto base = parseCppDeclarator();
            return RefPtr(new PtrDeclarator(base));
        }
        else
        {
            return parseCppPostfixDeclarator();
        }
    }

    void parseCppDeclaratorBasedDecl(List<RefPtr<ModifierNode>> const& fiddleModifiers)
    {
        auto typeSpecifier = parseCppTypeSpecifier();
        auto declarator = parseCppDeclarator();

        auto unwrapped = unwrapDeclarator(declarator, typeSpecifier);

        auto varDecl = RefPtr(new VarDecl());
        varDecl->nameToken = unwrapped.nameToken;
        varDecl->type = unwrapped.type;
        addDecl(varDecl);

        if (advanceIf(TokenType::OpAssign))
        {
            varDecl->initExpr = parseCppExpr();
        }
        expect(TokenType::Semicolon);
    }

    void parseNativeDeclaration(List<RefPtr<ModifierNode>> const& fiddleModifiers)
    {
        auto keyword = peek();
        if (advanceIf("namespace"))
        {
            RefPtr<PhysicalNamespaceDecl> namespaceDecl = new PhysicalNamespaceDecl();
            namespaceDecl->modifiers = fiddleModifiers;


            // read the namespace name
            namespaceDecl->nameToken = expect(TokenType::Identifier);

            expect(TokenType::LBrace);

            addDecl(namespaceDecl);
            WithParentDecl withNamespace(this, namespaceDecl);

            parseCppDecls(namespaceDecl);

            expect(TokenType::RBrace);
        }
        else if (advanceIf("class"))
        {
            auto decl = RefPtr(new ClassDecl());
            decl->modifiers = fiddleModifiers;
            parseCppAggTypeDecl(decl);
        }
        else if (advanceIf("struct"))
        {
            auto decl = RefPtr(new StructDecl());
            decl->modifiers = fiddleModifiers;
            parseCppAggTypeDecl(decl);
        }
        else if (peekType() == TokenType::Identifier)
        {
            // try to parse a declarator-based declaration
            // (which for now is probably a field);
            //
            parseCppDeclaratorBasedDecl(fiddleModifiers);
        }
        else
        {
            _sink.diagnose(peekLoc(), fiddle::Diagnostics::unexpected, peekType(), "OTHER");
            _sink.diagnose(SourceLoc(), fiddle::Diagnostics::internalError);
        }
    }

    List<RefPtr<ModifierNode>> parseFiddleModifiers()
    {
        List<RefPtr<ModifierNode>> modifiers;

        for (;;)
        {
            switch (peekType())
            {
            default:
                return modifiers;

            case TokenType::Identifier:
                break;
            }

            if (advanceIf("abstract"))
            {
                modifiers.add(new AbstractModifier());
            }
            else if (advanceIf("hidden"))
            {
                modifiers.add(new HiddenModifier());
            }
            else
            {
                return modifiers;
            }
        }

        return modifiers;
    }

    RefPtr<Expr> parseFiddlePrimaryExpr()
    {
        switch (peekType())
        {
        case TokenType::Identifier:
            return new NameExpr(read());

        case TokenType::LParent:
            {
                expect(TokenType::LParent);
                auto expr = parseFiddleExpr();
                expect(TokenType::RParent);
                return expr;
            }

        default:
            expect(TokenType::Identifier);
            return nullptr;
        }
    }

    List<RefPtr<Arg>> parseFiddleArgs()
    {
        List<RefPtr<Arg>> args;
        for (;;)
        {
            switch (peekType())
            {
            case TokenType::RBrace:
            case TokenType::RBracket:
            case TokenType::RParent:
            case TokenType::EndOfFile:
                return args;

            default:
                break;
            }

            auto arg = parseFiddleExpr();
            args.add(arg);

            if (!advanceIf(TokenType::Comma))
                return args;
        }
    }

    RefPtr<Expr> parseFiddlePostifxExpr()
    {
        auto result = parseFiddlePrimaryExpr();

        for (;;)
        {
            switch (peekType())
            {
            default:
                return result;

            case TokenType::Dot:
                {
                    expect(TokenType::Dot);
                    auto memberName = expect(TokenType::Identifier);

                    result = new MemberExpr(result, memberName);
                }
                break;

            case TokenType::LParent:
                {
                    expect(TokenType::LParent);
                    auto args = parseFiddleArgs();
                    expect(TokenType::RParent);

                    result = new CallExpr(result, args);
                }
                break;
            }
        }
    }
    RefPtr<Expr> parseFiddleExpr() { return parseFiddlePostifxExpr(); }

    RefPtr<Expr> parseFiddleTypeExpr() { return parseFiddleExpr(); }

    void parseFiddleAggTypeDecl(RefPtr<AggTypeDecl> decl)
    {
        decl->mode = Mode::Fiddle;

        // read the type name
        decl->nameToken = expect(TokenType::Identifier);

        // Read the bases clause.
        if (advanceIf(TokenType::Colon))
        {
            decl->directBaseType = parseFiddleTypeExpr();
        }

        addDecl(decl);
        WithParentDecl withParent(this, decl);

        if (advanceIf(TokenType::LBrace))
        {
            parseOptionalFiddleModeDecls();

            expect(TokenType::RBrace);
        }
        else
        {
            expect(TokenType::Semicolon);
        }
    }

    void parseFiddleModeDecl(List<RefPtr<ModifierNode>> modifiers)
    {
        if (advanceIf("class"))
        {
            auto decl = RefPtr(new ClassDecl());
            decl->modifiers = modifiers;
            parseFiddleAggTypeDecl(decl);
        }
        else
        {
            _sink.diagnose(
                peekLoc(),
                Diagnostics::unexpected,
                peekType(),
                "fiddle-mode declaration");
        }
    }

    void parseFiddleModeDecl()
    {
        auto modifiers = parseFiddleModifiers();
        parseFiddleModeDecl(modifiers);
    }

    void parseOptionalFiddleModeDecls()
    {
        for (;;)
        {
            switch (peekType())
            {
            case TokenType::RParent:
            case TokenType::RBrace:
            case TokenType::RBracket:
            case TokenType::EndOfFile:
                return;
            }

            parseFiddleModeDecl();
        }
    }

    void parseFiddleModeDecls(List<RefPtr<ModifierNode>> modifiers)
    {
        parseFiddleModeDecl(modifiers);
        parseOptionalFiddleModeDecls();
    }

    void parseFiddleNode()
    {
        auto fiddleToken = expect("FIDDLE");

        // We will capture the token at this invocation site,
        // because later on we will generate a macro that
        // this invocation will expand into.
        //
        auto fiddleMacroInvocation = RefPtr(new FiddleMacroInvocation());
        fiddleMacroInvocation->fiddleToken = fiddleToken;
        addDecl(fiddleMacroInvocation);

        // The `FIDDLE` keyword can be followed by parentheses around a bunch of
        // fiddle-mode modifiers.
        List<RefPtr<ModifierNode>> fiddleModifiers;
        if (advanceIf(TokenType::LParent))
        {
            if (advanceIf(TokenType::Ellipsis))
            {
                // A `FIDDLE(...)` invocation is a hook for
                // our expansion step to insert the generated
                // declarations that go into the body of
                // the parent declaration.

                fiddleMacroInvocation->node = _currentParentDecl;

                expect(TokenType::RParent);
                return;
            }


            // We start off by parsing optional modifiers
            fiddleModifiers = parseFiddleModifiers();

            if (peekType() != TokenType::RParent)
            {
                // In this case we are expecting a fiddle-mode declaration
                // to appear, in which case we will allow any number of full
                // fiddle-mode declarations, but won't expect a C++-mode
                // declaration to follow.

                // TODO: We should associate these declarations
                // as children of the `FiddleMacroInvocation`,
                // so that they can be emitted as part of its
                // expansion (if we decide to make more use
                // of the `FIDDLE()` approach...).

                parseFiddleModeDecls(fiddleModifiers);
                expect(TokenType::RParent);
                return;
            }
            expect(TokenType::RParent);
        }
        else
        {
            // TODO: diagnose this!
        }

        // Any tokens from here on are expected to be in C++-mode

        parseNativeDeclaration(fiddleModifiers);
    }

    void addDecl(ContainerDecl* parentDecl, Decl* memberDecl)
    {
        if (!memberDecl)
            return;

        parentDecl->members.add(memberDecl);

        auto physicalParent = as<PhysicalContainerDecl>(parentDecl);
        if (!physicalParent)
            return;

        auto logicalParent = physicalParent->logicalVersion;
        if (!logicalParent)
            return;

        if (auto physicalNamespace = as<PhysicalNamespaceDecl>(memberDecl))
        {
            auto namespaceName = physicalNamespace->nameToken.getContent();
            auto logicalNamespace = findDecl<LogicalNamespace>(logicalParent, namespaceName);
            if (!logicalNamespace)
            {
                logicalNamespace = new LogicalNamespace();

                logicalNamespace->nameToken = physicalNamespace->nameToken;

                logicalParent->members.add(logicalNamespace);
                logicalParent->mapNameToMember.add(namespaceName, logicalNamespace);
            }
            physicalNamespace->logicalVersion = logicalNamespace;
        }
        else
        {
            logicalParent->members.add(memberDecl);
        }
    }

    void addDecl(RefPtr<Decl> decl) { addDecl(_currentParentDecl, decl); }

    void parseCppDecls(RefPtr<ContainerDecl> parentDecl)
    {
        for (;;)
        {
            switch (peekType())
            {
            case TokenType::EndOfFile:
            case TokenType::RBrace:
            case TokenType::RBracket:
            case TokenType::RParent:
                return;

            default:
                break;
            }

            parseCppDecl();
        }
    }

    void readBalanced()
    {
        Count skipCount = read().getSkipCount();
        _cursor = _cursor + skipCount;
    }

    void parseCppDecl()
    {
        // We consume raw tokens until we see something
        // that ought to start a reflected/extracted declaration.
        //
        for (;;)
        {
            switch (peekType())
            {
            default:
                {
                    readBalanced();
                    continue;
                }

            case TokenType::RBrace:
            case TokenType::RBracket:
            case TokenType::RParent:
            case TokenType::EndOfFile:
                return;

            case TokenType::Identifier:
                break;

            case TokenType::Pound:
                // a `#` means we have run into a preprocessor directive
                // (or, somehow, we are already *inside* one...).
                //
                // We don't want to try to intercept anything to do with
                // these lines, so we will read until the next end-of-line.
                //
                read();
                while (!(peek().getToken().flags & TokenFlag::AtStartOfLine))
                {
                    if (peekType() == TokenType::EndOfFile)
                        break;
                    read();
                }
                continue;
            }

            // Okay, we have an identifier, but is its name
            // one that we want to pay attention to?
            //
            //
            auto name = peek().getContent();
            if (name == "FIDDLE")
            {
                // If the `FIDDLE` is the first token we are seeing, then we will
                // start parsing a construct in fiddle-mode:
                //
                parseFiddleNode();
            }
            else
            {
                // If the name isn't one we recognize, then
                // we are just reading raw tokens as usual.
                //
                readBalanced();
                continue;
            }
        }
    }

    RefPtr<SourceUnit> parseSourceUnit()
    {
        RefPtr<SourceUnit> sourceUnit = new SourceUnit();
        sourceUnit->logicalVersion = _module;

        WithParentDecl withSourceUnit(this, sourceUnit);
        while (_cursor != _end)
        {
            parseCppDecl();

            switch (peekType())
            {
            default:
                break;

            case TokenType::RBrace:
            case TokenType::RBracket:
            case TokenType::RParent:
            case TokenType::EndOfFile:
                read();
                break;
            }
        }
        read();

        return sourceUnit;
    }
};


// Check

struct CheckContext
{
private:
    DiagnosticSink& sink;

public:
    CheckContext(DiagnosticSink& sink)
        : sink(sink)
    {
    }

    void checkModule(LogicalModule* module) { checkMemberDecls(module); }

private:
    struct Scope
    {
    public:
        Scope(ContainerDecl* containerDecl, Scope* outer)
            : containerDecl(containerDecl), outer(outer)
        {
        }

        ContainerDecl* containerDecl = nullptr;
        Scope* outer = nullptr;
    };
    Scope* currentScope = nullptr;

    struct WithScope : Scope
    {
        WithScope(CheckContext* context, ContainerDecl* containerDecl)
            : Scope(containerDecl, context->currentScope)
            , _context(context)
            , _saved(context->currentScope)
        {
            context->currentScope = this;
        }

        ~WithScope() { _context->currentScope = _saved; }

    private:
        CheckContext* _context = nullptr;
        Scope* _saved = nullptr;
    };


    //
    void checkDecl(Decl* decl)
    {
        if (auto aggTypeDecl = as<AggTypeDecl>(decl))
        {
            checkTypeExprInPlace(aggTypeDecl->directBaseType);

            if (auto baseType = aggTypeDecl->directBaseType)
            {
                if (auto baseDeclRef = as<DirectDeclRef>(baseType))
                {
                    auto baseDecl = baseDeclRef->decl;
                    if (auto baseAggTypeDecl = as<AggTypeDecl>(baseDecl))
                    {
                        baseAggTypeDecl->directSubTypeDecls.add(aggTypeDecl);
                    }
                }
            }

            checkMemberDecls(aggTypeDecl);
        }
        else if (auto namespaceDecl = as<LogicalNamespace>(decl))
        {
            checkMemberDecls(namespaceDecl);
        }
        else if (auto varDecl = as<VarDecl>(decl))
        {
            // Note: for now we aren't trying to check the type
            // or the initial-value expression of a field.
        }
        else if (as<FiddleMacroInvocation>(decl))
        {
        }
        else
        {
            sink.diagnose(SourceLoc(), Diagnostics::unexpected, "case in checkDecl", "known type");
        }
    }

    void checkMemberDecls(ContainerDecl* containerDecl)
    {
        WithScope moduleScope(this, containerDecl);
        for (auto memberDecl : containerDecl->members)
        {
            checkDecl(memberDecl);
        }
    }

    void checkTypeExprInPlace(RefPtr<Expr>& ioTypeExpr)
    {
        if (!ioTypeExpr)
            return;
        ioTypeExpr = checkTypeExpr(ioTypeExpr);
    }

    RefPtr<Expr> checkTypeExpr(Expr* expr) { return checkExpr(expr); }

    RefPtr<Expr> checkExpr(Expr* expr)
    {
        if (auto nameExpr = as<NameExpr>(expr))
        {
            return lookUp(nameExpr->nameToken.getContent());
        }
        else
        {
            sink.diagnose(SourceLoc(), Diagnostics::unexpected, "case in checkExpr", "known type");
            return nullptr;
        }
    }

    RefPtr<Expr> lookUp(UnownedStringSlice const& name)
    {
        for (auto scope = currentScope; scope; scope = scope->outer)
        {
            auto containerDecl = scope->containerDecl;
            // TODO: accelerate lookup with a dictionary on the container...
            for (auto memberDecl : containerDecl->members)
            {
                if (memberDecl->nameToken.getContent() == name)
                {
                    return new DirectDeclRef(memberDecl);
                }
            }
        }
        sink.diagnose(SourceLoc(), Diagnostics::undefinedIdentifier, name);
        return nullptr;
    }
};


// Emit

struct EmitContext
{
private:
    SourceManager& _sourceManager;
    RefPtr<LogicalModule> _module;
    StringBuilder& _builder;

public:
    EmitContext(
        StringBuilder& builder,
        DiagnosticSink& sink,
        SourceManager& sourceManager,
        LogicalModule* module)
        : _builder(builder), _sourceManager(sourceManager), _module(module)
    {
        SLANG_UNUSED(sink);
    }

    void emitMacrosRec(Decl* decl)
    {
        emitMacrosForDecl(decl);
        if (auto container = as<ContainerDecl>(decl))
        {
            for (auto member : container->members)
                emitMacrosRec(member);
        }
    }

private:
    void emitMacrosForDecl(Decl* decl)
    {
        if (auto fiddleMacroInvocation = as<FiddleMacroInvocation>(decl))
        {
            emitMacroForFiddleInvocation(fiddleMacroInvocation);
        }
        else
        {
            // do nothing with most decls
        }
    }

    void emitMacroForFiddleInvocation(FiddleMacroInvocation* fiddleInvocation)
    {
        SourceLoc loc = fiddleInvocation->fiddleToken.getLoc();
        auto humaneLoc = _sourceManager.getHumaneLoc(loc);
        auto lineNumber = humaneLoc.line;

#define MACRO_LINE_ENDING " \\\n"

        // Un-define the old `FIDDLE_#` macro for the
        // given line number, since this file might
        // be pulling in another generated header
        // via one of its dependencies.
        //
        _builder.append("#ifdef FIDDLE_");
        _builder.append(lineNumber);
        _builder.append("\n#undef FIDDLE_");
        _builder.append(lineNumber);
        _builder.append("\n#endif\n");

        _builder.append("#define FIDDLE_");
        _builder.append(lineNumber);
        _builder.append("(...)");
        _builder.append(MACRO_LINE_ENDING);

        auto decl = as<AggTypeDecl>(fiddleInvocation->node);
        if (decl)
        {
            if (auto base = decl->directBaseType)
            {
                _builder.append("private: typedef ");
                emitTypedDecl(base, "Super");
                _builder.append(";" MACRO_LINE_ENDING);
            }

            if (decl->isSubTypeOf("NodeBase"))
            {
                _builder.append("friend class ::Slang::ASTBuilder;" MACRO_LINE_ENDING);
                _builder.append("friend struct ::Slang::SyntaxClassInfo;" MACRO_LINE_ENDING);

                _builder.append("public: static const ::Slang::SyntaxClassInfo "
                                "kSyntaxClassInfo;" MACRO_LINE_ENDING);

                _builder.append("public: static constexpr ASTNodeType kType = ASTNodeType::");
                _builder.append(decl->nameToken.getContent());
                _builder.append(";" MACRO_LINE_ENDING);

                _builder.append("public: ");
                _builder.append(decl->nameToken.getContent());
                _builder.append("() {}" MACRO_LINE_ENDING);
            }
            _builder.append("public:" MACRO_LINE_ENDING);
        }
        _builder.append("/* end */\n\n");
    }

    void emitTypedDecl(Expr* expr, const char* name)
    {
        if (auto declRef = as<DirectDeclRef>(expr))
        {
            _builder.append(declRef->decl->nameToken.getContent());
            _builder.append(" ");
            _builder.append(name);
        }
    }

#if 0
        void emitLineDirective(Token const& lexeme)
        {
            SourceLoc loc = lexeme.getLoc();
            auto humaneLoc = _sourceManager.getHumaneLoc(loc);
            _builder.append("\n#line ");
            _builder.append(humaneLoc.line);
            _builder.append(" \"");
            for (auto c : humaneLoc.pathInfo.getName())
            {
                if (c == '\\') _builder.append("\\\\");
                else _builder.append(c);
            }
            _builder.append("\"\n");
        }

        void emitLineDirective(TokenWithTrivia const& token)
        {
            if (token.getLeadingTrivia().getCount() != 0)
                emitLineDirective(token.getLeadingTrivia()[0]);
            else
                emitLineDirective(token.getToken());
        }

        void emitLineDirective(RawNode* node)
        {
            emitLineDirective(node->tokens[0]);
        }


        void emitTrivia(List<Token> const& trivia)
        {
            for (auto trivium : trivia)
                _builder.append(trivium.getContent());
        }

        void emitRawNode(RawNode* rawNode)
        {
            for (auto token : rawNode->tokens)
            {
                emitTrivia(token.getLeadingTrivia());
                _builder.append(token.getContent());
                emitTrivia(token.getTrailingTrivia());
            }
        }

        void emitTopLevelNode(Decl* node)
        {
            if (!node)
                return;

            if (node->findModifier<HiddenModifier>())
                return;

            if (auto rawNode = as<RawNode>(node))
            {
                // TODO: should emit a `#line` to point back to
                // the original source file...
                emitLineDirective(rawNode);

                emitRawNode(rawNode);
            }
            else if (auto decl = as<PhysicalNamespaceDecl>(node))
            {
                for (auto child : decl->members)
                {
                    emitTopLevelNode(child);
                }
            }
            else if (auto decl = as<AggTypeDecl>(node))
            {
                emitExtraMembersForAggTypeDecl(decl);

                for (auto child : decl->members)
                {
                    emitTopLevelNode(child);
                }
            }
            else if (auto varDecl = as<VarDecl>(node))
            {
                // Note: nothing to be done here...
            }
            else
            {
                _sink.diagnose(SourceLoc(), fiddle::Diagnostics::unexpected, "emitTopLevelNode", "unhandled case");
            }
        }

        void emitSourceUnit(SourceUnit* sourceUnit)
        {
            for (auto node : sourceUnit->members)
            {
                emitTopLevelNode(node);
            }
        }

    private:
#endif
};


Decl* findDecl_(ContainerDecl* outerDecl, UnownedStringSlice const& name)
{
    for (auto memberDecl : outerDecl->members)
    {
        if (memberDecl->nameToken.getContent() == name)
            return memberDecl;
    }
    return nullptr;
}

bool AggTypeDecl::isSubTypeOf(char const* name)
{
    Decl* decl = this;
    while (decl)
    {
        if (decl->nameToken.getContent() == UnownedTerminatedStringSlice(name))
        {
            return true;
        }

        auto aggType = as<AggTypeDecl>(decl);
        if (!aggType)
            break;

        auto baseTypeExpr = aggType->directBaseType;
        if (!baseTypeExpr)
            break;

        auto declRef = as<DirectDeclRef>(baseTypeExpr);
        if (!declRef)
            break;

        decl = declRef->decl;
    }
    return false;
}

bool isTrivia(TokenType lexemeType)
{
    switch (lexemeType)
    {
    default:
        return false;

    case TokenType::LineComment:
    case TokenType::BlockComment:
    case TokenType::NewLine:
    case TokenType::WhiteSpace:
        return true;
    }
}

List<TokenWithTrivia> collectTokensWithTrivia(TokenList const& lexemes)
{
    TokenReader reader(lexemes);

    List<TokenWithTrivia> allTokensWithTrivia;
    for (;;)
    {
        RefPtr<TokenWithTriviaNode> currentTokenWithTriviaNode = new TokenWithTriviaNode();
        TokenWithTrivia currentTokenWithTrivia = currentTokenWithTriviaNode;
        allTokensWithTrivia.add(currentTokenWithTrivia);

        while (isTrivia(reader.peekTokenType()))
        {
            auto trivia = reader.advanceToken();
            currentTokenWithTriviaNode->leadingTrivia.add(trivia);
        }

        auto token = reader.advanceToken();
        currentTokenWithTriviaNode->token = token;

        if (token.type == TokenType::EndOfFile)
            return allTokensWithTrivia;

        while (isTrivia(reader.peekTokenType()))
        {
            auto trivia = reader.advanceToken();
            currentTokenWithTriviaNode->trailingTrivia.add(trivia);

            if (trivia.type == TokenType::NewLine)
                break;
        }
    }
}

void readTokenTree(List<TokenWithTrivia> const& tokens, Index& ioIndex);

void readBalancedToken(List<TokenWithTrivia> const& tokens, Index& ioIndex, TokenType closeType)
{
    auto open = tokens[ioIndex++];
    auto openNode = (TokenWithTriviaNode*)open;

    Index startIndex = ioIndex;
    for (;;)
    {
        auto token = tokens[ioIndex];
        if (token.getType() == closeType)
        {
            ioIndex++;
            break;
        }

        switch (token.getType())
        {
        default:
            readTokenTree(tokens, ioIndex);
            continue;

        case TokenType::RBrace:
        case TokenType::RBracket:
        case TokenType::RParent:
        case TokenType::EndOfFile:
            break;
        }
        break;
    }
    openNode->skipCount = ioIndex - startIndex;
}

void readTokenTree(List<TokenWithTrivia> const& tokens, Index& ioIndex)
{
    switch (tokens[ioIndex].getType())
    {
    default:
        ioIndex++;
        return;

    case TokenType::LBrace:
        return readBalancedToken(tokens, ioIndex, TokenType::RBrace);

    case TokenType::LBracket:
        return readBalancedToken(tokens, ioIndex, TokenType::RBracket);

    case TokenType::LParent:
        return readBalancedToken(tokens, ioIndex, TokenType::RParent);
    }
}

void matchBalancedTokens(List<TokenWithTrivia> tokens)
{
    Index index = 0;
    for (;;)
    {
        auto& token = tokens[index];
        switch (token.getType())
        {
        case TokenType::EndOfFile:
            return;

        default:
            readTokenTree(tokens, index);
            break;

        case TokenType::RBrace:
        case TokenType::RBracket:
        case TokenType::RParent:
            // error!!!
            index++;
            break;
        }
    }
}

bool findOutputFileIncludeDirective(List<TokenWithTrivia> tokens, String outputFileName)
{
    auto cursor = tokens.begin();
    auto end = tokens.end() - 1;

    while (cursor != end)
    {
        if (cursor->getType() != TokenType::Pound)
        {
            cursor++;
            continue;
        }
        cursor++;

        if (cursor->getContent() != "include")
            continue;
        cursor++;

        if (cursor->getType() != TokenType::StringLiteral)
            continue;

        auto includedFileName = getStringLiteralTokenValue(cursor->getToken());
        if (includedFileName == outputFileName)
            return true;
    }
    return false;
}

RefPtr<SourceUnit> parseSourceUnit(
    SourceView* inputSourceView,
    LogicalModule* logicalModule,
    RootNamePool* rootNamePool,
    DiagnosticSink* sink,
    SourceManager* sourceManager,
    String outputFileName)
{
    Lexer lexer;
    NamePool namePool;
    namePool.setRootNamePool(rootNamePool);

    // We suppress any diagnostics that might get emitted during lexing,
    // so that we can ignore any files we don't understand.
    //
    DiagnosticSink lexerSink;
    lexer.initialize(inputSourceView, &lexerSink, &namePool, sourceManager->getMemoryArena());

    auto inputTokens = lexer.lexAllTokens();
    auto tokensWithTrivia = collectTokensWithTrivia(inputTokens);
    matchBalancedTokens(tokensWithTrivia);

    Parser parser(*sink, tokensWithTrivia, logicalModule);
    auto sourceUnit = parser.parseSourceUnit();

    // As a quick validation check, if the source file had
    // any `FIDDLE()` invocations in it, then we check to
    // make sure it also has a `#include` of the corresponding
    // output file name...
    if (hasAnyFiddleInvocations(sourceUnit))
    {
        if (!findOutputFileIncludeDirective(tokensWithTrivia, outputFileName))
        {
            sink->diagnose(
                inputSourceView->getRange().begin,
                fiddle::Diagnostics::expectedIncludeOfOutputHeader,
                outputFileName);
        }
    }

    return sourceUnit;
}

void push(lua_State* L, Val* val);

void push(lua_State* L, UnownedStringSlice const& text)
{
    lua_pushlstring(L, text.begin(), text.getLength());
}

template<typename T>
void push(lua_State* L, List<T> const& values)
{
    // Note: Lua tables are naturally indexed starting at 1.
    Index nextIndex = 1;
    lua_newtable(L);
    for (auto value : values)
    {
        Index index = nextIndex++;

        push(L, value);
        lua_seti(L, -2, index);
    }
}

void getAllSubclasses(AggTypeDecl* decl, List<RefPtr<AggTypeDecl>>& ioSubclasses)
{
    ioSubclasses.add(decl);
    for (auto subclass : decl->directSubTypeDecls)
        getAllSubclasses(subclass, ioSubclasses);
}

List<RefPtr<AggTypeDecl>> getAllSubclasses(AggTypeDecl* decl)
{
    List<RefPtr<AggTypeDecl>> result;
    getAllSubclasses(decl, result);
    return result;
}

int _toStringVal(lua_State* L)
{
    Val* val = (Val*)lua_touserdata(L, 1);

    if (auto directDeclRef = as<DirectDeclRef>(val))
    {
        val = directDeclRef->decl;
    }

    if (auto decl = as<Decl>(val))
    {
        push(L, decl->nameToken.getContent());
        return 1;
    }

    lua_pushfstring(L, "fiddle::Val @ 0x%p", val);
    return 1;
}

int _indexVal(lua_State* L)
{
    Val* val = (Val*)lua_touserdata(L, 1);
    char const* name = lua_tostring(L, 2);

    if (auto containerDecl = as<ContainerDecl>(val))
    {
        for (auto m : containerDecl->members)
        {
            if (m->nameToken.getContent() == UnownedTerminatedStringSlice(name))
            {
                push(L, m);
                return 1;
            }
        }
    }

    if (auto classDecl = as<ClassDecl>(val))
    {
        if (strcmp(name, "subclasses") == 0)
        {
            auto value = getAllSubclasses(classDecl);
            push(L, value);
            return 1;
        }

        if (strcmp(name, "directSuperClass") == 0)
        {
            push(L, classDecl->directBaseType);
            return 1;
        }

        if (strcmp(name, "directFields") == 0)
        {
            List<RefPtr<Decl>> fields;
            for (auto m : classDecl->members)
            {
                if (auto f = as<VarDecl>(m))
                    fields.add(f);
            }
            push(L, fields);
            return 1;
        }
    }

    if (auto decl = as<Decl>(val))
    {
        if (strcmp(name, "isAbstract") == 0)
        {
            lua_pushboolean(L, decl->findModifier<AbstractModifier>() != nullptr);
            return 1;
        }
    }

    return 0;
}

void push(lua_State* L, Val* val)
{
    if (!val)
    {
        lua_pushnil(L);
        return;
    }

    lua_pushlightuserdata(L, val);
    if (luaL_newmetatable(L, "fiddle::Val"))
    {
        lua_pushcfunction(L, &_indexVal);
        lua_setfield(L, -2, "__index");

        lua_pushcfunction(L, &_toStringVal);
        lua_setfield(L, -2, "__tostring");
    }
    lua_setmetatable(L, -2);
}

void registerValWithScript(String name, Val* val)
{
    auto L = getLuaState();

    push(L, val);
    lua_setglobal(L, name.getBuffer());
}


void registerScrapedStuffWithScript(LogicalModule* logicalModule)
{
    for (auto decl : logicalModule->members)
    {
        if (!decl->nameToken)
            continue;

        registerValWithScript(decl->nameToken.getContent(), decl);
    }
}

bool _hasAnyFiddleInvocationsRec(Decl* decl)
{
    if (as<FiddleMacroInvocation>(decl))
        return true;

    if (auto container = as<ContainerDecl>(decl))
    {
        for (auto m : container->members)
        {
            if (_hasAnyFiddleInvocationsRec(m))
                return true;
        }
    }
    return false;
}

bool hasAnyFiddleInvocations(SourceUnit* sourceUnit)
{
    return _hasAnyFiddleInvocationsRec(sourceUnit);
}

void checkModule(LogicalModule* module, DiagnosticSink* sink)
{
    CheckContext context(*sink);
    context.checkModule(module);
}


void emitSourceUnitMacros(
    SourceUnit* sourceUnit,
    StringBuilder& builder,
    DiagnosticSink* sink,
    SourceManager* sourceManager,
    LogicalModule* logicalModule)
{
    // The basic task here is to find each of the
    // `FIDDLE()` macro invocations, and for each
    // of them produce a matching definition that
    // will be used as the expansion of that one
    //

    EmitContext context(builder, *sink, *sourceManager, logicalModule);
    context.emitMacrosRec(sourceUnit);
}

} // namespace fiddle
