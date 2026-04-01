// slang-preprocessor.cpp
#include "slang-preprocessor.h"

// This file implements a C/C++-style preprocessor. While it does not aim for 100%
// compatibility with the preprocessor for those languages, it does strive to provide
// the same semantics in most cases users will care about around macros, toking pasting, etc.
//
// The main conceptual difference from a fully C-compatible preprocessor is that
// we do *not* implement distinct tokenization/lexing rules for the preprocessor and
// later compiler stages. Instead, our preprocessor uses the same lexer as the rest
// of the compiler, and operates as logical transformation from one stream of tokens
// to another.

#include "../compiler-core/slang-lexer.h"
#include "slang-compiler.h"
#include "slang-diagnostics.h"

#include <assert.h>

namespace Slang
{

//
// PreprocessorHandler
//

// The `PreprocessorHandler` interface allows other layers of the compielr to intercept
// important events during preprocessing. The following are the default (empty) implementations
// of the callbacks.

void PreprocessorHandler::handleEndOfTranslationUnit(Preprocessor* preprocessor)
{
    SLANG_UNUSED(preprocessor);
}

void PreprocessorHandler::handleFileDependency(SourceFile* sourceFile)
{
    SLANG_UNUSED(sourceFile);
}

// In order to simplify the naming scheme, we will nest the implementaiton of the
// preprocessor under an additional namesspace, so taht we can have, e.g.,
// `MacroDefinition` instead of `PreprocessorMacroDefinition`.
//
namespace preprocessor
{

//
// Forward Declarations
//

struct MacroDefinition;
struct MacroInvocation;

//
// Utility Types
//

/// A preprocessor conditional construct that is currently active.
///
/// This type handles preprocessor conditional structures like
/// `#if` / `#elif` / `#endif`. A single top-level input file
/// will have some number of "active" conditionals at one time,
/// based on the nesting depth of those conditional structures.
///
/// Each conditional may be in a distinct state, which decides
/// whether tokens should be skipped or not.
///
struct Conditional
{
    /// A state that a preprocessor conditional can be in.
    ///
    /// The state of a conditional depends both on what directives
    /// have been encountered so far (e.g., just an `#if`, or an
    /// `#if` and then an `#else`), as well as what the value
    /// of any conditions related to those directives have been.
    ///
    enum class State
    {
        /// Indicates that this conditional construct has not yet encountered a branch with a `true`
        /// condition.
        ///
        /// The preprocessor should skip tokens, but should keep scanning and evaluating branch
        /// conditions.
        Before,

        /// Indicates that this conditional construct is nested inside the branch with a `true`
        /// condition
        ///
        /// The preprocessor should not skip tokens, and should not bother evaluating subsequent
        /// branch conditions.
        During,

        /// Indicates that this conditional has already seen the branch with a `true` condition
        ///
        /// The preprocessor should skip tokens, and should not bother evaluating subsequent branch
        /// conditions.
        After,
    };

    /// The next outer conditional in the current input file, or NULL if this is the outer-most
    /// conditional.
    Conditional* parent;

    /// The token that started the conditional (e.g., an `#if` or `#ifdef`)
    Token ifToken;

    /// The `#else` directive token, if one has been seen (otherwise has `TokenType::Unknown`)
    Token elseToken;

    /// The state of the conditional
    State state;
};

/// An environment used for mapping macro names to their definitions during preprocessing.
///
struct Environment
{
    /// The "outer" environment, to be used if lookup in this env fails
    Environment* parent = NULL;

    /// Macros defined in this environment
    Dictionary<Name*, MacroDefinition*> macros;

    /// Clean up the environment, releasing all macros allocated into it
    ~Environment();
};

//
// Input Streams
//

// A fundamental action in the preprocessor is to transform a stream of
// input tokens to produce a stream of output tokens. The term "macro expansion"
// is used to describe two inter-related transformations of this kind:
//
// * Given an invocation of a macro `M`, we can "play back" the tokens in the
//   definition of `M` to produce a stream of tokens, potentially substituting
//   in argument values for parameters, pasting tokens, etc.
//
// * Given an input stream, we can scan its tokens looking for macro invocations,
//   and upon finding them expand those invocations using the first approach
//   outlined here.
//
// In practice, the second kind of expansion needs to abstract over where it
// is reading tokens from: an input file, an existing macro invocation, etc.
// In order to support reading from streams of tokens without knowing their
// exact implementation, we will define an abstract base class for input
// streams.

/// A logical stream of tokens.
struct InputStream
{
    /// Initialize an input stream, and assocaite with a specific `preprocessor`
    InputStream(Preprocessor* preprocessor)
        : m_preprocessor(preprocessor)
    {
    }

    // The two fundamental operations that every input stream must support
    // are reading one token from the stream, and "peeking" one token into
    // the stream to see what will be read next.

    /// Read one token from the input stream
    ///
    /// At the end of the stream should return a token with `TokenType::EndOfFile`.
    ///
    virtual Token readToken() = 0;

    /// Peek at the next token in the input stream
    ///
    /// This function should return whatever `readToken()` will return next.
    ///
    /// At the end of the stream should return a token with `TokenType::EndOfFile`.
    ///
    virtual Token peekToken() = 0;

    // Because different implementations of this abstract base class will
    // store differnet amounts of data, we need a virtual descritor to
    // ensure that we can clean up after them.

    /// Clean up an input stream
    virtual ~InputStream() = default;

    // Based on `peekToken()` we can define a few more utility functions
    // for cases where we only care about certain details of the input.

    /// Peek the type of the next token in the input stream.
    TokenType peekTokenType() { return peekToken().type; }

    /// Peek the location of the next token in the input stream.
    SourceLoc peekLoc() { return peekToken().loc; }

    /// Get the diagnostic sink to use for messages related to this stream
    DiagnosticSink* getSink();

    InputStream* getParent() { return m_parent; }

    void setParent(InputStream* parent) { m_parent = parent; }

    MacroInvocation* getFirstBusyMacroInvocation() { return m_firstBusyMacroInvocation; }

    virtual SourceLoc findNextLineEndImpl(SourceLoc from, UInt& lineCount) const = 0;

protected:
    /// The preprocessor that this input stream is being used by
    Preprocessor* m_preprocessor = nullptr;

    /// Parent stream in the stack of secondary input streams
    InputStream* m_parent = nullptr;

    /// Macro expansions that should be considered "busy" during expansion of this stream
    MacroInvocation* m_firstBusyMacroInvocation = nullptr;
};

// The simplest types of input streams are those that simply "play back"
// a list of tokens that was already captures. These types of streams
// are primarily used for playing back the tokens inside of a macro body.

/// An input stream that reads from a list of tokens that had already been tokenized before.
///
struct PretokenizedInputStream : InputStream
{
    typedef InputStream Super;

    /// Initialize an input stream, and assocaite with a specific `preprocessor` and list of
    /// `tokens`
    PretokenizedInputStream(Preprocessor* preprocessor, TokenReader const& tokens)
        : Super(preprocessor), m_tokenReader(tokens)
    {
    }

    // A pretokenized stream implements the key read/peek operations
    // by delegating to the underlying token reader.

    virtual Token readToken() SLANG_OVERRIDE { return m_tokenReader.advanceToken(); }

    virtual Token peekToken() SLANG_OVERRIDE { return m_tokenReader.peekToken(); }

    virtual SourceLoc findNextLineEndImpl(SourceLoc from, UInt& lineCount) const SLANG_OVERRIDE
    {
        // Not implemented
        SLANG_UNUSED(from)
        SLANG_UNUSED(lineCount)
        return {};
    }

protected:
    /// Initialize an input stream, and assocaite with a specific `preprocessor`
    PretokenizedInputStream(Preprocessor* preprocessor)
        : Super(preprocessor)
    {
    }

    /// Reader for pre-tokenized input
    TokenReader m_tokenReader;
};

// While macro bodies are the main use case for pre-tokenized input strams,
// we also use them for a few one-off cases where the preprocessor needs to
// construct one or more tokens on the fly (e.g., when stringizing or pasting
// tokens). These streams differ in that they own the storage for the tokens
// they will play back, because they are effectively "one-shot."

/// A pre-tokenized input stream that will only be used once, and which therefore owns the memory
/// for its tokens.
struct SingleUseInputStream : PretokenizedInputStream
{
    typedef PretokenizedInputStream Super;

    SingleUseInputStream(Preprocessor* preprocessor, TokenList const& lexedTokens)
        : Super(preprocessor), m_lexedTokens(lexedTokens)
    {
        m_tokenReader = TokenReader(m_lexedTokens);
    }

    /// A list of raw tokens that will provide input
    TokenList m_lexedTokens;
};

// During macro expansion, or the substitution of parameters into a macro body
// we end up needing to track multiple active input streams, and this is most
// easily done by having a distinct type to represent a stack of input streams.

/// A stack of input streams, that will always read the next available token from the top-most
/// stream
///
/// An input stream stack assumes ownership of all streams pushed onto it, and will clean them
/// up when they are no longer active or when the stack gets destructed.
///
struct InputStreamStack
{
    InputStreamStack() {}

    /// Clean up after an input stream stack
    ~InputStreamStack() { popAll(); }

    /// Push an input stream onto the stack
    void push(InputStream* stream)
    {
        stream->setParent(m_top);
        m_top = stream;
    }

    /// Pop all input streams on the stack
    void popAll()
    {
        // We need to delete any input streams still on the stack.
        //
        InputStream* parent = nullptr;
        for (InputStream* s = m_top; s; s = parent)
        {
            parent = s->getParent();
            delete s;
        }
        m_top = nullptr;
    }

    /// Read a token from the top-most input stream with input
    ///
    /// If there is no input remaining, will return the EOF token
    /// of the bottom-most stream.
    ///
    /// At least one input stream must have been `push()`ed before
    /// it is valid to call this operation.
    ///
    Token readToken()
    {
        SLANG_ASSERT(m_top);
        for (;;)
        {
            // We always try to read from the top-most stream, and if
            // it is not at its end, then we return its next token.
            //
            auto token = m_top->readToken();
            if (token.type != TokenType::EndOfFile)
                return token;

            // If the top stream has run out of input we try to
            // switch to its parent, if any.
            //
            auto parent = m_top->getParent();
            if (parent)
            {
                // This stack has taken ownership of the streams,
                // and must therefore delete the top stream before
                // popping it.
                //
                delete m_top;
                m_top = parent;
                continue;
            }

            // If the top stream did *not* have a parent (meaning
            // it was also the bottom stream), then we don't try
            // to pop it and instead return its EOF token as-is.
            //
            return token;
        }
    }

    /// Peek a token from the top-most input stream with input
    ///
    /// If there is no input remaining, will return the EOF token
    /// of the bottom-most stream.
    ///
    /// At least one input stream must have been `push()`ed before
    /// it is valid to call this operation.
    ///
    Token peekToken()
    {
        // The logic here mirrors `readToken()`, but we do not
        // modify the `m_top` value or delete streams when they
        // are at their end, so that we don't disrupt any state
        // that might depend on which streams are present on
        // the stack.
        //
        // Note: One might ask why we cannot just pop input
        // streams that are at their end immediately. The basic
        // reason has to do with determining what macros were
        // "busy" when considering expanding a new one.
        // Consider:
        //
        //      #define BAD A B C BAD
        //
        //      BAD X Y Z
        //
        // When expanding the invocation of `BAD`, we will eventually
        // reach a point where the `BAD` in the expansion has been read
        // and we are considering whether to consider it as a macro
        // invocation.
        //
        // In this case it is clear that the Right Answer is that the
        // original invocation of `BAD` is still active, and thus
        // the macro is busy. To ensure that behavior, we want to
        // be able to detect that the stream representing the
        // expansion of `BAD` is still active even after we read
        // the `BAD` token.
        //
        // TODO: Consider whether we can streamline the implementaiton
        // an remove this wrinkle.
        //
        auto top = m_top;
        for (;;)
        {
            SLANG_ASSERT(top);
            auto token = top->peekToken();
            if (token.type != TokenType::EndOfFile)
                return token;

            auto parent = top->getParent();
            if (parent)
            {
                top = parent;
                continue;
            }

            return token;
        }
    }

    /// Return type of the token that `peekToken()` will return
    TokenType peekTokenType() { return peekToken().type; }

    /// Return location of the token that `peekToken()` will return
    SourceLoc peekLoc() { return peekToken().loc; }

    /// Skip over all whitespace tokens in the input stream(s) to arrive at the next non-whitespace
    /// token
    void skipAllWhitespace()
    {
        for (;;)
        {
            switch (peekTokenType())
            {
            default:
                return;

                // Note: We expect `NewLine` to be the only case of whitespace we
                // encounter right now, because all the other cases will have been
                // filtered out by the `LexerInputStream`.
                //
            case TokenType::NewLine:
            case TokenType::WhiteSpace:
            case TokenType::BlockComment:
            case TokenType::LineComment:
                readToken();
                break;
            }
        }
    }

    /// Get the top stream of the input stack
    InputStream* getTopStream() { return m_top; }

    /// Get the input stream that the next token would come from
    ///
    /// If the input stack is at its end, this will just be the top-most stream.
    InputStream* getNextStream()
    {
        SLANG_ASSERT(m_top);
        auto top = m_top;
        for (;;)
        {
            auto token = top->peekToken();
            if (token.type != TokenType::EndOfFile)
                return top;

            auto parent = top->getParent();
            if (parent)
            {
                top = parent;
                continue;
            }

            return top;
        }
    }

    SourceLoc findNextLineEnd(SourceLoc from, UInt lineCount = 1) const
    {
        auto top = m_top;
        SourceLoc res = from;
        res = top->findNextLineEndImpl(res, lineCount);
        if (lineCount > 0)
        {
            // We did not consume all the lines, but arrived at the end of the current Stream
            // For now, return none.
            // We could pop the inputStream stack and continue looking,
            // but it is an unlikely edge case.
            res = {};
        }
        return res;
    }

private:
    /// The top of the stack of input streams
    InputStream* m_top = nullptr;
};

// Another (relatively) simple case of an input stream is one that reads
// tokens directly from the lexer.
//
// It might seem like we could simplify things even further by always lexing
// a file into tokens first, and then using the earlier input-stream cases
// for pre-tokenized input. The main reason we don't use that strategy is
// that when dealing with preprocessor conditionals we will often want to
// suppress diagnostic messages coming from the lexer when inside of disabled
// conditional branches.
//
// TODO: We might be able to simplify the logic here by having the lexer buffer
// up the issues it diagnoses along with a list of tokens, rather than diagnose
// them directly, and then have the preprocessor or later compilation stages
// take responsibility for actually emitting those diagnostics.

/// An input stream that reads tokens directly using the Slang `Lexer`
struct LexerInputStream : InputStream
{
    typedef InputStream Super;

    LexerInputStream(Preprocessor* preprocessor, SourceView* sourceView);

    Lexer* getLexer() { return &m_lexer; }

    // A common thread to many of the input stream implementations is to
    // use a single token of lookahead in order to suppor the `peekToken()`
    // operation with both simplicity and efficiency.

    Token readToken() SLANG_OVERRIDE
    {
        auto result = m_lookaheadToken;
        m_lookaheadToken = _readTokenImpl();
        return result;
    }

    Token peekToken() SLANG_OVERRIDE { return m_lookaheadToken; }

    virtual SourceLoc findNextLineEndImpl(SourceLoc from, UInt& lineCount) const SLANG_OVERRIDE
    {
        return m_lexer.findNextLineEnd(from, lineCount);
    }

private:
    /// Read a token from the lexer, bypassing lookahead
    Token _readTokenImpl()
    {
        for (;;)
        {
            Token token = m_lexer.lexToken();
            switch (token.type)
            {
            default:
                return token;

            case TokenType::WhiteSpace:
            case TokenType::BlockComment:
            case TokenType::LineComment:
                break;
            }
        }
    }

    /// The lexer state that will provide input
    Lexer m_lexer;

    /// One token of lookahead
    Token m_lookaheadToken;
};

// The remaining input stream cases deal with macro expansion, so it is
// probalby a good idea to discuss how macros are represented by the
// preprocessor as a first step.
//
// Note that there is an important distinction between a macro *definition*
// and a macro *invocation*, similar to how we distinguish a function definition
// from a call to that function.

/// A definition of a macro
struct MacroDefinition
{
    /// The "flavor" / type / kind of a macro definition
    enum class Flavor
    {
        /// A function-like macro (e.g., `#define INC(x) (x)++`)
        FunctionLike,

        /// An user-defiend object-like macro (e.g., `#define N 100`)
        ObjectLike,

        /// An object-like macro that is built in to the copmiler (e.g., `__LINE__`)
        BuiltinObjectLike,
    };

    // The body of a macro definition is input as a stream of tokens, but
    // when "playing back" a macro it is helpful to process those tokens
    // into a form where a lot of the semantic questions have been answered.
    //
    // We will chop up the tokens that macro up a macro definition/body into
    // distinct *ops* where each op has an *opcode* that defines how that
    // token or range of tokens behaves.

    /// Opcode for an `Op` in a macro definition
    enum class Opcode
    {
        /// A raw span of tokens from the macro body (no subsitution needed)
        ///
        /// The `index0` and `index1` fields form a begin/end pair of tokens
        RawSpan,

        /// A parameter of the macro, which should have expansion applied to it
        ///
        /// The `index0` opcode is the index of the token that named the parameter
        /// The `index1` field is the zero-based index of the chosen parameter
        ExpandedParam,

        /// A parameter of the macro, which should *not* have expansion applied to it
        ///
        /// The `index0` opcode is the index of the token that named the parameter
        /// The `index1` field is the zero-based index of the chosen parameter
        UnexpandedParam,

        /// A parameter of the macro, stringized (and not expanded)
        ///
        /// The `index0` opcode is the index of the token that named the parameter
        /// The `index1` field is the zero-based index of the chosen parameter
        StringizedParam,

        /// A paste of the last token of the preceding op and the first token of the next
        ///
        /// The `index0` opcode is the index of the `##` token
        TokenPaste,

        /// builtin expansion behavior for `__LINE__`
        BuiltinLine,

        /// builtin expansion behavior for `__FILE__`
        BuiltinFile,
    };

    /// A single op in the definition of the macro
    struct Op
    {
        /// The opcode that defines how to interpret this op
        Opcode opcode = Opcode::RawSpan;

        /// Two operands, with interpretation depending on the `opcode`
        Index index0 = 0;
        Index index1 = 0;
    };

    struct Param
    {
        NameLoc nameLoc;
        bool isVariadic = false;
    };

    /// The flavor of macro
    MacroDefinition::Flavor flavor;

    /// The name under which the macro was `#define`d
    NameLoc nameAndLoc;

    /// The tokens that make up the macro body
    TokenList tokens;

    /// List ops that describe how this macro expands
    List<Op> ops;

    /// Parameters of the macro, in case of a function-like macro
    List<Param> params;

    Name* getName() { return nameAndLoc.name; }

    SourceLoc getLoc() { return nameAndLoc.loc; }

    bool isBuiltin() { return flavor == MacroDefinition::Flavor::BuiltinObjectLike; }

    /// Is this a variadic macro?
    bool isVariadic()
    {
        // A macro is variadic if it has a last parameter and
        // that last parameter is a variadic parameter.
        //
        auto paramCount = params.getCount();
        if (paramCount == 0)
            return false;
        return params[paramCount - 1].isVariadic;
    }
};

// When a macro is invoked, we conceptually want to "play back" the ops
// that make up the macro's definition. The `MacroInvocation` type logically
// represents an invocation of a macro and handles the complexities of
// playing back its definition with things like argument substiution.

/// An invocation/call of a macro, which can provide tokens of its expansion
struct MacroInvocation : InputStream
{
    typedef InputStream Super;

    /// Create a new expansion of `macro`
    MacroInvocation(
        Preprocessor* preprocessor,
        MacroDefinition* macro,
        SourceLoc macroInvocationLoc,
        SourceLoc initiatingMacroInvocationLoc);

    /// Prime the input stream
    ///
    /// This operation *must* be called before the first `readToken()` or `peekToken()`
    void prime(MacroInvocation* nextBusyMacroInvocation);

    // The `readToken()` and `peekToken()` operations for a macro invocation
    // will be implemented by using one token of lookahead, which makes the
    // operations relatively simple.

    virtual Token readToken() SLANG_OVERRIDE
    {
        Token result = m_lookaheadToken;
        m_lookaheadToken = _readTokenImpl();
        return result;
    }

    virtual Token peekToken() SLANG_OVERRIDE { return m_lookaheadToken; }

    /// Is the given `macro` considered "busy" during the given macroinvocation?
    static bool isBusy(MacroDefinition* macro, MacroInvocation* duringMacroInvocation);

    Index getArgCount() { return m_args.getCount(); }

    SourceLoc getInvocationLoc() { return m_macroInvocationLoc; }

    MacroDefinition* getMacroDefinition() { return m_macro; }

    virtual SourceLoc findNextLineEndImpl(SourceLoc from, UInt& lineCount) const SLANG_OVERRIDE
    {
        // There are no actual lines inside of a macro invocation
        SLANG_UNUSED(from)
        SLANG_UNUSED(lineCount)
        return {};
    }

private:
    // Macro invocations are created as part of applying macro expansion
    // to a stream, so the `ExpansionInputStream` type takes responsibility
    // for setting up much of the state of a `MacroInvocation`.
    //
    friend struct ExpansionInputStream;

    /// The macro being expanded
    MacroDefinition* m_macro;

    /// A single argument to the macro invocation
    ///
    /// Each argument is represented as a begin/end pair of indices
    /// into the sequence of tokens that make up the macro arguments.
    ///
    struct Arg
    {
        Index beginTokenIndex = 0;
        Index endTokenIndex = 0;
    };

    /// Tokens that make up the macro arguments, in case of function-like macro expansion
    List<Token> m_argTokens;

    /// Arguments to the macro, in the case of a function-like macro expansion
    List<Arg> m_args;

    /// Additional macros that should be considered "busy" during this expansion
    MacroInvocation* m_nextBusyMacroInvocation = nullptr;

    /// Locatin of the macro invocation that led to this expansion
    SourceLoc m_macroInvocationLoc;

    /// Location of the "iniating" macro invocation in cases where multiple
    /// nested macro invocations might be in flight.
    SourceLoc m_initiatingMacroInvocationLoc;

    /// One token of lookahead
    Token m_lookaheadToken;

    /// Actually read a new token (not just using the lookahead)
    Token _readTokenImpl();

    // In order to play back a macro definition, we will play back the ops
    // in its body one at a time. Each op may expand to a stream of zero or
    // more tokens, so we need some state to track all of that.

    /// One or more input streams representing the current "op" being expanded
    InputStreamStack m_currentOpStreams;

    /// The index into the macro's list of the current operation being played back
    Index m_macroOpIndex = 0;

    /// Initialize the input stream for the current macro op
    void _initCurrentOpStream();

    /// Get a reader for the tokens that make up the macro argument at the given `paramIndex`
    TokenReader _getArgTokens(Index paramIndex);

    /// Push a stream onto `m_currentOpStreams` that consists of a single token
    void _pushSingleTokenStream(
        TokenType tokenType,
        SourceLoc tokenLoc,
        UnownedStringSlice const& content);

    /// Push a stream for a source-location builtin (`__FILE__` or `__LINE__`), with content set up
    /// by `valueBuilder`
    template<typename F>
    void _pushStreamForSourceLocBuiltin(TokenType tokenType, F const& valueBuilder);
};

// Playing back macro bodies for macro invocations is one part of the expansion process, and the
// other is scanning through a token stream and identifying macro invocations that need to be
// expanded. Rather than have one stream type try to handle both parts of the process, we use a
// distinct type to handle scanning for macro invocations.
//
// By using two distinct stream types we are able to handle intriciate details of the C/C++
// preprocessor like how the argument tokens to a macro are expanded before they are subsituted into
// the body, and then are subject to another round of macro expansion *after* substitution.

/// An input stream that applies macro expansion to another stream
struct ExpansionInputStream : InputStream
{
    typedef InputStream Super;

    /// Construct an input stream that applies macro expansion to `base`
    ExpansionInputStream(Preprocessor* preprocessor, InputStream* base)
        : Super(preprocessor), m_base(base)
    {
        m_inputStreams.push(base);
        m_lookaheadToken = _readTokenImpl();
    }

    Token readToken() SLANG_OVERRIDE
    {
        // Reading a token from an expansion strema amounts to checking
        // whether the current state of the input stream marks the start
        // of a macro invocation (in which case we push the resulting
        // invocation onto the input stack), and then reading a token
        // from whatever stream is on top of the stack.

        _maybeBeginMacroInvocation();

        Token result = m_lookaheadToken;
        m_lookaheadToken = _readTokenImpl();
        return result;
    }

    Token peekToken() SLANG_OVERRIDE
    {
        _maybeBeginMacroInvocation();
        return m_lookaheadToken;
    }

    // The "raw" read operations on an expansion input strema bypass
    // macro expansion and just read whatever token is next in the
    // input. These are useful for the top-level input stream of
    // a file, since we often want to read unexpanded tokens for
    // preprocessor directives.

    Token readRawToken()
    {
        Token result = m_lookaheadToken;
        m_lookaheadToken = _readTokenImpl();
        return result;
    }

    Token peekRawToken() { return m_lookaheadToken; }

    TokenType peekRawTokenType() { return peekRawToken().type; }

    void setInitiatingMacroSourceLoc(SourceLoc loc)
    {
        m_initiatingMacroInvocationLoc = loc;
        m_isInExpansion = true;
    }

    virtual SourceLoc findNextLineEndImpl(SourceLoc from, UInt& lineCount) const SLANG_OVERRIDE
    {
        // Should not be here / not implemented
        SLANG_UNUSED(from)
        SLANG_UNUSED(lineCount)
        return {};
    }

    SourceLoc findNextLineEnd(SourceLoc from, UInt lineCount = 1) const
    {
        return m_inputStreams.findNextLineEnd(from, lineCount);
    }

private:
    /// The base stream that macro expansion is being applied to
    InputStream* m_base = nullptr;

    /// A stack of the base stream and active macro invocation in flight
    InputStreamStack m_inputStreams;

    /// Location of the "iniating" macro invocation in cases where multiple
    /// nested macro invocations might be in flight.
    SourceLoc m_initiatingMacroInvocationLoc;

    /// Whether this ExpansionStream is created in the middle of
    /// another macro expansion.
    bool m_isInExpansion = false;

    /// One token of lookahead
    Token m_lookaheadToken;

    /// Read a token, bypassing lookahead
    Token _readTokenImpl()
    {
        Token token = m_inputStreams.readToken();
        return token;
    }

    /// Look at current input state and decide whether it represents a macro invocation
    void _maybeBeginMacroInvocation();

    /// Parse one argument to a macro invocation
    MacroInvocation::Arg _parseMacroArg(MacroInvocation* macroInvocation);

    /// Parse all arguments to a macro invocation
    void _parseMacroArgs(MacroDefinition* macro, MacroInvocation* macroInvocation);

    /// Push the given macro invocation into the stack of input streams
    void _pushMacroInvocation(MacroInvocation* macroInvocation);
};

// The top-level flow of the preprocessor is that it processed *input files*
// that contain both directives and ordinary tokens.
//
// Input files are a bit like token streams, but they don't fit neatly into
// the same abstraction due to all the special-case handling that directives
// and conditionals require.

/// An input file being processed by the preprocessor.
///
/// An input file manages both the expansion of lexed tokens
/// from the source file, and also state related to preprocessor
/// directives, including skipping of code due to `#if`, etc.
///
struct InputFile
{
    InputFile(Preprocessor* preprocessor, SourceView* sourceView);

    ~InputFile();

    /// Is this input file skipping tokens (because the current location is inside a disabled
    /// condition)?
    bool isSkipping();

    /// Get the inner-most conditional that is in efffect at the current location
    Conditional* getInnerMostConditional() { return m_conditional; }

    /// Push a new conditional onto the stack of conditionals in effect
    void pushConditional(Conditional* conditional)
    {
        conditional->parent = m_conditional;
        m_conditional = conditional;
    }

    /// Pop the inner-most conditional
    void popConditional()
    {
        auto conditional = m_conditional;
        SLANG_ASSERT(conditional);
        m_conditional = conditional->parent;
        delete conditional;
    }

    /// Read one token using all the expansion and directive-handling logic
    Token readToken() { return m_expansionStream->readToken(); }

    Lexer* getLexer() { return m_lexerStream->getLexer(); }

    ExpansionInputStream* getExpansionStream() { return m_expansionStream; }

    bool isIncludedFile() { return m_parent != nullptr; }

private:
    friend struct Preprocessor;

    /// The parent preprocessor
    Preprocessor* m_preprocessor = nullptr;

    /// The next outer input file
    ///
    /// E.g., if this file was `#include`d from another file, then `m_parent` would be
    /// the file with the `#include` directive.
    ///
    InputFile* m_parent = nullptr;

    /// The inner-most preprocessor conditional active for this file.
    Conditional* m_conditional = nullptr;

    /// The lexer input stream that unexpanded tokens will be read from
    LexerInputStream* m_lexerStream;

    /// An input stream that applies macro expansion to `m_lexerStream`
    ExpansionInputStream* m_expansionStream;
};

enum class PragmaWarningSpecifier
{
    Default,
    Disable,
    Error,
    Once,
    Suppress,
};

struct WarningTimeline
{
    struct Entry
    {
        PragmaWarningSpecifier specifier = {};
        SourceLoc::RawValue location = {}; // Absolute location
        // Used for the once specifier
        // -1 points to this, but not consumed
        // -2 points to this, but was consumed
        // >= 0 points to a previous once entry (sharing the payload)
        // Used for the suppress specifier to store the original SourceLoc needed to emit a warning
        union
        {
            int payload = 0;
            SourceLoc::RawValue debugLocation; // Store the raw value for trivial copy
        };

        bool operator<(Entry const& other) const { return location < other.location; }
    };
    // Sorted by location
    List<Entry> entries = {};

    const Entry* findEntry(SourceLoc::RawValue location) const
    {
        const Entry* res = nullptr;
        if (entries.getCount() && location >= entries.getFirst().location)
        {
            auto nextEntryIndex = ::std::upper_bound(
                entries.begin(),
                entries.end(),
                location,
                [](SourceLoc::RawValue const& lhs, Entry const& rhs)
                { return lhs < rhs.location; });
            res = nextEntryIndex - 1;
        }
        return res;
    }

    PragmaWarningSpecifier consumeSpecifier(SourceLoc::RawValue location)
    {
        PragmaWarningSpecifier res = PragmaWarningSpecifier::Default;
        Entry* entry = const_cast<Entry*>(findEntry(location));
        if (entry)
        {
            PragmaWarningSpecifier& spec = entry->specifier;
            res = spec;
            if (res == PragmaWarningSpecifier::Once)
            {
                int* payload = &entry->payload;
                if (*payload >= 0)
                {
                    payload = &entries[*payload].payload;
                }
                if (*payload == -1)
                {
                    res = PragmaWarningSpecifier::Default;
                    --(*payload);
                }
                else if (*payload < -1)
                {
                    res = PragmaWarningSpecifier::Disable;
                }
            }
        }
        return res;
    }

    const Entry* getLatestEntry() const
    {
        const Entry* res = nullptr;
        if (entries.getCount())
        {
            res = &entries.getLast();
        }
        return res;
    }

    PragmaWarningSpecifier getLatestSpecifier() const
    {
        PragmaWarningSpecifier res = PragmaWarningSpecifier::Default;
        if (entries.getCount())
        {
            res = entries.getLast().specifier;
        }
        return res;
    }

    void addEntry(
        SourceLoc::RawValue location,
        PragmaWarningSpecifier specifier,
        const Entry* poppingFrom,
        DiagnosticSink* sink,
        int id,
        SourceLoc debugLoc)
    {
        SourceLoc::RawValue maxKnownLocation =
            entries.getCount() ? entries.getLast().location : SourceLoc::RawValue(0);
        // Add on top
        if (location > maxKnownLocation)
        {
            // Add a new entry only if necessary
            if (getLatestSpecifier() != specifier || specifier == PragmaWarningSpecifier::Once)
            {
                Entry e;
                e.specifier = specifier;
                e.location = location;
                if (specifier == PragmaWarningSpecifier::Once)
                {
                    if (poppingFrom)
                    {
                        SLANG_ASSERT(poppingFrom->specifier == PragmaWarningSpecifier::Once);
                        if (poppingFrom->payload >= 0)
                        {
                            e.payload = poppingFrom->payload;
                        }
                        else
                        {
                            e.payload = static_cast<int>(poppingFrom - entries.begin());
                        }
                    }
                    else
                    {
                        e.payload = -1;
                    }
                }
                else
                {
                    e.debugLocation = debugLoc.getRaw();
                }
                entries.add(e);
            }
        }
        else
        {
            if (sink)
            {
                sink->diagnose(debugLoc, Diagnostics::pragmaWarningCannotInsertHere, id);
                const Entry* prevEntry = findEntry(location);
                if (prevEntry && prevEntry->specifier == PragmaWarningSpecifier::Suppress)
                {
                    sink->diagnose(
                        SourceLoc::fromRaw(prevEntry->debugLocation),
                        Diagnostics::pragmaWarningPointSuppress,
                        id);
                }
            }
        }
    }

    void addEntryForPragmaPop(
        SourceLoc::RawValue location,
        SourceLoc::RawValue pushedLocation,
        DiagnosticSink* sink,
        int id,
        SourceLoc debugLoc)
    {
        const Entry* poppingFrom = findEntry(pushedLocation);
        addEntry(
            location,
            poppingFrom ? poppingFrom->specifier : PragmaWarningSpecifier::Default,
            poppingFrom,
            sink,
            id,
            debugLoc);
    }
};

struct WarningStateTracker : SourceWarningStateTrackerBase
{
    SourceManager* sourceManager = nullptr;
    Dictionary<int, WarningTimeline> mapDiagnosticIdToTimeline = {};
    List<SourceLoc> stack = {};

    WarningStateTracker(SourceManager* sourceManager = nullptr)
        : sourceManager(sourceManager)
    {
    }

    SourceLoc::RawValue getAbsoluteLocation(SourceLoc loc) const
    {
        return sourceManager ? sourceManager->getAbsoluteLocation(loc) : loc.getRaw();
    }

    virtual Severity consumeWarningSeverity(SourceLoc location, int id, Severity severity) override
    {
        Severity res = severity;
        WarningTimeline* timeline = mapDiagnosticIdToTimeline.tryGetValue(id);
        if (!timeline)
            return res;
        SourceLoc::RawValue absoluteLoc = getAbsoluteLocation(location);
        PragmaWarningSpecifier spec = timeline->consumeSpecifier(absoluteLoc);
        if (spec == PragmaWarningSpecifier::Disable || spec == PragmaWarningSpecifier::Suppress)
        {
            res = Severity::Disable;
        }
        else if (spec == PragmaWarningSpecifier::Error)
        {
            res = Severity::Error;
        }
        return res;
    }

    void addEntry(
        SourceLoc location,
        SourceLoc nextLineEnd,
        int id,
        PragmaWarningSpecifier specifier,
        DiagnosticSink* sink = nullptr)
    {
        WarningTimeline& timeline = mapDiagnosticIdToTimeline[id];
        auto absLoc = getAbsoluteLocation(location);
        PragmaWarningSpecifier prev = timeline.getLatestSpecifier();
        auto lastEntry = timeline.getLatestEntry();
        timeline.addEntry(absLoc, specifier, nullptr, sink, id, location);
        if (specifier == PragmaWarningSpecifier::Suppress)
        {
            auto nextAbsLoc = getAbsoluteLocation(nextLineEnd);
            timeline.addEntry(nextAbsLoc, prev, lastEntry, sink, id, nextLineEnd);
        }
    }

    void addPragmaPush(SourceLoc location) { stack.add(location); }

    void addPragmaPop(SourceLoc location, DiagnosticSink* sink = nullptr)
    {
        if (stack.getCount())
        {
            const SourceLoc pushed = stack.getLast();
            stack.removeLast();
            if (mapDiagnosticIdToTimeline.getCount())
            {
                const SourceLoc::RawValue absLoc = getAbsoluteLocation(location);
                const SourceLoc::RawValue absPushed = getAbsoluteLocation(pushed);
                for (auto& [id, timeline] : mapDiagnosticIdToTimeline)
                {
                    timeline.addEntryForPragmaPop(absLoc, absPushed, sink, id, location);
                }
            }
        }
        else if (sink)
        {
            sink->diagnose(location, Diagnostics::pragmaWarningPopEmpty);
        }
    }
};

/// State of the preprocessor
struct Preprocessor
{
    /// Diagnostics sink to use when writing messages
    DiagnosticSink* sink = nullptr;

    /// Functionality for looking up files in a `#include` directive
    IncludeSystem* includeSystem = nullptr;

    /// A stack of "active" input files
    InputFile* m_currentInputFile = nullptr;

    // TODO: We could split the macro environment into a `globalEnv`
    // and a `superGlobalEnv` such that built-in macros like `__FILE__`
    // and `__LINE__` are defined in the super-global environment so
    // that they can be shadowed by user-defined macros but will again
    // be available after an `#undef`.

    /// Currently-defined macros
    Environment globalEnv;

    /// A pre-allocated token that can be returned to represent end-of-input situations.
    Token endOfFileToken;

    /// Callback handlers
    PreprocessorHandler* handler = nullptr;

    /// The unique identities of any paths that have issued `#pragma once` directives to
    /// stop them from being included again.
    HashSet<String> pragmaOnceUniqueIdentities;

    WarningStateTracker* warningStateTracker = nullptr;

    /// Name pool to use when creating `Name`s from strings
    NamePool* namePool = nullptr;

    /// File system to use when looking up files
    ISlangFileSystemExt* fileSystem = nullptr;

    /// Source manager to use when loading source files
    SourceManager* sourceManager = nullptr;

    /// Stores the initiating macro source location.
    SourceLoc initiatingMacroSourceLoc;

    /// Detected source language.
    SourceLanguage language = SourceLanguage::Unknown;

    /// Stores macro definition and invocation info for language server.
    PreprocessorContentAssistInfo* contentAssistInfo = nullptr;

    NamePool* getNamePool() { return namePool; }
    SourceManager* getSourceManager() { return sourceManager; }

    SourceLoc::RawValue absoluteSourceLocCounter = 0;

    /// Push a new input file onto the input stack of the preprocessor
    void pushInputFile(InputFile* inputFile, SourceLoc location);

    /// Pop the inner-most input file from the stack of input files
    void popInputFile();
};

static void reportMacroDefinitionForContentAssist(Preprocessor* preprocessor, MacroDefinition* def)
{
    if (!preprocessor->contentAssistInfo)
        return;

    MacroDefinitionContentAssistInfo info;
    info.name = def->getName();
    info.loc = def->getLoc();
    info.tokenList = def->tokens.m_tokens;
    for (auto param : def->params)
    {
        MacroDefinitionContentAssistInfo::Param p;
        p.isVariadic = param.isVariadic;
        p.name = param.nameLoc.name;
        info.params.add(p);
    }
    preprocessor->contentAssistInfo->macroDefinitions.add(info);
}

static void reportMacroInvocationForContentAssist(
    Preprocessor* preprocessor,
    MacroInvocation* invocation)
{
    if (!preprocessor->contentAssistInfo)
        return;
    if (preprocessor->m_currentInputFile && preprocessor->m_currentInputFile->isIncludedFile())
        return;
    MacroInvocationContentAssistInfo info;
    info.name = invocation->getMacroDefinition()->getName();
    info.loc = invocation->getInvocationLoc();

    preprocessor->contentAssistInfo->macroInvocations.add(info);
}

static void reportIncludeFileForContentAssist(Preprocessor* preprocessor, Token token, String path)
{
    if (!preprocessor->contentAssistInfo)
        return;
    if (preprocessor->m_currentInputFile && preprocessor->m_currentInputFile->isIncludedFile())
        return;
    FileIncludeContentAssistInfo info;
    info.loc = token.loc;
    info.length = (int)token.getContentLength();
    info.path = path;
    preprocessor->contentAssistInfo->fileIncludes.add(info);
}

// static Token AdvanceToken(Preprocessor* preprocessor);

// Convenience routine to access the diagnostic sink
static DiagnosticSink* GetSink(Preprocessor* preprocessor)
{
    return preprocessor->sink;
}

DiagnosticSink* InputStream::getSink()
{
    return GetSink(m_preprocessor);
}

//
// Basic Input Handling
//

LexerInputStream::LexerInputStream(Preprocessor* preprocessor, SourceView* sourceView)
    : Super(preprocessor)
{
    MemoryArena* memoryArena = sourceView->getSourceManager()->getMemoryArena();
    m_lexer.initialize(sourceView, GetSink(preprocessor), preprocessor->getNamePool(), memoryArena);
    m_lookaheadToken = _readTokenImpl();
}

InputFile::InputFile(Preprocessor* preprocessor, SourceView* sourceView)
{
    m_preprocessor = preprocessor;

    m_lexerStream = new LexerInputStream(preprocessor, sourceView);
    m_expansionStream = new ExpansionInputStream(preprocessor, m_lexerStream);
}

InputFile::~InputFile()
{
    // We start by deleting any remaining conditionals on the conditional stack.
    //
    // Note: This should only come up in the case where a conditional was not
    // terminated before the end of the file.
    //
    Conditional* parentConditional = nullptr;
    for (auto conditional = m_conditional; conditional; conditional = parentConditional)
    {
        parentConditional = conditional->parent;
        delete conditional;
    }

    // Note: We only delete the expansion strema here because the lexer
    // stream is being used as the "base" stream of the expansion stream,
    // and the expansion stream takes responsibility for deleting it.
    //
    delete m_expansionStream;
}

//
// Macros
//


// Find the currently-defined macro of the given name, or return NULL
static MacroDefinition* LookupMacro(Environment* environment, Name* name)
{
    for (Environment* e = environment; e; e = e->parent)
    {
        MacroDefinition* macro = NULL;
        if (e->macros.tryGetValue(name, macro))
            return macro;
    }

    return NULL;
}

bool MacroInvocation::isBusy(MacroDefinition* macro, MacroInvocation* duringMacroInvocation)
{
    for (auto busyMacroInvocation = duringMacroInvocation; busyMacroInvocation;
         busyMacroInvocation = busyMacroInvocation->m_nextBusyMacroInvocation)
    {
        if (busyMacroInvocation->m_macro == macro)
            return true;
    }
    return false;
}

MacroInvocation::MacroInvocation(
    Preprocessor* preprocessor,
    MacroDefinition* macro,
    SourceLoc macroInvocationLoc,
    SourceLoc initiatingMacroInvocationLoc)
    : Super(preprocessor)
{
    m_macro = macro;
    m_firstBusyMacroInvocation = this;
    m_macroInvocationLoc = macroInvocationLoc;
    m_initiatingMacroInvocationLoc = initiatingMacroInvocationLoc;
}

void MacroInvocation::prime(MacroInvocation* nextBusyMacroInvocation)
{
    m_nextBusyMacroInvocation = nextBusyMacroInvocation;

    _initCurrentOpStream();
    m_lookaheadToken = _readTokenImpl();

    reportMacroInvocationForContentAssist(m_preprocessor, this);
}

void ExpansionInputStream::_pushMacroInvocation(MacroInvocation* expansion)
{
    m_inputStreams.push(expansion);
    m_lookaheadToken = m_inputStreams.readToken();
}

/// Parse one macro argument and return it in the form of a macro
///
/// Assumes as a precondition that the caller has already checked
/// for a closing `)` or end-of-input token.
///
/// Does not consume any closing `)` or `,` for the argument.
///
MacroInvocation::Arg ExpansionInputStream::_parseMacroArg(MacroInvocation* macroInvocation)
{
    // Create the argument, represented as a special flavor of macro
    //
    MacroInvocation::Arg arg;
    arg.beginTokenIndex = macroInvocation->m_argTokens.getCount();

    // We will now read the tokens that make up the argument.
    //
    // We need to keep track of the nesting depth of parentheses,
    // because arguments should only break on a `,` that is
    // not properly nested in balanced parentheses.
    //
    int nestingDepth = 0;
    for (;;)
    {
        arg.endTokenIndex = macroInvocation->m_argTokens.getCount();

        m_inputStreams.skipAllWhitespace();
        Token token = m_inputStreams.peekToken();
        macroInvocation->m_argTokens.add(token);

        switch (token.type)
        {
        case TokenType::EndOfFile:
            // End of input means end of the argument.
            // It is up to the caller to diagnose the
            // lack of a closing `)`.
            return arg;

        case TokenType::RParent:
            // If we see a right paren when we aren't nested
            // then we are at the end of an argument.
            //
            if (nestingDepth == 0)
            {
                return arg;
            }
            // Otherwise we decrease our nesting depth, add
            // the token, and keep going
            nestingDepth--;
            break;

        case TokenType::Comma:
            // If we see a comma when we aren't nested
            // then we are at the end of an argument
            if (nestingDepth == 0)
            {
                return arg;
            }
            // Otherwise we add it as a normal token
            break;

        case TokenType::LParent:
            // If we see a left paren then we need to
            // increase our tracking of nesting
            nestingDepth++;
            break;

        default:
            break;
        }

        // Add the token and continue parsing.
        m_inputStreams.readToken();
    }
}

/// Parse the arguments to a function-like macro invocation.
///
/// This function assumes the opening `(` has already been parsed,
/// and it leaves the closing `)`, if any, for the caller to consume.
///
void ExpansionInputStream::_parseMacroArgs(MacroDefinition* macro, MacroInvocation* expansion)
{
    // There is a subtle case here, which is when a macro expects
    // exactly one non-variadic parameter, but the argument list is
    // empty. E.g.:
    //
    //      #define M(x) /* whatever */
    //
    //      M()
    //
    // In this case we should parse a single (empty) argument, rather
    // than issue an error because of there apparently being zero
    // arguments.
    //
    // In all other cases (macros that do not have exactly one
    // parameter, plus macros with a single variadic parameter) we
    // should treat an empty argument list as zero
    // arguments for the purposes of error messages (since that is
    // how a programmer is likely to view/understand it).
    //
    Index paramCount = macro->params.getCount();
    if (paramCount != 1 || macro->isVariadic())
    {
        // If there appear to be no arguments because the next
        // token would close the argument list, then we bail
        // out immediately.
        //
        switch (m_inputStreams.peekTokenType())
        {
        case TokenType::RParent:
        case TokenType::EndOfFile:
            return;
        }
    }

    // Otherwise, we have one or more arguments.
    for (;;)
    {
        // Parse an argument.
        MacroInvocation::Arg arg = _parseMacroArg(expansion);
        expansion->m_args.add(arg);

        // After consuming one macro argument, we look at
        // the next token to decide what to do.
        //
        switch (m_inputStreams.peekTokenType())
        {
        case TokenType::RParent:
        case TokenType::EndOfFile:
            // if we see a closing `)` or the end of
            // input, we know we are done with arguments.
            //
            return;

        case TokenType::Comma:
            // If we see a comma, then we will
            // continue scanning for more macro
            // arguments.
            //
            readRawToken();
            break;

        default:
            // Any other token represents a syntax error.
            //
            // TODO: We could try to be clever here in deciding
            // whether to break out of parsing macro arguments,
            // or whether to "recover" and continue to scan
            // ahead for a closing `)`. For now it is simplest
            // to just bail.
            //
            getSink()->diagnose(
                m_inputStreams.peekLoc(),
                Diagnostics::errorParsingToMacroInvocationArgument,
                paramCount,
                macro->getName());
            return;
        }
    }
}

// Check whether the current token on the given input stream should be
// treated as a macro invocation, and if so set up state for expanding
// that macro.
void ExpansionInputStream::_maybeBeginMacroInvocation()
{
    auto preprocessor = m_preprocessor;

    // We iterate because the first token in the expansion of one
    // macro may be another macro invocation.
    for (;;)
    {
        // The "next" token to be read is already in our `m_lookeadToken`
        // member, so we can simply inspect it.
        //
        // We also care about where that token came from (which input stream).
        //
        Token token = m_lookaheadToken;

        // If the token is not an identifier, then it can't possibly name a macro.
        //
        if (token.type != TokenType::Identifier)
        {
            return;
        }

        // We will look for a defined macro matching the name.
        //
        // If there isn't one this couldn't possibly be the start of a macro
        // invocation.
        //
        Name* name = token.getName();
        MacroDefinition* macro = LookupMacro(&preprocessor->globalEnv, name);
        if (!macro)
        {
            return;
        }

        // Now we get to the slightly trickier cases.
        //
        // *If* the identifier names a macro, but we are currently in the
        // process of expanding the same macro (possibly via multiple
        // nested expansions) then we don't want to expand it again.
        //
        // We determine which macros are currently being expanded
        // by looking at the input stream assocaited with that one
        // token of lookahead.
        //
        // Note: it is critical here that `m_inputStreams.getTopStream()`
        // returns the top-most stream that was active when `m_lookaheadToken`
        // was consumed. This means that an `InputStreamStack` cannot
        // "pop" an input stream that it at its end until after something
        // tries to read an additional token.
        //
        auto activeStream = m_inputStreams.getTopStream();

        // Each input stream keeps track of a linked list of the `MacroInvocation`s
        // that are considered "busy" while reading from that stream.
        //
        auto busyMacros = activeStream->getFirstBusyMacroInvocation();

        // If the macro is busy (already being expanded), we don't try to expand
        // it again, becaues that would trigger recursive/infinite expansion.
        //
        if (MacroInvocation::isBusy(macro, busyMacros))
            return;

        // At this point we know that the lookahead token names a macro
        // definition that is not busy. it is *very* likely that we are
        // going to be expanding a macro.
        //
        // If we aren't already expanding a macro (meaning that the
        // current stream tokens are being read from is the "base" stream
        // that expansion is being applied to), then we want to consider
        // the location of this invocation as the "initiating" macro
        // invocation location for things like `__LINE__` uses inside
        // of macro bodies.
        //
        if (!m_isInExpansion && activeStream == m_base)
        {
            m_initiatingMacroInvocationLoc = token.loc;
        }

        // The next steps depend on whether or not we are dealing
        // with a funciton-like macro.
        //
        switch (macro->flavor)
        {
        default:
            {
                // Object-like macros (whether builtin or user-defined) are the easy case.
                //
                // We simply create a new macro invocation based on the macro definition,
                // prime its input stream, and then push it onto our stack of active
                // macro invocations.
                //
                // Note: the macros that should be considered "busy" during the invocation
                // are all those that were busy at the time we read the name of the macro
                // to be expanded.
                //
                MacroInvocation* invocation = new MacroInvocation(
                    preprocessor,
                    macro,
                    token.loc,
                    m_initiatingMacroInvocationLoc);
                invocation->prime(busyMacros);
                _pushMacroInvocation(invocation);
            }
            break;

        case MacroDefinition::Flavor::FunctionLike:
            {
                // The function-like macro case is more complicated, primarily because
                // of the need to handle arguments. The arguments of a function-like
                // macro are expected to be tokens inside of balanced `()` parentheses.
                //
                // One special-case rule of the C/C++ preprocessor is that if the
                // name of a function-like macro is *not* followed by a `(`, then
                // it will not be subject to macro expansion. This design choice is
                // motivated by wanting to be able to create a macro that handles
                // direct calls to some primitive, along with a true function that handles
                // cases where it is used in other ways. E.g.:
                //
                //      extern int coolFunction(int x);
                //
                //      #define coolFunction(x) x^0xABCDEF
                //
                //      int x = coolFunction(3); // uses the macro
                //      int (*functionPtr)(int) f = coolFunction; // uses the function
                //
                // While we don't expect users to make heavy use of this feature in Slang,
                // it is worthwhile to try to stay compatible.
                //
                // Because the macro name is already in `m_lookaheadToken`, we can peak
                // at the underlying input stream to see if the next non-whitespace
                // token after the lookahead is a `(`.
                //
                m_inputStreams.skipAllWhitespace();
                Token maybeLeftParen = m_inputStreams.peekToken();
                if (maybeLeftParen.type != TokenType::LParent)
                {
                    // If we see a token other then `(` then we aren't suppsoed to be
                    // expanding the macro after all. Luckily, there is no state
                    // that we have to rewind at this point, because we never committed
                    // to macro expansion or consumed any (non-whitespace) tokens after
                    // the lookahead.
                    //
                    // We can simply bail out of looking for macro invocations, and the
                    // next read of a token will consume the lookahead token (the macro
                    // name) directly.
                    //
                    return;
                }

                // If we saw an opening `(`, then we know we are starting some kind of
                // macro invocation, although we don't yet know if it is well-formed.
                //
                MacroInvocation* invocation = new MacroInvocation(
                    preprocessor,
                    macro,
                    token.loc,
                    m_initiatingMacroInvocationLoc);

                // We start by consuming the opening `(` that we checked for above.
                //
                Token leftParen = m_inputStreams.readToken();
                SLANG_ASSERT(leftParen.type == TokenType::LParent);

                // Next we parse any arguments to the macro invocation, which will
                // consist of `()`-balanced sequences of tokens separated by `,`s.
                //
                _parseMacroArgs(macro, invocation);
                Index argCount = invocation->getArgCount();

                // We expect th arguments to be followed by a `)` to match the opening
                // `(`, and if we don't find one we need to diagnose the issue.
                //
                if (m_inputStreams.peekTokenType() == TokenType::RParent)
                {
                    m_inputStreams.readToken();
                }
                else
                {
                    GetSink(preprocessor)
                        ->diagnose(
                            m_inputStreams.peekLoc(),
                            Diagnostics::expectedTokenInMacroArguments,
                            TokenType::RParent,
                            m_inputStreams.peekTokenType());
                }

                // The number of arguments at the macro invocation site might not
                // match the number of arguments declared for the macro. In this
                // case we diagnose an issue *and* skip expansion of this invocation
                // (it effectively expands to zero new tokens).
                //
                const Index paramCount = Index(macro->params.getCount());
                if (!macro->isVariadic())
                {
                    // The non-variadic case is simple enough: either the argument
                    // count exactly matches the required parameter count, or we
                    // diagnose an error.
                    //
                    if (argCount != paramCount)
                    {
                        GetSink(preprocessor)
                            ->diagnose(
                                leftParen.loc,
                                Diagnostics::wrongNumberOfArgumentsToMacro,
                                paramCount,
                                argCount);
                        return;
                    }
                }
                else
                {
                    // In the variadic case, we only require arguments for the
                    // non-variadic parameters (all but the last one). In addition,
                    // we do not consider it an error to have more than the required
                    // number of arguments.
                    //
                    Index requiredArgCount = paramCount - 1;
                    if (argCount < requiredArgCount)
                    {
                        GetSink(preprocessor)
                            ->diagnose(
                                leftParen.loc,
                                Diagnostics::wrongNumberOfArgumentsToMacro,
                                requiredArgCount,
                                argCount);
                        return;
                    }
                }

                // Now that the arguments have been parsed and validated,
                // we are ready to proceed with expansion of the macro invocation.
                //
                // The main subtle thing we have to figure out is which macros should be considered
                // "busy" during the expansion of this function-like macro invocation.
                //
                // In the case of an object-like macro invocation:
                //
                //      1 + M + 2
                //            ^
                //
                // Input will have just read in the `M` token that names the macro
                // so we needed to consider whatever macro invocations had been in
                // flight (even if they were at their end) when checking if `M`
                // was busy.
                //
                // In contrast, for a function-like macro invocation:
                //
                //      1 + F ( A, B, C ) + 2
                //                        ^
                //
                // We will have just read in the `)` from the argument list, but
                // we don't actually need/want to worry about any macro invocation
                // that might have yielded the `)` token, since expanding that macro
                // again would *not* be able to lead to a recursive case.
                //
                // Instead, we really only care about the active stream that the
                // next token would be read from.
                //
                auto nextStream = m_inputStreams.getNextStream();
                auto busyMacrosForFunctionLikeInvocation =
                    nextStream->getFirstBusyMacroInvocation();

                invocation->prime(busyMacrosForFunctionLikeInvocation);
                _pushMacroInvocation(invocation);
            }
            break;
        }
    }
}

Token MacroInvocation::_readTokenImpl()
{
    // The `MacroInvocation` type maintains an invariant that after each
    // call to `_readTokenImpl`:
    //
    // * The `m_currentOpStreams` stack will be non-empty
    //
    // * The input state in `m_currentOpStreams` will correspond to the
    //   macro definition op at index `m_macroOpIndex`
    //
    // * The next token read from `m_currentOpStreams` will not be an EOF
    //   *unless* the expansion has reached the end of the macro invocaiton
    //
    // The first time `_readTokenImpl()` is called, it will only be able
    // to rely on the weaker invariant guaranteed by `_initCurrentOpStream()`:
    //
    // * The `m_currentOpStreams` stack will be non-empty
    //
    // * The input state in `m_currentOpStreams` will correspond to the
    //   macro definition op at index `m_macroOpIndex`
    //
    // * The next token read from `m_currentOpStreams` may be an EOF if
    //   the current op has an empty expansion.
    //
    // In either of those cases, we can start by reading the next token
    // from the expansion of the current op.
    //
    Token token = m_currentOpStreams.readToken();
    Index tokenOpIndex = m_macroOpIndex;

    // Once we've read that `token`, we need to work to establish or
    // re-establish our invariant, which we do by looping until we are
    // in a valid state.
    //
    for (;;)
    {
        // At the start of the loop, we already have the weaker invariant
        // guaranteed by `_initCurrentOpStream()`: the current op stream
        // is in a consistent state, but it *might* be at its end.
        //
        // If the current stream is *not* at its end, then we seem to
        // have the stronger invariant as well, and we can return.
        //
        if (m_currentOpStreams.peekTokenType() != TokenType::EndOfFile)
        {
            // We know that we have tokens remaining to read from
            // `m_currentOpStreams`, and we thus expect that the
            // `token` we just read must also be a non-EOF token.
            //
            // Note: This case is subtle, because this might be the first invocation
            // of `_readTokenImpl()` after the `_initCurrentOpStream()` call
            // as part of `prime()`. It seems that if the first macro op had
            // an empty expansion, then `token` might be the EOF for that op.
            //
            // That detail is handled below in the logic for switching to a new
            // macro op.
            //
            SLANG_ASSERT(token.type != TokenType::EndOfFile);

            // We can safely return with our invaraints intact, because
            // the next attempt to read a token will read a non-EOF.
            //
            return token;
        }

        // Otherwise, we have reached the end of the tokens coresponding
        // to the current op, and we want to try to advance to the next op
        // in the macro definition.
        //
        Index currentOpIndex = m_macroOpIndex;
        Index nextOpIndex = currentOpIndex + 1;

        // However, if we are already working on the last op in the macro
        // definition, then the next op index is out of range and we don't
        // want to advance. Instead we will keep the state of the macro
        // invocation where it is: at the end of the last op, returning
        // EOF tokens forever.
        //
        // Note that in this case we do not care whether `token` is an EOF
        // or not, because we expect the last op to yield an EOF at the
        // end of the macro expansion.
        //
        if (nextOpIndex == m_macro->ops.getCount())
            return token;

        // Because `m_currentOpStreams` is at its end, we can pop all of
        // those streams to reclaim their memory before we push any new
        // ones.
        //
        m_currentOpStreams.popAll();

        // Now we've commited to moving to the next op in the macro
        // definition, and we want to push appropriate streams onto
        // the stack of input streams to represent that op.
        //
        m_macroOpIndex = nextOpIndex;
        auto const& nextOp = m_macro->ops[nextOpIndex];

        // What we do depends on what the next op's opcode is.
        //
        switch (nextOp.opcode)
        {
        default:
            {
                // All of the easy cases are handled by `_initCurrentOpStream()`
                // which also gets invoked in the logic of `MacroInvocation::prime()`
                // to handle the first op in the definition.
                //
                // This operation will set up `m_currentOpStreams` so that it
                // accurately reflects the expansion of the op at index `m_macroOpIndex`.
                //
                // What it will *not* do is guarantee that the expansion for that
                // op is non-empty. We will thus continue the outer `for` loop which
                // checks whether the current op (which we just initialized here) is
                // already at its end.
                //
                _initCurrentOpStream();

                // Before we go back to the top of the loop, we need to deal with the
                // important corner case where `token` might have been an EOF because
                // the very first op in a macro body had an empty expansion, e.g.:
                //
                //      #define TWELVE(X) X 12 X
                //      TWELVE()
                //
                // In this case, the first `X` in the body of the macro will expand
                // to nothing, so once that op is set up by `_initCurrentOpStrem()`
                // the `token` we read here will be an EOF.
                //
                // The solution is to detect when all preceding ops considered by
                // this loop have been EOFs, and setting the value to the first
                // non-EOF token read.
                //
                if (token.type == TokenType::EndOfFile)
                {
                    token = m_currentOpStreams.readToken();
                    tokenOpIndex = m_macroOpIndex;
                }
            }
            break;

        case MacroDefinition::Opcode::TokenPaste:
            {
                // The more complicated case is a token paste (`##`).
                //
                Index tokenPasteTokenIndex = nextOp.index0;
                SourceLoc tokenPasteLoc = m_macro->tokens.m_tokens[tokenPasteTokenIndex].loc;

                // A `##` must always appear between two macro ops (whether literal tokens
                // or macro parameters) and it is supposed to paste together the last
                // token from the left op with the first token from the right op.
                //
                // We will accumulate the pasted token as a string and then re-lex it.
                //
                StringBuilder pastedContent;

                // Note that this is *not* the same as saying that we paste together the
                // last token the preceded the `##` with the first token that follows it.
                // In particular, if you have `L ## R` and either `L` or `R` has an empty
                // expansion, then the `##` should treat that operand as empty.
                //
                // As such, there's a few cases to consider here.

                // TODO: An extremely special case is the gcc-specific extension that allows
                // the use of `##` for eliding a comma when there are no arguments for a
                // variadic paameter, e.g.:
                //
                //      #define DEBUG(VALS...) debugImpl(__FILE__, __LINE__, ## VALS)
                //
                // Without the `##`, that case would risk producing an expression with a trailing
                // comma when invoked with no arguments (e.g., `DEBUG()`). The gcc-specific
                // behavior for `##` in this case discards the comma instead if the `VALS`
                // parameter had no arguments (which is *not* the same as having a single empty
                // argument).
                //
                // We could implement matching behavior in Slang with special-case logic here, but
                // doing so adds extra complexity so we may be better off avoiding it.
                //
                // The Microsoft C++ compiler automatically discards commas in a case like this
                // whether or not `##` has been used, except when certain flags to enable strict
                // compliance to standards are used. Emulating this behavior would be another
                // option.
                //
                // Later version of the C++ standard add `__VA_OPT__(...)` which can be used to
                // include/exclude tokens in an expansion based on whether or not any arguments
                // were provided for a variadic parameter. This is a relatively complicated feature
                // to try and replicate
                //
                // For Slang it may be simplest to solve this problem at the parser level, by
                // allowing trailing commas in argument lists without error/warning. However, if we
                // *do* decide to implement the gcc extension for `##` it would be logical to try to
                // detect and intercept that special case here.

                // If the `tokenOpIndex` that `token` was read from is the op right
                // before the `##`, then we know it is the last token produced by
                // the preceding op (or possibly an EOF if that op's expansion was empty).
                //
                if (tokenOpIndex == nextOpIndex - 1)
                {
                    if (token.type != TokenType::EndOfFile)
                    {
                        pastedContent << token.getContent();
                    }
                }
                else
                {
                    // Otherwise, the op that preceded the `##` was *not* the same op
                    // that produced `token`, which could only happen if that preceding
                    // op was one that was initialized by this loop and then found to
                    // have an empty expansion. As such, we don't need to add anything
                    // onto `pastedContent` in this case.
                }

                // Once we've dealt with the token to the left of the `##` (if any)
                // we can turn our attention to the token to the right.
                //
                // This token will be the first token (if any) to be produced by whatever
                // op follows the `##`. We will thus start by initialiing the `m_currentOpStrems`
                // for reading from that op.
                //
                m_macroOpIndex++;
                _initCurrentOpStream();

                // If the right operand yields at least one non-EOF token, then we need
                // to append that content to our paste result.
                //
                Token rightToken = m_currentOpStreams.readToken();
                if (rightToken.type != TokenType::EndOfFile)
                    pastedContent << rightToken.getContent();

                // Now we need to re-lex the token(s) that resulted from pasting, which requires
                // us to create a fresh source file to represent the paste result.
                //
                PathInfo pathInfo = PathInfo::makeTokenPaste();
                SourceManager* sourceManager = m_preprocessor->getSourceManager();
                SourceFile* sourceFile = sourceManager->createSourceFileWithString(
                    pathInfo,
                    pastedContent.produceString());
                SourceView* sourceView =
                    sourceManager->createSourceView(sourceFile, nullptr, tokenPasteLoc);

                Lexer lexer;
                lexer.initialize(
                    sourceView,
                    GetSink(m_preprocessor),
                    m_preprocessor->getNamePool(),
                    sourceManager->getMemoryArena());
                auto lexedTokens = lexer.lexAllSemanticTokens();

                // The `lexedTokens` will always contain at least one token, representing an EOF for
                // the end of the lexed token squence.
                //
                // Because we have concatenated together the content of zero, one, or two different
                // tokens, there are many cases for what the result could be:
                //
                // * The content could lex as zero tokens, followed by an EOF. This would happen if
                //   both the left and right operands to `##` were empty.
                //
                // * The content could lex to one token, followed by an EOF. This could happen if
                //   one operand was empty but not the other, or if the left and right tokens
                //   concatenated to form a single valid token.
                //
                // * The content could lex to more than one token, for cases like `+` pasted with
                // `-`,
                //   where the result is not a valid single token.
                //
                // The first two cases are both considered valid token pastes, while the latter
                // should be diagnosed as a warning, even if it is clear how we can handle it.
                //
                if (lexedTokens.m_tokens.getCount() > 2)
                {
                    getSink()->diagnose(
                        tokenPasteLoc,
                        Diagnostics::invalidTokenPasteResult,
                        pastedContent);
                }

                // No matter what sequence of tokens we got, we can create an input stream to
                // represent them and push it as the representation of the `##` macro definition op.
                //
                // Note: the stream(s) created for the right operand will be on the stack under the
                // new one we push for the pasted tokens, and as such the input state is capable of
                // reading from both the input stream for the `##` through to the input for the
                // right-hand-side op, which is consistent with `m_macroOpIndex`.
                //
                SingleUseInputStream* inputStream =
                    new SingleUseInputStream(m_preprocessor, lexedTokens);
                m_currentOpStreams.push(inputStream);

                // There's one final detail to cover before we move on. *If* we used `token` as part
                // of the content of the token paste, *or* if `token` is an EOF, then we need to
                // replace it with the first token read from the expansion.
                //
                // (Otherwise, the `##` is being initialized as part of advancing through ops with
                // empty expansion to the right of the op for a non-EOF `token`)
                //
                if ((tokenOpIndex == nextOpIndex - 1) || token.type == TokenType::EndOfFile)
                {
                    // Note that `tokenOpIndex` is being set here to the op index for the
                    // right-hand operand to the `##`. This is appropriate for cases where
                    // you might have chained `##` ops:
                    //
                    //      #define F(X,Y,Z) X ## Y ## Z
                    //
                    // If `Y` expands to a single token, then `X ## Y` should be treated
                    // as the left operand to the `Y ## Z` paste.
                    //
                    token = m_currentOpStreams.readToken();
                    tokenOpIndex = m_macroOpIndex;
                }

                // At this point we are ready to head back to the top of the loop and see
                // if our invariants have been re-established.
            }
            break;
        }
    }
}

void MacroInvocation::_pushSingleTokenStream(
    TokenType tokenType,
    SourceLoc tokenLoc,
    UnownedStringSlice const& content)
{
    // The goal here is to push a token stream that represents a single token
    // with exactly the given `content`, etc.
    //
    // We are going to keep the content alive using the slice pool for the source
    // manager, which will also lead to it being shared if used multiple times.
    //
    SourceManager* sourceManager = m_preprocessor->getSourceManager();
    auto& pool = sourceManager->getStringSlicePool();
    auto poolHandle = pool.add(content);
    auto slice = pool.getSlice(poolHandle);

    Token token;
    token.type = tokenType;
    token.setContent(slice);
    token.loc = tokenLoc;

    TokenList lexedTokens;
    lexedTokens.add(token);

    // Every token list needs to be terminated with an EOF,
    // so we will construct one that matches the location
    // for the `token`.
    //
    Token eofToken;
    eofToken.type = TokenType::EndOfFile;
    eofToken.loc = token.loc;
    eofToken.flags = TokenFlag::AfterWhitespace | TokenFlag::AtStartOfLine;
    lexedTokens.add(eofToken);

    SingleUseInputStream* inputStream = new SingleUseInputStream(m_preprocessor, lexedTokens);
    m_currentOpStreams.push(inputStream);
}

template<typename F>
void MacroInvocation::_pushStreamForSourceLocBuiltin(TokenType tokenType, F const& valueBuilder)
{
    // The `__LINE__` and `__FILE__` macros will always expand based on
    // the "initiating" source location, which should come from the
    // top-level file instead of any nested macros being expanded.
    //
    const SourceLoc initiatingLoc = m_initiatingMacroInvocationLoc;
    if (!initiatingLoc.isValid())
    {
        // If we cannot find a valid source location for the initiating
        // location, then we will not expand the macro.
        //
        // TODO: Maybe we should issue a diagnostic here?
        //
        return;
    }

    SourceManager* sourceManager = m_preprocessor->getSourceManager();
    HumaneSourceLoc humaneInitiatingLoc = sourceManager->getHumaneLoc(initiatingLoc);

    // The `valueBuilder` provided by the caller will determine what the content
    // of the token will be based on the source location (either to generate the
    // `__LINE__` or the `__FILE__` value).
    //
    StringBuilder content;
    valueBuilder(content, humaneInitiatingLoc);

    // Next we constuct and push an input stream with exactly the token type and content we want.
    //
    _pushSingleTokenStream(tokenType, m_macroInvocationLoc, content.getUnownedSlice());
}

TokenReader MacroInvocation::_getArgTokens(Index paramIndex)
{
    SLANG_ASSERT(paramIndex >= 0);
    SLANG_ASSERT(paramIndex < m_macro->params.getCount());

    // How we determine the range of argument tokens for a parameter
    // depends on whether or not it is a variadic parameter.
    //
    auto& param = m_macro->params[paramIndex];
    auto argTokens = m_argTokens.getBuffer();
    if (!param.isVariadic)
    {
        // The non-variadic case is, as expected, the simpler one.
        //
        // We expect that there must be an argument at the index corresponding
        // to the parameter, and we construct a `TokenReader` that will play
        // back the tokens of that argument.
        //
        // Special case: If we have no arguments but the macro expects one parameter,
        // return an empty token range
        if (m_args.getCount() == 0 && m_macro->params.getCount() == 1)
        {
            return TokenReader(argTokens, argTokens);
        }

        SLANG_ASSERT(paramIndex < m_args.getCount());
        auto arg = m_args[paramIndex];
        return TokenReader(argTokens + arg.beginTokenIndex, argTokens + arg.endTokenIndex);
    }
    else
    {
        // In the variadic case, it is possible that we have zero or more
        // arguments that will all need to be played back in any place where
        // the variadic parameter is referenced.
        //
        // The first relevant argument is the one at the index coresponding
        // to the variadic parameter, if any. The last relevant argument is
        // the last argument to the invocation, *if* there was a first
        // relevant argument.
        //
        Index firstArgIndex = paramIndex;
        Index lastArgIndex = m_args.getCount() - 1;

        // One special case is when there are *no* arguments coresponding
        // to the variadic parameter.
        //
        if (firstArgIndex > lastArgIndex)
        {
            // When there are no arguments for the varaidic parameter we will
            // construct an empty token range that comes after the other arguments.
            //
            if (lastArgIndex >= 0)
            {
                auto arg = m_args[lastArgIndex];
                return TokenReader(argTokens + arg.endTokenIndex, argTokens + arg.endTokenIndex);
            }
            else
            {
                return TokenReader(argTokens, argTokens);
            }
        }

        // Because the `m_argTokens` array includes the commas between arguments,
        // we can get the token sequence we want simply by making a reader that spans
        // all the tokens between the first and last argument (inclusive) that correspond
        // to the variadic parameter.
        //
        auto firstArg = m_args[firstArgIndex];
        auto lastArg = m_args[lastArgIndex];
        return TokenReader(argTokens + firstArg.beginTokenIndex, argTokens + lastArg.endTokenIndex);
    }
}

void MacroInvocation::_initCurrentOpStream()
{
    // The job of this function is to make sure that `m_currentOpStreams` is set up
    // to refelct the state of the op at `m_macroOpIndex`.
    //
    Index opIndex = m_macroOpIndex;
    auto& op = m_macro->ops[opIndex];

    // As one might expect, the setup logic to apply depends on the opcode for the op.
    //
    switch (op.opcode)
    {
    default:
        SLANG_UNEXPECTED("unhandled macro opcode case");
        break;

    case MacroDefinition::Opcode::RawSpan:
        {
            // A raw span of tokens (no use of macro parameters, etc.) is easy enough
            // to handle. The operands of the op give us the begin/end index of the
            // tokens in the macro definition that we'd like to use.
            //
            Index beginTokenIndex = op.index0;
            Index endTokenIndex = op.index1;

            // Because the macro definition stores its definition tokens directly, we
            // can simply construct a token reader for reading from the tokens in
            // the chosen range, and push a matching input stream.
            //
            auto tokenBuffer = m_macro->tokens.begin();
            auto tokenReader =
                TokenReader(tokenBuffer + beginTokenIndex, tokenBuffer + endTokenIndex);
            PretokenizedInputStream* stream =
                new PretokenizedInputStream(m_preprocessor, tokenReader);
            m_currentOpStreams.push(stream);
        }
        break;

    case MacroDefinition::Opcode::UnexpandedParam:
        {
            // When a macro parameter is referenced as an operand of a token paste (`##`)
            // it is not subjected to macro expansion.
            //
            // In this case, the zero-based index of the macro parameter was stored in
            // the `index1` operand to the macro op.
            //
            Index paramIndex = op.index1;

            // We can look up the corresponding argument to the macro invocation,
            // which stores a begin/end pair of indices into the raw token stream
            // that makes up the macro arguments.
            //
            auto tokenReader = _getArgTokens(paramIndex);

            // Because expansion doesn't apply to this parameter reference, we can simply
            // play back those tokens exactly as they appeared in the argument list.
            //
            PretokenizedInputStream* stream =
                new PretokenizedInputStream(m_preprocessor, tokenReader);
            m_currentOpStreams.push(stream);
        }
        break;

    case MacroDefinition::Opcode::ExpandedParam:
        {
            // Most uses of a macro parameter will be subject to macro expansion.
            //
            // The initial logic here is similar to the unexpanded case above.
            //
            Index paramIndex = op.index1;
            auto tokenReader = _getArgTokens(paramIndex);
            PretokenizedInputStream* stream =
                new PretokenizedInputStream(m_preprocessor, tokenReader);

            // The only interesting addition to the unexpanded case is that we wrap
            // the stream that "plays back" the argument tokens with a stream that
            // applies macro expansion to them.
            //
            ExpansionInputStream* expansion = new ExpansionInputStream(m_preprocessor, stream);
            expansion->setInitiatingMacroSourceLoc(m_initiatingMacroInvocationLoc);
            m_currentOpStreams.push(expansion);
        }
        break;

    case MacroDefinition::Opcode::StringizedParam:
        {
            // A macro parameter can also be "stringized" in which case the (unexpanded)
            // argument tokens will be concatenated and escaped to form the content of
            // a string literal.
            //
            // Much of the initial logic is shared with the other parameter cases above.
            //
            Index tokenIndex = op.index0;
            auto loc = m_macro->tokens.m_tokens[tokenIndex].loc;

            Index paramIndex = op.index1;
            auto tokenReader = _getArgTokens(paramIndex);

            // A stringized parameter is always a `"`-enclosed string literal
            // (there is no way to stringize things to form a character literal).
            //
            StringBuilder builder;
            builder.appendChar('"');
            for (bool first = true; !tokenReader.isAtEnd(); first = false)
            {
                auto token = tokenReader.advanceToken();

                // Any whitespace between the tokens of argument must be collapsed into
                // a single space character. Fortunately for us, the lexer has tracked
                // for each token whether it was immediately preceded by whitespace,
                // so we can check for whitespace that precedes any token except the first.
                //
                if (!first && (token.flags & TokenFlag::AfterWhitespace))
                {
                    builder.appendChar(' ');
                }

                // We need to rememember to apply escaping to the content of any tokens
                // being pulled into the string. E.g., this would come up if we end up
                // trying to stringize a literal like `"this"` because we need the resulting
                // token to be `"\"this\""` which includes the quote characters in the string
                // literal value.
                //
                auto handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);
                handler->appendEscaped(token.getContent(), builder);
            }
            builder.appendChar('"');

            // Once we've constructed the content of the stringized result, we need to push
            // a new single-token stream that represents that content.
            //
            _pushSingleTokenStream(TokenType::StringLiteral, loc, builder.getUnownedSlice());
        }
        break;

    case MacroDefinition::Opcode::BuiltinLine:
        {
            // This is a special opcode used only in the definition of the built-in `__LINE__` macro
            // (note that *uses* of `__LINE__` do not map to this opcode; only the definition of
            // `__LINE__` itself directly uses it).
            //
            // Most of the logic for generating a token from the current source location is wrapped
            // up in a helper routine so that we don't need to duplicate it between this and the
            // `__FILE__` case below.
            //
            // The only key details here are that we specify the type of the token
            // (`IntegerLiteral`) and its content (the value of `loc.line`).
            //
            _pushStreamForSourceLocBuiltin(
                TokenType::IntegerLiteral,
                [=](StringBuilder& builder, HumaneSourceLoc const& loc) { builder << loc.line; });
        }
        break;

    case MacroDefinition::Opcode::BuiltinFile:
        {
            // The `__FILE__` case is quite similar to `__LINE__`, except for the type of token it
            // yields, and the way it computes the desired token content.
            //
            _pushStreamForSourceLocBuiltin(
                TokenType::StringLiteral,
                [=](StringBuilder& builder, HumaneSourceLoc const& loc)
                {
                    auto escapeHandler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp);
                    StringEscapeUtil::appendQuoted(
                        escapeHandler,
                        loc.pathInfo.foundPath.getUnownedSlice(),
                        builder);
                });
        }
        break;

    case MacroDefinition::Opcode::TokenPaste:
        // Note: If we ever end up in this case for `Opcode::TokenPaste`, then it implies
        // something went very wrong.
        //
        // A `##` op should not be allowed to appear as the first (or last) token in
        // a macro body, and consecutive `##`s should be treated as a single `##`.
        //
        // When `_initCurrentOpStream()` gets called it is either:
        //
        // * called on the first op in the body of a macro (can't be a token paste)
        //
        // * called on the first op *after* a `##` (can't be another `##`)
        //
        // * explicitly tests for an handles token pastes spearately
        //
        // If we end up hitting the error here, then `_initCurrentOpStream()` is getting
        // called in an inappropriate case.
        //
        SLANG_UNEXPECTED("token paste op in macro expansion");
        break;
    }
}

//
// Preprocessor Directives
//

// When reading a preprocessor directive, we use a context
// to wrap the direct preprocessor routines defines so far.
//
// One of the most important things the directive context
// does is give us a convenient way to read tokens with
// a guarantee that we won't read past the end of a line.
struct PreprocessorDirectiveContext
{
    // The preprocessor that is parsing the directive.
    Preprocessor* m_preprocessor;

    // The directive token (e.g., the `if` in `#if`).
    // Useful for reference in diagnostic messages.
    Token m_directiveToken;

    // Has any kind of parse error been encountered in
    // the directive so far?
    bool m_parseError;

    // Have we done the necessary checks at the end
    // of the directive already?
    bool m_haveDoneEndOfDirectiveChecks;

    /// The input file that the directive appeared in
    ///
    InputFile* m_inputFile;
};

// Get the token for  the preprocessor directive being parsed.
inline Token const& GetDirective(PreprocessorDirectiveContext* context)
{
    return context->m_directiveToken;
}

// Get the name of the directive being parsed.
inline UnownedStringSlice GetDirectiveName(PreprocessorDirectiveContext* context)
{
    return context->m_directiveToken.getContent();
}

// Get the location of the directive being parsed.
inline SourceLoc const& GetDirectiveLoc(PreprocessorDirectiveContext* context)
{
    return context->m_directiveToken.loc;
}

// Wrapper to get the diagnostic sink in the context of a directive.
static inline DiagnosticSink* GetSink(PreprocessorDirectiveContext* context)
{
    return GetSink(context->m_preprocessor);
}

static InputFile* getInputFile(PreprocessorDirectiveContext* context)
{
    return context->m_inputFile;
}

static ExpansionInputStream* getInputStream(PreprocessorDirectiveContext* context)
{
    return context->m_inputFile->getExpansionStream();
}

// Wrapper to get a "current" location when parsing a directive
static SourceLoc PeekLoc(PreprocessorDirectiveContext* context)
{
    auto inputStream = getInputStream(context);
    return inputStream->peekLoc();
}

// Wrapper to look up a macro in the context of a directive.
static MacroDefinition* LookupMacro(PreprocessorDirectiveContext* context, Name* name)
{
    auto preprocessor = context->m_preprocessor;
    return LookupMacro(&preprocessor->globalEnv, name);
}

// Determine if we have read everything on the directive's line.
static bool IsEndOfLine(PreprocessorDirectiveContext* context)
{
    auto inputStream = getInputStream(context);
    switch (inputStream->peekRawTokenType())
    {
    case TokenType::EndOfFile:
    case TokenType::NewLine:
        return true;

    default:
        return false;
    }
}


// Peek one raw token in a directive, without going past the end of the line.
static Token PeekRawToken(PreprocessorDirectiveContext* context)
{
    auto inputStream = getInputStream(context);
    return inputStream->peekRawToken();
}

// Read one raw token in a directive, without going past the end of the line.
static Token AdvanceRawToken(PreprocessorDirectiveContext* context)
{
    auto inputStream = getInputStream(context);
    return inputStream->readRawToken();
}

// Peek next raw token type, without going past the end of the line.
static TokenType PeekRawTokenType(PreprocessorDirectiveContext* context)
{
    auto inputStream = getInputStream(context);
    return inputStream->peekRawTokenType();
}

// Read one token, with macro-expansion, without going past the end of the line.
static Token AdvanceToken(PreprocessorDirectiveContext* context)
{
    if (IsEndOfLine(context))
        return PeekRawToken(context);
    return getInputStream(context)->readToken();
}

// Peek one token, with macro-expansion, without going past the end of the line.
static Token PeekToken(PreprocessorDirectiveContext* context)
{
    auto inputStream = getInputStream(context);
    return inputStream->peekToken();
}

// Peek next token type, with macro-expansion, without going past the end of the line.
static TokenType PeekTokenType(PreprocessorDirectiveContext* context)
{
    auto inputStream = getInputStream(context);
    return inputStream->peekTokenType();
}

// Skip to the end of the line (useful for recovering from errors in a directive)
static void SkipToEndOfLine(PreprocessorDirectiveContext* context)
{
    while (!IsEndOfLine(context))
    {
        AdvanceRawToken(context);
    }
}

static SourceLoc FindNextEndOfLine(
    PreprocessorDirectiveContext* context,
    SourceLoc from,
    UInt lineCount = 1)
{
    auto inputStream = getInputStream(context);
    return inputStream->findNextLineEnd(from, lineCount);
}

static bool ExpectRaw(
    PreprocessorDirectiveContext* context,
    TokenType tokenType,
    DiagnosticInfo const& diagnostic,
    Token* outToken = NULL)
{
    if (PeekRawTokenType(context) != tokenType)
    {
        // Only report the first parse error within a directive
        if (!context->m_parseError)
        {
            GetSink(context)
                ->diagnose(PeekLoc(context), diagnostic, tokenType, GetDirectiveName(context));
        }
        context->m_parseError = true;
        return false;
    }
    Token const& token = AdvanceRawToken(context);
    if (outToken)
        *outToken = token;
    return true;
}

static bool Expect(
    PreprocessorDirectiveContext* context,
    TokenType tokenType,
    DiagnosticInfo const& diagnostic,
    Token* outToken = NULL)
{
    if (PeekTokenType(context) != tokenType)
    {
        // Only report the first parse error within a directive
        if (!context->m_parseError)
        {
            GetSink(context)
                ->diagnose(PeekLoc(context), diagnostic, tokenType, GetDirectiveName(context));
            context->m_parseError = true;
        }
        return false;
    }
    Token const& token = AdvanceToken(context);
    if (outToken)
        *outToken = token;
    return true;
}


//
// Preprocessor Conditionals
//

bool InputFile::isSkipping()
{
    // If we are not inside a preprocessor conditional, then don't skip
    Conditional* conditional = m_conditional;
    if (!conditional)
        return false;

    // skip tokens unless the conditional is inside its `true` case
    return conditional->state != Conditional::State::During;
}

// Wrapper for use inside directives
static inline bool isSkipping(PreprocessorDirectiveContext* context)
{
    return getInputFile(context)->isSkipping();
}

// Create a preprocessor conditional
static Conditional* CreateConditional(Preprocessor* /*preprocessor*/)
{
    // TODO(tfoley): allocate these more intelligently (for example,
    // pool them on the `Preprocessor`.
    return new Conditional();
}

static void _setLexerDiagnosticSuppression(InputFile* inputFile, bool shouldSuppressDiagnostics)
{
    if (shouldSuppressDiagnostics)
    {
        inputFile->getLexer()->m_lexerFlags |= kLexerFlag_SuppressDiagnostics;
    }
    else
    {
        inputFile->getLexer()->m_lexerFlags &= ~kLexerFlag_SuppressDiagnostics;
    }
}


static void updateLexerFlagsForConditionals(InputFile* inputFile)
{
    _setLexerDiagnosticSuppression(inputFile, inputFile->isSkipping());
}

/// Start a preprocessor conditional, with an initial enable/disable state.
static void beginConditional(PreprocessorDirectiveContext* context, bool enable)
{
    Preprocessor* preprocessor = context->m_preprocessor;
    InputFile* inputFile = getInputFile(context);

    Conditional* conditional = CreateConditional(preprocessor);

    conditional->ifToken = context->m_directiveToken;

    // Set state of this condition appropriately.
    //
    // Default to the "haven't yet seen a `true` branch" state.
    Conditional::State state = Conditional::State::Before;
    //
    // If we are nested inside a `false` branch of another condition, then
    // we never want to enable, so we act as if we already *saw* the `true` branch.
    //
    if (inputFile->isSkipping())
        state = Conditional::State::After;
    //
    // Otherwise, if our condition was true, then set us to be inside the `true` branch
    else if (enable)
        state = Conditional::State::During;

    conditional->state = state;

    // Push conditional onto the stack
    inputFile->pushConditional(conditional);

    updateLexerFlagsForConditionals(inputFile);
}

//
// Preprocessor Conditional Expressions
//

// Conditional expressions are always of type `int`
typedef int PreprocessorExpressionValue;

// Forward-declaretion
static PreprocessorExpressionValue _parseAndEvaluateExpression(
    PreprocessorDirectiveContext* context);

// Parse a unary (prefix) expression inside of a preprocessor directive.
static PreprocessorExpressionValue ParseAndEvaluateUnaryExpression(
    PreprocessorDirectiveContext* context)
{
    switch (PeekTokenType(context))
    {
    case TokenType::EndOfFile:
    case TokenType::NewLine:
        GetSink(context)->diagnose(
            PeekLoc(context),
            Diagnostics::syntaxErrorInPreprocessorExpression);
        return 0;
    }

    auto token = AdvanceToken(context);
    switch (token.type)
    {
    // handle prefix unary ops
    case TokenType::OpSub:
        return -ParseAndEvaluateUnaryExpression(context);
    case TokenType::OpNot:
        return !ParseAndEvaluateUnaryExpression(context);
    case TokenType::OpBitNot:
        return ~ParseAndEvaluateUnaryExpression(context);

    // handle parenthized sub-expression
    case TokenType::LParent:
        {
            Token leftParen = token;
            PreprocessorExpressionValue value = _parseAndEvaluateExpression(context);
            if (!Expect(
                    context,
                    TokenType::RParent,
                    Diagnostics::expectedTokenInPreprocessorExpression))
            {
                GetSink(context)->diagnose(leftParen.loc, Diagnostics::seeOpeningToken, leftParen);
            }
            return value;
        }

    case TokenType::IntegerLiteral:
        return stringToInt(token.getContent());

    case TokenType::Identifier:
        {
            if (token.getContent() == "defined")
            {
                // handle `defined(someName)`

                // Possibly parse a `(`
                Token leftParen;
                if (PeekRawTokenType(context) == TokenType::LParent)
                {
                    leftParen = AdvanceRawToken(context);
                }

                // Expect an identifier
                Token nameToken;
                if (!ExpectRaw(
                        context,
                        TokenType::Identifier,
                        Diagnostics::expectedTokenInDefinedExpression,
                        &nameToken))
                {
                    return 0;
                }
                Name* name = nameToken.getName();

                // If we saw an opening `(`, then expect one to close
                if (leftParen.type != TokenType::Unknown)
                {
                    if (!ExpectRaw(
                            context,
                            TokenType::RParent,
                            Diagnostics::expectedTokenInDefinedExpression))
                    {
                        GetSink(context)->diagnose(
                            leftParen.loc,
                            Diagnostics::seeOpeningToken,
                            leftParen);
                        return 0;
                    }
                }

                return LookupMacro(context, name) != NULL;
            }
            else if (token.getContent() == "__has_feature")
            {
                // handle `defined(someName)`

                // Possibly parse a `(`
                Token leftParen;
                if (PeekRawTokenType(context) == TokenType::LParent)
                {
                    leftParen = AdvanceRawToken(context);
                }

                // Expect an identifier
                Token nameToken;
                if (!ExpectRaw(
                        context,
                        TokenType::Identifier,
                        Diagnostics::expectedTokenInDefinedExpression,
                        &nameToken))
                {
                    return 0;
                }

                // If we saw an opening `(`, then expect one to close
                if (leftParen.type != TokenType::Unknown)
                {
                    if (!ExpectRaw(
                            context,
                            TokenType::RParent,
                            Diagnostics::expectedTokenInDefinedExpression))
                    {
                        GetSink(context)->diagnose(
                            leftParen.loc,
                            Diagnostics::seeOpeningToken,
                            leftParen);
                        return 0;
                    }
                }

                if (nameToken.getContent() == "hlsl_vk_buffer_pointer")
                {
                    return 1;
                }
                return 0;
            }
            // An identifier here means it was not defined as a macro (or
            // it is defined, but as a function-like macro. These should
            // just evaluate to zero (possibly with a warning)
            GetSink(context)->diagnose(
                token.loc,
                Diagnostics::undefinedIdentifierInPreprocessorExpression,
                token.getName());
            return 0;
        }

    default:
        GetSink(context)->diagnose(token.loc, Diagnostics::syntaxErrorInPreprocessorExpression);
        return 0;
    }
}

// Determine the precedence level of an infix operator
// for use in parsing preprocessor conditionals.
static int GetInfixOpPrecedence(Token const& opToken)
{
    // If token is on another line, it is not part of the
    // expression
    if (opToken.flags & TokenFlag::AtStartOfLine)
        return -1;

    // otherwise we look at the token type to figure
    // out what precedence it should be parse with
    switch (opToken.type)
    {
    default:
        // tokens that aren't infix operators should
        // cause us to stop parsing an expression
        return -1;

    case TokenType::OpMul:
        return 10;
    case TokenType::OpDiv:
        return 10;
    case TokenType::OpMod:
        return 10;

    case TokenType::OpAdd:
        return 9;
    case TokenType::OpSub:
        return 9;

    case TokenType::OpLsh:
        return 8;
    case TokenType::OpRsh:
        return 8;

    case TokenType::OpLess:
        return 7;
    case TokenType::OpGreater:
        return 7;
    case TokenType::OpLeq:
        return 7;
    case TokenType::OpGeq:
        return 7;

    case TokenType::OpEql:
        return 6;
    case TokenType::OpNeq:
        return 6;

    case TokenType::OpBitAnd:
        return 5;
    case TokenType::OpBitOr:
        return 4;
    case TokenType::OpBitXor:
        return 3;
    case TokenType::OpAnd:
        return 2;
    case TokenType::OpOr:
        return 1;
    }
};

// Evaluate one infix operation in a preprocessor
// conditional expression
static PreprocessorExpressionValue EvaluateInfixOp(
    PreprocessorDirectiveContext* context,
    Token const& opToken,
    PreprocessorExpressionValue left,
    PreprocessorExpressionValue right)
{
    switch (opToken.type)
    {
    default:
        //        SLANG_INTERNAL_ERROR(getSink(preprocessor), opToken);
        return 0;
        break;

    case TokenType::OpMul:
        return left * right;
    case TokenType::OpDiv:
        {
            if (right == 0)
            {
                if (!context->m_parseError)
                {
                    GetSink(context)->diagnose(
                        opToken.loc,
                        Diagnostics::divideByZeroInPreprocessorExpression);
                }
                return 0;
            }
            return left / right;
        }
    case TokenType::OpMod:
        {
            if (right == 0)
            {
                if (!context->m_parseError)
                {
                    GetSink(context)->diagnose(
                        opToken.loc,
                        Diagnostics::divideByZeroInPreprocessorExpression);
                }
                return 0;
            }
            return left % right;
        }
    case TokenType::OpAdd:
        return left + right;
    case TokenType::OpSub:
        return left - right;
    case TokenType::OpLsh:
        return left << right;
    case TokenType::OpRsh:
        return left >> right;
    case TokenType::OpLess:
        return left < right ? 1 : 0;
    case TokenType::OpGreater:
        return left > right ? 1 : 0;
    case TokenType::OpLeq:
        return left <= right ? 1 : 0;
    case TokenType::OpGeq:
        return left >= right ? 1 : 0;
    case TokenType::OpEql:
        return left == right ? 1 : 0;
    case TokenType::OpNeq:
        return left != right ? 1 : 0;
    case TokenType::OpBitAnd:
        return left & right;
    case TokenType::OpBitOr:
        return left | right;
    case TokenType::OpBitXor:
        return left ^ right;
    case TokenType::OpAnd:
        return left && right;
    case TokenType::OpOr:
        return left || right;
    }
}

// Parse the rest of an infix preprocessor expression with
// precedence greater than or equal to the given `precedence` argument.
// The value of the left-hand-side expression is provided as
// an argument.
// This is used to form a simple recursive-descent expression parser.
static PreprocessorExpressionValue ParseAndEvaluateInfixExpressionWithPrecedence(
    PreprocessorDirectiveContext* context,
    PreprocessorExpressionValue left,
    int precedence)
{
    for (;;)
    {
        // Look at the next token, and see if it is an operator of
        // high enough precedence to be included in our expression
        Token opToken = PeekToken(context);
        int opPrecedence = GetInfixOpPrecedence(opToken);

        // If it isn't an operator of high enough precedence, we are done.
        if (opPrecedence < precedence)
            break;

        // Otherwise we need to consume the operator token.
        AdvanceToken(context);

        // Next we parse a right-hand-side expression by starting with
        // a unary expression and absorbing and many infix operators
        // as possible with strictly higher precedence than the operator
        // we found above.
        PreprocessorExpressionValue right = ParseAndEvaluateUnaryExpression(context);
        for (;;)
        {
            // Look for an operator token
            Token rightOpToken = PeekToken(context);
            int rightOpPrecedence = GetInfixOpPrecedence(rightOpToken);

            // If no operator was found, or the operator wasn't high
            // enough precedence to fold into the right-hand-side,
            // exit this loop.
            if (rightOpPrecedence <= opPrecedence)
                break;

            // Now invoke the parser recursively, passing in our
            // existing right-hand side to form an even larger one.
            right =
                ParseAndEvaluateInfixExpressionWithPrecedence(context, right, rightOpPrecedence);
        }

        // Now combine the left- and right-hand sides using
        // the operator we found above.
        left = EvaluateInfixOp(context, opToken, left, right);
    }
    return left;
}

/// Parse a complete (infix) preprocessor expression, and return its value
static PreprocessorExpressionValue _parseAndEvaluateExpression(
    PreprocessorDirectiveContext* context)
{
    // First read in the left-hand side (or the whole expression in the unary case)
    PreprocessorExpressionValue value = ParseAndEvaluateUnaryExpression(context);

    // Try to read in trailing infix operators with correct precedence
    return ParseAndEvaluateInfixExpressionWithPrecedence(context, value, 0);
}
/// Parse a preprocessor expression, or skip it if we are in a disabled conditional
static PreprocessorExpressionValue _skipOrParseAndEvaluateExpression(
    PreprocessorDirectiveContext* context)
{
    auto inputStream = getInputFile(context);

    // If we are skipping, we want to ignore the expression (including
    // anything in it that would lead to a failure in parsing).
    //
    // We can simply treat the expression as `0` in this case, since its
    // value won't actually matter.
    //
    if (inputStream->isSkipping())
    {
        // Consume everything until the end of the line
        SkipToEndOfLine(context);
        return 0;
    }

    // Otherwise, we will need to parse an expression and return
    // its evaluated value.
    //
    return _parseAndEvaluateExpression(context);
}

// Handle a `#if` directive
static void HandleIfDirective(PreprocessorDirectiveContext* context)
{
    // Read a preprocessor expression (if not skipping), and begin a conditional
    // based on the value of that expression.
    //
    PreprocessorExpressionValue value = _skipOrParseAndEvaluateExpression(context);
    beginConditional(context, value != 0);
}

// Handle a `#ifdef` directive
static void HandleIfDefDirective(PreprocessorDirectiveContext* context)
{
    // Expect a raw identifier, so we can check if it is defined
    Token nameToken;
    if (!ExpectRaw(
            context,
            TokenType::Identifier,
            Diagnostics::expectedTokenInPreprocessorDirective,
            &nameToken))
        return;
    Name* name = nameToken.getName();

    // Check if the name is defined.
    beginConditional(context, LookupMacro(context, name) != NULL);
}

// Handle a `#ifndef` directive
static void HandleIfNDefDirective(PreprocessorDirectiveContext* context)
{
    // Expect a raw identifier, so we can check if it is defined
    Token nameToken;
    if (!ExpectRaw(
            context,
            TokenType::Identifier,
            Diagnostics::expectedTokenInPreprocessorDirective,
            &nameToken))
        return;
    Name* name = nameToken.getName();

    // Check if the name is defined.
    beginConditional(context, LookupMacro(context, name) == NULL);
}

// Handle a `#else` directive
static void HandleElseDirective(PreprocessorDirectiveContext* context)
{
    InputFile* inputFile = getInputFile(context);
    SLANG_ASSERT(inputFile);

    // if we aren't inside a conditional, then error
    Conditional* conditional = inputFile->getInnerMostConditional();
    if (!conditional)
    {
        GetSink(context)->diagnose(
            GetDirectiveLoc(context),
            Diagnostics::directiveWithoutIf,
            GetDirectiveName(context));
        return;
    }

    // if we've already seen a `#else`, then it is an error
    if (conditional->elseToken.type != TokenType::Unknown)
    {
        GetSink(context)->diagnose(
            GetDirectiveLoc(context),
            Diagnostics::directiveAfterElse,
            GetDirectiveName(context));
        GetSink(context)->diagnose(conditional->elseToken.loc, Diagnostics::seeDirective);
        return;
    }
    conditional->elseToken = context->m_directiveToken;

    switch (conditional->state)
    {
    case Conditional::State::Before:
        conditional->state = Conditional::State::During;
        break;

    case Conditional::State::During:
        conditional->state = Conditional::State::After;
        break;

    default:
        break;
    }

    updateLexerFlagsForConditionals(inputFile);
}

// Handle a `#elif` directive
static void HandleElifDirective(PreprocessorDirectiveContext* context)
{
    // Need to grab current input stream *before* we try to parse
    // the conditional expression.
    InputFile* inputFile = getInputFile(context);
    SLANG_ASSERT(inputFile);

    // HACK(tfoley): handle an empty `elif` like an `else` directive
    //
    // This is the behavior expected by at least one input program.
    // We will eventually want to be pedantic about this.
    switch (PeekRawTokenType(context))
    {
    case TokenType::EndOfFile:
    case TokenType::NewLine:
        GetSink(context)->diagnose(
            GetDirectiveLoc(context),
            Diagnostics::directiveExpectsExpression,
            GetDirectiveName(context));
        HandleElseDirective(context);
        return;
    }

    // if we aren't inside a conditional, then error
    Conditional* conditional = inputFile->getInnerMostConditional();
    if (!conditional)
    {
        GetSink(context)->diagnose(
            GetDirectiveLoc(context),
            Diagnostics::directiveWithoutIf,
            GetDirectiveName(context));
        return;
    }

    // if we've already seen a `#else`, then it is an error
    if (conditional->elseToken.type != TokenType::Unknown)
    {
        GetSink(context)->diagnose(
            GetDirectiveLoc(context),
            Diagnostics::directiveAfterElse,
            GetDirectiveName(context));
        GetSink(context)->diagnose(conditional->elseToken.loc, Diagnostics::seeDirective);
        return;
    }

    switch (conditional->state)
    {
    case Conditional::State::Before:
        {
            // Only evaluate the expression if we are in the before state.
            const PreprocessorExpressionValue value = _parseAndEvaluateExpression(context);
            if (value)
            {
                conditional->state = Conditional::State::During;
            }
            break;
        }
    case Conditional::State::During:
        {
            // Consume to end of line, ignoring expression
            SkipToEndOfLine(context);
            conditional->state = Conditional::State::After;
            break;
        }
    default:
        {
            // Consume to end of line, ignoring expression
            SkipToEndOfLine(context);
            break;
        }
    }

    updateLexerFlagsForConditionals(inputFile);
}

// Handle a `#endif` directive
static void HandleEndIfDirective(PreprocessorDirectiveContext* context)
{
    InputFile* inputFile = getInputFile(context);
    SLANG_ASSERT(inputFile);

    // if we aren't inside a conditional, then error
    Conditional* conditional = inputFile->getInnerMostConditional();
    if (!conditional)
    {
        GetSink(context)->diagnose(
            GetDirectiveLoc(context),
            Diagnostics::directiveWithoutIf,
            GetDirectiveName(context));
        return;
    }

    inputFile->popConditional();

    updateLexerFlagsForConditionals(inputFile);
}

// Helper routine to check that we find the end of a directive where
// we expect it.
//
// Most directives do not need to call this directly, since we have
// a catch-all case in the main `HandleDirective()` function.
// The `#include` case will call it directly to avoid complications
// when it switches the input stream.
static void expectEndOfDirective(PreprocessorDirectiveContext* context)
{
    if (context->m_haveDoneEndOfDirectiveChecks)
        return;

    context->m_haveDoneEndOfDirectiveChecks = true;

    if (!IsEndOfLine(context))
    {
        // If we already saw a previous parse error, then don't
        // emit another one for the same directive.
        if (!context->m_parseError)
        {
            GetSink(context)->diagnose(
                PeekLoc(context),
                Diagnostics::unexpectedTokensAfterDirective,
                GetDirectiveName(context));
        }
        SkipToEndOfLine(context);
    }

    // Clear out the end-of-line token
    AdvanceRawToken(context);
}

/// Read a file in the context of handling a preprocessor directive
static SlangResult readFile(
    PreprocessorDirectiveContext* context,
    String const& path,
    ISlangBlob** outBlob)
{
    // The actual file loading will be handled by the file system
    // associated with the parent linkage.
    //
    auto fileSystemExt = context->m_preprocessor->fileSystem;
    SLANG_RETURN_ON_FAIL(fileSystemExt->loadFile(path.getBuffer(), outBlob));

    return SLANG_OK;
}

void Preprocessor::pushInputFile(InputFile* inputFile, SourceLoc loc)
{
    if (m_currentInputFile)
    {
        SourceView* sourceView = m_currentInputFile->getLexer()->m_sourceView;
        SourceLoc::RawValue offset = SourceRange(sourceView->getLastSegment().begin, loc).getSize();
        absoluteSourceLocCounter += offset;
    }

    {
        SourceView* sourceView = inputFile->getLexer()->m_sourceView;
        sourceView->setAbsoluteLocationBase(absoluteSourceLocCounter);
    }

    inputFile->m_parent = m_currentInputFile;
    m_currentInputFile = inputFile;
}

// Handle a `#include` directive
static void HandleIncludeDirective(PreprocessorDirectiveContext* context)
{
    // Consume the directive
    AdvanceRawToken(context);

    Token pathToken;
    String path;
    if (PeekRawTokenType(context) == TokenType::OpLess)
    {
        StringBuilder pathSB;
        Expect(
            context,
            TokenType::OpLess,
            Diagnostics::expectedTokenInPreprocessorDirective,
            &pathToken);
        while (PeekRawTokenType(context) != TokenType::OpGreater &&
               PeekRawTokenType(context) != TokenType::EndOfFile)
        {
            pathSB << AdvanceRawToken(context).getContent();
        }
        if (!Expect(
                context,
                TokenType::OpGreater,
                Diagnostics::expectedTokenInPreprocessorDirective))
            return;
        path = pathSB.produceString();
    }
    else
    {
        Expect(
            context,
            TokenType::StringLiteral,
            Diagnostics::expectedTokenInPreprocessorDirective,
            &pathToken);
        path = getFileNameTokenValue(pathToken);
    }

    auto directiveLoc = GetDirectiveLoc(context);

    PathInfo includedFromPathInfo = context->m_preprocessor->getSourceManager()->getPathInfo(
        directiveLoc,
        SourceLocType::Actual);

    IncludeSystem* includeSystem = context->m_preprocessor->includeSystem;
    if (!includeSystem)
    {
        GetSink(context)->diagnose(pathToken.loc, Diagnostics::includeFailed, path);
        GetSink(context)->diagnose(pathToken.loc, Diagnostics::noIncludeHandlerSpecified);
        return;
    }

    /* Find the path relative to the foundPath */
    PathInfo filePathInfo;
    if (SLANG_FAILED(includeSystem->findFile(path, includedFromPathInfo.foundPath, filePathInfo)))
    {
        GetSink(context)->diagnose(pathToken.loc, Diagnostics::includeFailed, path);
        return;
    }

    // We must have a uniqueIdentity to be compare
    if (!filePathInfo.hasUniqueIdentity())
    {
        GetSink(context)->diagnose(pathToken.loc, Diagnostics::noUniqueIdentity, path);
        return;
    }

    reportIncludeFileForContentAssist(context->m_preprocessor, pathToken, filePathInfo.foundPath);

    // Do all checking related to the end of this directive before we push a new stream,
    // just to avoid complications where that check would need to deal with
    // a switch of input stream
    expectEndOfDirective(context);

    // Check whether we've previously included this file and seen a `#pragma once` directive
    if (context->m_preprocessor->pragmaOnceUniqueIdentities.contains(filePathInfo.uniqueIdentity))
    {
        return;
    }

    // Simplify the path
    filePathInfo.foundPath = includeSystem->simplifyPath(filePathInfo.foundPath);

    // Push the new file onto our stack of input streams
    // TODO(tfoley): check if we have made our include stack too deep
    auto sourceManager = context->m_preprocessor->getSourceManager();

    // See if this an already loaded source file
    SourceFile* sourceFile = sourceManager->findSourceFileRecursively(filePathInfo.uniqueIdentity);
    // If not create a new one, and add to the list of known source files
    if (!sourceFile)
    {
        ComPtr<ISlangBlob> foundSourceBlob;
        if (SLANG_FAILED(readFile(context, filePathInfo.foundPath, foundSourceBlob.writeRef())))
        {
            GetSink(context)->diagnose(pathToken.loc, Diagnostics::includeFailed, path);
            return;
        }

        sourceFile = sourceManager->createSourceFileWithBlob(filePathInfo, foundSourceBlob);
        sourceManager->addSourceFile(filePathInfo.uniqueIdentity, sourceFile);
    }

    // If we are running the preprocessor as part of compiling a
    // specific module, then we must keep track of the file we've
    // read as yet another file that the module will depend on.
    //
    if (auto handler = context->m_preprocessor->handler)
    {
        handler->handleFileDependency(sourceFile);
    }

    // This is a new parse (even if it's a pre-existing source file), so create a new SourceView
    SourceView* sourceView =
        sourceManager->createSourceView(sourceFile, &filePathInfo, directiveLoc);

    InputFile* inputFile = new InputFile(context->m_preprocessor, sourceView);

    context->m_preprocessor->pushInputFile(inputFile, directiveLoc);
}

static void _parseMacroOps(
    Preprocessor* preprocessor,
    MacroDefinition* macro,
    Dictionary<Name*, Index> const& mapParamNameToIndex)
{
    // Scan through the tokens to recognize the "ops" that make up
    // the macro body.
    //
    Index spanBeginIndex = 0;
    Index cursor = 0;
    for (;;)
    {
        Index spanEndIndex = cursor;
        Index tokenIndex = cursor++;
        Token const& token = macro->tokens.m_tokens[tokenIndex];
        MacroDefinition::Op newOp;
        switch (token.type)
        {
        default:
            // Most tokens just continue our current span.
            continue;

        case TokenType::Identifier:
            {
                auto paramName = token.getName();
                Index paramIndex = -1;
                if (!mapParamNameToIndex.tryGetValue(paramName, paramIndex))
                {
                    continue;
                }

                newOp.opcode = MacroDefinition::Opcode::ExpandedParam;
                newOp.index0 = tokenIndex;
                newOp.index1 = paramIndex;
            }
            break;

        case TokenType::Pound:
            {
                auto paramNameTokenIndex = cursor;
                auto paramNameToken = macro->tokens.m_tokens[paramNameTokenIndex];
                if (paramNameToken.type != TokenType::Identifier)
                {
                    GetSink(preprocessor)
                        ->diagnose(token.loc, Diagnostics::expectedMacroParameterAfterStringize);
                    continue;
                }
                auto paramName = paramNameToken.getName();
                Index paramIndex = -1;
                if (!mapParamNameToIndex.tryGetValue(paramName, paramIndex))
                {
                    GetSink(preprocessor)
                        ->diagnose(token.loc, Diagnostics::expectedMacroParameterAfterStringize);
                    continue;
                }

                cursor++;

                newOp.opcode = MacroDefinition::Opcode::StringizedParam;
                newOp.index0 = tokenIndex;
                newOp.index1 = paramIndex;
            }
            break;

        case TokenType::PoundPound:
            if (macro->ops.getCount() == 0 && (spanBeginIndex == spanEndIndex))
            {
                GetSink(preprocessor)->diagnose(token.loc, Diagnostics::tokenPasteAtStart);
                continue;
            }

            if (macro->tokens.m_tokens[cursor].type == TokenType::EndOfFile)
            {
                GetSink(preprocessor)->diagnose(token.loc, Diagnostics::tokenPasteAtEnd);
                continue;
            }

            newOp.opcode = MacroDefinition::Opcode::TokenPaste;
            newOp.index0 = tokenIndex;
            newOp.index1 = 0;

            // Okay, we need to do something here!

            break;

        case TokenType::EndOfFile:
            break;
        }

        if (spanBeginIndex != spanEndIndex ||
            ((token.type == TokenType::EndOfFile) && (macro->ops.getCount() == 0)))
        {
            MacroDefinition::Op spanOp;
            spanOp.opcode = MacroDefinition::Opcode::RawSpan;
            spanOp.index0 = spanBeginIndex;
            spanOp.index1 = spanEndIndex;
            macro->ops.add(spanOp);
        }
        if (token.type == TokenType::EndOfFile)
            break;

        macro->ops.add(newOp);
        spanBeginIndex = cursor;
    }

    Index opCount = macro->ops.getCount();
    SLANG_ASSERT(opCount != 0);
    for (Index i = 1; i < opCount - 1; ++i)
    {
        if (macro->ops[i].opcode == MacroDefinition::Opcode::TokenPaste)
        {
            if (macro->ops[i - 1].opcode == MacroDefinition::Opcode::ExpandedParam)
                macro->ops[i - 1].opcode = MacroDefinition::Opcode::UnexpandedParam;
            if (macro->ops[i + 1].opcode == MacroDefinition::Opcode::ExpandedParam)
                macro->ops[i + 1].opcode = MacroDefinition::Opcode::UnexpandedParam;
        }
    }
}

// Handle a `#define` directive
static void HandleDefineDirective(PreprocessorDirectiveContext* context)
{
    Token nameToken;
    if (!ExpectRaw(
            context,
            TokenType::Identifier,
            Diagnostics::expectedTokenInPreprocessorDirective,
            &nameToken))
        return;
    Name* name = nameToken.getName();

    MacroDefinition* oldMacro = LookupMacro(&context->m_preprocessor->globalEnv, name);
    if (oldMacro)
    {
        auto sink = GetSink(context);

        if (oldMacro->isBuiltin())
        {
            sink->diagnose(nameToken.loc, Diagnostics::builtinMacroRedefinition, name);
        }
        else
        {
            sink->diagnose(nameToken.loc, Diagnostics::macroRedefinition, name);
            sink->diagnose(oldMacro->getLoc(), Diagnostics::seePreviousDefinitionOf, name);
        }

        delete oldMacro;
    }

    MacroDefinition* macro = new MacroDefinition();

    Dictionary<Name*, Index> mapParamNameToIndex;

    // If macro name is immediately followed (with no space) by `(`,
    // then we have a function-like macro
    auto maybeOpenParen = PeekRawToken(context);
    if (maybeOpenParen.type == TokenType::LParent &&
        !(maybeOpenParen.flags & TokenFlag::AfterWhitespace))
    {
        // This is a function-like macro, so we need to remember that
        // and start capturing parameters
        macro->flavor = MacroDefinition::Flavor::FunctionLike;

        AdvanceRawToken(context);

        // If there are any parameters, parse them
        if (PeekRawTokenType(context) != TokenType::RParent)
        {
            for (;;)
            {
                // A macro parameter should follow one of three shapes:
                //
                //      NAME
                //      NAME...
                //      ...
                //
                // If we don't see an ellipsis ahead, we know we ought
                // to find one of the two cases that starts with an
                // identifier.
                //
                Token paramNameToken;
                if (PeekRawTokenType(context) != TokenType::Ellipsis)
                {
                    if (!ExpectRaw(
                            context,
                            TokenType::Identifier,
                            Diagnostics::expectedTokenInMacroParameters,
                            &paramNameToken))
                        break;
                }

                // Whether or not a name was seen, we allow an ellipsis
                // to indicate a variadic macro parameter.
                //
                // Note: a variadic parameter, if any, should always be
                // the last parameter of a macro, but we do not enforce
                // that requirement here.
                //
                Token ellipsisToken;
                MacroDefinition::Param param;
                if (PeekRawTokenType(context) == TokenType::Ellipsis)
                {
                    ellipsisToken = AdvanceRawToken(context);
                    param.isVariadic = true;
                }

                if (paramNameToken.type != TokenType::Unknown)
                {
                    // If we read an explicit name for the parameter, then we can use
                    // that name directly.
                    //
                    param.nameLoc.name = paramNameToken.getName();
                    param.nameLoc.loc = paramNameToken.loc;
                }
                else
                {
                    // If an explicit name was not read for the parameter, we *must*
                    // have an unnamed variadic parameter. We know this because the
                    // only case where the logic above doesn't require a name to
                    // be read is when it already sees an ellipsis ahead.
                    //
                    SLANG_ASSERT(ellipsisToken.type != TokenType::Unknown);

                    // Any unnamed variadic parameter is treated as one named `__VA_ARGS__`
                    //
                    param.nameLoc.name =
                        context->m_preprocessor->getNamePool()->getName("__VA_ARGS__");
                    param.nameLoc.loc = ellipsisToken.loc;
                }

                // TODO(tfoley): The C standard seems to disallow certain identifiers
                // (e.g., `defined` and `__VA_ARGS__`) from being used as the names
                // of user-defined macros or macro parameters. This choice seemingly
                // supports implementation flexibility in how the special meanings of
                // those identifiers are handled.
                //
                // We could consider issuing diagnostics for cases where a macro or parameter
                // uses such names, or we could simply provide guarantees about what those
                // names *do* in the context of the Slang preprocessor.

                // Add the parameter to the macro being deifned
                auto paramIndex = macro->params.getCount();
                macro->params.add(param);

                auto paramName = param.nameLoc.name;
                if (mapParamNameToIndex.containsKey(paramName))
                {
                    GetSink(context)->diagnose(
                        param.nameLoc.loc,
                        Diagnostics::duplicateMacroParameterName,
                        name);
                }
                else
                {
                    mapParamNameToIndex[paramName] = paramIndex;
                }


                // If we see `)` then we are done with arguments
                if (PeekRawTokenType(context) == TokenType::RParent)
                    break;

                ExpectRaw(context, TokenType::Comma, Diagnostics::expectedTokenInMacroParameters);
            }
        }

        ExpectRaw(context, TokenType::RParent, Diagnostics::expectedTokenInMacroParameters);

        // Once we have parsed the macro parameters, we can perform the additional validation
        // step of checking that any parameters before the last parameter are not variadic.
        //
        Index lastParamIndex = macro->params.getCount() - 1;
        for (Index i = 0; i < lastParamIndex; ++i)
        {
            auto& param = macro->params[i];
            if (!param.isVariadic)
                continue;

            GetSink(context)->diagnose(
                param.nameLoc.loc,
                Diagnostics::variadicMacroParameterMustBeLast,
                param.nameLoc.name);

            // As a precaution, we will unmark the variadic-ness of the parameter, so that
            // logic downstream from this step doesn't have to deal with the possibility
            // of a variadic parameter in the middle of the parameter list.
            //
            param.isVariadic = false;
        }
    }
    else
    {
        macro->flavor = MacroDefinition::Flavor::ObjectLike;
    }

    macro->nameAndLoc = NameLoc(nameToken);

    context->m_preprocessor->globalEnv.macros[name] = macro;

    // consume tokens until end-of-line
    for (;;)
    {
        Token token = PeekRawToken(context);
        switch (token.type)
        {
        default:
            // In the ordinary case, we just add the token to the definition,
            // and keep consuming more tokens.
            AdvanceRawToken(context);
            macro->tokens.add(token);
            continue;

        case TokenType::EndOfFile:
        case TokenType::NewLine:
            // The end of the current line/file ends the directive, and serves
            // as the end-of-file marker for the macro's definition as well.
            //
            token.type = TokenType::EndOfFile;
            macro->tokens.add(token);
            break;
        }
        break;
    }

    _parseMacroOps(context->m_preprocessor, macro, mapParamNameToIndex);

    reportMacroDefinitionForContentAssist(context->m_preprocessor, macro);
}

// Handle a `#undef` directive
static void HandleUndefDirective(PreprocessorDirectiveContext* context)
{
    Token nameToken;
    if (!ExpectRaw(
            context,
            TokenType::Identifier,
            Diagnostics::expectedTokenInPreprocessorDirective,
            &nameToken))
        return;
    Name* name = nameToken.getName();

    Environment* env = &context->m_preprocessor->globalEnv;
    MacroDefinition* macro = LookupMacro(env, name);
    if (macro != NULL)
    {
        // name was defined, so remove it
        env->macros.remove(name);

        delete macro;
    }
    else
    {
        // name wasn't defined
        GetSink(context)->diagnose(nameToken.loc, Diagnostics::macroNotDefined, name);
    }
}

static String _readDirectiveMessage(PreprocessorDirectiveContext* context)
{
    StringBuilder result;

    while (!IsEndOfLine(context))
    {
        Token token = AdvanceRawToken(context);
        if (token.flags & TokenFlag::AfterWhitespace)
        {
            if (result.getLength() != 0)
            {
                result.append(" ");
            }
        }
        result.append(token.getContent());
    }

    return result;
}

// Handle a `#warning` directive
static void HandleWarningDirective(PreprocessorDirectiveContext* context)
{
    _setLexerDiagnosticSuppression(getInputFile(context), true);

    // Consume the directive
    AdvanceRawToken(context);

    // Read the message.
    String message = _readDirectiveMessage(context);

    _setLexerDiagnosticSuppression(getInputFile(context), false);

    // Report the custom error.
    GetSink(context)->diagnose(GetDirectiveLoc(context), Diagnostics::userDefinedWarning, message);
}

// Handle a `#error` directive
static void HandleErrorDirective(PreprocessorDirectiveContext* context)
{
    _setLexerDiagnosticSuppression(getInputFile(context), true);

    // Consume the directive
    AdvanceRawToken(context);

    // Read the message.
    String message = _readDirectiveMessage(context);

    _setLexerDiagnosticSuppression(getInputFile(context), false);

    // Report the custom error.
    GetSink(context)->diagnose(GetDirectiveLoc(context), Diagnostics::userDefinedError, message);
}

static void _handleDefaultLineDirective(PreprocessorDirectiveContext* context)
{
    SourceLoc directiveLoc = GetDirectiveLoc(context);
    auto inputStream = getInputFile(context);
    auto sourceView = inputStream->getLexer()->m_sourceView;
    sourceView->addDefaultLineDirective(directiveLoc);
}

static void _diagnoseInvalidLineDirective(PreprocessorDirectiveContext* context)
{
    GetSink(context)->diagnose(
        PeekLoc(context),
        Diagnostics::expected2TokensInPreprocessorDirective,
        TokenType::IntegerLiteral,
        "default",
        GetDirectiveName(context));
    context->m_parseError = true;
}

// Handle a `#line` directive
static void HandleLineDirective(PreprocessorDirectiveContext* context)
{
    auto inputStream = getInputFile(context);

    int line = 0;

    SourceLoc directiveLoc = GetDirectiveLoc(context);

    switch (PeekTokenType(context))
    {
    case TokenType::IntegerLiteral:
        line = stringToInt(AdvanceToken(context).getContent());
        break;

    case TokenType::EndOfFile:
    case TokenType::NewLine:
        // `#line`
        _handleDefaultLineDirective(context);
        return;

    case TokenType::Identifier:
        if (PeekToken(context).getContent() == "default")
        {
            AdvanceToken(context);
            _handleDefaultLineDirective(context);
            return;
        }
        [[fallthrough]];
    default:
        _diagnoseInvalidLineDirective(context);
        return;
    }

    auto sourceManager = context->m_preprocessor->getSourceManager();

    String file;
    switch (PeekTokenType(context))
    {
    case TokenType::EndOfFile:
    case TokenType::NewLine:
        file = sourceManager->getPathInfo(directiveLoc).foundPath;
        break;

    case TokenType::StringLiteral:
        file = getStringLiteralTokenValue(AdvanceToken(context));
        break;

    case TokenType::IntegerLiteral:
        // Note(tfoley): GLSL allows the "source string" to be indicated by an integer
        // TODO(tfoley): Figure out a better way to handle this, if it matters
        file = AdvanceToken(context).getContent();
        break;

    default:
        Expect(
            context,
            TokenType::StringLiteral,
            Diagnostics::expectedTokenInPreprocessorDirective);
        return;
    }

    auto sourceView = inputStream->getLexer()->m_sourceView;
    sourceView->addLineDirective(directiveLoc, file, line);
}

#define SLANG_PRAGMA_DIRECTIVE_CALLBACK(NAME) \
    void NAME(PreprocessorDirectiveContext* context, Token subDirectiveToken)

// Callback interface used by `#pragma` directives
typedef SLANG_PRAGMA_DIRECTIVE_CALLBACK((*PragmaDirectiveCallback));

SLANG_PRAGMA_DIRECTIVE_CALLBACK(handleUnknownPragmaDirective)
{
    GetSink(context)->diagnose(
        subDirectiveToken,
        Diagnostics::unknownPragmaDirectiveIgnored,
        subDirectiveToken.getName());
    SkipToEndOfLine(context);
    return;
}

SLANG_PRAGMA_DIRECTIVE_CALLBACK(handlePragmaOnceDirective)
{
    // We need to identify the path of the file we are preprocessing,
    // so that we can avoid including it again.
    //
    // We are using the 'uniqueIdentity' as determined by the ISlangFileSystemEx interface to
    // determine file identities.

    auto directiveLoc = GetDirectiveLoc(context);
    auto issuedFromPathInfo = context->m_preprocessor->getSourceManager()->getPathInfo(
        directiveLoc,
        SourceLocType::Actual);

    // Must have uniqueIdentity for a #pragma once to work
    if (!issuedFromPathInfo.hasUniqueIdentity())
    {
        GetSink(context)->diagnose(subDirectiveToken, Diagnostics::pragmaOnceIgnored);
        return;
    }

    context->m_preprocessor->pragmaOnceUniqueIdentities.add(issuedFromPathInfo.uniqueIdentity);
}

SLANG_PRAGMA_DIRECTIVE_CALLBACK(handlePragmaWarningDirective)
{
    auto directiveLoc = GetDirectiveLoc(context);
    SLANG_UNUSED(subDirectiveToken)
    SLANG_UNUSED(directiveLoc);
    Expect(context, TokenType::LParent, Diagnostics::syntaxError);
    Token tk = PeekToken(context);
    auto finish = [&]() -> void { SkipToEndOfLine(context); };
    if (tk.type == TokenType::Identifier)
    {
        // #pragma warning (push)
        if (tk.getContent() == "push")
        {
            AdvanceToken(context);
            context->m_preprocessor->warningStateTracker->addPragmaPush(tk.loc);
        }
        // #pragma warning (pop)
        else if (tk.getContent() == "pop")
        {
            AdvanceToken(context);
            context->m_preprocessor->warningStateTracker->addPragmaPop(tk.loc, GetSink(context));
        }
        else
        {
            // #pragma warning (spec : id-list [; ...]), examples:
            // (disable : 123) : disables 123
            // (disable : 1 2 ; error 4 5 6) disables 1, then disables 2, then errors 4, ...
            // (disable : 1 ; default : 1) disables 1, then defaults 1
            // Parse a list of 'specifier : id-list', separated by ';'
            while (true)
            {
                // Read the specifier
                // We need the raw token location because if the token is a #definition,
                // PeekToken().loc would be where the definition is, not the invocation
                // PeekRawToken().loc is where the invocation of the macro is located:
                // Example:
                // #define SPEC suppress // (a)
                // #pragma warning (SPEC : 12) (b)
                // Here the raw token is 'SPEC' located at its invocation on line (b)
                // and the token is 'suppress' located at the macro definition on line (a)
                // The #pragma warning should take effect from line (b), not line (a),
                // So we need the raw token location.
                SourceLoc specifierLocation = PeekRawToken(context).loc;
                Token id;
                Expect(context, TokenType::Identifier, Diagnostics::syntaxError, &id);
                PragmaWarningSpecifier specifier;
                SourceLoc nextLineEnd = {}; // Needed for suppress
                if (id.getContent() == "default")
                {
                    specifier = PragmaWarningSpecifier::Default;
                }
                else if (id.getContent() == "disable")
                {
                    specifier = PragmaWarningSpecifier::Disable;
                }
                else if (id.getContent() == "error")
                {
                    specifier = PragmaWarningSpecifier::Error;
                }
                else if (id.getContent() == "once")
                {
                    specifier = PragmaWarningSpecifier::Once;
                }
                else if (id.getContent() == "suppress")
                {
                    specifier = PragmaWarningSpecifier::Suppress;
                    // We need to start from subDirectiveToken.loc because the next tokens
                    // might be macro invocations, and located before this #pragma warning line.
                    nextLineEnd = FindNextEndOfLine(context, specifierLocation, 2);
                    if (!nextLineEnd.isValid())
                    {
                        GetSink(context)->diagnose(
                            specifierLocation,
                            Diagnostics::pragmaWarningSuppressCannotIdentifyNextLine);
                        return finish();
                    }
                }
                else
                {
                    GetSink(context)->diagnose(
                        specifierLocation,
                        Diagnostics::pragmaWarningUnknownSpecifier,
                        id.getContent());
                    return finish();
                }
                Expect(context, TokenType::Colon, Diagnostics::syntaxError);
                // Read the id list
                while (true)
                {
                    // Same logic as for the specifierLocation
                    SourceLoc idLocation = PeekRawToken(context).loc;
                    Token warningNumberToken = PeekToken(context);
                    if (warningNumberToken.type == TokenType::IntegerLiteral)
                    {
                        AdvanceToken(context);
                        int warningNumber = stringToInt(warningNumberToken.getContent());
                        context->m_preprocessor->warningStateTracker->addEntry(
                            idLocation,
                            nextLineEnd,
                            warningNumber,
                            specifier,
                            GetSink(context));
                    }
                    else
                    {
                        break;
                    }
                }
                SourceLoc endLoc = PeekRawToken(context).loc;
                Token end = PeekToken(context);
                if (end.type == TokenType::Semicolon)
                {
                    // We need to parse the next 'spec : id-list'
                    AdvanceToken(context);
                    continue;
                }
                else if (end.type == TokenType::RParent)
                {
                    break;
                }
                else
                {
                    GetSink(context)->diagnose(
                        endLoc,
                        Diagnostics::unexpectedToken,
                        end.getContent());
                    return finish();
                }
            }
        }
    }
    else
    {
        GetSink(context)->diagnose(tk, Diagnostics::syntaxError);
        return finish();
    }
    Expect(context, TokenType::RParent, Diagnostics::syntaxError);
}

// Information about a specific `#pragma` directive
struct PragmaDirective
{
    // name of the directive
    char const* name;

    // Callback to handle the directive
    PragmaDirectiveCallback callback;
};

// A simple array of all the  `#pragma` directives we know how to handle.
static const PragmaDirective kPragmaDirectives[] = {
    {"once", &handlePragmaOnceDirective},

    {"warning", &handlePragmaWarningDirective},

    {NULL, NULL},
};

static const PragmaDirective kUnknownPragmaDirective = {
    NULL,
    &handleUnknownPragmaDirective,
};

// Look up the `#pragma` directive with the given name.
static PragmaDirective const* findPragmaDirective(String const& name)
{
    char const* nameStr = name.getBuffer();
    for (int ii = 0; kPragmaDirectives[ii].name; ++ii)
    {
        if (strcmp(kPragmaDirectives[ii].name, nameStr) != 0)
            continue;

        return &kPragmaDirectives[ii];
    }

    return &kUnknownPragmaDirective;
}

// Handle a `#pragma` directive
static void HandlePragmaDirective(PreprocessorDirectiveContext* context)
{
    // Try to read the sub-directive name.
    Token subDirectiveToken = PeekRawToken(context);

    // The sub-directive had better be an identifier
    if (subDirectiveToken.type != TokenType::Identifier)
    {
        GetSink(context)->diagnose(
            GetDirectiveLoc(context),
            Diagnostics::expectedPragmaDirectiveName);
        SkipToEndOfLine(context);
        return;
    }
    AdvanceRawToken(context);

    // Look up the handler for the sub-directive.
    PragmaDirective const* subDirective = findPragmaDirective(subDirectiveToken.getName()->text);

    // Apply the sub-directive-specific callback
    (subDirective->callback)(context, subDirectiveToken);
}

static void HandleExtensionDirective(PreprocessorDirectiveContext* context)
{
    SkipToEndOfLine(context);
}

static void HandleVersionDirective(PreprocessorDirectiveContext* context)
{
    [[maybe_unused]] int version;
    switch (PeekTokenType(context))
    {
    case TokenType::IntegerLiteral:
        version = stringToInt(AdvanceToken(context).getContent());
        break;
    default:
        GetSink(context)->diagnose(
            GetDirectiveLoc(context),
            Diagnostics::expectedIntegralVersionNumber);
        break;
    }

    SkipToEndOfLine(context);
    context->m_preprocessor->language = SourceLanguage::GLSL;
    // TODO, just skip the version for now
}

// Handle an invalid directive
static void HandleInvalidDirective(PreprocessorDirectiveContext* context)
{
    GetSink(context)->diagnose(
        GetDirectiveLoc(context),
        Diagnostics::unknownPreprocessorDirective,
        GetDirectiveName(context));
    SkipToEndOfLine(context);
}

// Callback interface used by preprocessor directives
typedef void (*PreprocessorDirectiveCallback)(PreprocessorDirectiveContext* context);

enum PreprocessorDirectiveFlag : unsigned int
{
    // Should this directive be handled even when skipping disbaled code?
    ProcessWhenSkipping = 1 << 0,

    /// Allow the handler for this directive to advance past the
    /// directive token itself, so that it can control lexer behavior
    /// more closely.
    DontConsumeDirectiveAutomatically = 1 << 1,
};

// Information about a specific directive
struct PreprocessorDirective
{
    // name of the directive
    char const* name;

    // Callback to handle the directive
    PreprocessorDirectiveCallback callback;

    unsigned int flags;
};

// A simple array of all the directives we know how to handle.
// TODO(tfoley): considering making this into a real hash map,
// and then make it easy-ish for users of the codebase to add
// their own directives as desired.
static const PreprocessorDirective kDirectives[] = {
    {"if", &HandleIfDirective, ProcessWhenSkipping},
    {"ifdef", &HandleIfDefDirective, ProcessWhenSkipping},
    {"ifndef", &HandleIfNDefDirective, ProcessWhenSkipping},
    {"else", &HandleElseDirective, ProcessWhenSkipping},
    {"elif", &HandleElifDirective, ProcessWhenSkipping},
    {"endif", &HandleEndIfDirective, ProcessWhenSkipping},

    {"include", &HandleIncludeDirective, DontConsumeDirectiveAutomatically},
    {"define", &HandleDefineDirective, 0},
    {"undef", &HandleUndefDirective, 0},
    {"warning", &HandleWarningDirective, DontConsumeDirectiveAutomatically},
    {"error", &HandleErrorDirective, DontConsumeDirectiveAutomatically},
    {"line", &HandleLineDirective, 0},
    {"pragma", &HandlePragmaDirective, 0},

    // GLSL
    {"version", &HandleVersionDirective, 0},
    {"extension", &HandleExtensionDirective, 0},


    {nullptr, nullptr, 0},
};

static const PreprocessorDirective kInvalidDirective = {
    nullptr,
    &HandleInvalidDirective,
    0,
};

// Look up the directive with the given name.
static PreprocessorDirective const* FindDirective(String const& name)
{
    char const* nameStr = name.getBuffer();
    for (int ii = 0; kDirectives[ii].name; ++ii)
    {
        if (strcmp(kDirectives[ii].name, nameStr) != 0)
            continue;

        return &kDirectives[ii];
    }

    return &kInvalidDirective;
}

// Process a directive, where the preprocessor has already consumed the
// `#` token that started the directive line.
static void HandleDirective(PreprocessorDirectiveContext* context)
{
    // Try to read the directive name.
    context->m_directiveToken = PeekRawToken(context);

    TokenType directiveTokenType = GetDirective(context).type;

    // An empty directive is allowed, and ignored.
    switch (directiveTokenType)
    {
    case TokenType::EndOfFile:
    case TokenType::NewLine:
        return;

    default:
        break;
    }

    // Otherwise the directive name had better be an identifier
    if (directiveTokenType != TokenType::Identifier)
    {
        GetSink(context)->diagnose(
            GetDirectiveLoc(context),
            Diagnostics::expectedPreprocessorDirectiveName);
        SkipToEndOfLine(context);
        return;
    }

    // Look up the handler for the directive.
    PreprocessorDirective const* directive = FindDirective(GetDirectiveName(context));

    // If we are skipping disabled code, and the directive is not one
    // of the small number that need to run even in that case, skip it.
    if (isSkipping(context) && !(directive->flags & PreprocessorDirectiveFlag::ProcessWhenSkipping))
    {
        SkipToEndOfLine(context);
        return;
    }

    if (!(directive->flags & PreprocessorDirectiveFlag::DontConsumeDirectiveAutomatically))
    {
        // Consume the directive name token.
        AdvanceRawToken(context);
    }

    // Apply the directive-specific callback
    (directive->callback)(context);

    // We expect the directive callback to consume the entire line, so if
    // it hasn't that is a parse error.
    expectEndOfDirective(context);
}

void Preprocessor::popInputFile()
{
    auto inputFile = m_currentInputFile;
    SLANG_ASSERT(inputFile);

    // We expect the file to be at its end, so that the
    // next token read would be an end-of-file token.
    //
    auto expansionStream = inputFile->getExpansionStream();
    Token eofToken = expansionStream->peekRawToken();
    SLANG_ASSERT(eofToken.type == TokenType::EndOfFile);

    // If there are any open preprocessor conditionals in the file, then
    // we need to diagnose them as an error, because they were not closed
    // at the end of the file.
    //
    for (auto conditional = inputFile->getInnerMostConditional(); conditional;
         conditional = conditional->parent)
    {
        GetSink(this)->diagnose(eofToken, Diagnostics::endOfFileInPreprocessorConditional);
        GetSink(this)->diagnose(
            conditional->ifToken,
            Diagnostics::seeDirective,
            conditional->ifToken.getContent());
    }

    {
        SourceView* sourceView = inputFile->getLexer()->m_sourceView;
        auto lastSegment = sourceView->getLastSegment();
        absoluteSourceLocCounter +=
            SourceRange(lastSegment.begin, sourceView->getRange().end).getSize();
    }

    // We will update the current file to the parent of whatever
    // the `inputFile` was (usually the file that `#include`d it).
    //
    auto parentFile = inputFile->m_parent;
    m_currentInputFile = parentFile;

    // As a subtle special case, if this is the *last* file to be popped,
    // then we will update the canonical EOF token used by the preprocessor
    // to be the EOF token for `inputFile`, so that the source location
    // information returned will be accurate.
    //
    if (!parentFile)
    {
        endOfFileToken = eofToken;
    }
    else
    {
        SourceView* sourceView = parentFile->getLexer()->m_sourceView;
        sourceView->addAbsoluteSegment(
            parentFile->getExpansionStream()->peekLoc(),
            absoluteSourceLocCounter);
    }

    delete inputFile;
}

// Read one token using the full preprocessor, with all its behaviors.
static Token ReadToken(Preprocessor* preprocessor)
{
    for (;;)
    {
        auto inputFile = preprocessor->m_currentInputFile;
        if (!inputFile)
            return preprocessor->endOfFileToken;

        auto expansionStream = inputFile->getExpansionStream();

        // Look at the next raw token in the input.
        Token token = expansionStream->peekRawToken();
        if (token.type == TokenType::EndOfFile)
        {
            preprocessor->popInputFile();
            continue;
        }

        // If we have a directive (`#` at start of line) then handle it
        if ((token.type == TokenType::Pound) && (token.flags & TokenFlag::AtStartOfLine))
        {
            // Skip the `#`
            expansionStream->readRawToken();

            // Create a context for parsing the directive
            PreprocessorDirectiveContext directiveContext;
            directiveContext.m_preprocessor = preprocessor;
            directiveContext.m_parseError = false;
            directiveContext.m_haveDoneEndOfDirectiveChecks = false;
            directiveContext.m_inputFile = inputFile;

            // Parse and handle the directive
            HandleDirective(&directiveContext);
            continue;
        }

        // otherwise, if we are currently in a skipping mode, then skip tokens
        if (inputFile->isSkipping())
        {
            expansionStream->readRawToken();
            continue;
        }

        token = expansionStream->peekToken();
        if (token.type == TokenType::EndOfFile)
        {
            preprocessor->popInputFile();
            continue;
        }

        expansionStream->readToken();
        return token;
    }
}

// clean up after an environment
Environment::~Environment()
{
    for (const auto& [_, macro] : this->macros)
        delete macro;
}

// Add a simple macro definition from a string (e.g., for a
// `-D` option passed on the command line
static void DefineMacro(Preprocessor* preprocessor, String const& key, String const& value)
{
    PathInfo pathInfo = PathInfo::makeCommandLine();

    MacroDefinition* macro = new MacroDefinition();
    macro->flavor = MacroDefinition::Flavor::ObjectLike;

    auto sourceManager = preprocessor->getSourceManager();

    SourceFile* keyFile = sourceManager->createSourceFileWithString(pathInfo, key);
    SourceFile* valueFile = sourceManager->createSourceFileWithString(pathInfo, value);

    // Note that we don't need to pass a special source loc to identify that these are defined on
    // the command line because the PathInfo on the SourceFile, is marked 'command line'.
    SourceView* keyView = sourceManager->createSourceView(keyFile, nullptr, SourceLoc::fromRaw(0));
    SourceView* valueView =
        sourceManager->createSourceView(valueFile, nullptr, SourceLoc::fromRaw(0));

    // Use existing `Lexer` to generate a token stream.
    Lexer lexer;
    lexer.initialize(
        valueView,
        GetSink(preprocessor),
        preprocessor->getNamePool(),
        sourceManager->getMemoryArena());
    macro->tokens = lexer.lexAllSemanticTokens();

    Dictionary<Name*, Index> mapParamNameToIndex;
    _parseMacroOps(preprocessor, macro, mapParamNameToIndex);

    Name* keyName = preprocessor->getNamePool()->getName(key);

    macro->nameAndLoc.name = keyName;
    macro->nameAndLoc.loc = keyView->getRange().begin;

    MacroDefinition* oldMacro = NULL;
    if (preprocessor->globalEnv.macros.tryGetValue(keyName, oldMacro))
    {
        delete oldMacro;
    }

    preprocessor->globalEnv.macros[keyName] = macro;
    reportMacroDefinitionForContentAssist(preprocessor, macro);
}

// read the entire input into tokens
static TokenList ReadAllTokens(Preprocessor* preprocessor)
{
    TokenList tokens;
    for (;;)
    {
        Token token = ReadToken(preprocessor);

        switch (token.type)
        {
        default:
            tokens.add(token);
            break;

        case TokenType::EndOfFile:
            // Note: we include the EOF token in the list,
            // since that is expected by the `TokenList` type.
            tokens.add(token);
            return tokens;

        case TokenType::WhiteSpace:
        case TokenType::NewLine:
        case TokenType::LineComment:
        case TokenType::BlockComment:
        case TokenType::Invalid:
            break;
        }
    }
}

static void finalCheckPragmaWarnings(Preprocessor* preprocessor)
{
    auto tracker = preprocessor->warningStateTracker;
    if (tracker)
    {
        auto sink = GetSink(preprocessor);
        for (const auto& pushed : tracker->stack)
        {
            sink->diagnose(pushed, Diagnostics::pragmaWarningPushNotPopped);
        }
        tracker->stack.clearAndDeallocate();
    }
}

} // namespace preprocessor

/// Try to look up a macro with the given `macroName` and produce its value as a string
Result findMacroValue(
    Preprocessor* preprocessor,
    char const* macroName,
    String& outValue,
    SourceLoc& outLoc)
{
    using namespace preprocessor;

    auto namePool = preprocessor->namePool;
    auto macro = LookupMacro(&preprocessor->globalEnv, namePool->getName(macroName));
    if (!macro)
        return SLANG_FAIL;
    if (macro->flavor != MacroDefinition::Flavor::ObjectLike)
        return SLANG_FAIL;

    MacroInvocation* invocation =
        new MacroInvocation(preprocessor, macro, SourceLoc(), SourceLoc());

    // Note: Since we are only expanding the one macro, we should not treat any
    // other macros as "busy" at the start of expansion.
    //
    invocation->prime(/*nextBusyMacroInvocation:*/ nullptr);

    String value;
    for (bool first = true;; first = false)
    {
        Token token = invocation->readToken();
        if (token.type == TokenType::EndOfFile)
            break;

        if (!first && (token.flags & TokenFlag::AfterWhitespace))
            value.append(" ");
        value.append(token.getContent());
    }

    delete invocation;

    outValue = value;
    outLoc = macro->getLoc();
    return SLANG_OK;
}

TokenList preprocessSource(
    SourceFile* file,
    DiagnosticSink* sink,
    IncludeSystem* includeSystem,
    Dictionary<String, String> const& defines,
    Linkage* linkage,
    SourceLanguage& outDetectedLanguage,
    PreprocessorHandler* handler)
{
    PreprocessorDesc desc;

    desc.sink = sink;
    desc.includeSystem = includeSystem;
    desc.handler = handler;

    desc.defines = &defines;

    desc.fileSystem = linkage->getFileSystemExt();
    desc.namePool = linkage->getNamePool();
    desc.sourceManager = linkage->getSourceManager();

    if (linkage->isInLanguageServer())
    {
        desc.contentAssistInfo = &linkage->contentAssistInfo.preprocessorInfo;
    }

    preprocessor::WarningStateTracker* wst =
        new preprocessor::WarningStateTracker(desc.sourceManager);
    desc.sink->setSourceWarningStateTracker(wst);

    return preprocessSource(file, desc, outDetectedLanguage);
}

TokenList preprocessSource(
    SourceFile* file,
    PreprocessorDesc const& desc,
    SourceLanguage& outDetectedLanguage)
{
    using namespace preprocessor;

    Preprocessor preprocessor;

    preprocessor.sink = desc.sink;
    preprocessor.includeSystem = desc.includeSystem;
    preprocessor.fileSystem = desc.fileSystem;
    preprocessor.namePool = desc.namePool;

    preprocessor.endOfFileToken.type = TokenType::EndOfFile;
    preprocessor.endOfFileToken.flags = TokenFlag::AtStartOfLine;
    preprocessor.contentAssistInfo = desc.contentAssistInfo;

    preprocessor.warningStateTracker =
        dynamicCast<preprocessor::WarningStateTracker>(desc.sink->getSourceWarningStateTracker());

    // Add builtin macros
    {
        auto namePool = desc.namePool;

        const char* const builtinNames[] = {"__FILE__", "__LINE__"};
        const MacroDefinition::Opcode builtinOpcodes[] = {
            MacroDefinition::Opcode::BuiltinFile,
            MacroDefinition::Opcode::BuiltinLine};

        for (Index i = 0; i < SLANG_COUNT_OF(builtinNames); i++)
        {
            auto name = namePool->getName(builtinNames[i]);

            MacroDefinition::Op op;
            op.opcode = builtinOpcodes[i];

            MacroDefinition* macro = new MacroDefinition();
            macro->flavor = MacroDefinition::Flavor::BuiltinObjectLike;
            macro->nameAndLoc = NameLoc(name);
            macro->ops.add(op);

            preprocessor.globalEnv.macros[name] = macro;
        }
    }

    auto sourceManager = desc.sourceManager;
    preprocessor.sourceManager = sourceManager;

    auto handler = desc.handler;
    preprocessor.handler = handler;

    if (desc.defines)
    {
        for (const auto& [key, value] : *desc.defines)
            DefineMacro(&preprocessor, key, value);
    }

    {
        // This is the originating source we are compiling - there is no 'initiating' source loc,
        // so pass SourceLoc(0) - meaning it has no initiating location.
        SourceView* sourceView =
            sourceManager->createSourceView(file, nullptr, SourceLoc::fromRaw(0));

        // create an initial input stream based on the provided buffer
        InputFile* primaryInputFile = new InputFile(&preprocessor, sourceView);
        preprocessor.pushInputFile(primaryInputFile, sourceView->getRange().begin);
    }

    TokenList tokens = ReadAllTokens(&preprocessor);

    if (handler)
    {
        handler->handleEndOfTranslationUnit(&preprocessor);
    }

    finalCheckPragmaWarnings(&preprocessor);

    // debugging: build the pre-processed source back together
#if 0
    StringBuilder sb;
    for (auto t : tokens)
    {
        if (t.flags & TokenFlag::AtStartOfLine)
        {
            sb << "\n";
        }
        else if (t.flags & TokenFlag::AfterWhitespace)
        {
            sb << " ";
        }

        sb << t.Content;
    }

    String s = sb.produceString();
#endif

    outDetectedLanguage = preprocessor.language;

    return tokens;
}

} // namespace Slang
