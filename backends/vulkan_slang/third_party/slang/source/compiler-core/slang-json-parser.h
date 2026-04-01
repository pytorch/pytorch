// slang-json-parser.h
#ifndef SLANG_JSON_PARSER_H
#define SLANG_JSON_PARSER_H

#include "slang-json-lexer.h"


namespace Slang
{

class JSONListener
{
public:
    /// Start an object
    virtual void startObject(SourceLoc loc) = 0;
    /// End an object
    virtual void endObject(SourceLoc loc) = 0;
    /// Start an array
    virtual void startArray(SourceLoc loc) = 0;
    /// End and array
    virtual void endArray(SourceLoc loc) = 0;


    /// Add the key. Must be followed by addXXXValue.
    virtual void addQuotedKey(const UnownedStringSlice& key, SourceLoc loc) = 0;
    virtual void addUnquotedKey(const UnownedStringSlice& key, SourceLoc loc) = 0;
    /// Can be performed in an array or after an addLexemeKey in an object
    virtual void addLexemeValue(
        JSONTokenType type,
        const UnownedStringSlice& value,
        SourceLoc loc) = 0;

    /// An integer value
    virtual void addIntegerValue(int64_t value, SourceLoc loc) = 0;
    /// Add a floating point value
    virtual void addFloatValue(double value, SourceLoc loc) = 0;
    /// Add a boolean value
    virtual void addBoolValue(bool value, SourceLoc loc) = 0;

    /// Add a string value. NOTE! string is unescaped/quoted
    virtual void addStringValue(const UnownedStringSlice& string, SourceLoc loc) = 0;

    /// Add a null value
    virtual void addNullValue(SourceLoc loc) = 0;
};

class JSONWriter : public JSONListener
{
public:
    /*
    https://en.wikipedia.org/wiki/Indentation_style
    */
    enum class IndentationStyle
    {
        Allman, ///< After every value, and opening, closing all other types
        KNR,    ///< K&R like. Fields have CR.
    };

    enum class LocationType : uint8_t
    {
        Object,
        Array,
        Comma,
    };

    // NOTE! Order must be kept the same without fixing is functions below
    enum class Location
    {
        BeforeOpenObject,
        BeforeCloseObject,
        AfterOpenObject,
        AfterCloseObject,

        BeforeOpenArray,
        BeforeCloseArray,
        AfterOpenArray,
        AfterCloseArray,

        FieldComma,
        Comma,

        CountOf,
    };

    static LocationType getLocationType(Location loc)
    {
        return isObject(loc) ? LocationType::Object
                             : (isComma(loc) ? LocationType::Comma : LocationType::Array);
    }

    static bool isObjectLike(Location loc)
    {
        return Index(loc) <= Index(Location::AfterCloseArray);
    }
    static bool isObject(Location loc) { return Index(loc) <= Index(Location::AfterCloseObject); }
    static bool isArray(Location loc)
    {
        return Index(loc) >= Index(Location::BeforeOpenArray) &&
               Index(loc) <= Index(Location::AfterCloseArray);
    }
    static bool isComma(Location loc) { return Index(loc) >= Index(Location::FieldComma); }
    static bool isOpen(Location loc) { return isObjectLike(loc) && (Index(loc) & 1) == 0; }
    static bool isClose(Location loc) { return isObjectLike(loc) && (Index(loc) & 1) != 0; }
    static bool isBefore(Location loc) { return isObjectLike(loc) && (Index(loc) & 2) == 0; }
    static bool isAfter(Location loc) { return isObjectLike(loc) && (Index(loc) & 2) != 0; }

    // Implement JSONListener
    virtual void startObject(SourceLoc loc) SLANG_OVERRIDE;
    virtual void endObject(SourceLoc loc) SLANG_OVERRIDE;
    virtual void startArray(SourceLoc loc) SLANG_OVERRIDE;
    virtual void endArray(SourceLoc loc) SLANG_OVERRIDE;
    virtual void addQuotedKey(const UnownedStringSlice& key, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addUnquotedKey(const UnownedStringSlice& key, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addLexemeValue(JSONTokenType type, const UnownedStringSlice& value, SourceLoc loc)
        SLANG_OVERRIDE;
    virtual void addIntegerValue(int64_t value, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addFloatValue(double value, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addBoolValue(bool value, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addStringValue(const UnownedStringSlice& string, SourceLoc loc) SLANG_OVERRIDE;
    virtual void addNullValue(SourceLoc loc) SLANG_OVERRIDE;

    /// Get the builder
    StringBuilder& getBuilder() { return m_builder; }

    JSONWriter(IndentationStyle format, Index lineLengthLimit = -1)
    {
        m_format = format;
        m_lineLengthLimit = lineLengthLimit;

        m_state.m_kind = State::Kind::Root;
        m_state.m_flags = 0;
    }

protected:
    struct State
    {
        enum class Kind : uint8_t
        {
            Root,
            Object,
            Array,
        };

        typedef uint8_t Flags;
        struct Flag
        {
            enum Enum : Flags
            {
                HasPrevious = 0x01,
                HasKey = 0x02,
            };
        };

        bool canEmitValue() const
        {
            switch (m_kind)
            {
            case Kind::Root:
                return (m_flags & Flag::HasPrevious) == 0;
            case Kind::Array:
                return true;
            case Kind::Object:
                return (m_flags & Flag::HasKey) != 0;
            default:
                return false;
            }
        }

        Kind m_kind;
        Flags m_flags;
    };

    void _maybeNextLine();
    void _nextLine();
    void _handleFormat(Location loc);

    Index _getLineLengthAfterIndent();

    /// Only emits the indent if at start of line
    void _maybeEmitIndent();
    void _emitIndent();

    void _maybeEmitComma();
    void _maybeEmitFieldComma();

    void _preValue(SourceLoc loc);
    void _postValue();

    void _indent() { m_currentIndent++; }
    void _dedent()
    {
        --m_currentIndent;
        SLANG_ASSERT(m_currentIndent >= 0);
    }

    /// True if the line is indented at the required level
    bool _hasIndent() { return m_emittedIndent >= 0 && m_emittedIndent == m_currentIndent; }

    Index m_currentIndent = 0;
    char m_indentChar = ' ';
    Index m_indentCharCount = 4;

    Index m_lineIndex = 0;
    Index m_lineStart = 0;
    Index m_emittedIndent = -1; /// If -1 for current line there is no indent emitted

    Index m_lineLengthLimit = -1; /// The limit is only applied *AFTER* indentation

    IndentationStyle m_format;

    StringBuilder m_builder;
    List<State> m_stack;
    State m_state;
};

class JSONParser
{
public:
    SlangResult parse(
        JSONLexer* lexer,
        SourceView* sourceView,
        JSONListener* listener,
        DiagnosticSink* sink);

protected:
    SlangResult _parseValue();
    SlangResult _parseObject();
    SlangResult _parseArray();

    SourceView* m_sourceView;
    DiagnosticSink* m_sink;
    JSONListener* m_listener;
    JSONLexer* m_lexer;
};


} // namespace Slang

#endif
