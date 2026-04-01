// slang-json-parser.cpp
#include "slang-json-parser.h"

#include "../core/slang-string-escape-util.h"
#include "slang-json-diagnostics.h"

/*
https://www.json.org/json-en.html
*/

namespace Slang
{

SlangResult JSONParser::_parseObject()
{
    {
        const SourceLoc loc = m_lexer->peekLoc();
        SLANG_RETURN_ON_FAIL(m_lexer->expect(JSONTokenType::LBrace));
        m_listener->startObject(loc);
    }

    {
        const SourceLoc loc = m_lexer->peekLoc();
        if (m_lexer->advanceIf(JSONTokenType::RBrace))
        {
            m_listener->endObject(loc);
            return SLANG_OK;
        }
    }

    while (true)
    {
        JSONToken keyToken;
        SLANG_RETURN_ON_FAIL(m_lexer->expect(JSONTokenType::StringLiteral, keyToken));
        m_listener->addQuotedKey(m_lexer->getLexeme(keyToken), keyToken.loc);

        SLANG_RETURN_ON_FAIL(m_lexer->expect(JSONTokenType::Colon));

        SLANG_RETURN_ON_FAIL(_parseValue());
        if (m_lexer->advanceIf(JSONTokenType::Comma))
        {
            continue;
        }

        break;
    }

    {
        const SourceLoc loc = m_lexer->peekLoc();
        SLANG_RETURN_ON_FAIL(m_lexer->expect(JSONTokenType::RBrace));
        m_listener->endObject(loc);
    }
    return SLANG_OK;
}

SlangResult JSONParser::_parseArray()
{
    {
        const SourceLoc loc = m_lexer->peekLoc();
        SLANG_RETURN_ON_FAIL(m_lexer->expect(JSONTokenType::LBracket));
        m_listener->startArray(loc);
    }

    {
        const SourceLoc loc = m_lexer->peekLoc();
        if (m_lexer->advanceIf(JSONTokenType::RBracket))
        {
            m_listener->endArray(loc);
            return SLANG_OK;
        }
    }

    while (true)
    {
        SLANG_RETURN_ON_FAIL(_parseValue());
        if (m_lexer->advanceIf(JSONTokenType::Comma))
        {
            continue;
        }
        break;
    }

    {
        const SourceLoc loc = m_lexer->peekLoc();
        SLANG_RETURN_ON_FAIL(m_lexer->expect(JSONTokenType::RBracket));
        m_listener->endArray(loc);
    }
    return SLANG_OK;
}

SlangResult JSONParser::_parseValue()
{
    switch (m_lexer->peekType())
    {
    case JSONTokenType::True:
    case JSONTokenType::False:
    case JSONTokenType::Null:
    case JSONTokenType::IntegerLiteral:
    case JSONTokenType::FloatLiteral:
    case JSONTokenType::StringLiteral:
        {
            const JSONToken& tok = m_lexer->peekToken();
            m_listener->addLexemeValue(tok.type, m_lexer->peekLexeme(), tok.loc);
            m_lexer->advance();
            return SLANG_OK;
        }
    case JSONTokenType::LBracket:
        {
            return _parseArray();
        }
    case JSONTokenType::LBrace:
        {
            return _parseObject();
        }
    default:
        {
            m_sink->diagnose(
                m_lexer->peekLoc(),
                JSONDiagnostics::unexpectedToken,
                getJSONTokenAsText(m_lexer->peekType()));
            return SLANG_FAIL;
        }
    case JSONTokenType::Invalid:
        {
            // It's a lex error, so just fail
            return SLANG_FAIL;
        }
    }
}

SlangResult JSONParser::parse(
    JSONLexer* lexer,
    SourceView* sourceView,
    JSONListener* listener,
    DiagnosticSink* sink)
{
    m_sourceView = sourceView;
    m_lexer = lexer;
    m_listener = listener;
    m_sink = sink;

    SLANG_RETURN_ON_FAIL(_parseValue());

    return m_lexer->expect(JSONTokenType::EndOfFile);
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

                               JSONWriter

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

Index JSONWriter::_getLineLengthAfterIndent()
{
    if (m_emittedIndent < 0)
    {
        return 0;
    }

    Index lineLength = m_builder.getLength() - m_lineStart;
    return lineLength - m_emittedIndent * m_indentCharCount;
}


void JSONWriter::_emitIndent()
{
    m_builder.appendRepeatedChar(m_indentChar, m_currentIndent * m_indentCharCount);
    m_emittedIndent = m_currentIndent;
    SLANG_ASSERT(m_emittedIndent >= 0);
}

void JSONWriter::_maybeEmitIndent()
{
    if (m_emittedIndent < 0)
    {
        _emitIndent();
    }
}

void JSONWriter::_nextLine()
{
    m_builder << "\n";
    m_lineStart = m_builder.getLength();
    m_lineIndex++;
    m_emittedIndent = -1;
}

void JSONWriter::_maybeNextLine()
{
    // Nothing has been emitted, because nothing has been indented, and we must indent before an
    // emit
    if (m_emittedIndent < 0)
    {
    }
    else
    {
        _nextLine();
    }
}

void JSONWriter::_handleFormat(Location loc)
{
    switch (m_format)
    {
    case IndentationStyle::Allman:
        {
            if (isComma(loc))
            {
                _maybeNextLine();
            }
            else
            {
                if (isBefore(loc))
                {
                    _maybeNextLine();
                    if (isClose(loc))
                    {
                        _dedent();
                    }
                }
                else
                {
                    _maybeNextLine();
                    if (isOpen(loc))
                    {
                        _indent();
                    }
                }
            }
            break;
        }
    case IndentationStyle::KNR:
        {
            if (isComma(loc))
            {
                if (loc == Location::FieldComma ||
                    (m_lineLengthLimit > 0 && _getLineLengthAfterIndent() > m_lineLengthLimit))
                {
                    _maybeNextLine();
                }
            }
            else
            {
                if (isBefore(loc))
                {
                    if (isClose(loc))
                    {
                        _maybeNextLine();
                        _dedent();
                    }
                }
                else
                {
                    _maybeNextLine();
                    if (isOpen(loc))
                    {
                        _indent();
                    }
                }
            }
            break;
        }
    }
}

void JSONWriter::_maybeEmitComma()
{
    if (m_state.m_flags & State::Flag::HasPrevious)
    {
        _maybeEmitIndent();
        m_builder << ", ";
        _handleFormat(Location::Comma);
    }
}

void JSONWriter::_maybeEmitFieldComma()
{
    if (m_state.m_flags & State::Flag::HasPrevious)
    {
        _maybeEmitIndent();
        m_builder << ", ";
        _handleFormat(Location::FieldComma);
    }
}

void JSONWriter::startObject(SourceLoc loc)
{
    SLANG_UNUSED(loc);
    SLANG_ASSERT(m_state.canEmitValue());

    _maybeEmitComma();

    _handleFormat(Location::BeforeOpenObject);
    _maybeEmitIndent();
    m_builder << "{";
    _handleFormat(Location::AfterOpenObject);

    m_state.m_flags |= State::Flag::HasPrevious;
    m_state.m_flags &= State::Flag::HasKey;

    m_stack.add(m_state);

    m_state.m_kind = State::Kind::Object;
    m_state.m_flags = 0;
}

void JSONWriter::endObject(SourceLoc loc)
{
    SLANG_UNUSED(loc);
    SLANG_ASSERT(m_state.m_kind == State::Kind::Object);

    _handleFormat(Location::BeforeCloseObject);
    _maybeEmitIndent();
    m_builder << "}";
    _handleFormat(Location::AfterCloseObject);

    m_state = m_stack.getLast();
    m_stack.removeLast();

    _postValue();
}

void JSONWriter::startArray(SourceLoc loc)
{
    SLANG_UNUSED(loc);
    SLANG_ASSERT(m_state.canEmitValue());

    _maybeEmitComma();

    _handleFormat(Location::BeforeOpenArray);
    _maybeEmitIndent();
    m_builder << "[";
    _handleFormat(Location::AfterOpenArray);

    m_state.m_flags |= State::Flag::HasPrevious;
    m_state.m_flags &= State::Flag::HasKey;

    m_stack.add(m_state);

    m_state.m_kind = State::Kind::Array;
    m_state.m_flags = 0;
}

void JSONWriter::endArray(SourceLoc loc)
{
    SLANG_UNUSED(loc);
    SLANG_ASSERT(m_state.m_kind == State::Kind::Array);

    _handleFormat(Location::BeforeCloseArray);
    _maybeEmitIndent();
    m_builder << "]";
    _handleFormat(Location::AfterCloseArray);

    m_state = m_stack.getLast();
    m_stack.removeLast();

    _postValue();
}

void JSONWriter::addUnquotedKey(const UnownedStringSlice& key, SourceLoc loc)
{
    SLANG_UNUSED(loc);
    SLANG_ASSERT(
        m_state.m_kind == State::Kind::Object && (m_state.m_flags & State::Flag::HasKey) == 0);

    _maybeEmitFieldComma();
    _maybeEmitIndent();

    // Output the key quoted
    StringEscapeHandler* handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::JSON);
    StringEscapeUtil::appendQuoted(handler, key, m_builder);

    m_builder << " : ";

    m_state.m_flags |= State::Flag::HasKey;
    // We don't want it to emit a , after the :
    m_state.m_flags &= ~State::Flag::HasPrevious;
}

void JSONWriter::addQuotedKey(const UnownedStringSlice& key, SourceLoc loc)
{
    SLANG_UNUSED(loc);
    SLANG_ASSERT(
        m_state.m_kind == State::Kind::Object && (m_state.m_flags & State::Flag::HasKey) == 0);

    // It should be quoted
    SLANG_ASSERT(key.getLength() >= 2 && key[0] == '"' && key[key.getLength() - 1] == '"');

    _maybeEmitFieldComma();
    _maybeEmitIndent();

    m_builder << key;

    m_builder << " : ";

    m_state.m_flags |= State::Flag::HasKey;
    // We don't want it to emit a , after the :
    m_state.m_flags &= ~State::Flag::HasPrevious;
}

void JSONWriter::_preValue(SourceLoc loc)
{
    SLANG_UNUSED(loc);
    SLANG_ASSERT(m_state.canEmitValue());

    _maybeEmitComma();
    _maybeEmitIndent();
}

void JSONWriter::_postValue()
{
    // We have a previous
    m_state.m_flags |= State::Flag::HasPrevious;
    // We don't have a key
    m_state.m_flags &= ~State::Flag::HasKey;
}


void JSONWriter::addLexemeValue(JSONTokenType type, const UnownedStringSlice& value, SourceLoc loc)
{
    _preValue(loc);

    switch (type)
    {
    case JSONTokenType::IntegerLiteral:
    case JSONTokenType::FloatLiteral:
    case JSONTokenType::StringLiteral:
        {
            m_builder << value;
            break;
        }
    case JSONTokenType::True:
        {
            m_builder << UnownedStringSlice::fromLiteral("true");
            break;
        }
    case JSONTokenType::False:
        {
            m_builder << UnownedStringSlice::fromLiteral("false");
            break;
        }
    case JSONTokenType::Null:
        {
            m_builder << UnownedStringSlice::fromLiteral("null");
            break;
        }
    default:
        {
            SLANG_ASSERT(!"Can only emit values");
        }
    }

    _postValue();
}

void JSONWriter::addIntegerValue(int64_t value, SourceLoc loc)
{
    _preValue(loc);
    m_builder << value;
    _postValue();
}

void JSONWriter::addFloatValue(double value, SourceLoc loc)
{
    _preValue(loc);
    m_builder << value;
    _postValue();
}

void JSONWriter::addBoolValue(bool inValue, SourceLoc loc)
{
    _preValue(loc);
    const UnownedStringSlice slice = inValue ? UnownedStringSlice::fromLiteral("true")
                                             : UnownedStringSlice::fromLiteral("false");
    m_builder << slice;
    _postValue();
}

void JSONWriter::addNullValue(SourceLoc loc)
{
    _preValue(loc);
    m_builder << UnownedStringSlice::fromLiteral("null");
    _postValue();
}

void JSONWriter::addStringValue(const UnownedStringSlice& slice, SourceLoc loc)
{
    _preValue(loc);
    StringEscapeHandler* handler = StringEscapeUtil::getHandler(StringEscapeUtil::Style::JSON);
    StringEscapeUtil::appendQuoted(handler, slice, m_builder);
    _postValue();
}

} // namespace Slang
