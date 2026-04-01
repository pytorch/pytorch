// slang-emit-source-writer.cpp
#include "slang-emit-source-writer.h"

#include "../core/slang-char-encode.h"


// Note: using C++ stdio just to get a locale-independent
// way to format floating-point values.
//
// TODO: Go ahead and implement the Dragon4 algorithm so
// that we can print floating-point values to arbitrary
// precision as needed.
#include <sstream>

namespace Slang
{

SourceWriter::SourceWriter(
    SourceManager* sourceManager,
    LineDirectiveMode lineDirectiveMode,
    IBoxValue<SourceMap>* sourceMap)
{
    m_sourceMap = sourceMap;
    m_lineDirectiveMode = lineDirectiveMode;
    m_sourceManager = sourceManager;
}

String SourceWriter::getContentAndClear()
{
    String content(getContent());
    clearContent();
    return content;
}

void SourceWriter::emitRawTextSpan(char const* textBegin, char const* textEnd)
{
    // TODO(tfoley): Need to make "corelib" not use `int` for pointer-sized things...
    auto len = textEnd - textBegin;
    m_builder.append(textBegin, len);
}

void SourceWriter::emitRawText(char const* text)
{
    emitRawTextSpan(text, text + strlen(text));
}

void SourceWriter::_emitTextSpan(char const* textBegin, char const* textEnd)
{
    // Don't change anything given an empty string
    if (textBegin == textEnd)
        return;

    // If the source location has changed in a way that required update,
    // do it now!
    _flushSourceLocationChange();

    // Note: we don't want to emit indentation on a line that is empty.
    // The logic in `Emit(textBegin, textEnd)` below will have broken
    // the text into lines, so we can simply check if a line consists
    // of just a newline.
    if (m_isAtStartOfLine && *textBegin != '\n')
    {
        // We are about to emit text (other than a newline)
        // at the start of a line, so we will emit the proper
        // amount of indentation to keep things looking nice.
        m_isAtStartOfLine = false;
        for (Int ii = 0; ii < m_indentLevel; ++ii)
        {
            char const* indentString = "    ";
            size_t indentStringSize = strlen(indentString);
            emitRawTextSpan(indentString, indentString + indentStringSize);

            // We will also update our tracking location, just in
            // case other logic needs it.
            //
            // TODO: We may need to have a switch that controls whether
            // we are in "pretty-printing" mode or "follow the locations
            // in the original code" mode.
            m_loc.column += indentStringSize;
        }
    }

    // Emit the raw text
    emitRawTextSpan(textBegin, textEnd);

    // Update our logical position
    auto len = int(textEnd - textBegin);
    m_loc.column += len;
}

void SourceWriter::indent()
{
    m_indentLevel++;
}

void SourceWriter::dedent()
{
    m_indentLevel--;
}

void SourceWriter::emitChar(char c)
{
    emit(&c, &c + 1);
}

void SourceWriter::emit(char const* textBegin, char const* textEnd)
{
    char const* spanBegin = textBegin;
    char const* spanEnd = spanBegin;
    for (;;)
    {
        if (spanEnd == textEnd)
        {
            // We have a whole range of text waiting to be flushed
            _emitTextSpan(spanBegin, spanEnd);
            return;
        }

        auto c = *spanEnd++;

        if (c == '\n')
        {
            // At the end of a line, we need to update our tracking
            // information on code positions
            _emitTextSpan(spanBegin, spanEnd);
            m_loc.line++;
            m_loc.column = 1;
            m_isAtStartOfLine = true;

            // Start a new span for emit purposes
            spanBegin = spanEnd;
        }
    }
}

void SourceWriter::emit(char const* text)
{
    emit(text, text + strlen(text));
}

void SourceWriter::emit(const String& text)
{
    emit(text.begin(), text.end());
}

void SourceWriter::emit(const UnownedStringSlice& text)
{
    emit(text.begin(), text.end());
}

void SourceWriter::emit(Name* name)
{
    emit(getText(name));
}

void SourceWriter::emit(const NameLoc& nameAndLoc)
{
    advanceToSourceLocationIfValid(nameAndLoc.loc);
    emit(getText(nameAndLoc.name));
}

void SourceWriter::emit(const StringSliceLoc& nameAndLoc)
{
    advanceToSourceLocationIfValid(nameAndLoc.loc);
    emit(nameAndLoc.name);
}

void SourceWriter::emitName(Name* name, const SourceLoc& locIn)
{
    advanceToSourceLocationIfValid(locIn);
    emit(name);
}

void SourceWriter::emitName(const NameLoc& nameAndLoc)
{
    emitName(nameAndLoc.name, nameAndLoc.loc);
}

void SourceWriter::emitName(const StringSliceLoc& nameAndLoc)
{
    emit(nameAndLoc);
}

void SourceWriter::emitName(Name* name)
{
    emitName(name, SourceLoc());
}

void SourceWriter::emitUInt64(uint64_t value)
{
    emit(value);
}

void SourceWriter::emitInt64(int64_t value)
{
    emit(value);
}

void SourceWriter::emit(Int32 value)
{
    char buffer[16];
    snprintf(buffer, sizeof(buffer), "%" PRId32, value);
    emit(buffer);
}

void SourceWriter::emit(Int64 value)
{
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%" PRId64, value);
    emit(buffer);
}

void SourceWriter::emit(UInt32 value)
{
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%" PRIu32, value);
    emit(buffer);
}

void SourceWriter::emit(UInt64 value)
{
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%" PRIu64, value);
    emit(buffer);
}

void SourceWriter::emit(double value)
{
    // There are a few different requirements here that we need to deal with:
    //
    // 1) We need to print something that is valid syntax in the target language
    //    (this means that hex floats are off the table for now)
    //
    // 2) We need our printing to be independent of the current global locale in C,
    //    so that we don't depend on the application leaving it as the default,
    //    and we also don't revert any changes they make.
    //    (this means that `sprintf` and friends are off the table)
    //
    // 3) We need to be sure that floating-point literals specified by the user will
    //    "round-trip" and turn into the same value when parsed back in. This means
    //    that we need to print a reasonable number of digits of precision.
    //
    // For right now, the easiest option that can balance these is to use
    // the C++ standard library `iostream`s, because they support an explicit locale,
    // and can (hopefully) print floating-point numbers accurately.
    //
    // Eventually, the right move here would be to implement proper floating-point
    // number formatting ourselves, but that would require extensive testing to
    // make sure we get it right.

    std::ostringstream stream;
    stream.imbue(std::locale::classic());

    int expBase2;
    std::frexp(value, &expBase2);
    // 2^17 = 131072 which is close to 10^5, so in that case we will
    // change to use scientific representation.
    std::ios::fmtflags flags = (std::abs(expBase2) >= 17) ? std::ios::scientific : std::ios::fixed;

    stream.setf(flags, std::ios::floatfield);
    stream.precision(std::numeric_limits<double>::max_digits10);
    stream << value;
    auto str = stream.str();

    std::size_t found = str.find_last_of("e");
    found = (found == std::string::npos) ? str.length() : found;

    // separate the mantissa and exponent part, as we want to remove the
    // trailing 0s from the mantissa part. If we selected the fixed format
    // above, the 'exponentStr' will be empty.
    std::string mantissaStr = str.substr(0, found);
    std::string exponentStr = str.substr(found, str.length());

    // Remove redundant trailing 0s.
    if (mantissaStr.end() > mantissaStr.begin())
    {
        auto lastChar = mantissaStr.end() - 1;
        while (lastChar > mantissaStr.begin() && *lastChar == '0')
            lastChar--;
        if (*lastChar == '.')
            lastChar++;
        if (lastChar > mantissaStr.end() - 1)
            lastChar = mantissaStr.end() - 1;
        mantissaStr = mantissaStr.substr(0, lastChar - mantissaStr.begin() + 1);
    }

    auto finalStr = mantissaStr + exponentStr;
    auto slice = UnownedStringSlice(finalStr.c_str());
    emit(slice);
}

void SourceWriter::advanceToSourceLocationIfValid(const SourceLoc& sourceLocation)
{
    if (sourceLocation.isValid())
    {
        advanceToSourceLocation(sourceLocation);
    }
}

void SourceWriter::advanceToSourceLocation(const SourceLoc& sourceLocation)
{
    // If we don't have any line directives *and* we don't want to output
    // source map, we can just ignore
    if (getLineDirectiveMode() == LineDirectiveMode::None && m_sourceMap == nullptr)
    {
        // Ignore if we aren't outputting directives
        return;
    }

    if (!sourceLocation.isValid())
    {
        // If it's not valid, we will just keep the current location.

        // The question now is if we want to trigger outputting the source location again. We do so
        // if
        // * The nextSourceLoc is valid
        // * The line number on the output is different from that location
        m_needToUpdateSourceLocation =
            m_needToUpdateSourceLocation ||
            (m_nextSourceLoc.isValid() && m_nextHumaneSourceLocation.line != m_loc.line);
        return;
    }

    // Even if the source location is the same, we still want to trigger output, as long
    // as we have a valid line location.
    if (sourceLocation == m_nextSourceLoc)
    {
        // This is important because we can end up with many instructions with the same source
        // location - for example when we have [__unsafeForceInlineEarly] all the inlined
        // instructions get the same location. When we output lines of source, the target sources
        // line numbers change, therefore we need to output  the same #line directive multiple
        // times.

        m_needToUpdateSourceLocation =
            m_needToUpdateSourceLocation || (m_nextHumaneSourceLocation.line > 0);
        return;
    }

    // Workout the humane source location.
    const HumaneSourceLoc humaneSourceLoc =
        getSourceManager()->getHumaneLoc(sourceLocation, SourceLocType::Emit);

    // If the location is valid, mark need to update, and the new location
    if (humaneSourceLoc.line > 0)
    {
        m_needToUpdateSourceLocation = true;
        m_nextHumaneSourceLocation = humaneSourceLoc;
    }

    // Either way set this as the current source location.
    m_nextSourceLoc = sourceLocation;
}

void SourceWriter::_flushSourceLocationChange()
{
    if (!m_needToUpdateSourceLocation)
        return;

    // Note: the order matters here, because trying to update
    // the source location may involve outputting text that
    // advances the location, and outputting text is what
    // triggers this flush operation.
    m_needToUpdateSourceLocation = false;

    _emitLineDirectiveIfNeeded(m_nextHumaneSourceLocation);

    // If we have a source map update state
    if (m_sourceMap)
    {
        _updateSourceMap(m_nextHumaneSourceLocation);
    }
}

void SourceWriter::_emitLineDirectiveAndUpdateSourceLocation(const HumaneSourceLoc& sourceLocation)
{
    _emitLineDirective(sourceLocation);

    HumaneSourceLoc newLoc = sourceLocation;
    newLoc.column = 1;

    m_loc = newLoc;
}

void SourceWriter::_updateSourceMap(const HumaneSourceLoc& sourceLocation)
{
    // Ignore invalid source locations
    if (sourceLocation.line <= 0)
        return;

    auto sourceMap = m_sourceMap->getPtr();

    // We need to work out the current column in the generated (ie being written) output
    Index generatedLineIndex, generatedColumnIndex;
    _calcLocation(generatedLineIndex, generatedColumnIndex);

    // Advance to the current output line
    sourceMap->advanceToLine(generatedLineIndex);

    // Add the entry into the map, mapping back to the original source
    SourceMap::Entry entry;
    entry.init();

    entry.sourceFileIndex =
        sourceMap->getSourceFileIndex(sourceLocation.pathInfo.getName().getUnownedSlice());
    entry.sourceLine = sourceLocation.line - 1;
    entry.sourceColumn = sourceLocation.column - 1;
    entry.generatedColumn = generatedColumnIndex;

    sourceMap->addEntry(entry);
}

void SourceWriter::_emitLineDirectiveIfNeeded(const HumaneSourceLoc& sourceLocation)
{
    if (m_supressLineDirective)
        return;

    // Don't do any of this work if the user has requested that we
    // not emit line directives.
    auto mode = getLineDirectiveMode();
    switch (mode)
    {
    case LineDirectiveMode::SourceMap:
    case LineDirectiveMode::None:
        return;

    case LineDirectiveMode::Default:
    default:
        break;
    }

    // Ignore invalid source locations
    if (sourceLocation.line <= 0)
        return;

    // If we are currently emitting code at a source location with
    // a differnet file or line, *or* if the source location is
    // somehow later on the line than what we want to emit,
    // then we need to emit a new `#line` directive.
    if (sourceLocation.pathInfo.foundPath != m_loc.pathInfo.foundPath ||
        sourceLocation.line != m_loc.line || sourceLocation.column < m_loc.column)
    {
        // Special case: if we are in the same file, and within a small number
        // of lines of the target location, then go ahead and output newlines
        // to get us caught up.
        enum
        {
            kSmallLineCount = 3
        };
        auto lineDiff = sourceLocation.line - m_loc.line;
        if (sourceLocation.pathInfo.foundPath == m_loc.pathInfo.foundPath &&
            sourceLocation.line > m_loc.line && lineDiff <= kSmallLineCount)
        {
            for (int ii = 0; ii < lineDiff; ++ii)
            {
                emit("\n");
            }
            SLANG_RELEASE_ASSERT(sourceLocation.line == m_loc.line);
        }
        else
        {
            // Go ahead and output a `#line` directive to get us caught up
            _emitLineDirectiveAndUpdateSourceLocation(sourceLocation);
        }
    }
}

void SourceWriter::_emitLineDirective(const HumaneSourceLoc& sourceLocation)
{
    emitRawText("\n#line ");

    char buffer[16];
    snprintf(buffer, sizeof(buffer), "%llu", (unsigned long long)sourceLocation.line);
    emitRawText(buffer);

    // Only emit the path part of a `#line` directive if needed
    if (sourceLocation.pathInfo.foundPath != m_loc.pathInfo.foundPath)
    {
        emitRawText(" ");

        auto mode = getLineDirectiveMode();
        switch (mode)
        {
        default:
        case LineDirectiveMode::SourceMap:
        case LineDirectiveMode::None:
            SLANG_UNEXPECTED("should not be trying to emit '#line' directive");
            return;
        case LineDirectiveMode::GLSL:
            {
                auto path = sourceLocation.pathInfo.foundPath;

                // GLSL doesn't support the traditional form of a `#line` directive without
                // an extension. Rather than depend on that extension we will output
                // a directive in the traditional GLSL fashion.
                //
                // TODO: Add some kind of configuration where we require the appropriate
                // extension and then emit a traditional line directive.

                int id = 0;
                if (!m_mapGLSLSourcePathToID.tryGetValue(path, id))
                {
                    id = m_glslSourceIDCount++;
                    m_mapGLSLSourcePathToID.add(path, id);
                }

                snprintf(buffer, sizeof(buffer), "%d", id);
                emitRawText(buffer);
                break;
            }
        case LineDirectiveMode::Default:
        case LineDirectiveMode::Standard:
            {
                // The simple case is to emit the path for the current source
                // location. We need to be a little bit careful with this,
                // because the path might include backslash characters if we
                // are on Windows, and we want to canonicalize those over
                // to forward slashes.
                //
                // TODO: Canonicalization like this should be done centrally
                // in a module that tracks source files.

                emitRawText("\"");
                const auto& path = sourceLocation.pathInfo.foundPath;
                for (auto c : path)
                {
                    char charBuffer[] = {c, 0};
                    switch (c)
                    {
                    default:
                        emitRawText(charBuffer);
                        break;

                        // The incoming file path might use `/` and/or `\\` as
                        // a directory separator. We want to canonicalize this.
                        //
                        // TODO: should probably canonicalize paths to not use backslash
                        // somewhere else in the compilation pipeline...
                    case '\\':
                        emitRawText("/");
                        break;
                    }
                }
                emitRawText("\"");
                break;
            }
        }
    }

    emitRawText("\n");
}

void SourceWriter::_calcLocation(Index& outLineIndex, Index& outColumnIndex)
{
    // If there are move chars we need to update
    if (m_currentOutputOffset < m_builder.getLength())
    {
        const char* cur = m_builder.getBuffer() + m_currentOutputOffset;
        const char* end = m_builder.end();

        const char* start = cur;

        while (cur < end)
        {
            // Reset start
            start = cur;

            // Look for the end of the line
            while (*cur != '\n' && *cur != '\r' && cur < end)
            {
                cur++;
            }

            // If we are not at the total end then we must have hit a \n or \r
            if (cur < end)
            {
                const auto c = *cur++;

                // Next line
                ++m_currentLineIndex;

                // Check the next char to see if it's part of a CR/LF combination
                if (cur < end)
                {
                    const auto d = *cur;
                    // If it is combination skip the next byte
                    cur += ((c ^ d) == ('\r' ^ '\n'));
                }

                // Calculate the offset to the start of this line
                m_currentColumnIndex = 0;
                start = cur;
            }
        }

        // Set the current offset to the end
        m_currentOutputOffset = m_builder.getLength();

        // Get the bytes remaining on this line (which may not be complete)
        const UnownedStringSlice lineRemaining(start, m_builder.end());

        // Offset the column index in codepoints
        m_currentColumnIndex += UTF8Util::calcCodePointCount(lineRemaining);
    }

    // Output the position
    outColumnIndex = m_currentColumnIndex;
    outLineIndex = m_currentLineIndex;
}

} // namespace Slang
