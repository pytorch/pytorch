// slang-diagnostic-sink.cpp
#include "slang-diagnostic-sink.h"

#include "../core/slang-char-util.h"
#include "../core/slang-dictionary.h"
#include "../core/slang-memory-arena.h"
#include "../core/slang-string-util.h"
#include "slang-core-diagnostics.h"
#include "slang-name-convention-util.h"
#include "slang-name.h"

namespace Slang
{

void printDiagnosticArg(StringBuilder& sb, char const* str)
{
    sb << str;
}

void printDiagnosticArg(StringBuilder& sb, int32_t val)
{
    sb << val;
}

void printDiagnosticArg(StringBuilder& sb, uint32_t val)
{
    sb << val;
}

void printDiagnosticArg(StringBuilder& sb, int64_t val)
{
    sb << val;
}

void printDiagnosticArg(StringBuilder& sb, uint64_t val)
{
    sb << val;
}

void printDiagnosticArg(StringBuilder& sb, double val)
{
    sb << val;
}

void printDiagnosticArg(StringBuilder& sb, Slang::String const& str)
{
    sb << str;
}

void printDiagnosticArg(StringBuilder& sb, Slang::UnownedStringSlice const& str)
{
    sb.append(str);
}


void printDiagnosticArg(StringBuilder& sb, Name* name)
{
    sb << getText(name);
}


void printDiagnosticArg(StringBuilder& sb, TokenType tokenType)
{
    sb << TokenTypeToString(tokenType);
}

void printDiagnosticArg(StringBuilder& sb, Token const& token)
{
    sb << token.getContent();
}

SourceLoc getDiagnosticPos(Token const& token)
{
    return token.loc;
}

// Take the format string for a diagnostic message, along with its arguments, and turn it into a
static void formatDiagnosticMessage(
    StringBuilder& sb,
    char const* format,
    int argCount,
    DiagnosticArg const* args)
{
    char const* spanBegin = format;
    for (;;)
    {
        char const* spanEnd = spanBegin;
        while (int c = *spanEnd)
        {
            if (c == '$')
                break;
            spanEnd++;
        }

        sb.append(spanBegin, int(spanEnd - spanBegin));
        if (!*spanEnd)
            return;

        SLANG_ASSERT(*spanEnd == '$');
        spanEnd++;
        int d = *spanEnd++;
        switch (d)
        {
        // A double dollar sign `$$` is used to emit a single `$`
        case '$':
            sb.append('$');
            break;

        // A single digit means to emit the corresponding argument.
        // TODO: support more than 10 arguments, and add options
        // to control formatting, etc.
        case '0':
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            {
                int index = d - '0';
                if (index >= argCount)
                {
                    // TODO(tfoley): figure out what a good policy will be for "panic" situations
                    // like this
                    SLANG_INVALID_OPERATION("too few arguments for diagnostic message");
                }
                else
                {
                    DiagnosticArg const& arg = args[index];
                    arg.printFunc(sb, arg.data);
                }
            }
            break;

        default:
            SLANG_INVALID_OPERATION("invalid diagnostic message format");
            break;
        }

        spanBegin = spanEnd;
    }
}

static void formatDiagnostic(
    const HumaneSourceLoc& humaneLoc,
    Diagnostic const& diagnostic,
    DiagnosticSink::Flags flags,
    StringBuilder& outBuilder)
{
    if (flags & DiagnosticSink::Flag::HumaneLoc)
    {
        outBuilder << humaneLoc.pathInfo.foundPath;
        outBuilder << "(";
        outBuilder << Int32(humaneLoc.line);
        if (flags & DiagnosticSink::Flag::LanguageServer)
        {
            outBuilder << ", " << humaneLoc.column;
        }
        outBuilder << "): ";
    }

    outBuilder << getSeverityName(diagnostic.severity);

    if ((flags & DiagnosticSink::Flag::LanguageServer) || diagnostic.ErrorID >= 0)
    {
        outBuilder << " ";
        outBuilder << diagnostic.ErrorID;
    }

    outBuilder << ": ";
    outBuilder << diagnostic.Message;
    outBuilder << "\n";
}

static void _replaceTabWithSpaces(const UnownedStringSlice& slice, Int tabSize, StringBuilder& out)
{
    const char* start = slice.begin();
    const char* const end = slice.end();

    const Index startLength = out.getLength();

    for (const char* cur = start; cur < end; cur++)
    {
        if (*cur == '\t')
        {
            if (start < cur)
            {
                out.append(start, cur);
            }

            // The amount of spaces we add depends on the current position.
            const Index lastPosition = out.getLength() - startLength;
            Index tabPosition = lastPosition;

            // Strip the tabPosition so it's back to the tab stop
            // Special case if tabSize is a power of 2
            if ((tabSize & (tabSize - 1)) == 0)
            {
                tabPosition = tabPosition & ~Index(tabSize - 1);
            }
            else
            {
                tabPosition -= tabPosition % tabSize;
            }

            // Move to next tab
            tabPosition += tabSize;

            // The amount of spaces to simulate the tab
            const Index spacesCount = tabPosition - lastPosition;

            // Add the spaces
            out.appendRepeatedChar(' ', spacesCount);

            // Set the start at the first character past
            start = cur + 1;
        }
    }

    if (start < end)
    {
        out.append(start, end);
    }
}

// Given multi-line text, and a position within the text (as a pointer into the memory of text)
// extract the line that contains pos
static UnownedStringSlice _extractLineContainingPosition(
    const UnownedStringSlice& text,
    const char* pos)
{
    SLANG_ASSERT(text.isMemoryContained(pos));

    const char* const contentStart = text.begin();
    const char* const contentEnd = text.end();

    // We want to determine the start of the line, and the end of the line
    const char* start = pos;
    for (; start > contentStart; --start)
    {
        const char c = *start;
        if (c == '\n' || c == '\r')
        {
            // We want the character after, but we can only do this if not already at pos
            start += int(start < pos);
            break;
        }
    }
    const char* end = pos;
    for (; end < contentEnd; ++end)
    {
        const char c = *end;
        if (c == '\n' || c == '\r')
        {
            break;
        }
    }

    return UnownedStringSlice(start, end);
}

static void _reduceLength(Index startIndex, const UnownedStringSlice& prefix, StringBuilder& ioBuf)
{
    StringBuilder buf;
    buf << prefix;
    buf.append(ioBuf.getUnownedSlice().tail(startIndex));
    ioBuf = buf;
}

static void _sourceLocationNoteDiagnostic(
    DiagnosticSink* sink,
    SourceView* sourceView,
    SourceLoc sourceLoc,
    StringBuilder& sb)
{
    SourceFile* sourceFile = sourceView->getSourceFile();
    if (!sourceFile)
    {
        return;
    }

    UnownedStringSlice content = sourceFile->getContent();

    // Make sure the offset is within content.
    // This is important because it's possible to have a 'SourceFile' that doesn't contain any
    // content (for example when reconstructed via serialization with just line offsets, the actual
    // source text 'content' isn't available).
    const int offset = sourceView->getRange().getOffset(sourceLoc);
    if (offset < 0 || offset >= content.getLength())
    {
        return;
    }

    // Work out the position of the SourceLoc in the source
    const char* const pos = content.begin() + offset;

    UnownedStringSlice line = _extractLineContainingPosition(content, pos);

    // Trim any trailing white space
    line = UnownedStringSlice(line.begin(), line.trim().end());

    // TODO(JS): The tab size should ideally be configurable from command line.
    // For now just go with 4.
    const Index tabSize = 4;

    StringBuilder sourceLine;
    StringBuilder caretLine;

    // First work out the sourceLine
    _replaceTabWithSpaces(line, tabSize, sourceLine);

    // Now the caretLine which appears underneath the sourceLine
    {
        // Produce the text up to the caret position (at pos), taking into account tabs
        _replaceTabWithSpaces(UnownedStringSlice(line.begin(), pos), tabSize, caretLine);

        // Now make all spaces
        const Index length = caretLine.getLength();
        caretLine.clear();
        caretLine.appendRepeatedChar(' ', length);

        Index caretIndex = caretLine.getLength();

        // Add caret
        caretLine << "^";

        auto lexer = sink->getSourceLocationLexer();
        if (lexer)
        {
            UnownedStringSlice token = lexer(UnownedStringSlice(pos, line.end()));

            if (token.getLength() > 1)
            {
                caretLine.appendRepeatedChar('~', token.getLength() - 1);
            }
        }

        const Index maxLength = sink->getSourceLineMaxLength();
        if (maxLength > 0)
        {
            const UnownedStringSlice ellipsis = UnownedStringSlice::fromLiteral("...");
            const UnownedStringSlice spaces = UnownedStringSlice::fromLiteral("   ");
            SLANG_ASSERT(ellipsis.getLength() == spaces.getLength());

            // We use the caretLine length if we have a lexer, because it will have underscores such
            // that it's end is the end of the item at issue. If we don't have the lexer, we
            // guesstimate using 1/4 of the maximum length
            const Index endIndex = lexer ? caretLine.getLength() : (caretIndex + (maxLength / 4));

            if (endIndex > maxLength)
            {
                const Index startIndex = endIndex - (maxLength - ellipsis.getLength());

                _reduceLength(startIndex, ellipsis, sourceLine);
                _reduceLength(startIndex, spaces, caretLine);
            }

            if (sourceLine.getLength() > maxLength)
            {
                StringBuilder buf;
                buf.append(sourceLine.getUnownedSlice().head(maxLength - ellipsis.getLength()));
                buf << ellipsis;
                sourceLine = buf;
            }
        }
    }

    // We could have handling here for if the line is too long, that we surround the important
    // section will ellipsis for example. For now we just output.

    sb << sourceLine << "\n";
    sb << caretLine << "\n";
}

// Output the length of the token at `sourceLoc`. This is used by language server.
static void _tokenLengthNoteDiagnostic(
    DiagnosticSink* sink,
    SourceView* sourceView,
    SourceLoc sourceLoc,
    StringBuilder& sb)
{
    SourceFile* sourceFile = sourceView->getSourceFile();
    if (!sourceFile)
    {
        return;
    }

    UnownedStringSlice content = sourceFile->getContent();

    // Make sure the offset is within content.
    // This is important because it's possible to have a 'SourceFile' that doesn't contain any
    // content (for example when reconstructed via serialization with just line offsets, the actual
    // source text 'content' isn't available).
    const int offset = sourceView->getRange().getOffset(sourceLoc);
    if (offset < 0 || offset >= content.getLength())
    {
        return;
    }

    // Work out the position of the SourceLoc in the source
    const char* const pos = content.begin() + offset;

    UnownedStringSlice line = _extractLineContainingPosition(content, pos);

    // Trim any trailing white space
    line = UnownedStringSlice(line.begin(), line.trim().end());

    auto lexer = sink->getSourceLocationLexer();
    if (lexer)
    {
        UnownedStringSlice token = lexer(UnownedStringSlice(pos, line.end()));

        if (token.getLength() > 1)
        {
            sb << "^+" << token.getLength() << "\n";
        }
    }
}

static void formatDiagnostic(DiagnosticSink* sink, Diagnostic const& diagnostic, StringBuilder& sb)
{
    auto sourceManager = sink->getSourceManager();

    SourceView* sourceView = nullptr;
    HumaneSourceLoc humaneLoc;
    const auto sourceLoc = diagnostic.loc;
    {
        if (sourceManager)
        {
            sourceView = sourceManager->findSourceViewRecursively(sourceLoc);
            if (sourceView)
            {
                humaneLoc = sourceView->getHumaneLoc(sourceLoc);
            }
        }

        formatDiagnostic(humaneLoc, diagnostic, sink->getFlags(), sb);

        {
            SourceView* currentView = sourceView;

            while (currentView && currentView->getInitiatingSourceLoc().isValid() &&
                   currentView->getSourceFile()->getPathInfo().type == PathInfo::Type::TokenPaste)
            {
                SourceView* initiatingView =
                    sourceManager
                        ? sourceManager->findSourceView(currentView->getInitiatingSourceLoc())
                        : nullptr;
                if (initiatingView == nullptr)
                {
                    break;
                }

                const DiagnosticInfo& diagnosticInfo = MiscDiagnostics::seeTokenPasteLocation;

                // Turn the message format into a message. For the moment it assumes no parameters.
                StringBuilder msg;
                formatDiagnosticMessage(msg, diagnosticInfo.messageFormat, 0, nullptr);

                // Set up the diagnostic.
                Diagnostic initiationDiagnostic;
                initiationDiagnostic.ErrorID = diagnosticInfo.id;
                initiationDiagnostic.Message = msg.produceString();
                initiationDiagnostic.loc = sourceView->getInitiatingSourceLoc();
                initiationDiagnostic.severity = diagnosticInfo.severity;

                // TODO(JS):
                // Not 100%  clear what the best sourceLoc type is most useful here - we will go
                // with default for now
                HumaneSourceLoc pasteHumaneLoc =
                    initiatingView->getHumaneLoc(sourceView->getInitiatingSourceLoc());

                // Okay we should output where the token paste took place
                formatDiagnostic(pasteHumaneLoc, initiationDiagnostic, sink->getFlags(), sb);

                // Make the initiatingView the current view
                currentView = initiatingView;
            }
        }
    }

    // If we are a language server, output additional token length info.
    if (sourceView && sink->isFlagSet(DiagnosticSink::Flag::LanguageServer))
    {
        _tokenLengthNoteDiagnostic(sink, sourceView, sourceLoc, sb);
    }

    if (sourceView && sink->isFlagSet(DiagnosticSink::Flag::SourceLocationLine) &&
        diagnostic.loc.isValid())
    {
        _sourceLocationNoteDiagnostic(sink, sourceView, sourceLoc, sb);
    }

    if (sourceView && sink->isFlagSet(DiagnosticSink::Flag::VerbosePath))
    {
        auto actualHumaneLoc = sourceView->getHumaneLoc(diagnostic.loc, SourceLocType::Actual);

        // Look up the path verbosely (will get the canonical path if necessary)
        actualHumaneLoc.pathInfo.foundPath = sourceView->getSourceFile()->calcVerbosePath();

        // Only output if it's actually different
        if (actualHumaneLoc.pathInfo.foundPath != humaneLoc.pathInfo.foundPath ||
            actualHumaneLoc.line != humaneLoc.line || actualHumaneLoc.column != humaneLoc.column)
        {
            formatDiagnostic(actualHumaneLoc, diagnostic, sink->getFlags(), sb);
        }
    }
}

void DiagnosticSink::init(SourceManager* sourceManager, SourceLocationLexer sourceLocationLexer)
{
    m_errorCount = 0;
    m_internalErrorLocsNoted = 0;

    m_sourceManager = sourceManager;
    m_sourceLocationLexer = sourceLocationLexer;
    m_sourceLineMaxLength = 0;

    m_flags = Flag::HumaneLoc;

    // If we have a source location lexer, we'll by default enable source location output
    if (sourceLocationLexer)
    {
        setFlag(Flag::SourceLocationLine);
    }
}

void DiagnosticSink::reset()
{
    m_errorCount = 0;
    m_internalErrorLocsNoted = 0;

    outputBuffer.clear();
}


void DiagnosticSink::noteInternalErrorLoc(SourceLoc const& loc)
{
    // Don't consider invalid source locations.
    if (!loc.isValid())
        return;

    if (m_parentSink)
    {
        m_parentSink->noteInternalErrorLoc(loc);
    }

    // If this is the first source location being noted,
    // then emit a message to help the user isolate what
    // code might have confused the compiler.
    if (m_internalErrorLocsNoted == 0)
    {
        diagnose(loc, MiscDiagnostics::noteLocationOfInternalError);
    }
    m_internalErrorLocsNoted++;
}

SlangResult DiagnosticSink::getBlobIfNeeded(ISlangBlob** outBlob)
{
    // If the client doesn't want an output blob, there is nothing to do.
    //
    if (!outBlob)
        return SLANG_OK;

    // For outputBuffer to be valid and hold diagnostics, writer must not be set
    SLANG_ASSERT(writer == nullptr);

    // If there were no errors, and there was no diagnostic output, there is nothing to do.
    if (getErrorCount() == 0 && outputBuffer.getLength() == 0)
    {
        return SLANG_OK;
    }

    Slang::ComPtr<ISlangBlob> blob = Slang::StringUtil::createStringBlob(outputBuffer);
    *outBlob = blob.detach();

    return SLANG_OK;
}

bool DiagnosticSink::diagnoseImpl(
    DiagnosticInfo const& info,
    const UnownedStringSlice& formattedMessage)
{
    if (info.severity >= Severity::Error)
    {
        m_errorCount++;
    }

    if (writer)
    {
        writer->write(formattedMessage.begin(), formattedMessage.getLength());
    }
    else
    {
        outputBuffer.append(formattedMessage);
    }

    if (m_parentSink)
    {
        m_parentSink->diagnoseImpl(info, formattedMessage);
    }

    if (info.severity >= Severity::Fatal)
    {
        // TODO: figure out a better policy for aborting compilation
        std::string message(formattedMessage.begin(), formattedMessage.end());
        SLANG_ABORT_COMPILATION(message.c_str());
    }
    return true;
}

Severity DiagnosticSink::getEffectiveMessageSeverity(
    DiagnosticInfo const& info,
    SourceLoc const& location)
{
    Severity effectiveSeverity = info.severity;

    if (effectiveSeverity <= Severity::Warning && m_sourceWarningStateTracker)
    {
        effectiveSeverity = m_sourceWarningStateTracker->consumeWarningSeverity(
            location,
            info.id,
            effectiveSeverity);
    }

    Severity* pSeverityOverride = m_severityOverrides.tryGetValue(info.id);

    // See if there is an override
    if (pSeverityOverride)
    {
        // Override the current severity, but don't allow lowering it if it's Error or Fatal
        if (effectiveSeverity < Severity::Error || *pSeverityOverride >= effectiveSeverity)
            effectiveSeverity = *pSeverityOverride;
    }

    if (isFlagSet(Flag::TreatWarningsAsErrors) && effectiveSeverity == Severity::Warning)
        effectiveSeverity = Severity::Error;

    return effectiveSeverity;
}

bool DiagnosticSink::diagnoseImpl(
    SourceLoc const& pos,
    DiagnosticInfo info,
    int argCount,
    DiagnosticArg const* args)
{
    // Override the severity in the 'info' structure to pass it further into formatDiagnostics
    info.severity = getEffectiveMessageSeverity(info, pos);

    if (info.severity == Severity::Disable)
        return false;

    StringBuilder messageBuilder;
    {
        StringBuilder sb;
        formatDiagnosticMessage(sb, info.messageFormat, argCount, args);

        Diagnostic diagnostic;
        diagnostic.ErrorID = info.id;
        diagnostic.Message = sb.produceString();
        diagnostic.loc = pos;
        diagnostic.severity = info.severity;

        // If so, pass the error string along to them
        formatDiagnostic(this, diagnostic, messageBuilder);
    }

    return diagnoseImpl(info, messageBuilder.getUnownedSlice());
}

void DiagnosticSink::diagnoseRaw(Severity severity, char const* message)
{
    return diagnoseRaw(severity, UnownedStringSlice(message));
}

void DiagnosticSink::diagnoseRaw(Severity severity, const UnownedStringSlice& message)
{
    if (severity >= Severity::Error)
    {
        m_errorCount++;
    }

    // Did the client supply a callback for us to use?
    if (writer)
    {
        // If so, pass the error string along to them.
        writer->write(message.begin(), message.getLength());
    }
    else
    {
        // If the user doesn't have a callback, then just
        // collect our diagnostic messages into a buffer.
        outputBuffer.append(message);
    }

    if (m_parentSink)
    {
        m_parentSink->diagnoseRaw(severity, message);
    }

    if (severity >= Severity::Fatal)
    {
        // TODO: figure out a better policy for aborting compilation
        SLANG_ABORT_COMPILATION("");
    }
}

void DiagnosticSink::overrideDiagnosticSeverity(
    int diagnosticId,
    Severity overrideSeverity,
    const DiagnosticInfo* info)
{
    if (info)
    {
        SLANG_ASSERT(info->id == diagnosticId);

        // If the override is the same as the default, we can just remove the override
        if (info->severity == overrideSeverity)
        {
            m_severityOverrides.remove(diagnosticId);
            return;
        }
    }

    // Set the override
    m_severityOverrides[diagnosticId] = overrideSeverity;
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DiagnosticLookup
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

Index DiagnosticsLookup::_findDiagnosticIndexByExactName(const UnownedStringSlice& slice) const
{
    const Index* indexPtr = m_nameMap.tryGetValue(slice);
    return indexPtr ? *indexPtr : -1;
}

void DiagnosticsLookup::_addName(const char* name, Index diagnosticIndex)
{
    UnownedStringSlice nameSlice(name);
    m_nameMap.add(nameSlice, diagnosticIndex);
}

void DiagnosticsLookup::addAlias(const char* name, const char* diagnosticName)
{
    const Index index = _findDiagnosticIndexByExactName(UnownedStringSlice(diagnosticName));
    SLANG_ASSERT(index >= 0);
    if (index >= 0)
    {
        _addName(name, index);
    }
}

const DiagnosticInfo* DiagnosticsLookup::getDiagnosticById(Int id) const
{
    const auto indexPtr = m_idMap.tryGetValue(id);
    return indexPtr ? m_diagnostics[*indexPtr] : nullptr;
}

const DiagnosticInfo* DiagnosticsLookup::findDiagnosticByExactName(
    const UnownedStringSlice& slice) const
{
    const Index* indexPtr = m_nameMap.tryGetValue(slice);
    return indexPtr ? m_diagnostics[*indexPtr] : nullptr;
}

const DiagnosticInfo* DiagnosticsLookup::findDiagnosticByName(const UnownedStringSlice& slice) const
{
    const auto convention = NameConventionUtil::inferConventionFromText(slice);
    switch (convention)
    {
    case NameConvention::Invalid:
        return nullptr;
    case NameConvention::LowerCamel:
        return findDiagnosticByExactName(slice);
    default:
        break;
    }

    StringBuilder buf;
    NameConventionUtil::convert(getNameStyle(convention), slice, NameConvention::LowerCamel, buf);

    return findDiagnosticByExactName(buf.getUnownedSlice());
}

Index DiagnosticsLookup::add(const DiagnosticInfo* info)
{
    // Check it's not already added
    SLANG_ASSERT(m_diagnostics.indexOf(info) < 0);

    const Index diagnosticIndex = m_diagnostics.getCount();
    m_diagnostics.add(info);

    _addName(info->name, diagnosticIndex);
    m_idMap.addIfNotExists(info->id, diagnosticIndex);

    return diagnosticIndex;
}

void DiagnosticsLookup::add(const DiagnosticInfo* const* infos, Index infosCount)
{
    for (Index i = 0; i < infosCount; ++i)
    {
        add(infos[i]);
    }
}

DiagnosticsLookup::DiagnosticsLookup()
    : m_arena(kArenaInitialSize)
{
}

DiagnosticsLookup::DiagnosticsLookup(
    const DiagnosticInfo* const* diagnostics,
    Index diagnosticsCount)
    : m_arena(kArenaInitialSize)
{
    // TODO: We should eventually have a more formal system for associating individual
    // diagnostics, or groups of diagnostics, with user-exposed names for use when
    // enabling/disabling warnings (or turning warnings into errors, etc.).
    //
    // For now we build a map from diagnostic name to it's entry.

    add(diagnostics, diagnosticsCount);
}

} // namespace Slang
