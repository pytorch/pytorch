#include "slang-language-server-auto-format.h"

#include "../compiler-core/slang-lexer.h"
#include "../core/slang-char-util.h"
#include "../core/slang-file-system.h"

namespace Slang
{

String findClangFormatTool()
{
    String processName = String("clang-format") + String(Process::getExecutableSuffix());
    if (File::exists(processName))
        return processName;
    RefPtr<Process> proc;
    CommandLine cmdLine;
    cmdLine.setExecutableLocation(ExecutableLocation(processName));
    if (Process::create(cmdLine, 0, proc) == SLANG_OK)
    {
        auto inStream = proc->getStream(StdStreamType::In);
        if (inStream)
            inStream->close();
        proc->kill(0);
        return processName;
    }
    auto fileName =
        Slang::SharedLibraryUtils::getSharedLibraryFileName((void*)slang_createGlobalSession);
    auto dirName = Slang::Path::getParentDirectory(fileName);
    auto localProcess = Path::combine(dirName, processName);
    if (File::exists(localProcess))
        return localProcess;
    auto extensionsStr = UnownedStringSlice("extensions");
    Index vsCodeLoc = dirName.indexOf(extensionsStr);
    if (vsCodeLoc != -1)
    {
        // If we still cannot find clang-format, try to use the clang-format bundled with VSCode's
        // C++ extension.
        String vsCodeExtDir = dirName.subString(0, vsCodeLoc + extensionsStr.getLength());
        struct CallbackContext
        {
            String foundPath;
            String parentDir;
            String processName;
        } callbackContext;
        callbackContext.processName = processName;
        callbackContext.parentDir = vsCodeExtDir;
        OSFileSystem::getExtSingleton()->enumeratePathContents(
            vsCodeExtDir.getBuffer(),
            [](SlangPathType /*pathType*/, const char* name, void* userData)
            {
                CallbackContext* context = (CallbackContext*)userData;
                if (UnownedStringSlice(name).indexOf(UnownedStringSlice("ms-vscode.cpptools-")) !=
                    -1)
                {
                    String candidateFileName = Path::combine(
                        Path::combine(context->parentDir, name, "LLVM"),
                        "bin",
                        context->processName);
                    if (File::exists(candidateFileName))
                        context->foundPath = candidateFileName;
                }
            },
            &callbackContext);
        if (callbackContext.foundPath.getLength())
            return callbackContext.foundPath;
    }
    return String();
}

List<TextRange> extractFormattingExclusionRanges(UnownedStringSlice text)
{
    List<TextRange> resultRanges;
    // Search the document and find all the spirv-asm blocks to exclude them from formatting.
    Lexer lexer;
    SourceManager manager;
    manager.initialize(nullptr, nullptr);
    auto sourceFile = manager.createSourceFileWithString(PathInfo(), text);
    auto sourceView = manager.createSourceView(sourceFile, nullptr, SourceLoc());
    DiagnosticSink sink;
    RootNamePool rootPool;
    NamePool namePool;
    namePool.setRootNamePool(&rootPool);
    MemoryArena memory;
    memory.init(1 << 16);
    lexer.initialize(sourceView, &sink, &namePool, &memory);
    for (;;)
    {
        auto headerToken = lexer.lexToken();
        if (headerToken.type == TokenType::EndOfFile)
            break;
        if (headerToken.type != TokenType::Identifier ||
            headerToken.getContent() != toSlice("spirv_asm"))
        {
            continue;
        }

        // We have found a spirv-asm block, now find the end of it.
        int braceCounter = 0;
        Token endToken;
        Token startToken;
        for (;;)
        {
            auto token = lexer.lexToken();
            if (token.type == TokenType::EndOfFile)
                break;
            switch (token.type)
            {
            case TokenType::RBrace:
                braceCounter--;
                if (braceCounter == 0)
                {
                    endToken = token;
                    goto breakLabel;
                }
                break;
            case TokenType::LBrace:
                {
                    if (braceCounter == 0)
                        startToken = token;
                    braceCounter++;
                }
                break;
            case TokenType::WhiteSpace:
            case TokenType::LineComment:
            case TokenType::BlockComment:
            case TokenType::NewLine:
                break;
            default:
                if (braceCounter == 0)
                    goto breakLabel;
                break;
            }
        }
    breakLabel:;
        if (endToken.type == TokenType::RBrace)
        {
            TextRange range;
            range.offsetStart = startToken.getLoc().getRaw();
            range.offsetEnd = endToken.getLoc().getRaw();
            resultRanges.add(range);
        }
    }
    return resultRanges;
}

void translateXmlEscape(StringBuilder& sb, UnownedStringSlice text)
{
    if (text.getLength() == 0)
        return;
    if (text[0] == '#')
    {
        Int charVal = 0;
        StringUtil::parseInt(text.tail(1), charVal);
        if (charVal != 0)
            sb.appendChar((char)charVal);
    }
    else if (text == "lt")
    {
        sb.appendChar('<');
    }
    else if (text == "gt")
    {
        sb.appendChar('>');
    }
    else if (text == "amp")
    {
        sb.appendChar('&');
    }
    else if (text == "apos")
    {
        sb.appendChar('\'');
    }
    else if (text == "quot")
    {
        sb.appendChar('\"');
    }
}

String parseXmlText(UnownedStringSlice text)
{
    StringBuilder sb;
    Index pos = 0;
    for (; pos < text.getLength();)
    {
        if (text[pos] == '&')
        {
            pos++;
            Index endPos = pos;
            while (endPos < text.getLength() && text[endPos] != ';')
                endPos++;
            auto escapedToken = text.subString(pos, endPos - pos);
            pos = endPos + 1;
            translateXmlEscape(sb, escapedToken);
        }
        else
        {
            sb.appendChar(text[pos]);
            pos++;
        }
    }
    return sb.produceString();
}

bool shouldUseFallbackStyle(const FormatOptions& options)
{
    if (!options.style.startsWith("file"))
        return false;

    String clangFormatFileName = ".clang-format";
    bool standardFileName = true;
    if (options.style.startsWith("file:"))
    {
        auto fileName = options.style.getUnownedSlice().tail(5);
        if (fileName.startsWith("\""))
            fileName = fileName.head(fileName.getLength() - 1).tail(1);
        clangFormatFileName = fileName;
        standardFileName = false;
    }
    auto path = options.fileName;
    do
    {
        path = Path::getParentDirectory(path);
        auto expectedFormatFileName = Path::combine(path, clangFormatFileName);
        if (File::exists(expectedFormatFileName))
        {
            return false;
        }
        if (standardFileName)
        {
            expectedFormatFileName = Path::combine(path, "_clang_format");
            if (File::exists(expectedFormatFileName))
            {
                return false;
            }
        }
    } while (path.getLength());
    return true;
}

List<Edit> formatSource(
    UnownedStringSlice text,
    Index lineStart,
    Index lineEnd,
    Index cursorOffset,
    const List<TextRange>& exclusionRanges,
    const FormatOptions& options)
{
    List<Edit> edits;

    String clangProcessName = options.clangFormatLocation;
    CommandLine cmdLine;
    cmdLine.setExecutableLocation(ExecutableLocation(clangProcessName));

    cmdLine.addArg("--assume-filename");
    cmdLine.addArg(options.fileName + ".cs");
    if (cursorOffset != -1)
    {
        cmdLine.addArg("--cursor=" + String(cursorOffset));
    }
    if (lineStart != -1)
    {
        cmdLine.addArg("--lines=" + String(lineStart) + ":" + String(lineEnd + 1));
    }
    cmdLine.addArg("--output-replacements-xml");
    // clang-format does not allow non-builtin style for `fallback-style`.
    // We detect if clang format file exists, and if not set style directly to user specified
    // fallback style.
    if (shouldUseFallbackStyle(options))
    {
        if (options.fallbackStyle.getLength())
        {
            cmdLine.addArg("-style");
            cmdLine.addArg(options.fallbackStyle);
        }
    }
    else if (options.style.getLength())
    {
        cmdLine.addArg("-style");
        cmdLine.addArg(options.style);
    }
    RefPtr<Process> proc;
    if (SLANG_FAILED(Process::create(cmdLine, 0, proc)))
        return edits;

    auto inStream = proc->getStream(StdStreamType::In);
    inStream->write(text.begin(), text.getLength());
    char terminator = '\0';
    inStream->write(&terminator, 1);
    inStream->flush();
    inStream->close();
    ExecuteResult result;
    ProcessUtil::readUntilTermination(proc, result);

    /*
    Example result of clang-format:

    <?xml version='1.0'?>
    <replacements xml:space='preserve' incomplete_format='false'>
    <replacement offset='26' length='2'>&#13;&#10;  </replacement>
    </replacements>
    */

    // Adhoc parsing of clang-format's result.
    List<UnownedStringSlice> lines;
    auto offsetStr = UnownedStringSlice("offset=");
    auto lengthStr = UnownedStringSlice("length=");
    auto endStr = UnownedStringSlice("</replacement>");
    auto replacementStr = UnownedStringSlice("<replacement ");
    StringUtil::calcLines(result.standardOutput.getUnownedSlice(), lines);
    for (auto line : lines)
    {
        line = line.trim();
        if (!line.startsWith(replacementStr))
            continue;
        line = line.tail(replacementStr.getLength());
        Index pos = line.indexOf(offsetStr);
        if (pos == -1)
            continue;
        pos += offsetStr.getLength();
        if (pos < line.getLength() && line[pos] == '\'')
            pos++;
        Edit edt;
        edt.offset = StringUtil::parseIntAndAdvancePos(line, pos);
        pos = line.indexOf(lengthStr);
        if (pos == -1)
            continue;
        Index exclusionRangeId = exclusionRanges.binarySearch(
            edt.offset,
            [](TextRange range, Index offset)
            {
                if (range.offsetEnd <= offset)
                    return -1;
                if (range.offsetStart > offset)
                    return 1;
                return 0;
            });
        if (exclusionRangeId != -1)
            continue;
        pos += lengthStr.getLength();
        if (pos < line.getLength() && line[pos] == '\'')
            pos++;
        edt.length = StringUtil::parseIntAndAdvancePos(line, pos);
        line = line.tail(pos);
        pos = line.indexOf('>');
        if (pos == -1)
            continue;
        line = line.tail(pos + 1);
        Index endPos = line.indexOf(endStr);
        if (endPos == -1)
            continue;
        line = line.head(endPos);
        edt.text = parseXmlText(line);
        // For on-type formatting, don't make any changes beyond the current cursor position.
        if (cursorOffset != -1 && edt.offset >= cursorOffset)
            break;
        // Never allow clang-format to put the semicolon after `}` in its own line.
        if (edt.offset < text.getLength() && edt.length == 0 && text[edt.offset] == ';' &&
            edt.offset > 0 && text[edt.offset - 1] == '}')
            continue;
        // If need to preserve line break, turn all edits with a line break into a space.
        if (options.behavior == FormatBehavior::PreserveLineBreak)
        {
            auto originalText = text.subString(edt.offset, edt.length);
            bool originalHasLineBreak = originalText.indexOf('\n') != -1;
            bool newHasLineBreak = edt.text.indexOf('\n') != -1;
            if (originalHasLineBreak == newHasLineBreak)
            {
            }
            else if (!originalHasLineBreak && newHasLineBreak)
            {
                if (edt.offset < text.getLength() && edt.offset >= 0 && text[edt.offset] == '}')
                    continue;
                edt.text = " ";
            }
            else
            {
                continue;
            }
        }
        edits.add(edt);
    }
    return edits;
}


} // namespace Slang
