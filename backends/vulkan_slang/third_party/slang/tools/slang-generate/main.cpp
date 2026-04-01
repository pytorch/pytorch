// main.cpp

#include "../../source/core/slang-io.h"
#include "../../source/core/slang-list.h"
#include "../../source/core/slang-secure-crt.h"
#include "../../source/core/slang-string-util.h"
#include "../../source/core/slang-string.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace Slang;

typedef Slang::UnownedStringSlice StringSpan;

struct Node
{
    enum class Flavor
    {
        text,   // Ordinary text to write to output
        escape, // Meta-level code (statements)
        splice, // Meta-level expression to splice into output
    };

    // What sort of node is this?
    Flavor flavor;

    // The text of this node for `Flavor::text`
    StringSpan span;

    // The body of this node for other flavors
    Node* body = nullptr;

    // The next node in the document
    Node* next = nullptr;

    Node() = default;
    ~Node()
    {
        if (body)
            delete body;
        if (next)
            delete next;
    }
};

// Information about a source file
struct SourceFile : public RefObject
{
    String inputPath;
    String linePath; ///< The path to this file for #line output

    StringSpan text;
    Node* node = nullptr;
    SourceFile() = default;
    ~SourceFile()
    {
        if (text.begin())
            free((void*)text.begin());

        // To avoid deep recursion in the Node destructor,
        // we delete the first level of the node tree iteratively.
        while (node)
        {
            Node* next = node->next;
            node->next = nullptr;
            delete node;
            node = next;
        }
    }
};

void addNode(Node**& ioLink, Node::Flavor flavor, char const* spanBegin, char const* spanEnd)
{
    Node* node = new Node();
    node->flavor = flavor;
    node->span = StringSpan(spanBegin, spanEnd);
    node->next = nullptr;

    *ioLink = node;
    ioLink = &node->next;
}

void addNode(Node**& ioLink, Node::Flavor flavor, Node* body)
{
    Node* node = new Node();
    node->flavor = flavor;
    node->body = body;
    node->next = nullptr;

    *ioLink = node;
    ioLink = &node->next;
}

bool isAlpha(int c)
{
    return ((c >= 'a') && (c <= 'z')) || ((c >= 'A') && (c <= 'Z')) || (c == '_');
}

void addTextSpan(Node**& ioLink, char const* spanBegin, char const* spanEnd)
{
    // Don't add an empty text span.
    if (spanBegin == spanEnd)
        return;

    addNode(ioLink, Node::Flavor::text, spanBegin, spanEnd);
}

void addSpliceSpan(Node**& ioLink, Node* body)
{
    addNode(ioLink, Node::Flavor::splice, body);
}

void addEscapeSpan(Node**& ioLink, Node* body)
{
    addNode(ioLink, Node::Flavor::escape, body);
}

void addEscapeSpan(Node**& ioLink, char const* spanBegin, char const* spanEnd)
{
    Node* body = nullptr;
    Node** link = &body;

    addTextSpan(link, spanBegin, spanEnd);

    return addEscapeSpan(ioLink, body);
}

bool isIdentifierChar(int c)
{
    if (c >= 'a' && c <= 'z')
        return true;
    if (c >= 'A' && c <= 'Z')
        return true;
    if (c == '_')
        return true;

    return false;
}

struct Reader
{
    char const* cursor;
    char const* end;
};

int peek(Reader const& reader)
{
    if (reader.cursor == reader.end)
        return EOF;

    return *reader.cursor;
}

int get(Reader& reader)
{
    if (reader.cursor == reader.end)
        return -1;

    return *reader.cursor++;
}

void handleNewline(Reader& reader, int c)
{
    int d = peek(reader);
    if ((c ^ d) == ('\r' ^ '\n'))
    {
        get(reader);
    }
}

bool isHorizontalSpace(int c)
{
    return (c == ' ') || (c == '\t');
}

void skipHorizontalSpace(Reader& reader)
{
    while (isHorizontalSpace(peek(reader)))
        get(reader);
}

void skipOptionalNewline(Reader& reader)
{
    switch (peek(reader))
    {
    default:
        break;

    case '\r':
    case '\n':
        {
            int c = get(reader);
            handleNewline(reader, c);
        }
        break;
    }
}

typedef unsigned int NodeReadFlags;
enum
{
    kNodeReadFlag_AllowEscape = 1 << 0,
};

Node* readBody(Reader& reader, NodeReadFlags flags, char openChar, int openCount, char closeChar)
{
    while (peek(reader) == openChar)
    {
        get(reader);
        openCount++;
    }

    Node* nodes = nullptr;
    Node** link = &nodes;

    bool atStartOfLine = true;
    int depth = 0;

    char const* spanBegin = reader.cursor;
    char const* lineBegin = reader.cursor;
    for (;;)
    {
        int c = get(reader);

        switch (c)
        {
        default:
            atStartOfLine = false;
            break;

        case EOF:
            {
                addTextSpan(link, spanBegin, reader.cursor);
                return nodes;
            }

        case '{':
        case '(':
            if (c == openChar)
            {
                depth++;
            }
            atStartOfLine = false;
            break;

        case ')':
        case '}':
            if (c == closeChar)
            {
                char const* spanEnd = reader.cursor - 1;

                if (openCount == 1)
                {
                    if (depth == 0)
                    {
                        // We are at the end of the body.
                        addTextSpan(link, spanBegin, spanEnd);
                        return nodes;
                    }

                    depth--;
                }
                else
                {
                    // Count how many closing chars are stacked up

                    int closeCount = 1;
                    while (peek(reader) == closeChar)
                    {
                        get(reader);
                        closeCount++;
                    }

                    if (closeCount == openCount)
                    {
                        // We are at the end of the body.
                        addTextSpan(link, spanBegin, spanEnd);
                        return nodes;
                    }
                }
            }
            atStartOfLine = false;
            break;


        case ' ':
        case '\t':
            break;

        case '\r':
        case '\n':
            {
                addTextSpan(link, spanBegin, reader.cursor);

                handleNewline(reader, c);

                lineBegin = reader.cursor;
                spanBegin = reader.cursor;
                atStartOfLine = true;
            }
            break;

        case '$':
            {
                // If this is the start of a splice, then
                // the end of the preceding raw-text space
                // will be the byte before `$`
                char const* spanEnd = reader.cursor - 1;

                if (peek(reader) == '(')
                {
                    // This appears to be an expression splice.
                    //
                    // We must end the preceding span.
                    //
                    addTextSpan(link, spanBegin, spanEnd);

                    Node* body = readBody(reader, 0, '(', 0, ')');

                    addSpliceSpan(link, body);

                    spanBegin = reader.cursor;
                    atStartOfLine = false;
                }
                else if (peek(reader) == '{')
                {
                    // This is the start of a block-structured escape, which will
                    // end at a matching `}`.

                    addTextSpan(link, spanBegin, lineBegin);

                    Node* body = readBody(reader, 0, '{', 0, '}');

                    addEscapeSpan(link, body);

                    spanBegin = reader.cursor;
                    atStartOfLine = false;
                }
                else if (atStartOfLine && peek(reader) == ':')
                {
                    // This is a statement escape, which will
                    // continue to the end of the line.
                    //
                    // The spliced text begins *after* the `:`
                    get(reader);
                    char const* spliceBegin = reader.cursor;

                    // The preceding text span will end at the
                    // start of this line.
                    addTextSpan(link, spanBegin, lineBegin);

                    // Any indentation on this line will be ignored.

                    // Read up to end of line.
                    for (;;)
                    {
                        int c = get(reader);
                        switch (c)
                        {
                        default:
                            continue;

                        case EOF:
                            break;

                        case '\r':
                        case '\n':
                            handleNewline(reader, c);
                            break;
                        }

                        break;
                    }

                    addEscapeSpan(link, spliceBegin, reader.cursor);

                    spanBegin = reader.cursor;
                    lineBegin = reader.cursor;
                }
                else if (atStartOfLine && isIdentifierChar(peek(reader)))
                {
                    // This is a statement splice, which will use a {}-enclosed
                    // body for the template to generate.

                    // Consume an optional identifier
                    while (isIdentifierChar(peek(reader)))
                        get(reader);

                    // Consume optional horizontal space
                    skipHorizontalSpace(reader);

                    // Consume an optional `()`-enclosed block (strip
                    // all but the outer-most `()`.

                    // optional space/newline/space before `{`
                    skipHorizontalSpace(reader);
                    skipOptionalNewline(reader);
                    skipHorizontalSpace(reader);

                    throw 99;
                }
                else
                {
                    // Doesn't seem to be a splice at all, just
                    // a literal `$` in the output.
                    atStartOfLine = false;
                }
            }
            break;
        }
    }
}

Node* readInput(char const* inputBegin, char const* inputEnd)
{
    Reader reader;
    reader.cursor = inputBegin;
    reader.end = inputEnd;

    return readBody(reader, kNodeReadFlag_AllowEscape, -2, 0, -2);
}

void emitRaw(FILE* stream, char const* begin, char const* end)
{
    // We will write the raw text to our output file.

    // TODO: need to output `#line` directives as well

    fputs("sb << \"", stream);
    for (char const* cc = begin; cc != end; ++cc)
    {
        int c = *cc;
        switch (c)
        {
        case '\\':
            fputs("\\\\", stream);
            break;

        case '\r':
            break;
        case '\t':
            fputs("\\t", stream);
            break;
        case '\"':
            fputs("\\\"", stream);
            break;
        case '\n':
            fputs("\\n\";\n", stream);
            fputs("sb << \"", stream);
            break;

        default:
            if ((c >= 32) && (c <= 126))
            {
                fputc(c, stream);
            }
            else
            {
                assert(false);
            }
        }
    }
    fprintf(stream, "\";\n");
}

void emitCode(FILE* stream, char const* begin, char const* end)
{
    for (auto cc = begin; cc != end; ++cc)
    {
        if (*cc == '\r')
            continue;

        fputc(*cc, stream);
    }
}

void emit(FILE* stream, char const* text)
{
    fprintf(stream, "%s", text);
}

void emit(FILE* stream, StringSpan const& span)
{
    fprintf(stream, "%.*s", int(span.end() - span.begin()), span.begin());
}

bool isASCIIPrintable(int c)
{
    return (c >= 0x20) && (c <= 0x7E);
}

void emitStringLiteralText(FILE* stream, StringSpan const& span)
{
    char const* cursor = span.begin();
    char const* end = span.end();

    while (cursor != end)
    {
        int c = *cursor++;
        switch (c)
        {
        case '\r':
        case '\n':
            fprintf(stream, "\\n");
            break;

        case '\t':
            fprintf(stream, "\\t");
            break;

        case ' ':
            fprintf(stream, " ");
            break;

        case '"':
            fprintf(stream, "\\\"");
            break;

        case '\\':
            fprintf(stream, "\\\\");
            break;

        default:
            if (isASCIIPrintable(c))
            {
                fprintf(stream, "%c", c);
            }
            else
            {
                fprintf(stream, "%03u", c);
            }
            break;
        }
    }
}

void emitSimpleText(FILE* stream, StringSpan const& span)
{
    UnownedStringSlice content(span), line;
    while (StringUtil::extractLine(content, line))
    {
        // Write the line
        fwrite(line.begin(), 1, line.getLength(), stream);

        // Specially handle the 'final line', excluding an empty line after \n.
        // We can detect, as if input ends with 'cr/lf' combination, content.begin == span.end(),
        // else if content.begin() == nullptr.
        if (content.begin() == nullptr || content.begin() == span.end())
        {
            break;
        }

        fprintf(stream, "\n");
    }
}

void emitCodeNodes(FILE* stream, Node* node)
{
    for (auto nn = node; nn; nn = nn->next)
    {
        switch (nn->flavor)
        {
        case Node::Flavor::text:
            emitSimpleText(stream, nn->span);
            emit(stream, "\n");
            break;

        default:
            throw "unexpected";
            break;
        }
    }
}

// Given line starts and a location, find the line number. Returns -1 if not found
static Index _findLineIndex(const List<UnownedStringSlice>& lineBreaks, const char* location)
{
    if (location == nullptr)
    {
        return -1;
    }

    // Use a binary chop to find the associated line
    Index lo = 0;
    Index hi = lineBreaks.getCount();

    while (lo + 1 < hi)
    {
        const auto mid = (hi + lo) >> 1;
        const auto midOffset = lineBreaks[mid].begin();
        if (midOffset <= location)
        {
            lo = mid;
        }
        else
        {
            hi = mid;
        }
    }

    return lo;
}

void emitTemplateNodes(SourceFile* sourceFile, FILE* stream, Node* node)
{
    // Work out
    List<UnownedStringSlice> lineBreaks;
    StringUtil::calcLines(sourceFile->text, lineBreaks);

    Node* prev = nullptr;
    for (auto nn = node; nn; prev = nn, nn = nn->next)
    {
        // If we transition from escape to text, insert line number directive
        bool enable = true;
        if (enable && prev && prev->flavor == Node::Flavor::escape &&
            nn->flavor == Node::Flavor::text)
        {
            // Find the line
            Index lineIndex = _findLineIndex(lineBreaks, nn->span.begin());
            // If found, output the directive
            if (lineIndex >= 0)
            {
                StringBuilder buf;
                buf << "SLANG_RAW(\"#line " << (lineIndex + 1) << " \\\"" << sourceFile->linePath
                    << "\\\"\")\n";

                emit(stream, buf.getUnownedSlice());
            }
        }

        switch (nn->flavor)
        {
        case Node::Flavor::text:
            emit(stream, "SLANG_RAW(\"");
            emitStringLiteralText(stream, nn->span);
            emit(stream, "\")\n");
            break;

        case Node::Flavor::splice:
            emit(stream, "SLANG_SPLICE(");
            emitCodeNodes(stream, nn->body);
            emit(stream, ")\n");
            break;

        case Node::Flavor::escape:
            emitCodeNodes(stream, nn->body);
            break;
        }
    }
}

void usage(char const* appName)
{
    fprintf(stderr, "usage: %s [FILE]... [--target-directory FILE]\n", appName);
}

SlangResult readAllText(char const* fileName, String& outString)
{
    FILE* f;
    fopen_s(&f, fileName, "rb");
    if (!f)
    {
        outString = "";
        return SLANG_FAIL;
    }
    else
    {
        fseek(f, 0, SEEK_END);
        auto size = ftell(f);

        StringRepresentation* stringRep =
            StringRepresentation::createWithCapacityAndLength(size, size);
        outString = String(stringRep);

        char* buffer = stringRep->getData();

        // Seems unnecessary
        // memset(buffer, 0, size);

        fseek(f, 0, SEEK_SET);
        size_t readCount = fread(buffer, sizeof(char), size, f);
        fclose(f);

        return (readCount == size) ? SLANG_OK : SLANG_FAIL;
    }
}

void writeAllText(char const* srcFileName, char const* fileName, const char* content)
{
    FILE* f = nullptr;
    fopen_s(&f, fileName, "wb");
    if (!f)
    {
        printf("%s(0): error G0001: cannot write file %s\n", srcFileName, fileName);
    }
    else
    {
        fwrite(content, 1, strlen(content), f);
        fclose(f);
    }
}

#define PARSE_HANDLER(NAME) Node* NAME(StringSpan const& text)

typedef PARSE_HANDLER((*ParseHandler));

PARSE_HANDLER(parseTemplateFile)
{
    // Read a template node!
    return readInput(text.begin(), text.end());
}

PARSE_HANDLER(parseCxxFile)
{
    // TODO: "scrape" the source file for metadata
    return nullptr;
}

PARSE_HANDLER(parseUnknownFile)
{
    // Don't process files we don't know how to handle.
    return nullptr;
}


Node* parseSourceFile(SourceFile* file)
{
    auto path = file->inputPath;
    auto text = file->text;

    static const struct
    {
        char const* extension;
        ParseHandler handler;
    } kHandlers[] = {
        {".meta.slang", &parseTemplateFile},
        {".meta.cpp", &parseTemplateFile},
        {".cpp", &parseCxxFile},
        {"", &parseUnknownFile},
    };

    for (auto hh : kHandlers)
    {
        if (path.endsWith(hh.extension))
        {
            return hh.handler(text);
        }
    }

    return nullptr;
}


SourceFile* parseSourceFile(const String& path)
{
    FILE* inputStream;
    fopen_s(&inputStream, path.getBuffer(), "rb");
    if (!inputStream)
    {
        fprintf(stderr, "unable to read input file: %s\n", path.getBuffer());
        return nullptr;
    }
    fseek(inputStream, 0, SEEK_END);
    size_t inputSize = ftell(inputStream);
    fseek(inputStream, 0, SEEK_SET);

    char* input = (char*)malloc(inputSize + 1);
    if (fread(input, inputSize, 1, inputStream) != 1)
    {
        fprintf(stderr, "unable to read input file: %s\n", path.getBuffer());
        return nullptr;
    }
    input[inputSize] = 0;

    char const* inputEnd = input + inputSize;
    StringSpan span = StringSpan(input, inputEnd);

    SourceFile* sourceFile = new SourceFile();

    sourceFile->inputPath = path;

    // We use the fileName as the line path, as the path as passed to the command could contain a
    // complicated depending on the project location.
    sourceFile->linePath = Path::getFileName(path);

    sourceFile->text = span;

    Node* node = parseSourceFile(sourceFile);

    sourceFile->node = node;

    fclose(inputStream);
    return sourceFile;
}

List<RefPtr<SourceFile>> gSourceFiles;

int main(int argc, const char* const* argv)
{
    // Parse command-line arguments.
    List<String> inputPaths;
    String outputDir;
    char const* appName = "slang-generate";

    {
        const char* const* argCursor = argv;
        const char* const* argEnd = argv + argc;
        // Copy the app name
        if (argCursor != argEnd)
        {
            appName = *argCursor++;
        }
        // Parse arguments
        for (; argCursor != argEnd; ++argCursor)
        {
            const auto arg = UnownedStringSlice(*argCursor);
            if (arg == "--target-directory")
            {
                argCursor++;
                if (argCursor == argEnd)
                {
                    usage(appName);
                    fprintf(stderr, "--target-directory expects an argument\n");
                    exit(1);
                }
                outputDir = Path::simplify(UnownedStringSlice(*argCursor));
            }
            else
            {
                // We simplify here because doing so also means paths separators are set to /
                // and that makes path emitting work correctly
                inputPaths.add(Path::simplify(arg));
            }
        }
    }

    if (inputPaths.getCount() == 0)
    {
        usage(appName);
        exit(1);
    }

    // Read each input file and process it according
    // to the type of treatment it requires.
    for (auto& inputPath : inputPaths)
    {
        SourceFile* sourceFile = parseSourceFile(inputPath);
        gSourceFiles.add(sourceFile);
    }

    for (auto sourceFile : gSourceFiles)
    {
        if (!sourceFile)
        {
            fprintf(stderr, "failed to parse source files\n");
            exit(1);
        }
    }

    // Once all inputs have been read, we can start
    // to produce output files by expanding templates.
    for (auto sourceFile : gSourceFiles)
    {
        auto inputPath = sourceFile->inputPath;
        auto node = sourceFile->node;

        // write output to a temporary file first
        StringBuilder outputPath;
        outputPath << inputPath << ".temp.h";

        FILE* outputStream;
        fopen_s(&outputStream, outputPath.getBuffer(), "w");
        if (!outputStream)
        {
            fprintf(stderr, "unable to open file for writing: %s.\n", outputPath.getBuffer());
            exit(1);
        }

        emitTemplateNodes(sourceFile, outputStream, node);

        fclose(outputStream);

        // update final output only when content has changed
        StringBuilder outputPathFinal;
        if (outputDir.getLength())
            outputPathFinal << outputDir << "/" << Slang::Path::getFileName(inputPath) << ".h";
        else
            outputPathFinal << inputPath << ".h";

        String allTextOld, allTextNew;
        readAllText(outputPathFinal.getBuffer(), allTextOld);
        readAllText(outputPath.getBuffer(), allTextNew);
        if (allTextOld != allTextNew)
        {
            writeAllText(
                inputPath.getBuffer(),
                outputPathFinal.getBuffer(),
                allTextNew.getBuffer());
        }
        remove(outputPath.getBuffer());
    }

    return 0;
}
