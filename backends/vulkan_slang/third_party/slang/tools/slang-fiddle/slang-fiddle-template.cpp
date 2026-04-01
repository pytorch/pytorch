// slang-fiddle-template.cpp
#include "slang-fiddle-template.h"

#include "slang-fiddle-script.h"

namespace fiddle
{
struct TextTemplateParserBase
{
protected:
    TextTemplateParserBase(
        SourceView* inputSourceView,
        DiagnosticSink* sink,
        UnownedStringSlice source)
        : _inputSourceView(inputSourceView)
        , _sink(sink)
        , _cursor(source.begin())
        , _end(source.end())
    {
    }

    SourceView* _inputSourceView = nullptr;
    DiagnosticSink* _sink = nullptr;
    char const* _cursor = nullptr;
    char const* _end = nullptr;

    bool atEnd() { return _cursor == _end; }

    UnownedStringSlice readLine()
    {
        auto lineBegin = _cursor;

        while (!atEnd())
        {
            char const* lineEnd = _cursor;
            switch (*_cursor)
            {
            default:
                _cursor++;
                continue;

            case '\r':
                _cursor++;
                if (*_cursor == '\n')
                    _cursor++;
                break;

            case '\n':
                _cursor++;
                break;
            }

            return UnownedStringSlice(lineBegin, lineEnd);
        }

        return UnownedStringSlice(lineBegin, _end);
    }
};

struct TextTemplateParser : TextTemplateParserBase
{
public:
    TextTemplateParser(
        SourceView* inputSourceView,
        DiagnosticSink* sink,
        UnownedStringSlice templateSource)
        : TextTemplateParserBase(inputSourceView, sink, templateSource)
    {
    }

    char const* findScriptStmtLine(UnownedStringSlice line)
    {
        char const* lineCursor = line.begin();
        char const* lineEnd = line.end();
        while (lineCursor != lineEnd)
        {
            switch (*lineCursor)
            {
            default:
                return nullptr;

            case ' ':
            case '\t':
                lineCursor++;
                continue;

            case '%':
                return lineCursor;
            }
        }
        return nullptr;
    }

    List<RefPtr<TextTemplateStmt>> stmts;

    void addRaw(char const* rawBegin, char const* rawEnd)
    {
        if (rawBegin == rawEnd)
            return;

        auto stmt = RefPtr(new TextTemplateRawStmt());
        stmt->text = UnownedStringSlice(rawBegin, rawEnd);
        stmts.add(stmt);
    }

    void addScriptStmtLine(char const* sourceBegin, char const* sourceEnd)
    {
        auto stmt = RefPtr(new TextTemplateScriptStmt());
        stmt->scriptSource = UnownedStringSlice(sourceBegin, sourceEnd);
        stmts.add(stmt);
    }

    void addScriptSpliceExpr(char const* sourceBegin, char const* sourceEnd)
    {
        auto stmt = RefPtr(new TextTemplateSpliceStmt());
        stmt->scriptExprSource = UnownedStringSlice(sourceBegin, sourceEnd);
        stmts.add(stmt);
    }

    bool isIdentifierStartChar(int c)
    {
        return (('a' <= c) && (c <= 'z')) || (('A' <= c) && (c <= 'Z')) || (c == '_');
    }

    bool isIdentifierChar(int c) { return isIdentifierStartChar(c) || (('0' <= c) && (c <= '9')); }

    RefPtr<TextTemplateStmt> parseTextTemplateBody()
    {
        bool isAtStartOfLine = true;
        bool isInScriptLine = false;
        int depthInSplice = 0;

        char const* currentLineBegin = _cursor;
        char const* currentSpanBegin = _cursor;
        while (!atEnd())
        {
            char const* currentSpanEnd = _cursor;

            bool wasAtStartOfLine = isAtStartOfLine;
            isAtStartOfLine = false;

            int c = *_cursor++;
            switch (c)
            {
            default:
                break;

            case '\r':
                if (*_cursor == '\n')
                {
                    _cursor++;
                }
            case '\n':
                isAtStartOfLine = true;
                currentLineBegin = _cursor;
                if (isInScriptLine)
                {
                    addScriptStmtLine(currentSpanBegin, currentSpanEnd);
                    isInScriptLine = false;
                    currentSpanBegin = currentSpanEnd;
                }
                break;

            case ' ':
            case '\t':
                isAtStartOfLine = wasAtStartOfLine;
                break;

            case '%':
                if (wasAtStartOfLine && !depthInSplice)
                {
                    addRaw(currentSpanBegin, currentLineBegin);
                    isInScriptLine = true;
                    currentSpanBegin = _cursor;
                }
                break;

            case '$':
                if (isInScriptLine)
                    continue;
                if (depthInSplice)
                    SLANG_ABORT_COMPILATION("fiddle encountered a '$' nested inside a splice");

                if (*_cursor == '(')
                {
                    _cursor++;
                    addRaw(currentSpanBegin, currentSpanEnd);
                    depthInSplice = 1;
                    currentSpanBegin = _cursor;
                    break;
                }
                else if (isIdentifierStartChar(*_cursor))
                {
                    addRaw(currentSpanBegin, currentSpanEnd);

                    auto spliceExprBegin = _cursor;
                    while (isIdentifierChar(*_cursor))
                        _cursor++;
                    auto spliceExprEnd = _cursor;
                    addScriptSpliceExpr(spliceExprBegin, spliceExprEnd);
                    currentSpanBegin = _cursor;
                    break;
                }
                break;

            case '(':
                if (!depthInSplice)
                    continue;
                depthInSplice++;
                break;

            case ')':
                if (!depthInSplice)
                    continue;
                depthInSplice--;
                if (depthInSplice == 0)
                {
                    addScriptSpliceExpr(currentSpanBegin, currentSpanEnd);
                    currentSpanBegin = _cursor;
                }
                break;
            }
        }
        addRaw(currentSpanBegin, _end);

        if (stmts.getCount() == 1)
            return stmts[0];
        else
        {
            auto stmt = RefPtr(new TextTemplateSeqStmt());
            stmt->stmts = stmts;
            return stmt;
        }
    }

private:
};


char const* templateStartMarker = "FIDDLE TEMPLATE";
char const* outputStartMarker = "FIDDLE OUTPUT";
char const* endMarker = "FIDDLE END";

struct TextTemplateFileParser : TextTemplateParserBase
{
public:
    TextTemplateFileParser(SourceView* inputSourceView, DiagnosticSink* sink)
        : TextTemplateParserBase(inputSourceView, sink, inputSourceView->getContent())
    {
    }

    RefPtr<TextTemplateFile> parseTextTemplateFile()
    {
        auto textTemplateFile = RefPtr(new TextTemplateFile());
        textTemplateFile->originalFileContent = _inputSourceView->getContent();
        while (!atEnd())
        {
            auto textTemplate = parseOptionalTextTemplate();
            if (textTemplate)
                textTemplateFile->textTemplates.add(textTemplate);
        }
        return textTemplateFile;
    }

private:
    Count _templateCounter = 0;

    bool matches(UnownedStringSlice const& line, char const* marker)
    {
        auto index = line.indexOf(UnownedTerminatedStringSlice(marker));
        return index >= 0;
    }

    bool findMatchingLine(char const* marker, UnownedStringSlice& outMatchingLine)
    {
        while (!atEnd())
        {
            auto line = readLine();
            if (!matches(line, marker))
            {
                // TODO: If the line doesn't match the expected marker,
                // but it *does* match one of the other markers, then
                // we should consider it a probable error.

                continue;
            }

            outMatchingLine = line;
            return true;
        }
        return false;
    }

    SourceLoc getLoc(char const* ptr)
    {
        auto offset = ptr - _inputSourceView->getContent().begin();
        auto startLoc = _inputSourceView->getRange().begin;
        auto loc = SourceLoc::fromRaw(startLoc.getRaw() + offset);
        return loc;
    }

    SourceLoc getLoc(UnownedStringSlice text) { return getLoc(text.begin()); }

    RefPtr<TextTemplateStmt> parseTextTemplateBody(UnownedStringSlice const& source)
    {
        TextTemplateParser parser(_inputSourceView, _sink, source);
        return parser.parseTextTemplateBody();
    }

    RefPtr<TextTemplate> parseOptionalTextTemplate()
    {
        // The idea is pretty simple; we scan through the source, one line at
        // a time, until we find a line that matches our template start pattern.
        //
        // If we *don't* find the start marker, then there must not be any
        // templates left.
        //
        UnownedStringSlice templateStartLine;
        if (!findMatchingLine(templateStartMarker, templateStartLine))
            return nullptr;

        char const* templateSourceBegin = _cursor;

        // If we *do* find a start line for a template, then we will expect
        // to find the other two kinds of lines, to round things out.

        UnownedStringSlice outputStartLine;
        if (!findMatchingLine(outputStartMarker, outputStartLine))
        {
            // TODO: need to diagnose a problem here...
            _sink->diagnose(
                getLoc(templateStartLine),
                fiddle::Diagnostics::expectedOutputStartMarker,
                outputStartMarker);
        }

        char const* templateSourceEnd = outputStartLine.begin();

        char const* existingOutputBegin = _cursor;

        UnownedStringSlice endLine;
        if (!findMatchingLine(endMarker, endLine))
        {
            // TODO: need to diagnose a problem here...
            _sink->diagnose(
                getLoc(templateStartLine),
                fiddle::Diagnostics::expectedEndMarker,
                endMarker);
        }
        char const* existingOutputEnd = endLine.begin();

        auto templateSource = UnownedStringSlice(templateSourceBegin, templateSourceEnd);
        auto templateBody = parseTextTemplateBody(templateSource);

        auto textTemplate = RefPtr(new TextTemplate());
        textTemplate->id = _templateCounter++;
        textTemplate->templateStartLine = templateStartLine;
        textTemplate->templateSource = templateSource;
        textTemplate->body = templateBody;
        textTemplate->outputStartLine = outputStartLine;
        textTemplate->existingOutputContent =
            UnownedStringSlice(existingOutputBegin, existingOutputEnd);
        textTemplate->endLine = endLine;
        return textTemplate;
    }
};

struct TextTemplateScriptCodeEmitter
{
public:
    TextTemplateScriptCodeEmitter(TextTemplateFile* templateFile)
        : _templateFile(templateFile)
    {
    }

    String emitScriptCodeForTextTemplateFile()
    {
        // We start by emitting the content of the template
        // file out as Lua code, so that we can evaluate
        // it all using the Lua VM.
        //
        // We go to some effort to make sure that the line
        // numbers in the generated Lua will match those
        // in the input.
        //

        char const* originalFileRawSpanStart = _templateFile->originalFileContent.begin();
        for (auto t : _templateFile->textTemplates)
        {
            flushOriginalFileRawSpan(originalFileRawSpanStart, t->templateSource.begin());

            evaluateTextTemplate(t);

            originalFileRawSpanStart = t->outputStartLine.begin();
        }
        flushOriginalFileRawSpan(
            originalFileRawSpanStart,
            _templateFile->originalFileContent.end());

        return _builder.produceString();
    }

private:
    TextTemplateFile* _templateFile = nullptr;
    StringBuilder _builder;

    void flushOriginalFileRawSpan(char const* begin, char const* end)
    {
        if (begin == end)
            return;

        // TODO: implement the important stuff...
        _builder.append("ORIGINAL [==[");
        _builder.append(UnownedStringSlice(begin, end));
        _builder.append("]==]");
    }

    void evaluateTextTemplate(TextTemplate* textTemplate)
    {
        // TODO: there really needs to be some framing around this...
        _builder.append("TEMPLATE(function() ");
        evaluateTextTemplateStmt(textTemplate->body);
        _builder.append(" end)");
    }

    void evaluateTextTemplateStmt(TextTemplateStmt* stmt)
    {
        if (auto seqStmt = as<TextTemplateSeqStmt>(stmt))
        {
            for (auto s : seqStmt->stmts)
                evaluateTextTemplateStmt(s);
        }
        else if (auto rawStmt = as<TextTemplateRawStmt>(stmt))
        {
            _builder.append("RAW [==[");
            _builder.append(rawStmt->text);
            _builder.append("]==]");
        }
        else if (auto scriptStmt = as<TextTemplateScriptStmt>(stmt))
        {
            _builder.append(scriptStmt->scriptSource);
            _builder.append(" ");
        }
        else if (auto spliceStmt = as<TextTemplateSpliceStmt>(stmt))
        {
            _builder.append("SPLICE(function()return(");
            _builder.append(spliceStmt->scriptExprSource);
            _builder.append(")end)");
        }
        else
        {
            SLANG_ABORT_COMPILATION(
                "fiddle encountered an unknown construct when converting a text template to Lua");
        }
    }
};


RefPtr<TextTemplateFile> parseTextTemplateFile(SourceView* inputSourceView, DiagnosticSink* sink)
{
    TextTemplateFileParser parser(inputSourceView, sink);
    return parser.parseTextTemplateFile();
}

void generateTextTemplateOutputs(
    String originalFileName,
    TextTemplateFile* file,
    StringBuilder& builder,
    DiagnosticSink* sink)
{
    TextTemplateScriptCodeEmitter emitter(file);
    String scriptCode = emitter.emitScriptCodeForTextTemplateFile();

    String output = evaluateScriptCode(originalFileName, scriptCode, sink);

    builder.append(output);
    builder.append("\n");
}

String generateModifiedInputFileForTextTemplates(
    String templateOutputFileName,
    TextTemplateFile* file,
    DiagnosticSink* sink)
{
    // The basic idea here is that we need to emit most of
    // the body of the file exactly as it originally
    // appeared, and then only modifify the few lines
    // that represent the text template output.
    //
    // TODO(tfoley): We could also use this as an opportunity
    // to insert the `FIDDLE(...)` markers that the scraping
    // tool needs, but that is more work than makes sense
    // right now.

    StringBuilder builder;


    char const* originalFileRawSpanStart = file->originalFileContent.begin();
    for (auto t : file->textTemplates)
    {
        builder.append(
            UnownedStringSlice(originalFileRawSpanStart, t->existingOutputContent.begin()));

        builder.append("#define FIDDLE_GENERATED_OUTPUT_ID ");
        builder.append(t->id);
        builder.append("\n");
        builder.append("#include \"");
        for (auto c : templateOutputFileName)
        {
            switch (c)
            {
            case '"':
            case '\\':
                builder.appendChar('\\');
                builder.appendChar(c);
                break;

            default:
                builder.appendChar(c);
                break;
            }
        }
        builder.append("\"\n");
        originalFileRawSpanStart = t->existingOutputContent.end();
    }
    builder.append(UnownedStringSlice(originalFileRawSpanStart, file->originalFileContent.end()));

    return builder.produceString();
}

} // namespace fiddle
