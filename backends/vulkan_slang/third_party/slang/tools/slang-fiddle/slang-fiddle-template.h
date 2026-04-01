// slang-fiddle-template.h
#pragma once

#include "compiler-core/slang-source-loc.h"
#include "core/slang-list.h"
#include "core/slang-string.h"
#include "slang-fiddle-diagnostics.h"

namespace fiddle
{
using namespace Slang;

class TextTemplateStmt : public RefObject
{
public:
};

class TextTemplateScriptStmt : public TextTemplateStmt
{
public:
    UnownedStringSlice scriptSource;
};

class TextTemplateRawStmt : public TextTemplateStmt
{
public:
    // TODO(tfoley): Add a `SourceLoc` here, so
    // that we can emit approriate `#line` directives
    // to the output...

    UnownedStringSlice text;
};

class TextTemplateSpliceStmt : public TextTemplateStmt
{
public:
    UnownedStringSlice scriptExprSource;
};

class TextTemplateSeqStmt : public TextTemplateStmt
{
public:
    List<RefPtr<TextTemplateStmt>> stmts;
};

class TextTemplate : public RefObject
{
public:
    /// ID of this template within the enclosing file
    Index id;

    UnownedStringSlice templateStartLine;
    UnownedStringSlice outputStartLine;
    UnownedStringSlice endLine;

    UnownedStringSlice templateSource;
    UnownedStringSlice existingOutputContent;

    RefPtr<TextTemplateStmt> body;
};

class TextTemplateFile : public RefObject
{
public:
    UnownedStringSlice originalFileContent;
    List<RefPtr<TextTemplate>> textTemplates;
};

RefPtr<TextTemplateFile> parseTextTemplateFile(SourceView* inputSourceView, DiagnosticSink* sink);

void generateTextTemplateOutputs(
    String originalFileName,
    TextTemplateFile* file,
    StringBuilder& builder,
    DiagnosticSink* sink);

String generateModifiedInputFileForTextTemplates(
    String templateOutputFileName,
    TextTemplateFile* file,
    DiagnosticSink* sink);
} // namespace fiddle
