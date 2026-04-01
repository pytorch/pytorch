#pragma once

#include "../compiler-core/slang-language-server-protocol.h"
#include "../core/slang-basic.h"
#include "../core/slang-com-object.h"
#include "slang-com-helper.h"
#include "slang-com-ptr.h"
#include "slang-compiler.h"
#include "slang-doc-ast.h"
#include "slang.h"

namespace Slang
{
class Workspace;

class DocumentVersion : public RefObject
{
private:
    URI uri;
    String path;
    String text;
    List<UnownedStringSlice> lines;
    List<List<Index>> mapUTF16CharIndexToCodePointIndex;
    List<List<Index>> mapCodePointIndexToUTF8ByteOffset;

public:
    void setPath(String filePath)
    {
        path = filePath;
        uri = URI::fromLocalFilePath(path.getUnownedSlice());
    }
    URI getURI() { return uri; }
    String getPath() { return path; }
    const String& getText() { return text; }
    void setText(const String& newText);

    void ensureUTFBoundsAvailable();
    ArrayView<Index> getUTF16Boundaries(Index line);
    ArrayView<Index> getUTF8Boundaries(Index line);

    void oneBasedUTF8LocToZeroBasedUTF16Loc(
        Index inLine,
        Index inCol,
        int64_t& outLine,
        int64_t& outCol);
    void oneBasedUTF8LocToZeroBasedUTF16Loc(
        Index inLine,
        Index inCol,
        int32_t& outLine,
        int32_t& outCol);
    void zeroBasedUTF16LocToOneBasedUTF8Loc(
        Index inLine,
        Index inCol,
        Index& outLine,
        Index& outCol);

    // Get starting offset of line.
    Index getLineStart(UnownedStringSlice line) { return line.begin() - text.begin(); }

    UnownedStringSlice peekIdentifier(Index line, Index col, Index& offset)
    {
        offset = getOffset(line, col);
        return peekIdentifier(offset);
    }

    UnownedStringSlice peekIdentifier(Index& offset);

    // Get offset from 1-based, utf-8 encoding location.
    Index getOffset(Index lineIndex, Index colIndex)
    {
        if (lineIndex < 0)
            return -1;
        if (lineIndex - 1 >= lines.getCount())
            return -1;
        if (lines.getCount() == 0)
            return -1;

        Index lineStart = lineIndex >= 1 ? getLineStart(lines[lineIndex - 1]) : 0;
        auto boundaries = getUTF8Boundaries(lineIndex);
        Index byteOffset = 0;
        if (colIndex > 0 && colIndex <= boundaries.getCount())
            byteOffset = boundaries[colIndex - 1];
        return lineStart + byteOffset;
    }

    // Get 1-based, utf-8 encoding location from offset.
    void offsetToLineCol(Index offset, Index& line, Index& col)
    {
        auto firstGreater = std::upper_bound(
            lines.begin(),
            lines.end(),
            offset,
            [this](Index first, UnownedStringSlice second)
            { return first < getLineStart(second); });
        line = Index(firstGreater - lines.begin());
        if (firstGreater == lines.begin())
        {
            col = offset + 1;
        }
        else
        {
            col = Index(offset - getLineStart(lines[line - 1])) + 1;
        }
        if (line > 0 && line <= lines.getCount())
            col = UTF8Util::calcCodePointCount(lines[line - 1].head(col));
    }

    // Get line from 1-based index.
    UnownedStringSlice getLine(Index lineIndex)
    {
        if (lineIndex < 0)
            return UnownedStringSlice();
        if (lineIndex - 1 >= lines.getCount())
            return UnownedStringSlice();
        if (lines.getCount() == 0)
            return UnownedStringSlice();

        return lineIndex > 0 ? lines[lineIndex - 1] : UnownedStringSlice();
    }

    // Get length of an identifier token starting at the specified position.
    int getTokenLength(Index line, Index col);
    int getTokenLength(Index offset);
};

struct DocumentDiagnostics
{
    OrderedHashSet<LanguageServerProtocol::Diagnostic> messages;
    String originalOutput;
};

enum class WorkspaceFlavor
{
    Standard,
    VFX,
};

class WorkspaceVersion : public RefObject
{
private:
    Dictionary<String, Module*> modules;
    Dictionary<ModuleDecl*, RefPtr<ASTMarkup>> markupASTs;
    Dictionary<Name*, MacroDefinitionContentAssistInfo*> macroDefinitions;
    void parseDiagnostics(String compilerOutput);

public:
    Workspace* workspace;
    WorkspaceFlavor flavor = WorkspaceFlavor::Standard;
    RefPtr<Linkage> linkage;
    Dictionary<String, DocumentDiagnostics> diagnostics;
    ASTMarkup* getOrCreateMarkupAST(ModuleDecl* module);
    Module* getOrLoadModule(String path);
    void ensureWorkspaceFlavor(UnownedStringSlice path);
    MacroDefinitionContentAssistInfo* tryGetMacroDefinition(UnownedStringSlice name);
};

struct OwnedPreprocessorMacroDefinition
{
    String name;
    String value;
};
class Workspace : public ISlangFileSystem, public ComObject
{
private:
    RefPtr<WorkspaceVersion> currentVersion;
    RefPtr<WorkspaceVersion> currentCompletionVersion;
    RefPtr<WorkspaceVersion> createWorkspaceVersion();

public:
    List<String> rootDirectories;
    List<String> additionalSearchPaths;
    OrderedHashSet<String> workspaceSearchPaths;
    List<OwnedPreprocessorMacroDefinition> predefinedMacros;
    bool searchInWorkspace = true;

    slang::IGlobalSession* slangGlobalSession;
    Dictionary<String, RefPtr<DocumentVersion>> openedDocuments;
    DocumentVersion* openDoc(String path, String text);
    void changeDoc(const String& path, LanguageServerProtocol::Range range, const String& text);
    void changeDoc(DocumentVersion* doc, const String& newText);

    void closeDoc(const String& path);

    // Update predefined macro settings. Returns true if the new settings are different from
    // existing ones.
    bool updatePredefinedMacros(List<String> predefinedMacros);
    bool updateSearchPaths(List<String> searchPaths);
    bool updateSearchInWorkspace(bool value);

    void init(List<URI> rootDirURI, slang::IGlobalSession* globalSession);
    void invalidate();
    WorkspaceVersion* getCurrentVersion();
    WorkspaceVersion* getCurrentCompletionVersion() { return currentCompletionVersion.Ptr(); }
    WorkspaceVersion* createVersionForCompletion();

public:
    // Inherited via ISlangFileSystem
    SLANG_COM_OBJECT_IUNKNOWN_ALL
    void* getInterface(const Guid& uuid);
    void* getObject(const Guid& uuid);

    // ISlangCastable
    virtual SLANG_NO_THROW void* SLANG_MCALL castAs(const Guid& guid) override;

    // ISlangFileSystem
    virtual SLANG_NO_THROW SlangResult SLANG_MCALL
    loadFile(const char* path, ISlangBlob** outBlob) override;
};
} // namespace Slang
