// slang-doc-markdown-writer.h
#ifndef SLANG_DOC_MARKDOWN_WRITER_H
#define SLANG_DOC_MARKDOWN_WRITER_H

#include "slang-ast-print.h"
#include "slang-compiler.h"
#include "slang-doc-ast.h"

namespace Slang
{

class ASTBuilder;

struct DocumentPage : public RefObject
{
    String title;
    String shortName;
    String path;
    String category;
    StringBuilder contentSB;
    Decl* decl = nullptr;
    bool skipWrite = false;
    DocumentPage* parentPage = nullptr;
    DocumentPage* findChildByShortName(const UnownedStringSlice& shortName);
    StringBuilder& get() { return contentSB; }
    List<RefPtr<DocumentPage>> children;

    // List of entries to document on this page.
    OrderedHashSet<ASTMarkup::Entry*> entries;
    ASTMarkup::Entry* getFirstEntry() { return *entries.begin(); }
    void writeToDisk();

    // Write summary on number of documented entries.
    void writeSummary(UnownedStringSlice fileName);
};

struct DocumentationConfig
{
    String preamble;
    String title;
    String libName;
    String rootDir;
    HashSet<String> visibleDeclNames;
    void parse(UnownedStringSlice configStr);
};

enum DocumentationSpanKind
{
    OrdinaryText,
    InlineCode,
};
struct ParsedDocumentationSpan
{
    String text;
    DocumentationSpanKind kind;
};
struct DocMarkdownWriter;

struct ParsedDescription
{
    String ownedText;
    List<ParsedDocumentationSpan> spans;
    void parse(UnownedStringSlice text);
    void write(DocMarkdownWriter* writer, Decl* decl, StringBuilder& out);
};

struct ParamDocumentation
{
    String name;
    String direction;
    ParsedDescription description;
};

enum class DocPageSection
{
    Description,
    Parameter,
    ReturnInfo,
    Remarks,
    Example,
    SeeAlso,
    InternalCallout,
    ExperimentalCallout,
    DeprecatedCallout,
};

struct DeclDocumentation
{
    Dictionary<String, ParamDocumentation> parameters;
    Dictionary<DocPageSection, ParsedDescription> sections;
    String categoryName;
    String categoryText;

    void parse(const UnownedStringSlice& text);
    void writeDescription(StringBuilder& out, DocMarkdownWriter* writer, Decl* decl);
    void writeGenericParameters(StringBuilder& out, DocMarkdownWriter* writer, Decl* decl);
    void writeSection(
        StringBuilder& sb,
        DocMarkdownWriter* writer,
        Decl* decl,
        DocPageSection section);
};

struct DocMarkdownWriter
{
    typedef ASTPrinter::Part Part;
    typedef ASTPrinter::PartPair PartPair;

    struct Signature
    {
        struct GenericParam
        {
            Part name;
            Part type;
        };

        Part returnType;
        List<PartPair> params;
        List<GenericParam> genericParams;
        Part name;
    };

    struct Requirement
    {
        typedef Requirement ThisType;

        bool operator==(const ThisType& rhs) const { return capabilitySet == rhs.capabilitySet; }
        SLANG_FORCE_INLINE bool operator!=(const ThisType& rhs) const
        {
            return !(capabilitySet == rhs.capabilitySet);
        }

        CapabilitySet capabilitySet;
    };

    /// Write out all documentation to the output buffer
    DocumentPage* writeAll(UnownedStringSlice configStr);
    String writeTOC();

    void ensureDeclPageCreated(ASTMarkup::Entry& entry);
    String translateToHTMLWithLinks(Decl* decl, String text);
    String translateToMarkdownWithLinks(String text, bool strictChildLookup = false);
    String escapeMarkdownText(String text);
    void generateSectionIndexPage(DocumentPage* page);
    void writePageRecursive(DocumentPage* page);
    void writePage(DocumentPage* page);

    /// This will write information about *all* of the overridden versions of a function/method
    void writeCallableOverridable(
        DocumentPage* page,
        const ASTMarkup::Entry& entry,
        CallableDecl* callable);

    void writeEnum(const ASTMarkup::Entry& entry, EnumDecl* enumDecl);
    void writeAggType(
        DocumentPage* page,
        const ASTMarkup::Entry& entry,
        AggTypeDeclBase* aggTypeDecl);
    void writeVar(const ASTMarkup::Entry& entry, VarDecl* varDecl);
    void writeProperty(const ASTMarkup::Entry& entry, PropertyDecl* propertyDecl);
    void writeTypeDef(const ASTMarkup::Entry& entry, TypeDefDecl* typeDefDecl);
    void writeAttribute(const ASTMarkup::Entry& entry, AttributeDecl* attributeDecl);

    void createPage(ASTMarkup::Entry& entry, Decl* decl);
    void registerCategory(DocumentPage* page, DeclDocumentation& doc);

    void writePreamble();

    void writeSignature(CallableDecl* callableDecl);
    void writeExtensionConditions(
        StringBuilder& sb,
        ExtensionDecl* decl,
        const char* prefix,
        bool isHtml);

    bool isVisible(const ASTMarkup::Entry& entry);
    bool isVisible(Decl* decl);
    bool isVisible(const Name* name);

    DocumentPage* findPageForToken(
        DocumentPage* currentPage,
        String token,
        String& outSectionName,
        Decl*& outDecl);
    String findLinkForToken(DocumentPage* currentPage, String token);

    /// Get the output string
    Dictionary<String, RefPtr<DocumentPage>>& getOutput() { return m_output; }

    DocumentPage* getPage(Decl* decl);
    StringBuilder* getBuilder(Decl* decl);

    /// Ctor.
    DocMarkdownWriter(ASTMarkup* markup, ASTBuilder* astBuilder, DiagnosticSink* sink)
        : m_markup(markup), m_astBuilder(astBuilder), m_sink(sink)
    {
    }

    struct StringListSet;

    /// Given a list of ASTPrinter::Parts, works out the different parts of the sig
    static void getSignature(const List<Part>& parts, Signature& outSig);

    struct NameAndText
    {
        Decl* decl = nullptr;
        String name;
        String text;
    };

    List<NameAndText> _getUniqueParams(const List<Decl*>& decls, DeclDocumentation* funcDoc);

    String _getName(Decl* decl);
    String _getFullName(Decl* decl);
    String _getDocFilePath(Decl* decl);
    String _getName(InheritanceDecl* decl);

    NameAndText _getNameAndText(ASTMarkup::Entry* entry, Decl* decl);
    NameAndText _getNameAndText(Decl* decl);

    template<typename T>
    List<NameAndText> _getAsNameAndTextList(const FilteredMemberList<T>& in)
    {
        List<NameAndText> out;
        for (auto decl : const_cast<FilteredMemberList<T>&>(in))
        {
            out.add(_getNameAndText(decl));
        }
        return out;
    }
    template<typename T>
    List<String> _getAsStringList(const List<T*>& in)
    {
        List<String> strings;
        for (auto decl : in)
        {
            strings.add(_getName(decl));
        }
        return strings;
    }

    List<NameAndText> _getAsNameAndTextList(const List<Decl*>& in);
    List<String> _getAsStringList(const List<Decl*>& in);

    void _appendAsBullets(const List<NameAndText>& values, bool insertLinkForName, char wrapChar);
    void _appendAsBullets(const List<String>& values, char wrapChar);

    void _appendCommaList(const List<String>& strings, char wrapChar);

    void _appendRequirements(const DocMarkdownWriter::Requirement& requirements);
    void _maybeAppendRequirements(
        const UnownedStringSlice& title,
        const List<DocMarkdownWriter::Requirement>& uniqueRequirements);

    void _appendExpr(StringBuilder& sb, Expr* expr);

    /// Appends prefix and the list of types derived from
    void _appendDerivedFrom(const UnownedStringSlice& prefix, AggTypeDeclBase* aggTypeDecl);
    void _appendEscaped(const UnownedStringSlice& text);

    void _appendAggTypeName(const ASTMarkup::Entry& entry, Decl* aggTypeDecl);

    ASTMarkup* m_markup;
    ASTBuilder* m_astBuilder;
    DiagnosticSink* m_sink;
    StringBuilder* m_builder = nullptr;
    DocumentPage* m_currentPage = nullptr;
    Dictionary<String, RefPtr<DocumentPage>> m_output;
    RefPtr<DocumentPage> m_rootPage;
    RefPtr<DocumentPage> m_typesPage;
    RefPtr<DocumentPage> m_attributesPage;
    RefPtr<DocumentPage> m_interfacesPage;
    RefPtr<DocumentPage> m_globalDeclsPage;

    DocumentationConfig m_config;
    Dictionary<String, String> m_categories;
};

} // namespace Slang

#endif
