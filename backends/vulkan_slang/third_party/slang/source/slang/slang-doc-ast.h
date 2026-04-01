// slang-doc-ast.h
#ifndef SLANG_DOC_AST_H
#define SLANG_DOC_AST_H

#include "../compiler-core/slang-doc-extractor.h"
#include "../core/slang-basic.h"
#include "slang-ast-all.h"
#include "slang-syntax.h"

namespace Slang
{

/* Holds the documentation markup that is associated with each node (typically a decl) from a module
 */
class ASTMarkup : public RefObject
{
public:
    typedef MarkupEntry Entry;

    /// Adds an entry, returns the reference to pre-existing node if there is one
    Entry& addEntry(NodeBase* base);
    /// Gets an entry for a node. Returns nullptr if there is no markup.
    Entry* getEntry(NodeBase* base);

    /// Get list of all of the entries in source order
    List<Entry>& getEntries() { return m_entries; }

    /// Attaches the markup to the AST nodes.
    void attachToAST();

protected:
    /// Map from AST nodes to documentation entries
    Dictionary<NodeBase*, Index> m_entryMap;
    /// All of the documentation entries in source order
    List<Entry> m_entries;
};

// ---------------------------------------------------------------------------
SLANG_INLINE ASTMarkup::Entry& ASTMarkup::addEntry(NodeBase* base)
{
    const Index count = m_entries.getCount();
    const Index index = m_entryMap.getOrAddValue(base, count);

    if (index == count)
    {
        Entry entry;
        entry.m_node = base;
        m_entries.add(entry);
    }
    return m_entries[index];
}

// ---------------------------------------------------------------------------
SLANG_INLINE ASTMarkup::Entry* ASTMarkup::getEntry(NodeBase* base)
{
    Index* indexPtr = m_entryMap.tryGetValue(base);
    return (indexPtr) ? &m_entries[*indexPtr] : nullptr;
}

SLANG_INLINE void ASTMarkup::attachToAST()
{
    for (auto& entry : m_entries)
    {
        if (auto decl = as<Decl>(entry.m_node))
        {
            decl->markup = new MarkupEntry(entry);
        }
    }
}

/* Extracts documentation markup from source.
The comments are extracted and associated in declarations. The association
is held in DocMarkup type. The comment style follows the doxygen style */
struct ASTMarkupUtil
{
    typedef DocMarkupExtractor Extractor;

    /// Given a module finds all the decls, and places in outDecls
    static void findDecls(ModuleDecl* moduleDecl, List<Decl*>& outDecls);

    /// Given a decl determines the search style that is appropriate. Returns None if can't
    /// determine a suitable style
    static Extractor::SearchStyle getSearchStyle(Decl* decl);

    /// Extracts documentation from the nodes held in the module using the source manager. Found
    /// documentation is placed in outMarkup
    static SlangResult extract(
        ModuleDecl* moduleDecl,
        SourceManager* sourceManager,
        DiagnosticSink* sink,
        ASTMarkup* outMarkup,
        bool searchOrindaryComments = false);
};

bool shouldDocumentDecl(Decl* decl);

} // namespace Slang

#endif
