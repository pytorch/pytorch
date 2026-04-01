#pragma once

#include "compiler-core/slang-lexer.h"
#include "diagnostics.h"
#include "identifier-lookup.h"
#include "node.h"

namespace CppParse
{
using namespace Slang;

class TypeSet : public RefObject
{
public:
    /// This is the looked up name.
    UnownedStringSlice
        m_macroName; ///< The name extracted from the macro SLANG_ABSTRACT_AST_CLASS -> AST

    String m_typeName; ///< The enum type name associated with this type for AST it is ASTNode
    String m_fileMark; ///< This 'mark' becomes of the output filename

    List<ClassLikeNode*> m_baseTypes; ///< The base types for this type set
};

class SourceOrigin : public RefObject
{
public:
    void addNode(Node* node)
    {
        if (auto classLike = as<ClassLikeNode>(node))
        {
            SLANG_ASSERT(classLike->m_origin == nullptr);
            classLike->m_origin = this;
        }

        m_nodes.add(node);
    }

    SourceOrigin(SourceFile* sourceFile, const String& macroOrigin)
        : m_sourceFile(sourceFile), m_macroOrigin(macroOrigin)
    {
    }

    String m_macroOrigin; ///< The macro text is inserted into the macro to identify the origin. It
                          ///< is based on the filename
    SourceFile* m_sourceFile; ///< The source file - also holds the path information

    /// All of the nodes defined in this file in the order they were defined
    /// Note that the same namespace may be listed multiple times.
    List<RefPtr<Node>> m_nodes;
};

struct Options;
class IdentifierLookup;

/* NodeTree holds nodes that have been parsed into a tree rooted on the 'rootNode'.
Also contains other state associated with or useful to a node tree */
class NodeTree
{
public:
    friend class Parser;
    /// Get all of the parsed source origins
    const List<RefPtr<SourceOrigin>>& getSourceOrigins() const { return m_sourceOrigins; }

    TypeSet* getTypeSet(const UnownedStringSlice& slice);
    TypeSet* getOrAddTypeSet(const UnownedStringSlice& slice);

    SourceOrigin* addSourceOrigin(SourceFile* sourceFile, const Options& options);

    /// Get all of the type sets
    const List<RefPtr<TypeSet>>& getTypeSets() const { return m_typeSets; }

    /// Get the root node
    Node* getRootNode() const { return m_rootNode; }

    /// When parsing we don't lookup all up super types/add derived types. This is because
    /// we allow files to be processed in any order, so we have to do the type lookup as a separate
    /// operation
    SlangResult calcDerivedTypes(DiagnosticSink* sink);

    NodeTree(StringSlicePool* typePool, NamePool* namePool, IdentifierLookup* identifierLookup);

    static String calcMacroOrigin(const String& filePath, const Options& options);

protected:
    SlangResult _calcDerivedTypesRec(ScopeNode* node, DiagnosticSink* sink);

    StringSlicePool m_typeSetPool;    ///< Pool for type set names
    List<RefPtr<TypeSet>> m_typeSets; ///< The type sets

    IdentifierLookup* m_identifierLookup;
    StringSlicePool* m_typePool; ///< Pool for just types

    NamePool* m_namePool;

    RefPtr<ScopeNode> m_rootNode; ///< The root scope

    List<RefPtr<SourceOrigin>> m_sourceOrigins;
};

} // namespace CppParse
