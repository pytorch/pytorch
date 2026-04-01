#include "node-tree.h"

#include "compiler-core/slang-name-convention-util.h"
#include "core/slang-io.h"
#include "identifier-lookup.h"
#include "options.h"

namespace CppParse
{
using namespace Slang;

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! NodeTree !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

NodeTree::NodeTree(
    StringSlicePool* typePool,
    NamePool* namePool,
    IdentifierLookup* identifierLookup)
    : m_typePool(typePool)
    , m_namePool(namePool)
    , m_identifierLookup(identifierLookup)
    , m_typeSetPool(StringSlicePool::Style::Empty)
{
    m_rootNode = new ScopeNode(Node::Kind::Namespace);
    m_rootNode->m_reflectionType = ReflectionType::Reflected;
}

TypeSet* NodeTree::getTypeSet(const UnownedStringSlice& slice)
{
    Index index = m_typeSetPool.findIndex(slice);
    if (index < 0)
    {
        return nullptr;
    }
    return m_typeSets[index];
}

TypeSet* NodeTree::getOrAddTypeSet(const UnownedStringSlice& slice)
{
    const Index index = Index(m_typeSetPool.add(slice));
    if (index >= m_typeSets.getCount())
    {
        SLANG_ASSERT(m_typeSets.getCount() == index);
        TypeSet* typeSet = new TypeSet;

        m_typeSets.add(typeSet);
        typeSet->m_macroName = m_typeSetPool.getSlice(StringSlicePool::Handle(index));
        return typeSet;
    }
    else
    {
        return m_typeSets[index];
    }
}

SourceOrigin* NodeTree::addSourceOrigin(SourceFile* sourceFile, const Options& options)
{
    // Calculate from the path, a 'macro origin' name.
    const String macroOrigin = calcMacroOrigin(sourceFile->getPathInfo().foundPath, options);

    SourceOrigin* origin = new SourceOrigin(sourceFile, macroOrigin);
    m_sourceOrigins.add(origin);
    return origin;
}

/* static */ String NodeTree::calcMacroOrigin(const String& filePath, const Options& options)
{
    // Get the filename without extension
    String fileName = Path::getFileNameWithoutExt(filePath);

    // We can work on just the slice
    UnownedStringSlice slice = fileName.getUnownedSlice();

    // Filename prefix
    if (options.m_stripFilePrefix.getLength() &&
        slice.startsWith(options.m_stripFilePrefix.getUnownedSlice()))
    {
        const Index len = options.m_stripFilePrefix.getLength();
        slice = UnownedStringSlice(slice.begin() + len, slice.end());
    }

    // Trim -
    slice = slice.trim('-');

    StringBuilder out;
    NameConventionUtil::convert(slice, NameConvention::UpperSnake, out);
    return out;
}

SlangResult NodeTree::_calcDerivedTypesRec(ScopeNode* inScopeNode, DiagnosticSink* sink)
{
    if (inScopeNode->isClassLike())
    {
        ClassLikeNode* classLikeNode = static_cast<ClassLikeNode*>(inScopeNode);

        if (classLikeNode->m_super.hasContent())
        {
            ScopeNode* parentScope = classLikeNode->m_parentScope;
            if (parentScope == nullptr)
            {
                sink->diagnoseRaw(
                    Severity::Error,
                    UnownedStringSlice::fromLiteral("Can't lookup in scope if there is none!"));
                return SLANG_FAIL;
            }

            Node* superNode = Node::lookup(parentScope, classLikeNode->m_super.getContent());

            if (!superNode)
            {
                if (classLikeNode->isReflected())
                {
                    sink->diagnose(
                        classLikeNode->m_name,
                        CPPDiagnostics::superTypeNotFound,
                        classLikeNode->getAbsoluteName());
                    return SLANG_FAIL;
                }
            }
            else
            {
                ClassLikeNode* superType = as<ClassLikeNode>(superNode);

                if (!superType)
                {
                    sink->diagnose(
                        classLikeNode->m_name,
                        CPPDiagnostics::superTypeNotAType,
                        classLikeNode->getAbsoluteName());
                    return SLANG_FAIL;
                }

                if (superType->m_typeSet != classLikeNode->m_typeSet)
                {
                    sink->diagnose(
                        classLikeNode->m_name,
                        CPPDiagnostics::typeInDifferentTypeSet,
                        classLikeNode->m_name.getContent(),
                        classLikeNode->m_typeSet->m_macroName,
                        superType->m_typeSet->m_macroName);
                    return SLANG_FAIL;
                }

                // The base class must be defined in same scope (as we didn't allow different scopes
                // for base classes)
                superType->addDerived(classLikeNode);
            }
        }
        else
        {
            // Add to it's own typeset
            if (classLikeNode->isReflected() && classLikeNode->m_typeSet)
            {
                classLikeNode->m_typeSet->m_baseTypes.add(classLikeNode);
            }
        }
    }

    for (Node* child : inScopeNode->m_children)
    {
        ScopeNode* childScope = as<ScopeNode>(child);
        if (childScope)
        {
            SLANG_RETURN_ON_FAIL(_calcDerivedTypesRec(childScope, sink));
        }
    }

    return SLANG_OK;
}

SlangResult NodeTree::calcDerivedTypes(DiagnosticSink* sink)
{
    return _calcDerivedTypesRec(m_rootNode, sink);
}


} // namespace CppParse
