#include "node.h"

#include "core/slang-string-escape-util.h"
#include "core/slang-string-util.h"
#include "file-util.h"

namespace CppParse
{

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Node Impl
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

SLANG_FORCE_INLINE static void _indent(Index indentCount, StringBuilder& out)
{
    FileUtil::indent(indentCount, out);
}

void Node::dumpMarkup(int indentCount, StringBuilder& out)
{
    if (m_markup.getLength() <= 0)
    {
        return;
    }

    List<UnownedStringSlice> lines;
    StringUtil::calcLines(m_markup.getUnownedSlice(), lines);

    // Remove empty lines from the end
    while (lines.getCount())
    {
        auto lastLine = lines.getLast();
        if (lastLine.trim().getLength() == 0)
        {
            lines.removeLast();
            continue;
        }
        break;
    }

    if (lines.getCount() == 0)
    {
        return;
    }

    for (auto line : lines)
    {
        _indent(indentCount, out);
        out << "// " << line << "\n";
    }
}

ScopeNode* Node::getRootScope()
{
    if (m_parentScope)
    {
        ScopeNode* scope = m_parentScope;
        while (scope->m_parentScope)
        {
            scope = scope->m_parentScope;
        }
        return scope;
    }
    else
    {
        return as<ScopeNode>(this);
    }
}

void Node::calcScopeDepthFirst(List<Node*>& outNodes)
{
    outNodes.add(this);
}

void Node::calcAbsoluteName(StringBuilder& outName) const
{
    List<Node*> path;
    calcScopePath(const_cast<Node*>(this), path);

    // 1 so we skip the global scope
    for (Index i = 1; i < path.getCount(); ++i)
    {
        Node* node = path[i];

        if (i > 1)
        {
            outName << "::";
        }

        if (node->m_kind == Kind::AnonymousNamespace)
        {
            outName << "{Anonymous}";
        }
        else
        {
            outName << node->m_name.getContent();
        }
    }
}

/* static */ void Node::calcScopePath(Node* node, List<Node*>& outPath)
{
    outPath.clear();

    while (node)
    {
        outPath.add(node);
        node = node->m_parentScope;
    }

    // reverse the order, so we go from root to the node
    outPath.reverse();
}

/* static */ void Node::filterImpl(Filter inFilter, List<Node*>& ioNodes)
{
    // Filter out all the unreflected nodes
    Index count = ioNodes.getCount();
    for (Index j = 0; j < count;)
    {
        Node* node = ioNodes[j];

        if (!inFilter(node))
        {
            ioNodes.removeAt(j);
            count--;
        }
        else
        {
            j++;
        }
    }
}

/* static */ Node* Node::lookupNameInScope(ScopeNode* scope, const UnownedStringSlice& name)
{
    // TODO(JS): Doesn't handle 'using namespace'.

    // Must be unqualified name
    SLANG_ASSERT(name.indexOf(UnownedStringSlice::fromLiteral("::")) < 0);

    Node* childNode = scope->findChild(name);
    if (childNode)
    {
        return childNode;
    }

    // If we have an anonymous namespace in this scope, try looking up in there..
    if (scope->m_anonymousNamespace)
    {
        Node* childNode = scope->m_anonymousNamespace->findChild(name);
        if (childNode)
        {
            return childNode;
        }
    }

    // I could have an enum (that's not an enum class)
    for (Node* node : scope->m_children)
    {
        EnumNode* enumNode = as<EnumNode>(node);
        if (enumNode && enumNode->m_kind == Node::Kind::Enum)
        {
            Node** nodePtr = enumNode->m_childMap.tryGetValue(name);
            if (nodePtr)
            {
                return *nodePtr;
            }
        }
    }

    return nullptr;
}

/* static */ Node* Node::lookupFromScope(
    ScopeNode* scope,
    const UnownedStringSlice* parts,
    Index partsCount)
{
    SLANG_ASSERT(partsCount > 0);
    if (partsCount == 1)
    {
        return lookupNameInScope(scope, parts[0]);
    }

    for (Index i = 0; i < partsCount; ++i)
    {
        const UnownedStringSlice& part = parts[i];

        Node* node = lookupNameInScope(scope, part);
        if (node == nullptr)
        {
            return node;
        }
        // If at end, then we are done
        if (i == partsCount - 1)
        {
            return node;
        }

        // If there are more elements, then node must be some kind of scope,
        // if we are going to find it
        scope = as<ScopeNode>(node);
        if (scope == nullptr)
        {
            break;
        }
    }

    return nullptr;
}

/* static */ void Node::splitPath(
    const UnownedStringSlice& inPath,
    List<UnownedStringSlice>& outParts)
{
    if (inPath.indexOf(UnownedStringSlice::fromLiteral("::")) >= 0)
    {
        StringUtil::split(inPath, UnownedStringSlice::fromLiteral("::"), outParts);
        // Remove any whitespace
        for (auto& part : outParts)
        {
            part = part.trim();
        }
    }
    else
    {
        outParts.clear();
        outParts.add(inPath.trim());
    }
}

/* static */ Node* Node::lookupFromScope(ScopeNode* scope, const UnownedStringSlice& inPath)
{
    if (inPath.indexOf(UnownedStringSlice::fromLiteral("::")) >= 0)
    {
        List<UnownedStringSlice> parts;
        splitPath(inPath, parts);

        return lookupFromScope(scope, parts.getBuffer(), parts.getCount());
    }
    else
    {
        return lookupNameInScope(scope, inPath);
    }
}

/* static */ Node* Node::lookup(ScopeNode* scope, const UnownedStringSlice& inPath)
{
    if (inPath.indexOf(UnownedStringSlice::fromLiteral("::")) >= 0)
    {
        List<UnownedStringSlice> parts;
        splitPath(inPath, parts);

        if (parts[0].getLength() == 0)
        {
            // It's a lookup from global scope
            ScopeNode* rootScope = scope->getRootScope();
            return lookupFromScope(rootScope, parts.getBuffer() + 1, parts.getCount() + 1);
        }

        // Okay lets try a lookup from each scope up to the global scope
        while (scope)
        {
            Node* node = lookupFromScope(scope, parts.getBuffer(), parts.getCount());
            if (node)
            {
                return node;
            }

            scope = scope->m_parentScope;
        }
    }
    else
    {
        while (scope)
        {
            // Lookup in this scope
            Node* node = lookupNameInScope(scope, inPath);
            if (node)
            {
                return node;
            }

            // Try parent scope
            scope = scope->m_parentScope;
        }
    }

    return nullptr;
}

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ScopeNode !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ScopeNode* ScopeNode::getAnonymousNamespace()
{
    if (!m_anonymousNamespace)
    {
        m_anonymousNamespace = new ScopeNode(Kind::AnonymousNamespace);
        m_anonymousNamespace->m_parentScope = this;
        m_children.add(m_anonymousNamespace);
    }

    return m_anonymousNamespace;
}

void ScopeNode::addChildIgnoringName(Node* child)
{
    SLANG_ASSERT(child->m_parentScope == nullptr);
    // Can't add anonymous namespace this way - should be added via getAnonymousNamespace
    SLANG_ASSERT(child->m_kind != Kind::AnonymousNamespace);

    child->m_parentScope = this;
    m_children.add(child);
}

void ScopeNode::addChild(Node* child)
{
    addChildIgnoringName(child);

    if (child->m_name.hasContent())
    {
        m_childMap.add(child->m_name.getContent(), child);
    }
}

Node* ScopeNode::findChild(const UnownedStringSlice& name) const
{
    Node* const* nodePtr = m_childMap.tryGetValue(name);
    if (nodePtr)
    {
        return *nodePtr;
    }
    return nullptr;
}

void ScopeNode::calcScopeDepthFirst(List<Node*>& outNodes)
{
    outNodes.add(this);
    for (Node* child : m_children)
    {
        child->calcScopeDepthFirst(outNodes);
    }
}

void ScopeNode::dump(int indentCount, StringBuilder& out)
{
    dumpMarkup(indentCount, out);

    _indent(indentCount, out);

    switch (m_kind)
    {
    case Kind::AnonymousNamespace:
        {
            out << "namespace {\n";
        }
    case Kind::Namespace:
        {
            if (m_name.hasContent())
            {
                out << "namespace " << m_name.getContent() << " {\n";
            }
            else
            {
                out << "{\n";
            }
            break;
        }
    }

    for (Node* child : m_children)
    {
        child->dump(indentCount + 1, out);
    }

    _indent(indentCount, out);
    out << "}\n";
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EnumCaseNode !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/* Returns true if needs space between the tokens.
It determines this based on the locs, and if they contain something between them.
*/
static bool _needsSpace(const Token& prevTok, const Token& tok)
{
    auto prevLoc = prevTok.getLoc();
    auto loc = tok.getLoc();

    auto prevContent = prevTok.getContent();

    if (prevLoc + prevContent.getLength() == loc)
    {
        return false;
    }

    return true;
}


static void _dumpTokens(const Token* toks, Index count, StringBuilder& out)
{
    if (count > 0)
    {
        out << toks[0].getContent();

        for (Index i = 1; i < count; ++i)
        {
            const auto& prevToken = toks[i - 1];
            const auto& token = toks[i];

            if (_needsSpace(prevToken, token))
            {
                out << " ";
            }

            out << token.getContent();
        }
    }
}

static void _dumpTokens(const List<Token>& toks, StringBuilder& out)
{
    _dumpTokens(toks.getBuffer(), toks.getCount(), out);
}


void EnumCaseNode::dump(int indent, StringBuilder& out)
{
    if (isReflected())
    {
        dumpMarkup(indent, out);

        _indent(indent, out);
        out << m_name.getContent();

        if (m_valueTokens.getCount())
        {
            out << " = ";
            _dumpTokens(m_valueTokens, out);
        }

        out << ",\n";
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EnumNode !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void TypeDefNode::dump(int indent, StringBuilder& out)
{
    if (isReflected())
    {
        dumpMarkup(indent, out);

        _indent(indent, out);

        out << "typedef ";
        _dumpTokens(m_targetTypeTokens, out);
        out << " " << m_name.getContent() << ";\n";
    }
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! EnumNode !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void EnumNode::dump(int indent, StringBuilder& out)
{
    if (!isReflected())
    {
        return;
    }

    dumpMarkup(indent, out);

    _indent(indent, out);

    out << "enum ";

    if (m_kind == Kind::EnumClass)
    {
        out << "class ";
    }

    if (m_name.type != TokenType::Invalid)
    {
        out << m_name.getContent();
    }

    if (m_backingTokens.getCount() > 0)
    {
        out << " : ";
        _dumpTokens(m_backingTokens, out);
    }

    out << "\n";
    _indent(indent, out);
    out << "{\n";

    for (Node* child : m_children)
    {
        child->dump(indent + 1, out);
    }

    _indent(indent, out);
    out << "}\n";
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!! CallableNode !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void CallableNode::dump(int indent, StringBuilder& out)
{
    if (!isReflected())
    {
        return;
    }

    dumpMarkup(indent, out);

    _indent(indent, out);

    if (m_isStatic)
    {
        out << "static ";
    }
    if (m_isVirtual)
    {
        out << "virtual ";
    }

    out << m_returnType << " ";
    out << m_name.getContent() << "(";

    const Index count = m_params.getCount();
    for (Index i = 0; i < count; ++i)
    {
        if (i > 0)
        {
            out << ", ";
        }

        const auto& param = m_params[i];
        out << param.m_type;
        if (param.m_name.type == TokenType::Identifier)
        {
            out << " " << param.m_name.getContent();
        }
    }

    out << ")";

    if (m_isPure)
    {
        out << " = 0";
    }

    out << "\n";
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! FieldNode !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

void FieldNode::dump(int indent, StringBuilder& out)
{
    if (!isReflected())
    {
        return;
    }

    dumpMarkup(indent, out);

    _indent(indent, out);

    if (m_isStatic)
    {
        out << "static ";
    }

    out << m_fieldType << " " << m_name.getContent() << "\n";
}

/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ClassLikeNode !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

/// Add a node that is derived from this
void ClassLikeNode::addDerived(ClassLikeNode* derived)
{
    SLANG_ASSERT(derived->m_superNode == nullptr);
    derived->m_superNode = this;
    m_derivedTypes.add(derived);
}

void ClassLikeNode::calcDerivedDepthFirst(List<ClassLikeNode*>& outNodes)
{
    outNodes.add(this);
    for (ClassLikeNode* derivedType : m_derivedTypes)
    {
        derivedType->calcDerivedDepthFirst(outNodes);
    }
}

void ClassLikeNode::dumpDerived(int indentCount, StringBuilder& out)
{
    if (isClassLike() && isReflected() && m_name.hasContent())
    {
        _indent(indentCount, out);
        out << m_name.getContent() << "\n";
    }

    for (ClassLikeNode* derivedType : m_derivedTypes)
    {
        derivedType->dumpDerived(indentCount + 1, out);
    }
}

Index ClassLikeNode::calcDerivedDepth() const
{
    const ClassLikeNode* node = this;
    Index count = 0;

    while (node)
    {
        count++;
        node = node->m_superNode;
    }

    return count;
}

ClassLikeNode* ClassLikeNode::findLastDerived()
{
    for (Index i = m_derivedTypes.getCount() - 1; i >= 0; --i)
    {
        ClassLikeNode* derivedType = m_derivedTypes[i];
        ClassLikeNode* found = derivedType->findLastDerived();
        if (found)
        {
            return found;
        }
    }
    return this;
}

bool ClassLikeNode::hasReflectedDerivedType() const
{
    for (ClassLikeNode* type : m_derivedTypes)
    {
        if (type->isReflected())
        {
            return true;
        }
    }
    return false;
}

void ClassLikeNode::getReflectedDerivedTypes(List<ClassLikeNode*>& out) const
{
    out.clear();
    for (ClassLikeNode* type : m_derivedTypes)
    {
        if (type->isReflected())
        {
            out.add(type);
        }
    }
}

void ClassLikeNode::dump(int indentCount, StringBuilder& out)
{
    dumpMarkup(indentCount, out);

    _indent(indentCount, out);

    const char* typeName = (m_kind == Kind::StructType) ? "struct" : "class";

    out << typeName << " ";

    if (!isReflected())
    {
        out << " (";
    }
    out << m_name.getContent();
    if (!isReflected())
    {
        out << ") ";
    }

    if (m_super.hasContent())
    {
        out << " : " << m_super.getContent();
    }

    out << " {\n";

    for (Node* child : m_children)
    {
        child->dump(indentCount + 1, out);
    }

    _indent(indentCount, out);
    out << "}\n";
}

} // namespace CppParse
