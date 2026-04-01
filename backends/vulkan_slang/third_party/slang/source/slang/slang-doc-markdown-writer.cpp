// slang-doc-markdown-writer.cpp
#include "slang-doc-markdown-writer.h"

#include "../core/slang-char-util.h"
#include "../core/slang-string-util.h"
#include "../core/slang-token-reader.h"
#include "../core/slang-type-text-util.h"
#include "slang-ast-builder.h"
#include "slang-lookup.h"

namespace Slang
{


/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! DocMarkDownWriter !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! */

template<typename T>
static void _getDecls(ContainerDecl* containerDecl, List<T*>& out)
{
    for (Decl* decl : containerDecl->members)
    {
        if (T* declAsType = as<T>(decl))
        {
            out.add(declAsType);
        }
    }
}

template<typename T>
static void _getDeclsOfType(
    DocMarkdownWriter* writer,
    ContainerDecl* containerDecl,
    List<Decl*>& out)
{
    for (Decl* decl : containerDecl->members)
    {
        if (as<T>(decl))
        {
            if (!writer->isVisible(decl))
                continue;
            out.add(decl);
        }
        else if (auto genericDecl = as<GenericDecl>(decl))
        {
            if (as<T>(genericDecl->inner))
            {
                if (!writer->isVisible(decl))
                    continue;
                out.add(genericDecl);
            }
        }
    }
}

template<typename T>
static void _getDeclsOfType(DocMarkdownWriter* writer, DocumentPage* page, List<Decl*>& out)
{
    // Collect all decls of type T from all entries for the page.
    List<Decl*> declList;
    for (auto entry : page->entries)
    {
        _getDeclsOfType<T>(writer, as<ContainerDecl>(entry->m_node), declList);
    }
    // Deduplicate based on name.
    Dictionary<Name*, Decl*> nameDict;
    for (auto decl : declList)
    {
        nameDict[decl->getName()] = decl;
    }
    // Sort by name.
    for (auto pair : nameDict)
    {
        if (pair.first)
            out.add(pair.second);
    }
    out.sort(
        [](Decl* a, Decl* b) -> bool { return getText(a->getName()) < getText(b->getName()); });
}

template<typename T>
static void _toList(FilteredMemberList<T>& list, List<Decl*>& out)
{
    for (Decl* decl : list)
    {
        out.add(decl);
    }
}

static void _appendAsSingleLine(const UnownedStringSlice& in, StringBuilder& out)
{
    List<UnownedStringSlice> lines;
    StringUtil::calcLines(in, lines);

    // Ideally we'd remove any extraneous whitespace, but for now just join
    StringUtil::join(lines.getBuffer(), lines.getCount(), ' ', out);
}

String getDocPath(const DocumentationConfig& config, String path)
{
    return config.rootDir + Path::getPathWithoutExt(path);
}

String getTocTreeEntry(const String& name, const String& fromPath, const String& toPath)
{
    StringBuilder sb;
    // Format: name <path>
    sb << name;
    sb << " <" << Path::getPathWithoutExt(Path::getRelativePath(fromPath, toPath)) << ">\n";
    return sb.produceString();
}

void DocMarkdownWriter::_appendAsBullets(
    const List<NameAndText>& values,
    bool insertLinkForName,
    char wrapChar)
{
    auto& out = *m_builder;
    for (const auto& value : values)
    {
        out << "#### ";
        if (value.decl)
        {
            // Add anchor ID for the decl.
            if (as<GenericTypeParamDeclBase>(value.decl))
            {
                out << " <a id=\"typeparam-" << getText(value.decl->getName()) << "\"></a>";
            }
            else
            {
                out << " <a id=\"decl-" << getText(value.decl->getName()) << "\"></a>";
            }
        }
        const String& name = value.name;
        auto path = findLinkForToken(m_currentPage, name);
        if (name.getLength())
        {
            if (wrapChar)
            {
                if (path.getLength())
                {
                    out << "[";
                }
                out.appendChar(wrapChar);
                out << name;
                out.appendChar(wrapChar);
                if (path.getLength())
                {
                    out << "](" << Path::getPathWithoutExt(path) << ")";
                }
            }
            else
            {
                if (insertLinkForName)
                    out << translateToMarkdownWithLinks(name);
                else
                {
                    auto spaceLoc = name.indexOf(' ');
                    if (spaceLoc == -1)
                        out << escapeMarkdownText(name);
                    else
                    {
                        auto first = name.getUnownedSlice().head(spaceLoc);
                        auto rest = name.getUnownedSlice().tail(spaceLoc + 1);
                        out << escapeMarkdownText(first) << " ";
                        out << translateToMarkdownWithLinks(rest);
                    }
                }
            }
        }
        if (value.text.getLength())
        {
            out.appendChar('\n');

            ParsedDescription desc;
            desc.parse(value.text.getUnownedSlice());
            desc.write(this, value.decl, out);
        }
        else
        {
            out << "\n";
        }
    }
}

void DocMarkdownWriter::_appendAsBullets(const List<String>& values, char wrapChar)
{
    auto& out = *m_builder;
    for (const auto& value : values)
    {
        out << "* ";
        const String& name = value;

        String path;
        path = findLinkForToken(m_currentPage, name);
        if (path.getLength() == 0)
        {
            Slang::Misc::TokenReader tokenReader(name);
            if (!tokenReader.IsEnd())
                path = findLinkForToken(m_currentPage, tokenReader.ReadToken().Content);
        }
        if (path.getLength())
        {
            out << "[";
        }
        if (name.getLength())
        {
            if (wrapChar)
            {
                out.appendChar(wrapChar);
                out << escapeMarkdownText(name);
                out.appendChar(wrapChar);
            }
            else
            {
                out << escapeMarkdownText(name);
            }
        }
        if (path.getLength())
        {
            out << "]("
                << Path::getPathWithoutExt(
                       Path::getRelativePath(Path::getParentDirectory(m_currentPage->path), path))
                << ")";
        }
        out << "\n";
    }
}

String DocMarkdownWriter::_getName(Decl* decl)
{
    StringBuilder buf;
    ASTPrinter::appendDeclName(decl, buf);
    return buf.produceString();
}

String DocMarkdownWriter::_getFullName(Decl* decl)
{
    ASTPrinter printer(m_astBuilder);
    printer.addDeclPath(decl);
    return printer.getStringBuilder().produceString();
}

String _translateNameToPath(const UnownedStringSlice& name)
{
    // We will generate an all-lowercase file name based on the name of the decl.
    // We do so by converting all capital letters to lowercase, and append a disambiguator
    // postfix at the end that tracks the location of the capital letters in the original name.
    // For example, `MyType` will be converted to `mytype-02`, because there is one capital letter
    // at location 0 and one at location 2.
    // To prevent URL issues, all leading underscores are replaced with '0', and we
    // will also append the location of the replaced character to the disambiguator postfix.
    // For example, `_MyType` will be converted to `0mytype-013`.
    // To keep disambiguators short, we use base 36 to represent each location.

    StringBuilder buf;
    StringBuilder disambiguatorSB;
    for (Index i = 0; i < name.getLength(); i++)
    {
        auto c = name[i];
        // Removing leading underscores to prevent url issues.
        if (c == '_' && buf.getLength() == 0)
        {
            disambiguatorSB.append(String(i, 36));
            buf.appendChar('0');
            continue;
        }

        if (c == ' ')
        {
            buf.appendChar('-');
        }
        else if (c >= 'A' && c <= 'Z')
        {
            // Convert to lower case.
            buf.appendChar((char)(c - 'A' + 'a'));
            disambiguatorSB.append(String(i, 36));
        }
        else if (CharUtil::isAlphaOrDigit(c) || c == '_')
        {
            buf.appendChar(c);
        }
        else
        {
            buf.appendChar('x');
            buf << String((int)c, 16);
        }
    }
    if (disambiguatorSB.getLength())
    {
        buf << "-";
        buf << disambiguatorSB.produceString().toLower();
    }
    return buf;
}

String DocMarkdownWriter::_getDocFilePath(Decl* decl)
{
    if (!decl)
        return "";

    StringBuilder sb;
    if (as<NamespaceDeclBase>(getParentDecl(decl)))
    {
        if (as<InterfaceDecl>(decl))
        {
            sb << "interfaces/";
        }
        else if (as<AggTypeDeclBase>(decl) || as<TypeDefDecl>(decl))
        {
            sb << "types/";
        }
        else if (as<AttributeDecl>(decl))
        {
            sb << "attributes/";
        }
        else
        {
            sb << "global-decls/";
        }
    }

    if (auto extDecl = as<ExtensionDecl>(decl))
    {
        if (auto declRef = isDeclRefTypeOf<Decl>(extDecl->targetType))
        {
            auto name = _translateNameToPath(_getName(declRef.getDecl()).getUnownedSlice());
            sb << name;
            sb << "/index.md";
            return sb.produceString();
        }
    }
    if (as<AggTypeDeclBase>(decl))
    {
        auto name = _translateNameToPath(_getName(decl).getUnownedSlice());
        sb << name;
        sb << "/index.md";
        return sb.produceString();
    }
    auto parentPath = _getDocFilePath(getParentDecl(decl));
    if (parentPath.endsWith(".md"))
    {
        parentPath = Path::getParentDirectory(parentPath);
    }
    if (parentPath.getLength() > 0)
    {
        sb << parentPath;
        sb << "/";
    }
    sb << _translateNameToPath(_getName(decl).getUnownedSlice());
    sb << ".md";
    return sb.produceString();
}

String DocMarkdownWriter::_getName(InheritanceDecl* decl)
{
    StringBuilder buf;
    buf.clear();
    buf << decl->base;
    return buf.produceString();
}

DocMarkdownWriter::NameAndText DocMarkdownWriter::_getNameAndText(
    ASTMarkup::Entry* entry,
    Decl* decl)
{
    NameAndText nameAndText;

    nameAndText.decl = decl;
    nameAndText.name = _getName(decl);

    StringBuilder sb;
    if (auto varDeclBase = as<VarDeclBase>(decl))
    {
        if (varDeclBase->type)
        {
            sb << " : " << varDeclBase->type->toString();

            if (varDeclBase->initExpr)
            {
                sb << " = ";
                _appendExpr(sb, varDeclBase->initExpr);
            }
        }
    }
    else if (auto typeParam = as<GenericTypeParamDeclBase>(decl))
    {
        bool isFirst = true;
        for (auto member : decl->parentDecl->members)
        {
            if (auto constraint = as<TypeConstraintDecl>(member))
            {
                if (isDeclRefTypeOf<Decl>(getSub(m_astBuilder, constraint)).getDecl() == typeParam)
                {
                    if (isFirst)
                    {
                        sb << ": ";
                        isFirst = false;
                    }
                    else
                    {
                        sb << ", ";
                    }
                    sb << constraint->getSup().type->toString();
                    break;
                }
            }
        }
        if (auto genericTypeParam = as<GenericTypeParamDecl>(decl))
        {
            if (genericTypeParam->initType.type)
            {
                sb << " = ";
                sb << genericTypeParam->initType.type->toString();
            }
        }
    }
    else if (auto enumCase = as<EnumCaseDecl>(decl))
    {
        if (enumCase->tagExpr)
        {
            sb << " = ";
            _appendExpr(sb, enumCase->tagExpr);
        }
    }
    nameAndText.name.append(sb.produceString());

    if (entry && entry->m_markup.getLength())
    {
        nameAndText.text = entry->m_markup;
    }

    return nameAndText;
}

DocMarkdownWriter::NameAndText DocMarkdownWriter::_getNameAndText(Decl* decl)
{
    ASTMarkup::Entry* entry = m_markup->getEntry(decl);
    return _getNameAndText(entry, decl);
}

List<DocMarkdownWriter::NameAndText> DocMarkdownWriter::_getAsNameAndTextList(const List<Decl*>& in)
{
    List<NameAndText> out;
    for (auto decl : in)
    {
        out.add(_getNameAndText(decl));
    }
    return out;
}

List<String> DocMarkdownWriter::_getAsStringList(const List<Decl*>& in)
{
    List<String> strings;
    for (auto decl : in)
    {
        strings.add(_getName(decl));
    }
    return strings;
}

void DocMarkdownWriter::_appendCommaList(const List<String>& strings, char wrapChar)
{
    auto& out = *m_builder;
    for (Index i = 0; i < strings.getCount(); ++i)
    {
        if (i > 0)
        {
            out << toSlice(", ");
        }
        if (wrapChar)
        {
            out.appendChar(wrapChar);
            out << strings[i];
            out.appendChar(wrapChar);
        }
        else
        {
            out << translateToMarkdownWithLinks(strings[i]);
        }
    }
}

/* static */ void DocMarkdownWriter::getSignature(const List<Part>& parts, Signature& outSig)
{
    const Index count = parts.getCount();
    for (Index i = 0; i < count; ++i)
    {
        const auto& part = parts[i];
        switch (part.type)
        {
        case Part::Type::ParamType:
            {
                PartPair pair;
                pair.first = part;
                if ((i + 1) < count && parts[i + 1].type == Part::Type::ParamName)
                {
                    pair.second = parts[i + 1];
                    i++;
                }
                outSig.params.add(pair);
                break;
            }
        case Part::Type::ReturnType:
            {
                outSig.returnType = part;
                break;
            }
        case Part::Type::DeclPath:
            {
                outSig.name = part;
                break;
            }
        case Part::Type::GenericParamValue:
        case Part::Type::GenericParamType:
            {
                Signature::GenericParam genericParam;
                genericParam.name = part;

                if ((i + 1) < count && parts[i + 1].type == Part::Type::GenericParamValueType)
                {
                    genericParam.type = parts[i + 1];
                    i++;
                }

                outSig.genericParams.add(genericParam);
                break;
            }

        default:
            break;
        }
    }
}

void escapeHTMLContent(StringBuilder& sb, UnownedStringSlice str)
{
    for (auto ch : str)
    {
        switch (ch)
        {
        case '<':
            sb << "&lt;";
            break;
        case '>':
            sb << "&gt;";
            break;
        case '&':
            sb << "&amp;";
            break;
        case '"':
            sb << "&quot;";
            break;
        default:
            sb.appendChar(ch);
            break;
        }
    }
}

void DocMarkdownWriter::writeVar(const ASTMarkup::Entry& entry, VarDecl* varDecl)
{
    auto& out = *m_builder;

    ASTPrinter printer(m_astBuilder);
    printer.addDeclPath(DeclRef<Decl>(varDecl));

    out << toSlice("# ") << printer.getSlice() << toSlice("\n\n");

    DeclDocumentation declDoc;
    declDoc.parse(entry.m_markup.getUnownedSlice());
    declDoc.writeDescription(out, this, varDecl);
    registerCategory(m_currentPage, declDoc);

    out << toSlice("## Signature\n");
    out << toSlice("<pre>\n");
    if (varDecl->hasModifier<HLSLStaticModifier>())
    {
        out << toSlice("<span class='code_keyword'>static</span> ");
    }
    if (varDecl->hasModifier<ConstModifier>())
    {
        out << toSlice("<span class='code_keyword'>const</span> ");
    }
    if (varDecl->hasModifier<ConstExprModifier>())
    {
        out << toSlice("<span class='code_keyword'>constexpr</span> ");
    }
    if (varDecl->hasModifier<InModifier>())
    {
        out << toSlice("<span class='code_keyword'>in</span> ");
    }
    if (varDecl->hasModifier<OutModifier>())
    {
        out << toSlice("<span class='code_keyword'>out</span> ");
    }
    StringBuilder typeSB;
    varDecl->type->toText(typeSB);
    out << translateToHTMLWithLinks(varDecl, typeSB.produceString()) << toSlice(" ");
    out << translateToHTMLWithLinks(varDecl, printer.getSlice());

    if (varDecl->initExpr)
    {
        out << toSlice(" = ");
        _appendExpr(out, varDecl->initExpr);
    }

    out << toSlice(";\n</pre>\n\n");

    declDoc.writeSection(out, this, varDecl, DocPageSection::Remarks);
    declDoc.writeSection(out, this, varDecl, DocPageSection::Example);
    declDoc.writeSection(out, this, varDecl, DocPageSection::SeeAlso);
}

void DocMarkdownWriter::writeProperty(const ASTMarkup::Entry& entry, PropertyDecl* propertyDecl)
{
    auto& out = *m_builder;
    out << toSlice("# property ");

    ASTPrinter printer(m_astBuilder);
    printer.addDeclPath(DeclRef<Decl>(propertyDecl));
    out << escapeMarkdownText(printer.getSlice()) << toSlice("\n\n");

    DeclDocumentation declDoc;
    declDoc.parse(entry.m_markup.getUnownedSlice());
    declDoc.writeDescription(out, this, propertyDecl);
    registerCategory(m_currentPage, declDoc);

    out << toSlice("## Signature\n\n");

    out << toSlice("<pre>\n<span class='code_keyword'>property</span> ");
    out << translateToHTMLWithLinks(propertyDecl, printer.getSlice());
    out << toSlice(" : ");
    StringBuilder typeSB;
    propertyDecl->type->toText(typeSB);
    out << translateToHTMLWithLinks(propertyDecl, typeSB.produceString());
    out << "\n{\n";
    for (auto member : propertyDecl->members)
    {
        if (as<GetterDecl>(member))
        {
            out << "    get;\n";
        }
        else if (as<SetterDecl>(member))
        {
            out << "    set;\n";
        }
        else if (as<RefAccessorDecl>(member))
        {
            out << "    ref;\n";
        }
    }
    out << "}\n</pre>\n\n";

    declDoc.writeSection(out, this, propertyDecl, DocPageSection::ReturnInfo);
    declDoc.writeSection(out, this, propertyDecl, DocPageSection::Remarks);
    declDoc.writeSection(out, this, propertyDecl, DocPageSection::Example);
    declDoc.writeSection(out, this, propertyDecl, DocPageSection::SeeAlso);
}

void DocMarkdownWriter::writeTypeDef(const ASTMarkup::Entry& entry, TypeDefDecl* typeDefDecl)
{
    auto& out = *m_builder;

    out << toSlice("# ");
    ASTMarkup::Entry newEntry = entry;
    _appendAggTypeName(newEntry, typeDefDecl);
    out << toSlice("\n\n");

    DeclDocumentation declDoc;
    declDoc.parse(entry.m_markup.getUnownedSlice());
    registerCategory(m_currentPage, declDoc);

    declDoc.writeDescription(out, this, typeDefDecl);

    out << toSlice("## Signature\n\n");

    out << toSlice("<pre>\n<span class='code_keyword'>typealias</span> ");
    ASTPrinter printer(m_astBuilder);
    printer.addDeclPath(typeDefDecl);
    out << translateToHTMLWithLinks(typeDefDecl, printer.getSlice());
    out << toSlice(" = ");

    // Insert a line break if the type name is already long.
    if (printer.getSlice().getLength() > 25)
    {
        out << "\n    ";
    }
    out << translateToHTMLWithLinks(typeDefDecl, typeDefDecl->type->toString());
    out << ";\n</pre>\n\n";

    declDoc.writeGenericParameters(out, this, typeDefDecl);

    declDoc.writeSection(out, this, typeDefDecl, DocPageSection::Remarks);
    declDoc.writeSection(out, this, typeDefDecl, DocPageSection::Example);
    declDoc.writeSection(out, this, typeDefDecl, DocPageSection::SeeAlso);
}

static String getAttributeName(AttributeDecl* decl)
{
    auto name = getText(decl->getName());
    if (name.startsWith("vk_"))
    {
        return String("vk::") + String(name.getUnownedSlice().tail(3));
    }
    return name;
}

void DocMarkdownWriter::writeAttribute(const ASTMarkup::Entry& entry, AttributeDecl* attributeDecl)
{
    auto& out = *m_builder;

    out << toSlice("# attribute [");
    out << escapeMarkdownText(getAttributeName(attributeDecl));
    out << toSlice("]\n\n");
    DeclDocumentation declDoc;
    declDoc.parse(entry.m_markup.getUnownedSlice());
    declDoc.writeDescription(out, this, attributeDecl);
    registerCategory(m_currentPage, declDoc);

    out << toSlice("## Signature\n\n");
    List<Decl*> paramDecls;
    for (auto param : attributeDecl->getMembersOfType<ParamDecl>())
    {
        paramDecls.add(param);
    }
    out << "<pre>\n";
    out << "[" << translateToHTMLWithLinks(attributeDecl, getAttributeName(attributeDecl));
    if (paramDecls.getCount() > 0)
    {
        out << "(";
        for (Index i = 0; i < paramDecls.getCount(); i++)
        {
            if (i > 0)
                out << ", ";
            auto param = paramDecls[i];
            out << translateToHTMLWithLinks(attributeDecl, _getName(param));
            auto type = as<ParamDecl>(param)->type;
            if (type)
            {
                out << " : ";
                out << translateToHTMLWithLinks(attributeDecl, type->toString());
            }
        }
        out << ")";
    }
    out << "]\n</pre>\n\n";

    if (paramDecls.getCount() > 0)
    {
        out << "## Parameters\n\n";

        // Document ordinary parameters
        _appendAsBullets(_getUniqueParams(paramDecls, &declDoc), false, 0);

        out << toSlice("\n");
    }

    declDoc.writeSection(out, this, attributeDecl, DocPageSection::Remarks);
    declDoc.writeSection(out, this, attributeDecl, DocPageSection::Example);
    declDoc.writeSection(out, this, attributeDecl, DocPageSection::SeeAlso);
}

void DocMarkdownWriter::writeExtensionConditions(
    StringBuilder& out,
    ExtensionDecl* extensionDecl,
    const char* prefix,
    bool isHtml)
{
    // Synthesize `where` clause for things defined in an extension.
    auto targetTypeDeclRef = isDeclRefTypeOf<ContainerDecl>(extensionDecl->targetType);
    if (!targetTypeDeclRef)
        return;

    if (auto genAppDeclRef = as<GenericAppDeclRef>(targetTypeDeclRef.declRefBase))
    {
        for (Index i = 0; i < genAppDeclRef->getArgCount(); i++)
        {
            auto arg = genAppDeclRef->getArg(i);
            Decl* genericParamDecl = nullptr;
            Index parameterIndex = i;
            if (auto extTypeParamDecl = isDeclRefTypeOf<GenericTypeParamDeclBase>(arg))
            {
                genericParamDecl = extTypeParamDecl.getDecl();
            }
            else if (auto extValueParamVal = as<GenericParamIntVal>(arg))
            {
                genericParamDecl = extValueParamVal->getDeclRef().getDecl();
            }

            // Locate the original generic parameter defined on the type being extended.
            Decl* originalParamDecl = nullptr;
            if (auto targetTypeParentGenericDecl =
                    as<GenericDecl>(targetTypeDeclRef.getDecl()->parentDecl))
            {
                for (auto member : targetTypeParentGenericDecl->members)
                {
                    if (auto typeParamDecl = as<GenericTypeParamDeclBase>(member))
                    {
                        if (typeParamDecl->parameterIndex == parameterIndex)
                        {
                            originalParamDecl = typeParamDecl;
                            break;
                        }
                    }
                    else if (auto valParamDecl = as<GenericValueParamDecl>(member))
                    {
                        if (valParamDecl->parameterIndex == parameterIndex)
                        {
                            originalParamDecl = valParamDecl;
                            break;
                        }
                    }
                }
            }

            // If we can't find such parameter, bail.
            if (!originalParamDecl)
                continue;

            bool isEqualityConstraint = false;
            Val* constraintVal = nullptr;
            if (genericParamDecl)
            {
                // If we have `TargetType<T>` the member belongs to `extension<X : C>
                // TargetType<X>`, We want to print a synthesized `where T : C` clause. Here
                // `extTypeParamDecl` is a reference to `X`, so we need to find the corresponding
                // `T`.

                // Find constraints on the originalParamDecl.
                for (auto member : genericParamDecl->parentDecl->members)
                {
                    if (auto typeConstraint = as<GenericTypeConstraintDecl>(member))
                    {
                        if (isDeclRefTypeOf<Decl>(typeConstraint->sub.type).getDecl() ==
                            genericParamDecl)
                        {
                            if (typeConstraint->isEqualityConstraint)
                            {
                                isEqualityConstraint = true;
                            }
                            constraintVal = typeConstraint->getSup().type;
                            break;
                        }
                    }
                }
            }
            else
            {
                // If we have `extension TargetType<Y>` where `Y` does not name a generic parameter
                // defined on the extension itself, we want to print a synthesized `where T == Y`
                // clause, where `T` is the original generic parameter on the target type.
                isEqualityConstraint = true;
                constraintVal = arg;
            }
            if (constraintVal)
            {
                out << prefix;
                if (isHtml)
                    out << translateToHTMLWithLinks(
                        originalParamDecl,
                        originalParamDecl->getName()->text);
                else
                    out << translateToMarkdownWithLinks(originalParamDecl->getName()->text);
                if (isEqualityConstraint)
                    out << " == ";
                else
                    out << " : ";
                if (isHtml)
                    out << translateToHTMLWithLinks(originalParamDecl, constraintVal->toString());
                else
                    out << translateToMarkdownWithLinks(constraintVal->toString());
            }
        }
    }
}

void DocMarkdownWriter::writeSignature(CallableDecl* callableDecl)
{
    StringBuilder& out = *getBuilder(callableDecl);

    if (callableDecl->hasModifier<HLSLStaticModifier>())
    {
        out << "<span class='code_keyword'>static</span> ";
    }

    List<ASTPrinter::Part> parts;

    ASTPrinter printer(
        m_astBuilder,
        ASTPrinter::OptionFlag::ParamNames | ASTPrinter::OptionFlag::NoSpecializedExtensionTypeName,
        &parts);
    printer.addDeclSignature(makeDeclRef(callableDecl));

    Signature signature;
    getSignature(parts, signature);

    const Index paramCount = signature.params.getCount();

    {
        // Some types (like constructors say) don't have any return type, so check before outputting
        const UnownedStringSlice returnType = printer.getPartSlice(signature.returnType);
        if (returnType.getLength() > 0)
        {
            out << translateToHTMLWithLinks(callableDecl, returnType) << toSlice(" ");
        }
    }

    out << translateToHTMLWithLinks(callableDecl, printer.getPartSlice(signature.name));

    switch (paramCount)
    {
    case 0:
        {
            // Has no parameters
            out << toSlice("()");
            break;
        }
    case 1:
        {
            if (signature.name.end - signature.name.start < 40)
            {
                // Place all on single line
                out.appendChar('(');
                const auto& param = signature.params[0];
                out << translateToHTMLWithLinks(callableDecl, printer.getPartSlice(param.first))
                    << toSlice(" ");
                out << translateToHTMLWithLinks(callableDecl, printer.getPartSlice(param.second));
                out << ")";
                break;
            }
            // If the name is already long, fall through to default.
            [[fallthrough]];
        }
    default:
        {
            // Put each parameter on a line on it's own
            out << toSlice("(\n");
            StringBuilder line;
            for (Index i = 0; i < paramCount; ++i)
            {
                const auto& param = signature.params[i];
                line.clear();

                line << "    "
                     << translateToHTMLWithLinks(callableDecl, printer.getPartSlice(param.first));

                line.appendChar(' ');
                line << translateToHTMLWithLinks(callableDecl, printer.getPartSlice(param.second));
                if (i < paramCount - 1)
                {
                    line << ",\n";
                }

                out << line;
            }

            out << ")";
            break;
        }
    }

    // Print `where` clause.
    Decl* parentDecl = callableDecl;
    while (parentDecl)
    {
        if (auto extensionDecl = as<ExtensionDecl>(parentDecl))
        {
            // Synthesize `where` clause for things defined in an extension.
            if (auto targetTypeDeclRef = isDeclRefTypeOf<ContainerDecl>(extensionDecl->targetType))
            {
                writeExtensionConditions(
                    out,
                    extensionDecl,
                    "\n    <span class='code_keyword'>where</span> ",
                    true);
                // We need to follow the parent of the target type instead of the parent of the
                // extension decl.
                parentDecl = getParentDecl(targetTypeDeclRef.getDecl());
                continue;
            }
        }

        if (auto genericParent = as<GenericDecl>(parentDecl->parentDecl))
        {
            for (auto member : genericParent->members)
            {
                if (auto typeConstraint = as<GenericTypeConstraintDecl>(member))
                {
                    out << toSlice("\n    <span class='code_keyword'>where</span> ");
                    out << translateToHTMLWithLinks(
                        parentDecl,
                        getSub(m_astBuilder, typeConstraint)->toString());
                    if (typeConstraint->isEqualityConstraint)
                        out << " == ";
                    else
                        out << toSlice(" : ");
                    out << translateToHTMLWithLinks(
                        parentDecl,
                        getSup(m_astBuilder, typeConstraint)->toString());
                }
            }
        }
        parentDecl = getParentDecl(parentDecl);
    }
    out << ";\n";
}

List<DocMarkdownWriter::NameAndText> DocMarkdownWriter::_getUniqueParams(
    const List<Decl*>& decls,
    DeclDocumentation* funcDoc)
{
    List<NameAndText> out;

    Dictionary<String, Index> nameDict;

    for (auto decl : decls)
    {
        Name* name = decl->getName();
        if (!name)
        {
            continue;
        }
        auto nameText = _getNameAndText(decl);
        Index index = nameDict.getOrAddValue(nameText.name, out.getCount());

        if (index >= out.getCount())
        {
            out.add(nameText);
        }

        // Extract text.
        NameAndText& nameAndMarkup = out[index];
        if (nameAndMarkup.text.getLength() > 0)
        {
            continue;
        }

        ParamDocumentation paramDoc;
        if (funcDoc->parameters.tryGetValue(getText(name), paramDoc))
        {
            StringBuilder sb;
            if (paramDoc.direction.getLength())
                sb << "\\[" << paramDoc.direction << "\\] ";
            sb << paramDoc.description.ownedText;
            nameAndMarkup.text = sb.produceString();
        }
        else
        {
            auto entry = m_markup->getEntry(decl);
            if (entry && entry->m_markup.getLength())
            {
                nameAndMarkup.text = entry->m_markup;
            }
        }
    }

    return out;
}

static Index _addRequirement(
    const DocMarkdownWriter::Requirement& req,
    List<DocMarkdownWriter::Requirement>& ioReqs)
{
    auto index = ioReqs.indexOf(req);
    if (index < 0)
    {
        ioReqs.add(req);
        return ioReqs.getCount() - 1;
    }
    return index;
}

static Index _addRequirement(CapabilitySet set, List<DocMarkdownWriter::Requirement>& ioReqs)
{
    return _addRequirement(DocMarkdownWriter::Requirement{set}, ioReqs);
}

static Index _addRequirements(Decl* decl, List<DocMarkdownWriter::Requirement>& ioReqs)
{
    StringBuilder buf;

    if (auto capAttr = decl->findModifier<RequireCapabilityAttribute>())
    {
        return _addRequirement(capAttr->capabilitySet, ioReqs);
    }
    return -1;
}

static String getCapabilityName(CapabilityName name)
{
    auto text = capabilityNameToString(name);
    if (text.startsWith("_"))
    {
        return text.tail(1);
    }
    return text;
}

static String getCapabilityName(CapabilityAtom atom)
{
    return getCapabilityName((CapabilityName)atom);
}

void DocMarkdownWriter::_appendExpr(StringBuilder& sb, Expr* expr)
{
    if (auto typeCast = as<TypeCastExpr>(expr))
        _appendExpr(sb, typeCast->arguments[0]);
    else if (auto declRefExpr = as<DeclRefExpr>(expr))
    {
        ASTPrinter printer(m_astBuilder);
        printer.addDeclPath(declRefExpr->declRef);
        sb << escapeMarkdownText(printer.getSlice());
    }
    else if (auto litExpr = as<LiteralExpr>(expr))
    {
        sb << litExpr->token.getContent();
    }
    else
    {
        sb << "...";
    }
}

void DocMarkdownWriter::_appendRequirements(const Requirement& requirement)
{
    auto capabilitySet = requirement.capabilitySet;
    m_builder->append("Defined for the following targets:\n\n");
    for (auto targetSet : capabilitySet.getCapabilityTargetSets())
    {
        m_builder->append("#### ");
        m_builder->append(getCapabilityName(targetSet.first));
        m_builder->append("\n");

        if (targetSet.second.shaderStageSets.getCount() == kCapabilityStageCount)
        {
            m_builder->append("Available in all stages.\n");
        }
        else if (targetSet.second.shaderStageSets.getCount() > 1)
        {
            m_builder->append("Available in stages: ");
            bool isFirst = true;
            for (auto& stage : targetSet.second.shaderStageSets)
            {
                if (!isFirst)
                {
                    m_builder->append(", ");
                }
                isFirst = false;
                m_builder->append("`");
                m_builder->append(getCapabilityName(stage.first));
                m_builder->append("`");
            }
            m_builder->append(".\n");
        }
        else if (targetSet.second.shaderStageSets.getCount() == 1)
        {
            m_builder->append("Available in `");
            m_builder->append(getCapabilityName(targetSet.second.shaderStageSets.begin()->first));
            m_builder->append("` stage only.\n");
        }

        m_builder->append("\n");

        // TODO: We should probably print the capabilities for each stage set if the requirements
        // differ between different stages, but for now we'll just print the first one, assuming the
        // rest are the same. This is currently true for most if not all of our core module decls.
        //
        if (targetSet.second.shaderStageSets.getCount() > 0 &&
            targetSet.second.shaderStageSets.begin()->second.atomSet.has_value())
        {
            List<String> capabilities;
            auto atomSet = targetSet.second.shaderStageSets.begin()
                               ->second.atomSet.value()
                               .newSetWithoutImpliedAtoms();
            for (auto atom : atomSet)
            {
                // If the requirement atom is the target or stage atom, don't repeat ourselves.
                if ((CapabilityAtom)atom == targetSet.first)
                    continue;
                if ((CapabilityAtom)atom == targetSet.second.shaderStageSets.begin()->first)
                    continue;
                String capabilityName = capabilityNameToString((CapabilityName)atom);
                if (!capabilityName.startsWith("_"))
                {
                    capabilities.add(capabilityName);
                }
            }
            if (capabilities.getCount() > 1)
            {
                m_builder->append("Requires capabilities: ");
                _appendCommaList(capabilities, '`');
                m_builder->append(".\n");
            }
            else if (capabilities.getCount() == 1)
            {
                m_builder->append("Requires capability: `");
                m_builder->append(capabilities[0]);
                m_builder->append("`");
                m_builder->append(".\n");
            }
        }
    }
}

void DocMarkdownWriter::_maybeAppendRequirements(
    const UnownedStringSlice& title,
    const List<DocMarkdownWriter::Requirement>& uniqueRequirements)
{
    auto& out = *m_builder;
    const Index uniqueCount = uniqueRequirements.getCount();

    if (uniqueCount <= 0)
    {
        return;
    }

    if (uniqueCount == 1)
    {
        const auto& reqs = uniqueRequirements[0];

        out << title;

        _appendRequirements(reqs);
        out << toSlice("\n");
    }
    else
    {
        out << title;

        for (Index i = 0; i < uniqueCount; ++i)
        {
            out << "### Capability Set " << (i + 1) << ("\n\n");
            _appendRequirements(uniqueRequirements[i]);
            out << toSlice("\n");
        }
    }

    out << toSlice("\n");
}

static Decl* _getSameNameDecl(ContainerDecl* parentDecl, Decl* decl)
{
    Decl* result = nullptr;
    parentDecl->getMemberDictionary().tryGetValue(decl->getName(), result);
    return result;
}

static bool _isFirstOverridden(Decl* decl)
{
    decl = _getSameNameDecl(as<ContainerDecl>(getParentDecl(decl)), decl);

    ContainerDecl* parentDecl = decl->parentDecl;

    Name* declName = decl->getName();
    if (declName)
    {
        Decl** firstDeclPtr = parentDecl->getMemberDictionary().tryGetValue(declName);
        return (firstDeclPtr && *firstDeclPtr == decl) || (firstDeclPtr == nullptr);
    }

    return false;
}

void ParsedDescription::write(DocMarkdownWriter* writer, Decl* decl, StringBuilder& out)
{
    for (auto span : spans)
    {
        switch (span.kind)
        {
        case DocumentationSpanKind::OrdinaryText:
            {
                out << span.text;
                break;
            }
        case DocumentationSpanKind::InlineCode:
            {
                out << "<span class='code'>" << writer->translateToHTMLWithLinks(decl, span.text)
                    << "</span>";
                break;
            }
        }
    }
}

void ParsedDescription::parse(UnownedStringSlice text)
{
    text = text.trim();
    ownedText = text;
    List<UnownedStringSlice> lines;
    StringUtil::calcLines(text, lines);
    Index codeBlockIndent = 0;
    bool isInCodeBlock = false;
    for (auto line : lines)
    {
        auto originalLine = line;
        line = line.trim();
        if (line.startsWith("```"))
        {
            isInCodeBlock = !isInCodeBlock;
            spans.add({line, DocumentationSpanKind::OrdinaryText});
            spans.add({toSlice("\n"), DocumentationSpanKind::OrdinaryText});
            codeBlockIndent = originalLine.indexOf('`');
            continue;
        }

        if (!isInCodeBlock)
        {
            bool isInCode = false;
            const char* currentSpanStart = line.begin();
            const char* currentSpanEnd = currentSpanStart;
            for (Index i = 0; i < line.getLength(); i++)
            {
                if (line[i] == '`')
                {
                    if (currentSpanEnd > currentSpanStart)
                    {
                        spans.add(
                            {UnownedStringSlice(currentSpanStart, line.begin() + i),
                             isInCode ? DocumentationSpanKind::InlineCode
                                      : DocumentationSpanKind::OrdinaryText});
                        currentSpanEnd = currentSpanStart = line.begin() + i + 1;
                    }
                    isInCode = !isInCode;
                }
                else
                {
                    currentSpanEnd = line.begin() + i + 1;
                }
            }
            if (currentSpanEnd > currentSpanStart)
            {
                spans.add(
                    {UnownedStringSlice(currentSpanStart, currentSpanEnd),
                     DocumentationSpanKind::OrdinaryText});
            }
            spans.add({toSlice("\n"), DocumentationSpanKind::OrdinaryText});
        }
        else
        {
            line = originalLine;
            for (Index i = 0; i < codeBlockIndent; i++)
            {
                if (line.startsWith(" "))
                {
                    line = line.tail(1);
                }
                else
                {
                    break;
                }
            }
            spans.add({line, DocumentationSpanKind::OrdinaryText});
            spans.add({toSlice("\n"), DocumentationSpanKind::OrdinaryText});
        }
    }
}

void DeclDocumentation::parse(const UnownedStringSlice& text)
{
    List<UnownedStringSlice> lines;
    StringUtil::calcLines(text, lines);
    DocPageSection currentSection = DocPageSection::Description;
    Dictionary<DocPageSection, StringBuilder> sectionBuilders;
    for (Index ptr = 0; ptr < lines.getCount(); ptr++)
    {
        auto originalLine = lines[ptr];
        auto line = originalLine.trim();
        if (line.startsWith("@param"))
        {
            currentSection = DocPageSection::Parameter;
            line = line.tail(6).trimStart();
            UnownedStringSlice paramDirection;
            UnownedStringSlice paramName;
            if (line.startsWith("["))
            {
                auto closingIndex = line.indexOf(']');
                if (closingIndex != -1)
                {
                    paramDirection = line.subString(1, closingIndex - 1);
                    line = line.tail(closingIndex + 1).trimStart();
                }
            }
            auto spaceIndex = line.indexOf(' ');
            if (spaceIndex != -1)
            {
                paramName = line.subString(0, spaceIndex);
                line = line.tail(spaceIndex + 1).trimStart();
            }
            StringBuilder paramSB;
            paramSB << line << "\n";
            ptr++;
            for (; ptr < lines.getCount(); ptr++)
            {
                auto nextLine = lines[ptr].trim();
                if (nextLine.getLength() == 0 || nextLine.startsWith("@"))
                {
                    ptr--;
                    break;
                }
                paramSB << nextLine << "\n";
            }
            ParamDocumentation paramDesc;
            paramDesc.description.parse(paramSB.getUnownedSlice());
            paramDesc.name = paramName;
            paramDesc.direction = paramDirection;
            parameters[paramDesc.name] = paramDesc;
            continue;
        }
        else if (line.startsWith("@return"))
        {
            currentSection = DocPageSection::ReturnInfo;
            line = line.tail(7).trim();
        }
        else if (line.startsWith("@returns"))
        {
            currentSection = DocPageSection::ReturnInfo;
            line = line.tail(8).trim();
        }
        else if (line.startsWith("@remarks"))
        {
            currentSection = DocPageSection::Remarks;
            line = line.tail(8).trim();
        }
        else if (line.startsWith("@example"))
        {
            currentSection = DocPageSection::Example;
            line = line.tail(8).trim();
        }
        else if (line.startsWith("@see"))
        {
            currentSection = DocPageSection::SeeAlso;
            line = line.tail(4).trim();
        }
        else if (line.startsWith("@experimental"))
        {
            currentSection = DocPageSection::ExperimentalCallout;
            line = line.tail(13).trim();
        }
        else if (line.startsWith("@internal"))
        {
            currentSection = DocPageSection::InternalCallout;
            line = line.tail(9).trim();
        }
        else if (line.startsWith("@deprecated"))
        {
            currentSection = DocPageSection::DeprecatedCallout;
            line = line.tail(11).trim();
        }
        else if (line.startsWith("@category"))
        {
            line = line.tail(9).trimStart();
            auto spaceIndex = line.indexOf(' ');
            if (spaceIndex != -1)
            {
                categoryName = line.subString(0, spaceIndex);
                categoryText = line.tail(spaceIndex + 1).trim();
            }
            else
            {
                categoryName = line.trim();
            }
            continue;
        }
        else
        {
            line = originalLine;
        }
        sectionBuilders[currentSection] << line << "\n";

        // If the current directive is a callout, set currentSection back
        // to Description after processing the directive line.
        switch (currentSection)
        {
        case DocPageSection::ExperimentalCallout:
        case DocPageSection::InternalCallout:
        case DocPageSection::DeprecatedCallout:
            currentSection = DocPageSection::Description;
            break;
        }
    }
    for (auto& kv : sectionBuilders)
    {
        sections[kv.first].parse(kv.second.getUnownedSlice());
    }
}

void DocMarkdownWriter::writeCallableOverridable(
    DocumentPage* page,
    const ASTMarkup::Entry& primaryEntry,
    CallableDecl* callableDecl)
{
    SLANG_UNUSED(primaryEntry);

    auto& out = *m_builder;
    {
        // Output the overridable path (ie without terminal generic parameters)
        ASTPrinter printer(m_astBuilder, ASTPrinter::OptionFlag::NoSpecializedExtensionTypeName);
        printer.addOverridableDeclPath(DeclRef<Decl>(callableDecl));
        // Extract the name
        out << toSlice("# ") << escapeMarkdownText(printer.getStringBuilder()) << toSlice("\n\n");
    }

    // Collect descriptions from all overloads.
    StringBuilder descriptionSB, additionalDescriptionSB;
    for (auto entry : page->entries)
    {
        auto markup = entry->m_markup.trim();
        if (markup.getLength() == 0)
            continue;
        if (entry->m_markup.startsWith("@"))
        {
            additionalDescriptionSB << markup << "\n";
        }
        else if (descriptionSB.getLength() != 0)
        {
            // We already have a main description, so this is potentially a duplicate.
            // If the content is not the same, we will report a warning.
            if (!descriptionSB.toString().startsWith(markup))
            {
                auto decl = as<Decl>(entry->m_node);
                m_sink->diagnose(
                    decl->loc,
                    Diagnostics::ignoredDocumentationOnOverloadCandidate,
                    decl);
            }
        }
        else
        {
            descriptionSB << markup << "\n";
        }
    }

    DeclDocumentation funcDoc;
    funcDoc.parse(descriptionSB.getUnownedSlice());
    funcDoc.parse(additionalDescriptionSB.getUnownedSlice());

    registerCategory(page, funcDoc);

    auto& descSection = funcDoc.sections[DocPageSection::Description];
    if (descSection.ownedText.getLength() > 0)
    {
        out << toSlice("## Description\n\n");
        descSection.write(this, callableDecl, out);
    }

    // Collect all overloads from all entries on the page.
    List<CallableDecl*> sigs;
    List<Requirement> requirements;
    HashSet<Decl*> sigSet;
    {
        for (auto& entry : page->entries)
        {
            Decl* sameNameDecl = _getSameNameDecl(
                as<ContainerDecl>(getParentDecl((Decl*)entry->m_node)),
                callableDecl);

            for (Decl* curDecl = sameNameDecl; curDecl;
                 curDecl = curDecl->nextInContainerWithSameName)
            {
                CallableDecl* sig = nullptr;
                if (GenericDecl* genericDecl = as<GenericDecl>(curDecl))
                {
                    sig = as<CallableDecl>(genericDecl->inner);
                }
                else
                {
                    sig = as<CallableDecl>(curDecl);
                }

                if (!sig)
                {
                    continue;
                }

                // Want to add only the primary sig
                if (sig->primaryDecl == nullptr || sig->primaryDecl == sig)
                {
                    if (sigSet.add(sig))
                        sigs.add(sig);
                }
            }
        }

        // Lets put back into source order
        sigs.sort(
            [](CallableDecl* a, CallableDecl* b) -> bool
            { return a->loc.getRaw() < b->loc.getRaw(); });
    }

    // Maps a sig index to a unique requirements set
    List<Index> requirementsMap;

    for (Index i = 0; i < sigs.getCount(); ++i)
    {
        CallableDecl* sig = sigs[i];

        // Add the requirements for all the different versions
        for (CallableDecl* curSig = sig; curSig; curSig = curSig->nextDecl)
        {
            requirementsMap.add(_addRequirements(sig, requirements));
        }
    }

    // Output the signature
    {
        out << toSlice("## Signature \n\n");
        out << toSlice("<pre>\n");

        const Int sigCount = sigs.getCount();
        for (Index i = 0; i < sigCount; ++i)
        {
            auto sig = sigs[i];
            // Get the requirements index for this sig
            const Index requirementsIndex = requirementsMap[i];

            // Output if needs unique requirements
            if (requirements.getCount() > 1 && requirementsIndex != -1)
            {
                out << toSlice("/// Requires Capability Set ") << (requirementsIndex + 1)
                    << toSlice(":\n");
            }

            writeSignature(sig);

            out << "\n";
        }
        out << "</pre>\n\n";
    }

    {
        // We will use the first documentation found for each parameter type
        {
            List<Decl*> paramDecls;
            List<Decl*> genericDecls;
            for (auto sig : sigs)
            {
                GenericDecl* genericDecl = as<GenericDecl>(sig->parentDecl);

                // NOTE!
                // Here we assume the names of generic parameters are such that they are

                // We list generic parameters, as types of parameters, if they are directly
                // associated with this callable.
                if (genericDecl)
                {
                    for (Decl* decl : genericDecl->members)
                    {
                        if (as<GenericTypeParamDeclBase>(decl) || as<GenericValueParamDecl>(decl))
                        {
                            genericDecls.add(decl);
                        }
                    }
                }

                for (ParamDecl* paramDecl : sig->getParameters())
                {
                    paramDecls.add(paramDecl);
                }
            }

            if (genericDecls.getCount() > 0)
            {
                out << "## Generic Parameters\n\n";

                // Document generic parameters
                _appendAsBullets(_getUniqueParams(genericDecls, &funcDoc), false, 0);

                out << toSlice("\n");
            }

            if (paramDecls.getCount() > 0)
            {
                out << "## Parameters\n\n";

                // Document ordinary parameters
                _appendAsBullets(_getUniqueParams(paramDecls, &funcDoc), false, 0);

                out << toSlice("\n");
            }
        }
    }

    auto& returnsSection = funcDoc.sections[DocPageSection::ReturnInfo];
    if (returnsSection.ownedText.getLength() > 0)
    {
        out << toSlice("## Return value\n");
        returnsSection.write(this, callableDecl, out);
    }

    auto& remarksSection = funcDoc.sections[DocPageSection::Remarks];
    if (remarksSection.ownedText.getLength() > 0)
    {
        out << toSlice("## Remarks\n");
        remarksSection.write(this, callableDecl, out);
    }

    auto& exampleSection = funcDoc.sections[DocPageSection::Example];
    if (exampleSection.ownedText.getLength() > 0)
    {
        out << toSlice("## Example\n");
        exampleSection.write(this, callableDecl, out);
    }

    _maybeAppendRequirements(toSlice("## Availability and Requirements\n\n"), requirements);

    auto& seeAlsoSection = funcDoc.sections[DocPageSection::SeeAlso];
    if (seeAlsoSection.ownedText.getLength() > 0)
    {
        out << toSlice("## See Also\n");
        seeAlsoSection.write(this, callableDecl, out);
    }
}

void DocMarkdownWriter::writeEnum(const ASTMarkup::Entry& entry, EnumDecl* enumDecl)
{
    auto& out = *m_builder;

    out << toSlice("# enum ");
    Name* name = enumDecl->getName();
    if (name)
    {
        out << name->text;
    }
    out << toSlice("\n\n");

    DeclDocumentation declDoc;
    declDoc.parse(entry.m_markup.getUnownedSlice());
    declDoc.writeDescription(out, this, enumDecl);
    registerCategory(m_currentPage, declDoc);

    out << toSlice("## Values \n\n");

    _appendAsBullets(_getAsNameAndTextList(enumDecl->getMembersOfType<EnumCaseDecl>()), false, '_');

    declDoc.writeSection(out, this, enumDecl, DocPageSection::Remarks);
    declDoc.writeSection(out, this, enumDecl, DocPageSection::Example);
    declDoc.writeSection(out, this, enumDecl, DocPageSection::SeeAlso);
}

void DocMarkdownWriter::_appendEscaped(const UnownedStringSlice& text)
{
    auto& out = *m_builder;

    const char* start = text.begin();
    const char* cur = start;
    const char* const end = text.end();

    for (; cur < end; ++cur)
    {
        const char c = *cur;

        switch (c)
        {
        case '<':
        case '>':
        case '&':
        case '"':
        case '_':
            {
                // Flush if any before
                if (cur > start)
                {
                    out.append(start, cur);
                }
                // Prefix with the
                out.appendChar('\\');

                // Start will still include the char, for later flushing
                start = cur;
                break;
            }
        default:
            break;
        }
    }

    // Flush any remaining
    if (cur > start)
    {
        out.append(start, cur);
    }
}


void DocMarkdownWriter::_appendDerivedFrom(
    const UnownedStringSlice& prefix,
    AggTypeDeclBase* aggTypeDecl)
{
    auto& out = *m_builder;

    List<InheritanceDecl*> inheritanceDecls;
    _getDecls(aggTypeDecl, inheritanceDecls);

    const Index count = inheritanceDecls.getCount();
    if (count)
    {
        out << prefix;
        for (Index i = 0; i < count; ++i)
        {
            InheritanceDecl* inheritanceDecl = inheritanceDecls[i];
            if (i > 0)
            {
                out << toSlice(", ");
            }
            out << escapeMarkdownText(inheritanceDecl->base->toString());
        }
    }
}

void DocMarkdownWriter::_appendAggTypeName(const ASTMarkup::Entry& entry, Decl* aggTypeDecl)
{
    SLANG_UNUSED(entry);

    auto& out = *m_builder;

#if 0
    // For extensions, try to see if the documentation defines a more readable title.
    if (as<ExtensionDecl>(aggTypeDecl))
    {
        auto trimStart = String(entry.m_markup.trimStart());
        if (trimStart.startsWith("@title"))
        {
            List<UnownedStringSlice> lines;
            StringUtil::calcLines(trimStart.getUnownedSlice(), lines);
            if (lines.getCount() > 0)
            {
                out << escapeMarkdownText(lines[0].tail(6).trim());

                // Remove @title directive from the description markup.
                StringBuilder restSB;
                for (Index i = 1; i < lines.getCount(); ++i)
                {
                    restSB << lines[i] << "\n";
                }
                entry.m_markup = restSB.produceString();
                return;
            }
        }
    }
#endif

    // This could be lots of different things - struct/class/extension/interface/..

    ASTPrinter printer(m_astBuilder);
    printer.addDeclPath(DeclRef<Decl>(aggTypeDecl));

    if (as<StructDecl>(aggTypeDecl))
    {
        out << toSlice("struct ") << escapeMarkdownText(printer.getStringBuilder().produceString());
    }
    else if (as<ClassDecl>(aggTypeDecl))
    {
        out << toSlice("class ") << escapeMarkdownText(printer.getStringBuilder().produceString());
    }
    else if (as<InterfaceDecl>(aggTypeDecl))
    {
        out << toSlice("interface ")
            << escapeMarkdownText(printer.getStringBuilder().produceString());
    }
    else if (ExtensionDecl* extensionDecl = as<ExtensionDecl>(aggTypeDecl))
    {
        out << toSlice("extension ") << escapeMarkdownText(extensionDecl->targetType->toString());
        _appendDerivedFrom(toSlice(" : "), extensionDecl);
    }
    else if (as<TypeDefDecl>(aggTypeDecl))
    {
        out << toSlice("typealias ")
            << escapeMarkdownText(printer.getStringBuilder().produceString());
    }
    else
    {
        out << toSlice("?");
    }
}

void DocMarkdownWriter::writeAggType(
    DocumentPage* page,
    const ASTMarkup::Entry& primaryEntry,
    AggTypeDeclBase* aggTypeDecl)
{
    auto& out = *m_builder;

    // We can write out he name using the printer
    out << toSlice("# ");
    _appendAggTypeName(primaryEntry, aggTypeDecl);
    out << toSlice("\n\n");
    List<ExtensionDecl*> conditionalConformanceExts;
    {
        List<InheritanceDecl*> inheritanceDecls;
        _getDecls<InheritanceDecl>(aggTypeDecl, inheritanceDecls);
        List<String> baseTypes;
        HashSet<String> conditionalBaseTypes;
        baseTypes = _getAsStringList(inheritanceDecls);
        for (auto entry : page->entries)
        {
            for (auto member : as<ContainerDecl>(entry->m_node)->members)
            {
                if (auto inheritanceDecl = as<InheritanceDecl>(member))
                {
                    if (auto extDecl = as<ExtensionDecl>(entry->m_node))
                    {
                        conditionalConformanceExts.add(extDecl);
                        conditionalBaseTypes.add(inheritanceDecl->base->toString());
                    }
                }
            }
        }
        if (baseTypes.getCount())
        {
            if (as<InterfaceDecl>(aggTypeDecl))
                out << "*Inherits from:* ";
            else
                out << "*Conforms to:* ";
            _appendCommaList(baseTypes, 0);
            out << toSlice("\n\n");
        }
        if (conditionalBaseTypes.getCount())
        {
            out << "*Conditionally conforms to:* ";
            List<String> list;
            for (auto t : conditionalBaseTypes)
                list.add(t);
            _appendCommaList(list, 0);
            out << toSlice("\n\n");
        }
    }

    DeclDocumentation declDoc;
    declDoc.parse(primaryEntry.m_markup.getUnownedSlice());
    declDoc.writeDescription(out, this, aggTypeDecl);
    registerCategory(page, declDoc);

    declDoc.writeGenericParameters(out, this, aggTypeDecl);

    {
        List<AssocTypeDecl*> assocTypeDecls;
        _getDecls<AssocTypeDecl>(aggTypeDecl, assocTypeDecls);

        if (assocTypeDecls.getCount())
        {
            out << toSlice("## Associated types\n\n");

            for (AssocTypeDecl* assocTypeDecl : assocTypeDecls)
            {
                out << "#### _" << escapeMarkdownText(assocTypeDecl->getName()->text) << "\n\n";

                // Look up markup
                ASTMarkup::Entry* assocTypeDeclEntry = m_markup->getEntry(assocTypeDecl);
                if (assocTypeDeclEntry)
                {
                    _appendAsSingleLine(assocTypeDeclEntry->m_markup.getUnownedSlice(), out);
                }

                List<TypeConstraintDecl*> inheritanceDecls;
                _getDecls<TypeConstraintDecl>(assocTypeDecl, inheritanceDecls);

                if (inheritanceDecls.getCount())
                {
                    out << toSlice("\n\nConstraints:\n\n");
                    for (auto inheritanceDecl : inheritanceDecls)
                    {
                        out << "  - ";
                        out << escapeMarkdownText(
                            getSub(m_astBuilder, inheritanceDecl)->toString());
                        out << " : ";
                        out << escapeMarkdownText(
                            getSup(m_astBuilder, inheritanceDecl)->toString());
                        out << toSlice("\n");
                    }
                }
            }
            out << toSlice("\n\n");
        }
    }

    {
        List<Decl*> fields;
        _getDeclsOfType<VarDecl>(this, page, fields);
        if (fields.getCount())
        {
            out << toSlice("## Fields\n\n");
            _appendAsBullets(_getAsNameAndTextList(fields), true, 0);
            out << toSlice("\n");
        }
    }

    {
        List<Decl*> properties;
        _getDeclsOfType<PropertyDecl>(this, page, properties);
        if (properties.getCount())
        {
            out << toSlice("## m_currentPage->path\n\n");
            _appendAsBullets(_getAsNameAndTextList(properties), true, 0);
            out << toSlice("\n");
        }
    }

    {
        List<Decl*> uniqueMethods;
        _getDeclsOfType<CallableDecl>(this, page, uniqueMethods);

        if (uniqueMethods.getCount())
        {
            // Put in source definition order
            uniqueMethods.sort(
                [](Decl* a, Decl* b) -> bool { return a->loc.getRaw() < b->loc.getRaw(); });

            out << "## Methods\n\n";
            _appendAsBullets(_getAsStringList(uniqueMethods), 0);
            out << toSlice("\n");
        }
    }

    if (conditionalConformanceExts.getCount())
    {
        out << "## Conditional Conformances\n\n";
        for (auto ext : conditionalConformanceExts)
        {
            for (auto member : ext->members)
            {
                auto inheritanceDecl = as<InheritanceDecl>(member);
                if (!inheritanceDecl)
                    continue;
                out << "### Conformance to ";
                out << escapeMarkdownText(inheritanceDecl->base.type->toString());
                out << "\n";
                StringBuilder sb;
                writeExtensionConditions(sb, ext, "\n", false);
                List<UnownedStringSlice> lines, nonEmptyLines;
                StringUtil::calcLines(sb.getUnownedSlice(), lines);
                for (auto line : lines)
                {
                    if (line.trim().getLength())
                        nonEmptyLines.add(line);
                }
                ASTPrinter printer(m_astBuilder);
                printer.addDeclPath(aggTypeDecl->getDefaultDeclRef());
                out << "`" << printer.getString() << "` additionally conforms to `";
                out << escapeMarkdownText(inheritanceDecl->base.type->toString());
                if (nonEmptyLines.getCount() != 0)
                {
                    out << "` when the following conditions are met:\n\n";
                    for (auto condition : nonEmptyLines)
                    {
                        out << "  * " << condition << "\n";
                    }
                }
                else
                {
                    out << "`.\n";
                }
            }
        }
    }
    declDoc.writeSection(out, this, aggTypeDecl, DocPageSection::Remarks);
    declDoc.writeSection(out, this, aggTypeDecl, DocPageSection::Example);
    declDoc.writeSection(out, this, aggTypeDecl, DocPageSection::SeeAlso);
}

String DocMarkdownWriter::escapeMarkdownText(String text)
{
    StringBuilder sb;
    for (auto c : text)
    {
        switch (c)
        {
        case '_':
        case '*':
        case '[':
        case ']':
        case '<':
        case '>':
        case '|':
        case '.':
        case '!':
        case '(':
        case ')':
            sb << '\\';
            sb.appendChar(c);
            break;
        default:
            sb.appendChar(c);
            break;
        }
    }
    return sb.produceString();
}

void DocMarkdownWriter::ensureDeclPageCreated(ASTMarkup::Entry& entry)
{
    auto page = getPage(as<Decl>(entry.m_node));
    page->entries.add(&entry);
}

Slang::Misc::Token treatLiteralsAsIdentifier(Slang::Misc::Token token)
{
    // If the token is a literal, we want to treat it as an identifier.
    if (token.Type == Slang::Misc::TokenType::StringLiteral)
    {
        token.Type = Slang::Misc::TokenType::Identifier;
        StringBuilder stringSB;
        StringEscapeUtil::appendQuoted(
            StringEscapeUtil::getHandler(StringEscapeUtil::Style::Cpp),
            token.Content.getUnownedSlice(),
            stringSB);
        token.Content = stringSB.produceString();
    }
    else if (
        token.Type == Slang::Misc::TokenType::IntLiteral ||
        token.Type == Slang::Misc::TokenType::DoubleLiteral)
    {
        token.Type = Slang::Misc::TokenType::Identifier;
    }
    return token;
}

String DocMarkdownWriter::translateToMarkdownWithLinks(String text, bool strictChildLookup)
{
    StringBuilder sb;
    List<DocumentPage*> currentPage;
    currentPage.add(m_currentPage);
    Slang::Misc::TokenReader reader(text);
    bool requireSpaceBeforeNextToken = false;
    bool isFirstToken = true;
    for (; !reader.IsEnd();)
    {
        auto token = treatLiteralsAsIdentifier(reader.ReadToken());


        if (token.Type == Slang::Misc::TokenType::Identifier)
        {
            if (requireSpaceBeforeNextToken)
                sb.append(' ');
            auto tokenContent = token.Content;
            if (tokenContent == "operator")
            {
                for (;;)
                {
                    auto operatorToken = reader.ReadToken();
                    tokenContent.append(operatorToken.Content);
                    if (operatorToken.Type != Slang::Misc::TokenType::LParent &&
                        operatorToken.Type != Slang::Misc::TokenType::LBracket)
                    {
                        break;
                    }
                }
            }
            String sectionName;
            Decl* referencedDecl = nullptr;
            auto page =
                findPageForToken(currentPage.getLast(), tokenContent, sectionName, referencedDecl);

            if (isFirstToken && strictChildLookup && page && page->parentPage != m_currentPage)
            {
                // If we are performing a strict child lookup (for displaying the member list of an
                // agg type), then we want to ignore any lookup results that refer to a different
                // parent page.
                page = nullptr;
            }

            if (page)
            {
                sb.append("[");
                sb << escapeMarkdownText(tokenContent.getUnownedSlice());
                sb.append("](");
                sb.append(Path::getPathWithoutExt(Path::getRelativePath(
                    Path::getParentDirectory(m_currentPage->path),
                    page->path)));
                if (sectionName.getLength())
                    sb << "#" << sectionName;
                sb.append(")");
                currentPage.getLast() = page;
                continue;
            }
            requireSpaceBeforeNextToken = true;
            isFirstToken = false;
        }
        else
        {
            switch (token.Type)
            {
            case Slang::Misc::TokenType::OpLess:
            case Slang::Misc::TokenType::OpGreater:
            case Slang::Misc::TokenType::Comma:
            case Slang::Misc::TokenType::Dot:
            case Slang::Misc::TokenType::IntLiteral:
            case Slang::Misc::TokenType::Semicolon:
                requireSpaceBeforeNextToken = false;
                break;
            default:
                requireSpaceBeforeNextToken = true;
                sb.appendChar(' ');
                break;
            }
        }
        // Maintain the `currentPage` stack so we can use the correct starting page
        // to lookup things like `Foo<int>.Bar`. When we look up `Bar`, we want to start
        // from the same page after looking up `Foo`, so we need to push the stack when we
        // see `<` and pop the stack when we see `>`.
        if (token.Type == Slang::Misc::TokenType::OpLess)
        {
            currentPage.add(currentPage.getLast());
        }
        else if (token.Type == Slang::Misc::TokenType::OpGreater)
        {
            if (currentPage.getCount() > 1)
                currentPage.removeLast();
        }
        sb << escapeMarkdownText(token.Content.getUnownedSlice());
        if (token.Type == Slang::Misc::TokenType::Comma)
            sb.appendChar(' ');
    }
    return sb.produceString();
}

// Implemented in slang-language-server-completion.cpp
bool isDeclKeyword(const UnownedStringSlice& slice);

bool isKeyword(const UnownedStringSlice& slice)
{
    if (isDeclKeyword(slice))
        return true;
    static const char* knownTypeNames[] =
        {"int", "float", "half", "double", "bool", "void", "uint"};
    for (auto typeName : knownTypeNames)
    {
        if (slice == typeName)
            return true;
    }
    return false;
}

String DocMarkdownWriter::translateToHTMLWithLinks(Decl* decl, String text)
{
    SLANG_UNUSED(decl);
    StringBuilder sb;
    List<DocumentPage*> currentPage;
    currentPage.add(m_currentPage);
    Slang::Misc::TokenReader reader(text);
    bool prevIsIdentifier = false;
    for (; !reader.IsEnd();)
    {
        auto token = treatLiteralsAsIdentifier(reader.ReadToken());

        if (token.Type == Slang::Misc::TokenType::Identifier)
        {
            if (prevIsIdentifier)
                sb.append(' ');
            String sectionName;
            Decl* referencedDecl = nullptr;
            auto page =
                findPageForToken(currentPage.getLast(), token.Content, sectionName, referencedDecl);
            if (page)
            {
                sb.append("<a href=\"");
                sb.append(Path::getPathWithoutExt(Path::getRelativePath(
                    Path::getParentDirectory(m_currentPage->path),
                    page->path)));
                sb.append(".html");
                if (sectionName.getLength())
                    sb << "#" << sectionName;
                sb.append("\"");
                if (isKeyword(token.Content.getUnownedSlice()))
                    sb.append(" class=\"code_keyword\"");
                else if (as<AggTypeDeclBase>(referencedDecl) || as<SimpleTypeDecl>(referencedDecl))
                    sb.append(" class=\"code_type\"");
                else if (as<ParamDecl>(referencedDecl))
                    sb.append(" class=\"code_param\"");
                else if (as<VarDeclBase>(referencedDecl) || as<EnumCaseDecl>(referencedDecl))
                    sb.append(" class=\"code_var\"");
                sb.append(">");
                escapeHTMLContent(sb, token.Content.getUnownedSlice());
                sb.append("</a>");
                currentPage.getLast() = page;
                continue;
            }
            prevIsIdentifier = true;
        }
        else
        {
            prevIsIdentifier = false;
        }
        // Maintain the `currentPage` stack so we can use the correct starting page
        // to lookup things like `Foo<int>.Bar`. When we look up `Bar`, we want to start
        // from the same page after looking up `Foo`, so we need to push the stack when we
        // see `<` and pop the stack when we see `>`.
        if (token.Type == Slang::Misc::TokenType::OpLess)
        {
            currentPage.add(m_currentPage);
        }
        else if (token.Type == Slang::Misc::TokenType::OpGreater)
        {
            if (currentPage.getCount() > 1)
                currentPage.removeLast();
        }
        bool shouldCloseSpan = false;
        if (isKeyword(token.Content.getUnownedSlice()))
        {
            sb.append("<span class=\"code_keyword\">");
            shouldCloseSpan = true;
        }
        escapeHTMLContent(sb, token.Content.getUnownedSlice());
        if (shouldCloseSpan)
            sb.append("</span>");
        if (token.Type == Slang::Misc::TokenType::Comma)
            sb.appendChar(' ');
    }
    return sb.produceString();
}

void DocMarkdownWriter::writePreamble()
{
    auto& out = *m_builder;
    if (out.getLength() == 0)
    {
        out << m_config.preamble;
        out << toSlice("\n");
    }
}

const char* getSectionTitle(DocPageSection section)
{
    switch (section)
    {
    case DocPageSection::Description:
        return "Description";
    case DocPageSection::Parameter:
        return "Parameters";
    case DocPageSection::ReturnInfo:
        return "Return value";
    case DocPageSection::Remarks:
        return "Remarks";
    case DocPageSection::Example:
        return "Example";
    case DocPageSection::SeeAlso:
        return "See also";
    default:
        return "";
    }
}

void DeclDocumentation::writeDescription(StringBuilder& out, DocMarkdownWriter* writer, Decl* decl)
{
    // Write all callout sections first.
    writeSection(out, writer, decl, DocPageSection::DeprecatedCallout);
    writeSection(out, writer, decl, DocPageSection::ExperimentalCallout);
    writeSection(out, writer, decl, DocPageSection::InternalCallout);

    // Write description section.
    writeSection(out, writer, decl, DocPageSection::Description);
}

void DeclDocumentation::writeGenericParameters(
    StringBuilder& out,
    DocMarkdownWriter* writer,
    Decl* decl)
{
    GenericDecl* genericDecl = as<GenericDecl>(decl->parentDecl);
    if (!genericDecl)
        return;

    // The parameters, in order
    List<Decl*> params;
    for (Decl* member : genericDecl->members)
    {
        if (as<GenericTypeParamDeclBase>(member) || as<GenericValueParamDecl>(member))
        {
            params.add(member);
        }
    }

    if (params.getCount())
    {
        out << toSlice("## Generic Parameters\n\n");
        auto paramList = writer->_getAsNameAndTextList(params);

        // Append names with constraints if any.
        for (Index i = 0; i < paramList.getCount(); i++)
        {
            auto param = params[i];
            if (paramList[i].text.getLength() == 0)
            {
                ParamDocumentation paramDoc;
                if (parameters.tryGetValue(getText(param->getName()), paramDoc))
                {
                    StringBuilder sb;
                    sb << paramDoc.description.ownedText;
                    paramList[i].text = sb.produceString();
                }
            }
        }
        writer->_appendAsBullets(paramList, false, 0);
        out << toSlice("\n");
    }
}

void DeclDocumentation::writeSection(
    StringBuilder& out,
    DocMarkdownWriter* writer,
    Decl* decl,
    DocPageSection section)
{
    SLANG_UNUSED(decl);
    ParsedDescription* sectionDoc = sections.tryGetValue(section);
    if (!sectionDoc)
        return;

    switch (section)
    {
    case DocPageSection::DeprecatedCallout:
        out << "> #### Deprecated Feature\n";
        out << "> The feature described in this page is marked as deprecated, and may be "
               "removed in a future release.\n";
        out << "> Users are advised to avoid using this feature, and to migrate to a newer "
               "alternative.\n";
        out << "\n";
        return;
    case DocPageSection::ExperimentalCallout:
        out << "> #### Experimental Feature\n";
        out << "> The feature described in this page is marked as experimental, and may be "
               "subject to change in future releases.\n";
        out << "> Users are advised that any code that depend on this feature may not be "
               "compilable by future versions of the compiler.\n";
        out << "\n";
        return;
    case DocPageSection::InternalCallout:
        out << "> #### Internal Feature\n";
        out << "> The feature described in this page is marked as an internal implementation "
               "detail, and is not intended for use by end-users.\n";
        out << "> Users are advised to avoid using this declaration directly, as it may be "
               "removed or changed in future releases.\n";
        out << "\n";
        return;
    }
    if (sectionDoc && sectionDoc->ownedText.getLength() > 0)
    {
        out << "## " << getSectionTitle(section) << "\n\n";
        sectionDoc->write(writer, decl, out);
    }
}

void DocMarkdownWriter::createPage(ASTMarkup::Entry& entry, Decl* decl)
{
    // Skip these they will be output as part of their respective 'containers'
    if (as<ParamDecl>(decl) || as<EnumCaseDecl>(decl) || as<AssocTypeDecl>(decl) ||
        as<TypeConstraintDecl>(decl) || as<ThisTypeDecl>(decl) || as<AccessorDecl>(decl))
    {
        return;
    }

    if (CallableDecl* callableDecl = as<CallableDecl>(decl))
    {
        if (_isFirstOverridden(callableDecl))
        {
            ensureDeclPageCreated(entry);
        }
    }
    else if (as<EnumDecl>(decl))
    {
        ensureDeclPageCreated(entry);
    }
    else if (as<AggTypeDeclBase>(decl))
    {
        ensureDeclPageCreated(entry);
    }
    else if (as<VarDecl>(decl))
    {
        // If part of aggregate type will be output there.
        ensureDeclPageCreated(entry);
    }
    else if (as<TypeDefDecl>(decl))
    {
        ensureDeclPageCreated(entry);
    }
    else if (as<PropertyDecl>(decl))
    {
        ensureDeclPageCreated(entry);
    }
    else if (as<AttributeDecl>(decl))
    {
        ensureDeclPageCreated(entry);
    }
    else if (as<GenericDecl>(decl))
    {
        // We can ignore as inner decls will be picked up, and written
    }
}

void DocMarkdownWriter::registerCategory(DocumentPage* page, DeclDocumentation& doc)
{
    if (doc.categoryText.getLength() != 0)
    {
        m_categories[doc.categoryName] = doc.categoryText;
    }
    else if (!m_categories.containsKey(doc.categoryName))
    {
        m_categories[doc.categoryName] = doc.categoryName;
    }
    page->category = doc.categoryName;
}


bool DocMarkdownWriter::isVisible(const Name* name)
{
    return name == nullptr || !name->text.startsWith(toSlice("__")) ||
           m_config.visibleDeclNames.contains(getText((Name*)name));
}

DocumentPage* DocMarkdownWriter::findPageForToken(
    DocumentPage* currentPage,
    String token,
    String& outSectionName,
    Decl*& outDecl)
{
    while (currentPage)
    {
        // Are there any children pages whose short title matches `token`?
        // If so, return the path of that page.
        if (currentPage->shortName == token)
        {
            outDecl = currentPage->decl;
            return currentPage;
        }
        if (auto rs = currentPage->findChildByShortName(token.getUnownedSlice()))
        {
            outDecl = rs->decl;
            return rs;
        }
        // Is `token` documented as a section on current page?
        // This will be the case for parameters and generic parameters.
        if (currentPage->decl)
        {
            for (auto entry : currentPage->entries)
            {
                auto containerDecl = as<ContainerDecl>(entry->m_node);
                if (!containerDecl)
                    continue;
                if (auto genericParent = as<GenericDecl>(containerDecl->parentDecl))
                {
                    for (auto member : genericParent->members)
                    {
                        if (getText(member->getName()) == token)
                        {
                            outDecl = member;
                            if (as<GenericTypeParamDeclBase>(member))
                                outSectionName = String("typeparam-") + token;
                            else if (as<GenericValueParamDecl>(member))
                                outSectionName = String("decl-") + token;
                            return currentPage;
                        }
                    }
                }
                for (auto member : containerDecl->members)
                {
                    if (as<ParamDecl>(member) || as<EnumCaseDecl>(member))
                    {
                        if (getText(member->getName()) == token)
                        {
                            outDecl = member;
                            outSectionName = String("decl-") + token;
                            return currentPage;
                        }
                    }
                }
            }
        }

        currentPage = currentPage->parentPage;
    }
    // Otherwise, try find in global decls.
    if (auto rs = m_typesPage->findChildByShortName(token.getUnownedSlice()))
    {
        outDecl = rs->decl;
        return rs;
    }
    if (auto rs = m_interfacesPage->findChildByShortName(token.getUnownedSlice()))
    {
        outDecl = rs->decl;
        return rs;
    }
    if (auto rs = m_globalDeclsPage->findChildByShortName(token.getUnownedSlice()))
    {
        outDecl = rs->decl;
        return rs;
    }
    return nullptr;
}

String DocMarkdownWriter::findLinkForToken(DocumentPage* currentPage, String token)
{
    String sectionName;
    Decl* decl = nullptr;
    if (auto page = findPageForToken(currentPage, token, sectionName, decl))
    {
        if (sectionName.getLength() == 0)
            return page->path;
        return page->path + "#" + sectionName;
    }
    return String();
}

bool DocMarkdownWriter::isVisible(const ASTMarkup::Entry& entry)
{
    // For now if it's not public it's not visible
    if (entry.m_visibility != MarkupVisibility::Public)
    {
        return false;
    }

    Decl* decl = as<Decl>(entry.m_node);
    return decl == nullptr || isVisible(decl);
}

bool DocMarkdownWriter::isVisible(Decl* decl)
{
    if (!isVisible(decl->getName()))
    {
        return false;
    }
    bool parentIsVisible = true;
    auto parent = decl;
    while (parent)
    {
        if (auto extDecl = as<ExtensionDecl>(parent))
        {
            if (auto targetDecl = isDeclRefTypeOf<Decl>(extDecl->targetType))
            {
                parentIsVisible = parentIsVisible && isVisible(targetDecl.getDecl());
                parent = targetDecl.getDecl();
            }
            else
            {
                parent = getParentDecl(parent);
            }
        }
        else
        {
            parent = getParentDecl(parent);
        }
        if (as<AggTypeDeclBase>(parent))
        {
            parentIsVisible = parentIsVisible && isVisible(parent);
        }
    }
    auto entry = m_markup->getEntry(decl);
    return parentIsVisible && (entry == nullptr || entry->m_visibility == MarkupVisibility::Public);
}

void DocumentationConfig::parse(UnownedStringSlice config)
{
    List<UnownedStringSlice> lines;
    StringUtil::calcLines(config, lines);
    Index ptr = 0;
    for (; ptr < lines.getCount(); ptr++)
    {
        auto line = lines[ptr];
        if (line.startsWith(toSlice("@preamble:")))
        {
            ptr++;
            StringBuilder preambleSB;
            for (; ptr < lines.getCount(); ptr++)
            {
                if (lines[ptr].startsWith("@end"))
                    break;
                preambleSB << lines[ptr] << "\n";
            }
            ptr++;
            preamble = preambleSB.produceString();
        }
        else if (line.startsWith(toSlice("@title:")))
        {
            title = line.tail(7).trim();
        }
        else if (line.startsWith(toSlice("@libname:")))
        {
            libName = line.tail(9).trim();
        }
        else if (line.startsWith(toSlice("@rootdir:")))
        {
            rootDir = line.tail(9).trim();
        }
        else if (line.startsWith("@includedecl:"))
        {
            ptr++;
            for (; ptr < lines.getCount(); ptr++)
            {
                if (lines[ptr].startsWith("@end"))
                    break;
                auto name = lines[ptr].trim();
                if (name.getLength())
                    visibleDeclNames.add(name);
            }
            ptr++;
        }
    }
}

void sortPages(DocumentPage* page)
{
    page->children.sort(
        [](DocumentPage* a, DocumentPage* b) -> bool { return a->shortName < b->shortName; });
}

void DocMarkdownWriter::generateSectionIndexPage(DocumentPage* page)
{
    // Generate the content for meta section index page.
    StringBuilder& sb = page->get();
    sb << m_config.preamble;
    sb << "# " << page->title;
    sb << "\n\n";
    sb << m_config.libName;
    sb << " defines the following " << String(page->title).toLower() << ":\n\n";
    sortPages(page);

    for (auto child : page->children)
    {
        sb << "- [" << escapeMarkdownText(child->shortName) << "]("
           << Path::getPathWithoutExt(
                  Path::getRelativePath(Path::getParentDirectory(page->path), child->path))
           << ")\n";
    }
}

DocumentPage* DocMarkdownWriter::writeAll(UnownedStringSlice configStr)
{
    m_config.parse(configStr);

    auto addBuiltinPage = [&](DocumentPage* parent,
                              UnownedStringSlice path,
                              UnownedStringSlice title,
                              UnownedStringSlice shortTitle)
    {
        RefPtr<DocumentPage> page = new DocumentPage();
        page->title = title;
        page->path = path;
        page->shortName = shortTitle;
        page->decl = nullptr;
        if (parent)
        {
            parent->children.add(page);
        }
        m_output[page->path] = page;
        return page.get();
    };
    m_rootPage = addBuiltinPage(
        nullptr,
        toSlice("index.md"),
        m_config.title.getUnownedSlice(),
        toSlice("Core Module Reference"));
    m_rootPage->skipWrite = true;

    m_interfacesPage = addBuiltinPage(
        m_rootPage.get(),
        toSlice("interfaces/index.md"),
        toSlice("Interfaces"),
        toSlice("Interfaces"));
    m_typesPage = addBuiltinPage(
        m_rootPage.get(),
        toSlice("types/index.md"),
        toSlice("Types"),
        toSlice("Types"));
    m_attributesPage = addBuiltinPage(
        m_rootPage.get(),
        toSlice("attributes/index.md"),
        toSlice("Attributes"),
        toSlice("Attributes"));
    m_globalDeclsPage = addBuiltinPage(
        m_rootPage.get(),
        toSlice("global-decls/index.md"),
        toSlice("Global Declarations"),
        toSlice("Global Declarations"));

    // In the first pass, we create all the pages so we can reference them
    // when writing the content.
    for (auto& entry : m_markup->getEntries())
    {
        Decl* decl = as<Decl>(entry.m_node);

        if (decl && isVisible(entry))
        {
            createPage(entry, decl);
        }
    }
    // In the second pass, actually writes the content to each page.
    writePageRecursive(m_rootPage.get());

    generateSectionIndexPage(m_interfacesPage);
    generateSectionIndexPage(m_typesPage);
    generateSectionIndexPage(m_attributesPage);
    generateSectionIndexPage(m_globalDeclsPage);

    return m_rootPage.get();
}

void DocMarkdownWriter::writePage(DocumentPage* page)
{
    if (page->skipWrite)
        return;
    if (page->entries.getCount() == 0)
        return;

    m_currentPage = page;
    m_builder = &(page->get());

    writePreamble();

    Decl* decl = (Decl*)page->getFirstEntry()->m_node;
    if (CallableDecl* callableDecl = as<CallableDecl>(decl))
    {
        writeCallableOverridable(page, *page->getFirstEntry(), callableDecl);
    }
    else if (EnumDecl* enumDecl = as<EnumDecl>(decl))
    {
        writeEnum(*page->getFirstEntry(), enumDecl);
    }
    else if (AggTypeDeclBase* aggTypeDeclBase = as<AggTypeDeclBase>(decl))
    {
        // Find the primary decl.
        ASTMarkup::Entry* primaryEntry = page->getFirstEntry();
        AggTypeDeclBase* primaryDecl = aggTypeDeclBase;
        for (auto entry : page->entries)
        {
            if (auto aggTypeDecl = as<AggTypeDecl>(entry->m_node))
            {
                primaryEntry = entry;
                primaryDecl = aggTypeDecl;
                break;
            }
        }
        writeAggType(page, *primaryEntry, primaryDecl);
    }
    else if (PropertyDecl* propertyDecl = as<PropertyDecl>(decl))
    {
        writeProperty(*page->getFirstEntry(), propertyDecl);
    }
    else if (VarDecl* varDecl = as<VarDecl>(decl))
    {
        writeVar(*page->getFirstEntry(), varDecl);
    }
    else if (TypeDefDecl* typeDefDecl = as<TypeDefDecl>(decl))
    {
        writeTypeDef(*page->getFirstEntry(), typeDefDecl);
    }
    else if (AttributeDecl* attributeDecl = as<AttributeDecl>(decl))
    {
        writeAttribute(*page->getFirstEntry(), attributeDecl);
    }
}

void DocMarkdownWriter::writePageRecursive(DocumentPage* page)
{
    writePage(page);
    for (auto child : page->children)
    {
        writePageRecursive(child);
    }
}

void writeTOCImpl(
    StringBuilder& sb,
    DocMarkdownWriter* writer,
    DocumentationConfig& config,
    DocumentPage* page);

void writeTOCChildren(
    StringBuilder& sb,
    DocMarkdownWriter* writer,
    DocumentationConfig& config,
    DocumentPage* page)
{
    if (page->children.getCount() == 0)
        return;

    sb << R"(<ul class="toc_list">)"
       << "\n";

    // Don't sort the root page.
    if (page->path != "index.md")
        sortPages(page);

    Dictionary<String, List<DocumentPage*>> categories;
    for (auto child : page->children)
    {
        categories[child->category].add(child);
    }
    List<String> categoryNames;
    for (auto& kv : categories)
    {
        categoryNames.add(kv.first);
    }
    categoryNames.sort();
    auto parentPath = Path::getParentDirectory(page->path);
    parentPath.append("/");

    // Create toctree for index pages
    if (page->path.endsWith("index.md"))
    {
        StringBuilder& tocSB = page->get();
        tocSB << "\n<!-- RTD-TOC-START\n";
        tocSB << "```{toctree}\n:titlesonly:\n:hidden:\n\n";

        // Add category landing pages to the toctree
        for (auto& cat : categoryNames)
        {
            // Skip non-categorized pages
            if (cat.getLength() == 0)
                continue;

            String landingPagePath = parentPath + cat + ".md";
            tocSB << getTocTreeEntry(writer->m_categories[cat], parentPath, landingPagePath);
        }

        // Add uncategorized pages to the toctree
        for (auto child : categories[""])
        {
            tocSB << getTocTreeEntry(child->shortName, parentPath, child->path);
        }
        tocSB << "```\n";
        tocSB << "RTD-TOC-END -->\n";
    }

    for (auto& cat : categoryNames)
    {
        // Skip non-categorized pages first.
        if (cat.getLength() == 0)
            continue;
        sb << "<li data-link=\"" << config.rootDir << parentPath << cat << "\"><span>";
        escapeHTMLContent(sb, writer->m_categories[cat].getUnownedSlice());
        sb << "</span>\n";
        sb << "<ul class=\"toc_list\">\n";
        for (auto child : categories[cat])
        {
            writeTOCImpl(sb, writer, config, child);
        }
        sb << "</ul>\n";
        sb << "</li>\n";

        // Create a landing page for the category.
        RefPtr<DocumentPage> landingPage = new DocumentPage();
        landingPage->title = writer->m_categories[cat];
        landingPage->path = parentPath + cat + ".md";
        landingPage->shortName = writer->m_categories[cat];
        landingPage->decl = nullptr;
        landingPage->parentPage = page;
        landingPage->contentSB << config.preamble;
        landingPage->contentSB << "# " << landingPage->title
                               << "\n\nThis category contains the following declarations:\n\n";
        for (auto child : categories[cat])
        {
            landingPage->contentSB
                << "#### [" << writer->escapeMarkdownText(child->title) << "]("
                << Path::getPathWithoutExt(
                       Path::getRelativePath(Path::getParentDirectory(page->path), child->path))
                << ")\n\n";
        }

        // Add the toctree for the category landing page.
        landingPage->contentSB << "\n<!-- RTD-TOC-START\n";
        landingPage->contentSB << "```{toctree}\n:titlesonly:\n:hidden:\n\n";
        for (auto child : categories[cat])
        {
            landingPage->contentSB << getTocTreeEntry(child->shortName, parentPath, child->path);
        }
        landingPage->contentSB << "```\n";
        landingPage->contentSB << "RTD-TOC-END -->\n";

        page->children.add(landingPage);
    }

    for (auto child : categories[""])
    {
        writeTOCImpl(sb, writer, config, child);
    }
    sb << "</ul>\n";
}

void writeTOCImpl(
    StringBuilder& sb,
    DocMarkdownWriter* writer,
    DocumentationConfig& config,
    DocumentPage* page)
{
    sb << R"(<li data-link=")" << getDocPath(config, page->path) << R"("><span>)";
    escapeHTMLContent(sb, page->shortName.getUnownedSlice());
    sb << "</span>\n";
    writeTOCChildren(sb, writer, config, page);
    sb << "</li>";
}

String DocMarkdownWriter::writeTOC()
{
    StringBuilder sb;
    sb << R"(<ul class="toc_root_list"><li data-link=")" << m_config.rootDir << R"(index"><span>)"
       << m_config.title << "</span>\n";
    writeTOCChildren(sb, this, m_config, m_rootPage);
    sb << "</li></ul>\n";
    return sb.produceString();
}

DocumentPage* DocMarkdownWriter::getPage(Decl* decl)
{
    auto path = _getDocFilePath(decl);
    RefPtr<DocumentPage> page;
    if (m_output.tryGetValue(path, page))
    {
        return page.get();
    }
    page = new DocumentPage();
    page->title = _getFullName(decl);
    page->path = path;
    page->shortName = _getName(decl);
    page->decl = decl;
    m_output[path] = page;

    // If parent page exists, add this page to it's children
    if (path.endsWith("index.md"))
        path = Path::getParentDirectory(path);
    auto parentPath = Path::getParentDirectory(path);
    parentPath = parentPath + "/index.md";
    RefPtr<DocumentPage> parentPage;
    if (m_output.tryGetValue(parentPath, parentPage))
    {
        parentPage->children.add(page);
        page->parentPage = parentPage.get();
    }
    return page.get();
}

StringBuilder* DocMarkdownWriter::getBuilder(Decl* decl)
{
    m_currentPage = getPage(decl);
    return &(m_currentPage->get());
}

void writePageToDisk(DocumentPage* page)
{
    if (!page->skipWrite)
    {
        auto dir = Path::getParentDirectory(page->path);
        if (dir.getLength())
            Path::createDirectoryRecursive(dir);
        File::writeAllText(page->path, page->contentSB.toString());
    }
    for (auto child : page->children)
    {
        writePageToDisk(child);
    }
}

void DocumentPage::writeToDisk()
{
    writePageToDisk(this);
}

struct DocumentationStats
{
    int documentedPageCount = 0;
    List<String> undocumentedPages;
};

static void _collectStats(DocumentationStats& stats, DocumentPage* page)
{
    if (!page->skipWrite && page->entries.getCount() != 0)
    {
        bool isDocumented = false;
        for (auto entry : page->entries)
        {
            if (entry->m_markup.getUnownedSlice().trim().getLength() > 0)
            {
                DeclDocumentation doc;
                doc.parse(entry->m_markup.getUnownedSlice());
                auto desc = doc.sections.tryGetValue(DocPageSection::Description);
                // A page is considered documented if it has a description section
                // with more than 10 characters and ends with a `.`.
                if (desc && desc->ownedText.trim().getLength() > 10 &&
                    String(desc->ownedText.trim()).endsWith("."))
                {
                    isDocumented = true;
                }
                break;
            }
        }
        if (isDocumented)
        {
            stats.documentedPageCount++;
        }
        else
        {
            stats.undocumentedPages.add(page->title);
        }
    }
    for (auto child : page->children)
    {
        _collectStats(stats, child);
    }
}

void DocumentPage::writeSummary(UnownedStringSlice fileName)
{
    DocumentationStats stats;
    _collectStats(stats, this);
    StringBuilder sb;
    sb << "documented pages: " << stats.documentedPageCount << "\n";
    sb << "undocumented pages: " << stats.undocumentedPages.getCount() << "("
       << String(
              stats.undocumentedPages.getCount() /
                  (float)(stats.documentedPageCount + stats.undocumentedPages.getCount()) * 100,
              "%.1f")
       << "%)\n\n";
    for (auto page : stats.undocumentedPages)
        sb << page << "\n";
    File::writeAllText(fileName, sb.produceString());
}

DocumentPage* DocumentPage::findChildByShortName(const UnownedStringSlice& name)
{
    for (auto child : children)
    {
        if (child->shortName == name)
            return child;
    }
    return nullptr;
}


} // namespace Slang
