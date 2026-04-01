// slang-doc-ast.cpp
#include "slang-doc-ast.h"

#include "../core/slang-string-util.h"
#include "slang-ast-support-types.h"
// #include "slang-ast-builder.h"
// #include "slang-ast-print.h"

namespace Slang
{

/* static */ DocMarkupExtractor::SearchStyle ASTMarkupUtil::getSearchStyle(Decl* decl)
{
    typedef Extractor::SearchStyle SearchStyle;

    if (const auto enumCaseDecl = as<EnumCaseDecl>(decl))
    {
        return SearchStyle::EnumCase;
    }
    if (const auto paramDecl = as<ParamDecl>(decl))
    {
        return SearchStyle::Param;
    }
    else if (const auto callableDecl = as<CallableDecl>(decl))
    {
        return SearchStyle::Function;
    }
    else if (as<VarDecl>(decl) || as<TypeDefDecl>(decl) || as<AssocTypeDecl>(decl))
    {
        return SearchStyle::Variable;
    }
    else if (auto genericDecl = as<GenericDecl>(decl))
    {
        return getSearchStyle(genericDecl->inner);
    }
    else if (as<GenericTypeParamDecl>(decl) || as<GenericValueParamDecl>(decl))
    {
        return SearchStyle::GenericParam;
    }
    else if (as<AttributeDecl>(decl))
    {
        return SearchStyle::Attribute;
    }
    else
    {
        // If can't determine just allow before
        return SearchStyle::Before;
    }
}

bool shouldDocumentDecl(Decl* decl)
{
    return !getText(decl->getName()).startsWith("$__syn") &&
           !decl->hasModifier<SynthesizedModifier>();
}

static void _addDeclRec(Decl* decl, List<Decl*>& outDecls)
{
    if (decl == nullptr || !shouldDocumentDecl(decl))
    {
        return;
    }

    // If we don't have a loc, we have no way of locating documentation.
    if (decl->loc.isValid() || decl->nameAndLoc.loc.isValid())
    {
        outDecls.add(decl);
    }

    if (GenericDecl* genericDecl = as<GenericDecl>(decl))
    {
        _addDeclRec(genericDecl->inner, outDecls);
    }

    if (ContainerDecl* containerDecl = as<ContainerDecl>(decl))
    {
        // Add the container - which could be a class, struct, enum, namespace, extension, generic
        // etc. Now add what the container contains
        for (Decl* childDecl : containerDecl->members)
        {
            _addDeclRec(childDecl, outDecls);
        }
    }
}

/* static */ void ASTMarkupUtil::findDecls(ModuleDecl* moduleDecl, List<Decl*>& outDecls)
{
    for (Decl* decl : moduleDecl->members)
    {
        _addDeclRec(decl, outDecls);
    }
}

SlangResult ASTMarkupUtil::extract(
    ModuleDecl* moduleDecl,
    SourceManager* sourceManager,
    DiagnosticSink* sink,
    ASTMarkup* outDoc,
    bool searchOrindaryComments)
{
    List<Decl*> decls;
    findDecls(moduleDecl, decls);

    const Index declsCount = decls.getCount();

    List<Extractor::SearchItemInput> inputItems;
    List<Extractor::SearchItemOutput> outItems;

    {
        inputItems.setCount(declsCount);

        for (Index i = 0; i < declsCount; ++i)
        {
            Decl* decl = decls[i];
            auto& item = inputItems[i];

            item.sourceLoc = decl->loc.isValid() ? decl->loc : decl->nameAndLoc.loc;
            // Has to be valid to be lookupable
            SLANG_ASSERT(item.sourceLoc.isValid());

            item.searchStyle = getSearchStyle(decl);

            // Don't generate documentation for synthesized members.
            if (!shouldDocumentDecl(decl))
                item.searchStyle = DocMarkupExtractor::SearchStyle::None;
        }

        DocMarkupExtractor extractor;
        extractor.setSearchInOrdinaryComments(searchOrindaryComments);

        List<SourceView*> views;
        SLANG_RETURN_ON_FAIL(
            extractor
                .extract(inputItems.getBuffer(), declsCount, sourceManager, sink, views, outItems));
    }

    // Set back
    for (Index i = 0; i < declsCount; ++i)
    {
        const auto& outputItem = outItems[i];
        const auto& inputItem = inputItems[outputItem.inputIndex];

        // If we don't know how to search add to the output
        if (inputItem.searchStyle != Extractor::SearchStyle::None)
        {
            Decl* decl = decls[outputItem.inputIndex];

            // Add to the documentation
            ASTMarkup::Entry& docEntry = outDoc->addEntry(decl);
            docEntry.m_markup = outputItem.text;
            docEntry.m_visibility = outputItem.visibilty;
        }
    }

    return SLANG_OK;
}

} // namespace Slang
