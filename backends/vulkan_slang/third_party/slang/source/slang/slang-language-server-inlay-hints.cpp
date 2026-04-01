#include "slang-language-server-inlay-hints.h"

#include "../core/slang-char-util.h"
#include "slang-ast-iterator.h"
#include "slang-ast-support-types.h"
#include "slang-language-server.h"
#include "slang-visitor.h"

namespace Slang
{
List<LanguageServerProtocol::InlayHint> getInlayHints(
    Linkage* linkage,
    Module* module,
    UnownedStringSlice fileName,
    DocumentVersion* doc,
    LanguageServerProtocol::Range range,
    const InlayHintOptions& options)
{
    List<LanguageServerProtocol::InlayHint> result;
    auto manager = linkage->getSourceManager();
    auto docText = doc->getText().getUnownedSlice();
    iterateASTWithLanguageServerFilter(
        fileName,
        manager,
        module->getModuleDecl(),
        [&](SyntaxNode* node)
        {
            if (auto invokeExpr = as<InvokeExpr>(node))
            {
                if (!options.showParameterNames)
                    return;
                auto humaneLoc = manager->getHumaneLoc(node->loc);
                if (humaneLoc.line - 1 < range.start.line || humaneLoc.line - 1 > range.end.line)
                    return;
                if (humaneLoc.pathInfo.foundPath != fileName)
                    return;
                auto funcExpr = as<DeclRefExpr>(invokeExpr->functionExpr);
                if (!funcExpr)
                    return;
                if (as<ConstructorDecl>(funcExpr->declRef.getDecl()))
                    return;
                auto callable = as<CallableDecl>(funcExpr->declRef.getDecl());
                if (!callable)
                    return;
                auto params = callable->getParameters();
                Index i = 0;
                for (auto param : params)
                {
                    if (i >= invokeExpr->argumentDelimeterLocs.getCount() - 1)
                        break;
                    if (auto name = param->getName())
                    {
                        LanguageServerProtocol::InlayHint hint;
                        auto loc = manager->getHumaneLoc(invokeExpr->argumentDelimeterLocs[i]);
                        auto offset = doc->getOffset(loc.line, loc.column);
                        offset++;
                        while (offset < docText.getLength() &&
                               CharUtil::isWhitespace(docText[offset]))
                            offset++;
                        Index posLine, posCol;
                        doc->offsetToLineCol(offset, posLine, posCol);
                        Index utf16line, utf16col;
                        doc->oneBasedUTF8LocToZeroBasedUTF16Loc(
                            posLine,
                            posCol,
                            utf16line,
                            utf16col);
                        hint.position.line = (int)utf16line;
                        hint.position.character = (int)utf16col;
                        hint.paddingLeft = false;
                        hint.kind = LanguageServerProtocol::kInlayHintKindParameter;
                        StringBuilder lblSb;
                        if (param->hasModifier<OutModifier>())
                            lblSb << "out ";
                        else if (param->hasModifier<InOutModifier>())
                            lblSb << "inout ";
                        else if (param->hasModifier<RefModifier>())
                            lblSb << "ref ";
                        else if (param->hasModifier<ConstRefModifier>())
                            lblSb << "constref ";
                        lblSb << name->text;
                        lblSb << ":";
                        hint.label = lblSb.produceString();
                        result.add(hint);
                    }
                    i++;
                }
            }
            else if (auto varDecl = as<VarDeclBase>(node))
            {
                if (!options.showDeducedType)
                    return;
                auto humaneLoc = manager->getHumaneLoc(node->loc);
                if (humaneLoc.line - 1 < range.start.line || humaneLoc.line - 1 > range.end.line)
                    return;
                if (humaneLoc.pathInfo.foundPath != fileName)
                    return;
                if (varDecl->type.exp)
                    return;
                if (!varDecl->type.type)
                    return;
                if (as<ErrorType>(varDecl->type.type))
                    return;
                if (!varDecl->getName())
                    return;

                LanguageServerProtocol::InlayHint hint;
                auto loc = manager->getHumaneLoc(varDecl->nameAndLoc.loc);
                auto offset = doc->getOffset(loc.line, loc.column);
                offset++;
                while (offset < docText.getLength() && _isIdentifierChar(docText[offset]))
                    offset++;
                Index posLine, posCol;
                doc->offsetToLineCol(offset, posLine, posCol);
                Index utf16line, utf16col;
                doc->oneBasedUTF8LocToZeroBasedUTF16Loc(posLine, posCol, utf16line, utf16col);
                hint.position.line = (int)utf16line;
                hint.position.character = (int)utf16col;
                hint.kind = LanguageServerProtocol::kInlayHintKindType;
                StringBuilder lblSb;
                lblSb << ": " << varDecl->type.type->toString();
                hint.label = lblSb.produceString();

                LanguageServerProtocol::TextEdit edit;
                edit.range.start = hint.position;
                edit.range.end = hint.position;
                edit.newText = " " + hint.label;
                hint.textEdits.add(edit);
                result.add(hint);
            }
        });
    return result;
}

} // namespace Slang
