#pragma once

#include "../core/slang-basic.h"
#include "slang-ast-all.h"
#include "slang-compiler.h"
#include "slang-syntax.h"
#include "slang-workspace-version.h"
#include "slang.h"

namespace Slang
{
List<LanguageServerProtocol::DocumentSymbol> getDocumentSymbols(
    Linkage* linkage,
    Module* module,
    UnownedStringSlice fileName,
    DocumentVersion* doc);
} // namespace Slang
