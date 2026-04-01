#pragma once

#include "../core/slang-basic.h"
#include "slang-ast-all.h"
#include "slang-compiler.h"
#include "slang-syntax.h"
#include "slang-workspace-version.h"
#include "slang.h"

namespace Slang
{

struct InlayHintOptions
{
    bool showDeducedType = false;
    bool showParameterNames = false;
};

List<LanguageServerProtocol::InlayHint> getInlayHints(
    Linkage* linkage,
    Module* module,
    UnownedStringSlice fileName,
    DocumentVersion* doc,
    LanguageServerProtocol::Range range,
    const InlayHintOptions& options);
} // namespace Slang
