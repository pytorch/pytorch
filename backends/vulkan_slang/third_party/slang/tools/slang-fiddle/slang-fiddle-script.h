// slang-fiddle-script.h
#pragma once

#include "../external/lua/lapi.h"
#include "../external/lua/lauxlib.h"
#include "compiler-core/slang-source-loc.h"
#include "core/slang-list.h"
#include "core/slang-string.h"
#include "slang-fiddle-diagnostics.h"
#include "slang-fiddle-scrape.h"

namespace fiddle
{
using namespace Slang;

lua_State* getLuaState();

String evaluateScriptCode(String originalFileName, String scriptSource, DiagnosticSink* sink);
} // namespace fiddle
