// slang-ir-obfuscate-loc.h
#ifndef SLANG_IR_OBFUSCATE_LOC_H_INCLUDED
#define SLANG_IR_OBFUSCATE_LOC_H_INCLUDED

#include "../compiler-core/slang-source-map.h"
#include "../core/slang-basic.h"
#include "slang-compiler.h"
#include "slang-ir.h"

namespace Slang
{

/*** Obfuscate locs in module. Store the mapping from obfuscated locs to actual locs in the form of
 * a source map */
SlangResult obfuscateModuleLocs(IRModule* module, SourceManager* sourceManager);

} // namespace Slang

#endif
