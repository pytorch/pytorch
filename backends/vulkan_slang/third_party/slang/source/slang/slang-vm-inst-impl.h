#ifndef SLANG_VM_INST_IMPL_H
#define SLANG_VM_INST_IMPL_H

#include "slang-vm-bytecode.h"

namespace Slang
{

slang::VMExtFunction mapInstToFunction(
    VMInstHeader* instHeader,
    VMModuleView* module,
    Dictionary<String, slang::VMExtFunction>& extInstHandlers);

} // namespace Slang

#endif
