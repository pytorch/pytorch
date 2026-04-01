#ifndef SLANG_INTERNAL_H_INCLUDED
#define SLANG_INTERNAL_H_INCLUDED

#include "slang.h"

namespace Slang
{
struct GlobalSessionInternalDesc
{
    bool isBootstrap = false;
};
} // namespace Slang

SLANG_API SlangResult slang_createGlobalSessionImpl(
    const SlangGlobalSessionDesc* desc,
    const Slang::GlobalSessionInternalDesc* internalDesc,
    slang::IGlobalSession** outGlobalSession);

#endif
