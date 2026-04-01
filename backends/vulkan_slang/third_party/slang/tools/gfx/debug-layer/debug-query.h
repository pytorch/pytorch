// debug-query.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugQueryPool : public DebugObject<IQueryPool>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

    IQueryPool::Desc desc;

public:
    IQueryPool* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getResult(GfxIndex index, GfxCount count, uint64_t* data) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL reset() override;
};

} // namespace debug
} // namespace gfx
