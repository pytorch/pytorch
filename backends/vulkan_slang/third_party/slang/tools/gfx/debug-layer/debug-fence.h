// debug-fence.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugFence : public DebugObject<IFence>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;
    IFence* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW Result SLANG_MCALL getCurrentValue(uint64_t* outValue) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL setCurrentValue(uint64_t value) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL getSharedHandle(InteropHandle* outHandle) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeHandle(InteropHandle* outNativeHandle) override;

public:
    uint64_t maxValueToSignal = 0;
};

} // namespace debug
} // namespace gfx
