// debug-resource-views.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugResourceView : public DebugObject<IResourceView>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    IResourceView* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW Desc* SLANG_MCALL getViewDesc() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeHandle(InteropHandle* outNativeHandle) override;
};

class DebugAccelerationStructure : public DebugObject<IAccelerationStructure>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    IAccelerationStructure* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW DeviceAddress SLANG_MCALL getDeviceAddress() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getNativeHandle(InteropHandle* outNativeHandle) override;
    virtual SLANG_NO_THROW Desc* SLANG_MCALL getViewDesc() override;
};

} // namespace debug
} // namespace gfx
