// debug-swap-chain.h
#pragma once
#include "debug-base.h"

namespace gfx
{
using namespace Slang;

namespace debug
{

class DebugSwapchain : public DebugObject<ISwapchain>
{
public:
    SLANG_COM_OBJECT_IUNKNOWN_ALL;

public:
    ISwapchain* getInterface(const Slang::Guid& guid);
    virtual SLANG_NO_THROW const Desc& SLANG_MCALL getDesc() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL
    getImage(GfxIndex index, ITextureResource** outResource) override;
    virtual SLANG_NO_THROW Result SLANG_MCALL present() override;
    virtual SLANG_NO_THROW int SLANG_MCALL acquireNextImage() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL resize(GfxCount width, GfxCount height) override;
    virtual SLANG_NO_THROW bool SLANG_MCALL isOccluded() override;
    virtual SLANG_NO_THROW Result SLANG_MCALL setFullScreenMode(bool mode) override;

public:
    Slang::RefPtr<DebugCommandQueue> queue;
    Desc desc;

private:
    Slang::List<Slang::RefPtr<DebugTextureResource>> m_images;
    void maybeRebuildImageList();
};

} // namespace debug
} // namespace gfx
