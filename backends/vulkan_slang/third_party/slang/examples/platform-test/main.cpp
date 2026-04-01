#include "examples/example-base/example-base.h"
#include "platform/window.h"
#include "slang.h"

using namespace gfx;
using namespace Slang;

struct PlatformTest : public WindowedAppBase
{

    void onSizeChanged() { printf("onSizeChanged\n"); }

    void onFocus() { printf("onFocus\n"); }

    void onLostFocus() { printf("onLostFocus\n"); }

    void onKeyDown(platform::KeyEventArgs args)
    {
        printf("onKeyDown(key=0x%02x, buttons=0x%02x)\n", (uint32_t)args.key, args.buttons);
    }

    void onKeyUp(platform::KeyEventArgs args)
    {
        printf("okKeyUp(key=0x%02x, buttons=0x%02x)\n", (uint32_t)args.key, args.buttons);
    }

    void onKeyPress(platform::KeyEventArgs args)
    {
        printf("onKeyPress(keyChar=0x%02x)\n", args.keyChar);
    }

    void onMouseMove(platform::MouseEventArgs args)
    {
        printf(
            "onMouseMove(x=%d, y=%d, delta=%d, buttons=0x%02x\n",
            args.x,
            args.y,
            args.delta,
            args.buttons);
    }

    void onMouseDown(platform::MouseEventArgs args)
    {
        printf(
            "onMouseDown(x=%d, y=%d, delta=%d, buttons=0x%02x\n",
            args.x,
            args.y,
            args.delta,
            args.buttons);
    }

    void onMouseUp(platform::MouseEventArgs args)
    {
        printf(
            "onMouseUp(x=%d, y=%d, delta=%d, buttons=0x%02x\n",
            args.x,
            args.y,
            args.delta,
            args.buttons);
    }

    void onMouseWheel(platform::MouseEventArgs args)
    {
        printf(
            "onMouseWheel(x=%d, y=%d, delta=%d, buttons=0x%02x\n",
            args.x,
            args.y,
            args.delta,
            args.buttons);
    }

    Slang::Result initialize()
    {
        SLANG_RETURN_ON_FAIL(initializeBase("platform-test", 1024, 768));

        // We may not have a window if we're running in test mode
        SLANG_ASSERT(isTestMode() || gWindow);
        if (gWindow)
        {
            gWindow->events.sizeChanged = [this]() { onSizeChanged(); };
            gWindow->events.focus = [this]() { onFocus(); };
            gWindow->events.lostFocus = [this]() { onLostFocus(); };
            gWindow->events.keyDown = [this](const platform::KeyEventArgs& e) { onKeyDown(e); };
            gWindow->events.keyUp = [this](const platform::KeyEventArgs& e) { onKeyUp(e); };
            gWindow->events.keyPress = [this](const platform::KeyEventArgs& e) { onKeyPress(e); };
            gWindow->events.mouseMove = [this](const platform::MouseEventArgs& e)
            { onMouseMove(e); };
            gWindow->events.mouseDown = [this](const platform::MouseEventArgs& e)
            { onMouseDown(e); };
            gWindow->events.mouseUp = [this](const platform::MouseEventArgs& e) { onMouseUp(e); };
            gWindow->events.mouseWheel = [this](const platform::MouseEventArgs& e)
            { onMouseWheel(e); };
        }

        return SLANG_OK;
    }

    virtual void renderFrame(int frameBufferIndex) override
    {
        ComPtr<ICommandBuffer> commandBuffer =
            gTransientHeaps[frameBufferIndex]->createCommandBuffer();

        auto renderEncoder =
            commandBuffer->encodeRenderCommands(gRenderPass, gFramebuffers[frameBufferIndex]);

        gfx::Viewport viewport = {};
        viewport.maxZ = 1.0f;
        viewport.extentX = (float)windowWidth;
        viewport.extentY = (float)windowHeight;
        renderEncoder->setViewportAndScissor(viewport);

        renderEncoder->endEncoding();
        commandBuffer->close();
        gQueue->executeCommandBuffer(commandBuffer);

        // We may not have a swapchain if we're running in test mode
        SLANG_ASSERT(isTestMode() || gSwapchain);
        if (gSwapchain)
            gSwapchain->present();
    }
};

// This macro instantiates an appropriate main function to
// run the application defined above.
EXAMPLE_MAIN(innerMain<PlatformTest>);
