#if !defined(_WIN32) && !defined(__APPLE__) && !defined(SLANG_ENABLE_XLIB)

#include "../window.h"

using namespace Slang;

namespace platform
{

void Application::init() {}

void Application::doEvents() {}

void Application::quit() {}

void Application::dispose() {}

void Application::run(Window* mainWindow, bool waitForEvents)
{
    SLANG_UNUSED(mainWindow);
    SLANG_UNUSED(waitForEvents);
}

Window* Application::createWindow(const WindowDesc& desc)
{
    return nullptr;
}


} // namespace platform

#endif
