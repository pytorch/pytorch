// window.h
#pragma once

#include "core/slang-basic.h"
#include "core/slang-func-ptr.h"
#include "platform-api.h"
#include "slang-com-ptr.h"

namespace platform
{

enum class KeyCode : uint32_t
{
    None = 0,
    Left = 0x25,
    Up = 0x26,
    Down = 0x28,
    Right = 0x27,
    Escape = 0x1B,
    Return = 0x0D,
    Space = 0x20,
    Shift = 0x10,
    Ctrl = 0x11,
    Alt = 0x12,
    Backspace = 0x08,
    Delete = 0x2E,
    Home = 0x24,
    End = 0x23,
    PageUp = 0x21,
    PageDown = 0x22,
    Insert = 0x2D,
    Tab = 0x09,
    A = 0x41,
    B = 0x42,
    C = 0x43,
    D = 0x44,
    E = 0x45,
    F = 0x46,
    G = 0x47,
    H = 0x48,
    I = 0x49,
    J = 0x4A,
    K = 0x4B,
    L = 0x4C,
    M = 0x4D,
    N = 0x4E,
    O = 0x4F,
    P = 0x50,
    Q = 0x51,
    R = 0x52,
    S = 0x53,
    T = 0x54,
    U = 0x55,
    V = 0x56,
    W = 0x57,
    X = 0x58,
    Y = 0x59,
    Z = 0x5A,
    Semicolon = 0xBA,
    Comma = 0xBC,
    Dot = 0xBE,
    Slash = 0xBF,
    Quote = 0xDE,
    LBracket = 0xDB,
    RBracket = 0xDD,
    Backslash = 0xDC,
    Minus = 0xBD,
    Plus = 0xBB,
    Tilde = 0xC0,
    Key0 = 0x30,
    Key1 = 0x31,
    Key2 = 0x32,
    Key3 = 0x33,
    Key4 = 0x34,
    Key5 = 0x35,
    Key6 = 0x36,
    Key7 = 0x37,
    Key8 = 0x38,
    Key9 = 0x39,
    F1 = 0x70,
    F2 = 0x71,
    F3 = 0x72,
    F4 = 0x73,
    F5 = 0x74,
    F6 = 0x75,
    F7 = 0x76,
    F8 = 0x77,
    F9 = 0x78,
    F10 = 0x79,
    F11 = 0x7A,
    F12 = 0x7B,
};

struct WindowHandle
{
    enum class Type
    {
        Unknown,
        Win32Handle,
        NSWindowHandle,
        XLibHandle,
    };
    Type type;
    intptr_t handleValues[2];
    static WindowHandle fromHwnd(void* hwnd)
    {
        WindowHandle handle = {};
        handle.type = WindowHandle::Type::Win32Handle;
        handle.handleValues[0] = (intptr_t)(hwnd);
        return handle;
    }
    static WindowHandle fromNSWindow(void* nswindow)
    {
        WindowHandle handle = {};
        handle.type = WindowHandle::Type::NSWindowHandle;
        handle.handleValues[0] = (intptr_t)(nswindow);
        return handle;
    }
    static WindowHandle fromXWindow(void* xdisplay, uint32_t xwindow)
    {
        WindowHandle handle = {};
        handle.type = WindowHandle::Type::XLibHandle;
        handle.handleValues[0] = (intptr_t)(xdisplay);
        handle.handleValues[1] = xwindow;
        return handle;
    }
    template<typename T>
    T convert()
    {
        T result;
        result.type = (decltype(result.type))type;
        result.handleValues[0] = handleValues[0];
        result.handleValues[1] = handleValues[1];
        return result;
    }
};

struct ButtonState
{
    enum Enum
    {
        None = 0,
        LeftButton = 1,
        RightButton = 2,
        MiddleButton = 4,
        Shift = 8,
        Control = 16,
        Alt = 32
    };
};

struct KeyEventArgs
{
    KeyCode key;
    wchar_t keyChar; // For KeyPress event
    ButtonState::Enum buttons;
    bool cancelEvent;
};

struct MouseEventArgs
{
    int x, y;
    int delta;
    ButtonState::Enum buttons;
};

struct Rect
{
    int x, y;
    int width, height;
};

enum class WindowStyle
{
    Default,
    FixedSize,
};

struct WindowDesc
{
    char const* title = nullptr;
    int width = 0;
    int height = 0;
    WindowStyle style = WindowStyle::Default;
};

class Window : public Slang::RefObject
{
public:
    struct Events
    {
        Slang::Action<> mainLoop;
        Slang::Action<> sizeChanged;
        Slang::Action<> focus;
        Slang::Action<> lostFocus;
        Slang::Action<KeyEventArgs&> keyDown;
        Slang::Action<KeyEventArgs&> keyUp;
        Slang::Action<KeyEventArgs&> keyPress;
        Slang::Action<MouseEventArgs> mouseMove;
        Slang::Action<MouseEventArgs> mouseUp;
        Slang::Action<MouseEventArgs> mouseDown;
        Slang::Action<MouseEventArgs> mouseWheel;
    };

    Events events;

    virtual void setClientSize(uint32_t width, uint32_t height) = 0;
    virtual Rect getClientRect() = 0;
    virtual void centerScreen() = 0;
    virtual void close() = 0;
    virtual bool getFocused() = 0;
    virtual bool getVisible() = 0;
    virtual WindowHandle getNativeHandle() = 0;
    virtual void setText(Slang::String text) = 0;
    virtual void show() = 0;
    virtual void hide() = 0;
    virtual int getCurrentDpi() = 0;
};

class Application
{
public:
    static SLANG_PLATFORM_API Window* createWindow(const WindowDesc& desc);
    static SLANG_PLATFORM_API void init();
    static SLANG_PLATFORM_API void run(Window* mainWindow, bool waitForEvents = false);
    static SLANG_PLATFORM_API void quit();
    static SLANG_PLATFORM_API void doEvents();
    static SLANG_PLATFORM_API void dispose();
};

} // namespace platform

#ifdef _WIN32

#ifdef _MSC_VER
#ifdef _DEBUG
#define GFX_DUMP_LEAK _CrtDumpMemoryLeaks();
#endif
#endif

#endif

#ifndef GFX_DUMP_LEAK
#define GFX_DUMP_LEAK
#endif

#define PLATFORM_UI_MAIN(APPLICATION_ENTRY)      \
    int exampleMain(int argc, char** argv)       \
    {                                            \
        platform::Application::init();           \
        auto rs = APPLICATION_ENTRY(argc, argv); \
        platform::Application::dispose();        \
        GFX_DUMP_LEAK                            \
        return rs;                               \
    }
