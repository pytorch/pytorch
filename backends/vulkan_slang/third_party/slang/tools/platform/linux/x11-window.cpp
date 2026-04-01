#ifdef SLANG_ENABLE_XLIB

#include "../window.h"

#include <X11/Xlib.h>
#include <X11/Xresource.h>
#include <X11/Xutil.h>

#ifdef None
#undef None
#endif

#include "core/slang-basic.h"

using namespace Slang;

namespace platform
{
typedef ::Window X11WindowHandle;
class X11PlatformWindow;

void initKeyCodeTranslationTable(Display* display);
void freeKeyCodeTranslationTable();
KeyCode translateKeyCode(int keyCode);
int getKeyChar(KeyCode keyCode, int keyState);

const int kKeyStateTableSize = 256;

enum class KeyState
{
    Released,
    Pressed,
    Hold
};

enum class KeyEvent
{
    Press,
    Release
};

enum class MouseEvent
{
    Move,
    Down,
    Up,
    Scroll
};

class X11AppContext
{
public:
    static bool isTerminated;
    static RefPtr<Window> mainWindow;
    static OrderedDictionary<X11WindowHandle, X11PlatformWindow*> windows;
    static X11WindowHandle mainWindowHandle;
    static Display* xdisplay;
    static KeyState keyStates[kKeyStateTableSize];
    static X11PlatformWindow* currentMouseEventWindow;
};

bool X11AppContext::isTerminated = false;
RefPtr<Window> X11AppContext::mainWindow;
OrderedDictionary<X11WindowHandle, X11PlatformWindow*> X11AppContext::windows;
X11WindowHandle X11AppContext::mainWindowHandle;
Display* X11AppContext::xdisplay = nullptr;
KeyState X11AppContext::keyStates[kKeyStateTableSize] = {};
X11PlatformWindow* X11AppContext::currentMouseEventWindow = nullptr;

void Application::init() {}

static void doEventsImpl(bool waitForEvents);

void Application::doEvents()
{
    doEventsImpl(false);
}

void Application::quit()
{
    X11AppContext::isTerminated = true;
}

void Application::dispose()
{
    X11AppContext::mainWindow = nullptr;
    X11AppContext::windows = decltype(X11AppContext::windows)();
    freeKeyCodeTranslationTable();
}

void Application::run(Window* mainWindow, bool waitForEvents)
{
    if (mainWindow)
    {
        X11AppContext::mainWindow = mainWindow;
        X11AppContext::mainWindowHandle =
            (X11WindowHandle)mainWindow->getNativeHandle().handleValues[1];
        mainWindow->show();
        while (!X11AppContext::isTerminated)
        {
            doEventsImpl(waitForEvents);
            if (!X11AppContext::isTerminated)
                mainWindow->events.mainLoop();
        }
    }
}

class X11PlatformWindow : public Window
{
public:
    X11WindowHandle handle;
    bool visible = false;
    int currentWidth = 0;
    int currentHeight = 0;
    bool fixedSized = false;
    X11PlatformWindow(const WindowDesc& desc)
    {
        currentWidth = desc.width;
        currentHeight = desc.height;

        int blackColor =
            BlackPixel(X11AppContext::xdisplay, DefaultScreen(X11AppContext::xdisplay));
        int whiteColor =
            WhitePixel(X11AppContext::xdisplay, DefaultScreen(X11AppContext::xdisplay));
        handle = XCreateSimpleWindow(
            X11AppContext::xdisplay,
            DefaultRootWindow(X11AppContext::xdisplay),
            0,
            0,
            desc.width,
            desc.height,
            0,
            blackColor,
            blackColor);
        X11AppContext::windows[handle] = this;
        Atom wmDelete = XInternAtom(X11AppContext::xdisplay, "WM_DELETE_WINDOW", True);
        XSetWMProtocols(X11AppContext::xdisplay, handle, &wmDelete, 1);
        XSelectInput(
            X11AppContext::xdisplay,
            handle,
            StructureNotifyMask | KeyPressMask | KeyReleaseMask | PointerMotionMask |
                ButtonPressMask | ButtonReleaseMask | ExposureMask | FocusChangeMask);

        if (desc.style == WindowStyle::FixedSize)
        {
            fixedSized = true;
            setFixedSizeHint(desc.width, desc.height);
        }
        setText(desc.title);
    }

    ~X11PlatformWindow() { close(); }

    void setFixedSizeHint(int w, int h)
    {
        auto sizeHints = XAllocSizeHints();
        sizeHints->flags = PMinSize | PMaxSize;
        sizeHints->min_width = sizeHints->max_width = w;
        sizeHints->min_height = sizeHints->max_height = h;
        XSetWMNormalHints(X11AppContext::xdisplay, handle, sizeHints);
        XFree(sizeHints);
    }

    virtual void setClientSize(uint32_t width, uint32_t height) override
    {
        if (fixedSized)
            setFixedSizeHint(width, height);
        XResizeWindow(X11AppContext::xdisplay, handle, width, height);
        handleResizeEvent(width, height);
    }

    virtual Rect getClientRect() override
    {
        Rect rect = {};
        if (!handle)
            return rect;
        X11WindowHandle winRoot = 0, winParent = 0;
        X11WindowHandle* winChildren = nullptr;
        unsigned int numChilren = 0;
        XQueryTree(
            X11AppContext::xdisplay,
            handle,
            &winRoot,
            &winParent,
            &winChildren,
            &numChilren);
        unsigned borderWidth, depth;
        XGetGeometry(
            X11AppContext::xdisplay,
            handle,
            &winRoot,
            &rect.x,
            &rect.y,
            (uint32_t*)&rect.width,
            (uint32_t*)&rect.height,
            &borderWidth,
            &depth);
        return rect;
    }

    virtual void centerScreen() override
    {
        auto currentRect = getClientRect();
        XWindowAttributes attributes;
        XGetWindowAttributes(X11AppContext::xdisplay, handle, &attributes);
        int screenWidth = WidthOfScreen(attributes.screen);
        int screenHeight = HeightOfScreen(attributes.screen);
        int x = (screenWidth - currentRect.width) / 2;
        int y = (screenHeight - currentRect.height) / 2;
        XMoveWindow(X11AppContext::xdisplay, handle, x, y);
    }
    virtual void close() override
    {
        if (handle)
        {
            X11AppContext::windows.remove(handle);
            XDestroyWindow(X11AppContext::xdisplay, handle);
            handle = 0;
        }
    }
    virtual bool getFocused() override
    {
        if (!handle)
            return false;
        int revertTo;
        X11WindowHandle focusedWindow;
        XGetInputFocus(X11AppContext::xdisplay, &focusedWindow, &revertTo);
        return focusedWindow == handle;
    }
    virtual WindowHandle getNativeHandle() override
    {
        WindowHandle rs;
        rs.type = WindowHandle::Type::XLibHandle;
        rs.handleValues[0] = (intptr_t)X11AppContext::xdisplay;
        rs.handleValues[1] = (intptr_t)handle;
        return rs;
    }
    virtual void setText(String text) override
    {
        if (!handle)
            return;
        XStoreName(X11AppContext::xdisplay, handle, text.getBuffer());
        XClassHint* hint = XAllocClassHint();
        hint->res_class = (char*)"Slang platform window";
        hint->res_name = (char*)"Slang platform window";
        XSetClassHint(X11AppContext::xdisplay, handle, hint);
        XFree(hint);
    }
    virtual bool getVisible() override { return visible; }
    virtual void show() override
    {
        XMapWindow(X11AppContext::xdisplay, handle);
        visible = true;
    }
    virtual void hide() override
    {
        if (!handle)
            return;
        XUnmapWindow(X11AppContext::xdisplay, handle);
        visible = false;
    }
    virtual int getCurrentDpi() override
    {
        char* resourceString = XResourceManagerString(X11AppContext::xdisplay);
        XrmDatabase db;
        XrmValue value;
        char* type = NULL;
        double dpi = 96.0;
        db = XrmGetStringDatabase(resourceString);
        if (resourceString)
        {
            if (XrmGetResource(db, "Xft.dpi", "String", &type, &value))
            {
                if (value.addr)
                {
                    dpi = atof(value.addr);
                }
            }
        }
        return (int)dpi;
    }
    void handleResizeEvent(int w, int h)
    {
        if (w != currentWidth || h != currentHeight)
        {
            currentWidth = w;
            currentHeight = h;
            events.sizeChanged();
        }
    }

    static void addButtonState(ButtonState::Enum& state, ButtonState::Enum newState)
    {
        state = ButtonState::Enum((int)state | (int)newState);
    }

    ButtonState::Enum getButtonState(int state)
    {
        ButtonState::Enum buttonState = ButtonState::Enum::None;
        if (state & ShiftMask)
            addButtonState(buttonState, ButtonState::Enum::Shift);
        if (state & ControlMask)
            addButtonState(buttonState, ButtonState::Enum::Control);
        if (state & Mod1Mask)
            addButtonState(buttonState, ButtonState::Enum::Alt);
        if (state & Button1Mask)
            addButtonState(buttonState, ButtonState::Enum::LeftButton);
        if (state & Button2Mask)
            addButtonState(buttonState, ButtonState::Enum::MiddleButton);
        if (state & Button3Mask)
            addButtonState(buttonState, ButtonState::Enum::RightButton);
        return buttonState;
    }

    void handleKeyEvent(KeyEvent eventType, KeyCode keyCode, int keyChar, int state)
    {
        KeyEventArgs e;
        e.buttons = getButtonState(state);
        e.cancelEvent = false;
        e.key = keyCode;
        e.keyChar = keyChar;
        if (eventType == KeyEvent::Press)
        {
            events.keyDown(e);
            if (keyChar)
                events.keyPress(e);
        }
        else
        {
            events.keyUp(e);
        }
    }

    void handleMouseEvent(
        MouseEvent eventType,
        int x,
        int y,
        int delta,
        int button,
        int state,
        unsigned long time)
    {
        auto buttonState = getButtonState(state);
        if (button == Button1)
            addButtonState(buttonState, ButtonState::Enum::LeftButton);
        else if (button == Button2)
            addButtonState(buttonState, ButtonState::Enum::MiddleButton);
        else if (button == Button3)
            addButtonState(buttonState, ButtonState::Enum::RightButton);
        MouseEventArgs e;
        e.buttons = buttonState;
        e.delta = delta;
        e.x = x;
        e.y = y;

        switch (eventType)
        {
        case MouseEvent::Down:
            events.mouseDown(e);
            break;
        case MouseEvent::Up:
            events.mouseUp(e);
            break;
        case MouseEvent::Move:
            events.mouseMove(e);
            break;
        case MouseEvent::Scroll:
            events.mouseWheel(e);
            break;
        default:
            break;
        }
    }

    void handleCloseEvent()
    {
        hide();
        if (X11AppContext::mainWindowHandle == handle)
            Application::quit();
    }

    void handleExposeEvent() {}

    void handleFocus(bool focus) {}
};

Window* Application::createWindow(const WindowDesc& desc)
{
    if (!X11AppContext::xdisplay)
    {
        XInitThreads();
        XrmInitialize();
        X11AppContext::xdisplay = XOpenDisplay(nullptr);
        initKeyCodeTranslationTable(X11AppContext::xdisplay);
        if (!X11AppContext::xdisplay)
            printf("Failed to open XDisplay.\n");
    }
    return new X11PlatformWindow(desc);
}

void doEventsImpl(bool waitForEvents)
{
    auto xdisplay = X11AppContext::xdisplay;
    if (!X11AppContext::xdisplay)
        return;

    static bool supressInvokeTasks = false;
    X11PlatformWindow* sysWindow = nullptr;
    KeyCode vKeyCode = KeyCode::None;
    int iKeyCode = 0;
    while (XPending(xdisplay))
    {
        XEvent nextEvent;
        XNextEvent(xdisplay, &nextEvent);
        switch (nextEvent.type)
        {
        case KeyPress:
            vKeyCode = translateKeyCode(nextEvent.xkey.keycode);
            iKeyCode = (int)vKeyCode;
            if (iKeyCode < kKeyStateTableSize)
            {
                if (X11AppContext::keyStates[iKeyCode] == KeyState::Released)
                    X11AppContext::keyStates[iKeyCode] = KeyState::Pressed;
                else if (X11AppContext::keyStates[iKeyCode] == KeyState::Pressed)
                    X11AppContext::keyStates[iKeyCode] = KeyState::Hold;
            }
            if (X11AppContext::windows.tryGetValue(nextEvent.xkey.window, sysWindow))
            {
                wchar_t keyChar = getKeyChar(vKeyCode, nextEvent.xkey.state);
                sysWindow->handleKeyEvent(KeyEvent::Press, vKeyCode, keyChar, nextEvent.xkey.state);
            }
            break;
        case KeyRelease:
            vKeyCode = translateKeyCode(nextEvent.xkey.keycode);
            iKeyCode = (int)vKeyCode;
            if (iKeyCode < kKeyStateTableSize)
            {
                X11AppContext::keyStates[iKeyCode] = KeyState::Released;
            }
            if (X11AppContext::windows.tryGetValue(nextEvent.xkey.window, sysWindow))
            {
                sysWindow->handleKeyEvent(KeyEvent::Release, vKeyCode, 0, nextEvent.xkey.state);
            }
            break;
        case MotionNotify:
            if (X11AppContext::windows.tryGetValue(nextEvent.xmotion.window, sysWindow))
            {
                X11AppContext::currentMouseEventWindow = sysWindow;
                sysWindow->handleMouseEvent(
                    MouseEvent::Move,
                    nextEvent.xmotion.x,
                    nextEvent.xmotion.y,
                    0,
                    0,
                    nextEvent.xmotion.state,
                    nextEvent.xmotion.time);
            }
            break;
        case ButtonPress:
            if (X11AppContext::windows.tryGetValue(nextEvent.xbutton.window, sysWindow))
            {
                X11AppContext::currentMouseEventWindow = sysWindow;
                if (nextEvent.xbutton.button <= Button3)
                    sysWindow->handleMouseEvent(
                        MouseEvent::Down,
                        nextEvent.xbutton.x,
                        nextEvent.xbutton.y,
                        0,
                        nextEvent.xbutton.button,
                        nextEvent.xbutton.state,
                        nextEvent.xbutton.time);
                else if (nextEvent.xbutton.button == Button4)
                    sysWindow->handleMouseEvent(
                        MouseEvent::Scroll,
                        nextEvent.xbutton.x,
                        nextEvent.xbutton.y,
                        120,
                        nextEvent.xbutton.button,
                        nextEvent.xbutton.state,
                        nextEvent.xbutton.time);
                else if (nextEvent.xbutton.button == Button5)
                    sysWindow->handleMouseEvent(
                        MouseEvent::Scroll,
                        nextEvent.xbutton.x,
                        nextEvent.xbutton.y,
                        -120,
                        nextEvent.xbutton.button,
                        nextEvent.xbutton.state,
                        nextEvent.xbutton.time);
            }
            break;
        case ButtonRelease:
            if (X11AppContext::windows.tryGetValue(nextEvent.xbutton.window, sysWindow))
            {
                X11AppContext::currentMouseEventWindow = sysWindow;
                sysWindow->handleMouseEvent(
                    MouseEvent::Up,
                    nextEvent.xbutton.x,
                    nextEvent.xbutton.y,
                    0,
                    nextEvent.xbutton.button,
                    nextEvent.xbutton.state,
                    nextEvent.xbutton.time);
            }
            break;
        case ConfigureNotify:
            if (X11AppContext::windows.tryGetValue(nextEvent.xconfigure.window, sysWindow))
            {
                sysWindow->handleResizeEvent(
                    nextEvent.xconfigure.width,
                    nextEvent.xconfigure.height);
            }
            break;
        case Expose:
            if (X11AppContext::windows.tryGetValue(nextEvent.xexpose.window, sysWindow))
            {
                sysWindow->handleExposeEvent();
            }
            break;
        case ClientMessage:
            if (X11AppContext::windows.tryGetValue(nextEvent.xclient.window, sysWindow))
            {
                Atom wmDelete = XInternAtom(X11AppContext::xdisplay, "WM_DELETE_WINDOW", True);
                if (nextEvent.xclient.data.l[0] == wmDelete)
                {
                    sysWindow->handleCloseEvent();
                }
            }
            break;
        case FocusIn:
            if (X11AppContext::windows.tryGetValue(nextEvent.xfocus.window, sysWindow))
            {
                sysWindow->handleFocus(true);
            }
            break;
        case FocusOut:
            if (X11AppContext::windows.tryGetValue(nextEvent.xfocus.window, sysWindow))
            {
                sysWindow->handleFocus(false);
            }
            break;
        }
    }
}

} // namespace platform

#endif
