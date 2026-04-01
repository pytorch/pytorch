#ifdef _WIN32

#include "../window.h"

#include <windows.h>
#include <windowsx.h>
using namespace Slang;

#pragma comment(lib, "Gdi32")

namespace platform
{

static const wchar_t* kWindowClassName = L"slang-platform-window";

typedef BOOL(WINAPI* EnableNonClientDpiScalingProc)(_In_ HWND hwnd);

class Win32AppContext
{
public:
    static EnableNonClientDpiScalingProc enableNonClientDpiScaling;
    static RefPtr<Window> mainWindow;
    static OrderedDictionary<HWND, Window*> windows;
    static HWND mainWindowHandle;
    static bool isTerminated;
    static bool isWindows81OrGreater;
};

EnableNonClientDpiScalingProc Win32AppContext::enableNonClientDpiScaling = nullptr;
HWND Win32AppContext::mainWindowHandle = nullptr;
RefPtr<Window> Win32AppContext::mainWindow;
OrderedDictionary<HWND, Window*> Win32AppContext::windows;
bool Win32AppContext::isTerminated = false;
bool Win32AppContext::isWindows81OrGreater = false;


ButtonState::Enum _addButtonState(ButtonState::Enum val, ButtonState::Enum newState)
{
    return (ButtonState::Enum)((int)val | (int)newState);
}

ButtonState::Enum getModifierState()
{
    ButtonState::Enum result = ButtonState::Enum::None;
    if (GetAsyncKeyState(VK_CONTROL))
        result = _addButtonState(result, ButtonState::Enum::Control);
    if (GetAsyncKeyState(VK_SHIFT))
        result = _addButtonState(result, ButtonState::Enum::Shift);
    if (GetAsyncKeyState(VK_MENU))
        result = _addButtonState(result, ButtonState::Enum::Alt);
    return result;
}

ButtonState::Enum getModifierState(WPARAM wParam)
{
    ButtonState::Enum result = ButtonState::Enum::None;
    if (wParam & MK_CONTROL)
        result = _addButtonState(result, ButtonState::Enum::Control);
    if (wParam & MK_MBUTTON)
        result = _addButtonState(result, ButtonState::Enum::MiddleButton);
    if (wParam & MK_RBUTTON)
        result = _addButtonState(result, ButtonState::Enum::RightButton);
    if (wParam & MK_SHIFT)
        result = _addButtonState(result, ButtonState::Enum::Shift);
    if (GetAsyncKeyState(VK_MENU))
        result = _addButtonState(result, ButtonState::Enum::Alt);
    return result;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
    bool useDefProc = true;
    Window* window = nullptr;
    Win32AppContext::windows.tryGetValue(hWnd, window);
    switch (message)
    {
    case WM_LBUTTONUP:
    case WM_MBUTTONUP:
    case WM_RBUTTONUP:
        {
            int mx = GET_X_LPARAM(lParam);
            int my = GET_Y_LPARAM(lParam);
            bool processed = false;
            if (window)
            {
                window->events.mouseUp(MouseEventArgs{mx, my, 0, getModifierState(wParam)});
            }
        }
        break;
    case WM_LBUTTONDOWN:
    case WM_MBUTTONDOWN:
    case WM_RBUTTONDOWN:
        {
            int mx = GET_X_LPARAM(lParam);
            int my = GET_Y_LPARAM(lParam);
            bool processed = false;
            if (window)
            {
                window->events.mouseDown(MouseEventArgs{mx, my, 0, getModifierState(wParam)});
            }
        }
        break;
    case WM_MOUSEMOVE:
        {
            int mx = GET_X_LPARAM(lParam);
            int my = GET_Y_LPARAM(lParam);
            if (window)
            {
                window->events.mouseMove(MouseEventArgs{mx, my, 0, getModifierState(wParam)});
            }
        }
        break;
    case WM_MOUSEWHEEL:
        {
            int delta = GET_WHEEL_DELTA_WPARAM(wParam);
            if (window)
            {
                window->events.mouseMove(MouseEventArgs{0, 0, delta, getModifierState(wParam)});
            }
        }
        break;
    case WM_CHAR:
        {
            if (window)
            {
                KeyEventArgs keyEventArgs =
                    {KeyCode::None, (wchar_t)(wParam), ButtonState::Enum::None, false};
                window->events.keyPress(keyEventArgs);
                if (keyEventArgs.cancelEvent)
                    useDefProc = false;
            }
        }
        break;
    case WM_KEYDOWN:
        {
            if (window)
            {
                KeyEventArgs keyEventArgs = {(KeyCode)(wParam), 0, getModifierState(), false};
                window->events.keyDown(keyEventArgs);
                if (keyEventArgs.cancelEvent)
                    useDefProc = false;
            }
        }
        break;
    case WM_KEYUP:
        {
            if (window)
            {
                KeyEventArgs keyEventArgs = {(KeyCode)(wParam), 0, getModifierState(), false};
                window->events.keyUp(keyEventArgs);
                if (keyEventArgs.cancelEvent)
                    useDefProc = false;
            }
        }
        break;
    case WM_SETFOCUS:
        {
            if (window)
            {
                window->events.focus();
            }
        }
        break;
    case WM_KILLFOCUS:
        {
            if (window)
            {
                window->events.lostFocus();
            }
        }
        break;
    case WM_SIZE:
        {
            if (window)
            {
                window->events.sizeChanged();
            }
        }
        break;
    case WM_NCCREATE:
        {
            if (Win32AppContext::enableNonClientDpiScaling)
                Win32AppContext::enableNonClientDpiScaling(hWnd);
            return DefWindowProc(hWnd, message, wParam, lParam);
        }
        break;
    default:
        break;
    }
    if (message == WM_DESTROY && hWnd == Win32AppContext::mainWindowHandle)
    {
        PostQuitMessage(0);
        return 0;
    }
    if (useDefProc)
        return DefWindowProc(hWnd, message, wParam, lParam);
    return 0;
}

void registerWindowClass()
{
    WNDCLASSEX wcex;

    wcex.cbSize = sizeof(WNDCLASSEX);

    wcex.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC | CS_DBLCLKS;
    wcex.lpfnWndProc = WndProc;
    wcex.cbClsExtra = 0;
    wcex.cbWndExtra = 0;
    wcex.hInstance = GetModuleHandle(NULL);
    wcex.hIcon = 0;
    wcex.hCursor = LoadCursor(NULL, IDC_ARROW);
    wcex.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
    wcex.lpszMenuName = 0;
    wcex.lpszClassName = kWindowClassName;
    wcex.hIconSm = 0;

    RegisterClassExW(&wcex);
}

void unregisterWindowClass()
{
    UnregisterClassW(kWindowClassName, GetModuleHandle(NULL));
}

HRESULT(WINAPI* getDpiForMonitor)
(void* hmonitor, int dpiType, unsigned int* dpiX, unsigned int* dpiY);

void Application::init()
{
    *(FARPROC*)&Win32AppContext::enableNonClientDpiScaling =
        GetProcAddress(GetModuleHandleA("User32"), "EnableNonClientDpiScaling");
    void*(WINAPI * RtlGetVersion)(LPOSVERSIONINFOEXW);
    OSVERSIONINFOEXW osInfo;
    *(FARPROC*)&RtlGetVersion = GetProcAddress(GetModuleHandleA("ntdll"), "RtlGetVersion");

    if (RtlGetVersion)
    {
        osInfo.dwOSVersionInfoSize = sizeof(osInfo);
        RtlGetVersion(&osInfo);
        if (osInfo.dwMajorVersion > 8 || (osInfo.dwMajorVersion == 8 && osInfo.dwMinorVersion >= 1))
            Win32AppContext::isWindows81OrGreater = true;
    }
    HRESULT(WINAPI * setProcessDpiAwareness)(int value);
    *(FARPROC*)&setProcessDpiAwareness =
        GetProcAddress(GetModuleHandleA("Shcore"), "SetProcessDpiAwareness");
    *(FARPROC*)&getDpiForMonitor = GetProcAddress(GetModuleHandleA("Shcore"), "GetDpiForMonitor");
    if (setProcessDpiAwareness)
    {
        if (Win32AppContext::isWindows81OrGreater)
            setProcessDpiAwareness(2); // PROCESS_PER_MONITOR_DPI_AWARE
        else
            setProcessDpiAwareness(1); // PROCESS_SYSTEM_DPI_AWARE
    }
    registerWindowClass();
}

void doEventsImpl(bool waitForEvents)
{
    int hasMsg = 0;
    do
    {
        MSG msg = {};
        hasMsg =
            (waitForEvents ? GetMessage(&msg, NULL, 0, 0) : PeekMessage(&msg, NULL, 0, 0, TRUE));
        if (hasMsg)
        {
            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }
        if (msg.message == WM_QUIT)
            Win32AppContext::isTerminated = true;
    } while (!Win32AppContext::isTerminated && hasMsg);
}

void Application::doEvents()
{
    doEventsImpl(false);
}

void Application::quit()
{
    Win32AppContext::isTerminated = true;
}

void Application::dispose()
{
    Win32AppContext::mainWindow = nullptr;
    Win32AppContext::windows = decltype(Win32AppContext::windows)();
    unregisterWindowClass();
}

void Application::run(Window* mainWindow, bool waitForEvents)
{
    if (mainWindow)
    {
        Win32AppContext::mainWindow = mainWindow;
        Win32AppContext::mainWindowHandle = (HWND)mainWindow->getNativeHandle().handleValues[0];
        ShowWindow(Win32AppContext::mainWindowHandle, SW_SHOW);
        UpdateWindow(Win32AppContext::mainWindowHandle);
    }
    while (!Win32AppContext::isTerminated)
    {
        doEventsImpl(waitForEvents);
        if (Win32AppContext::isTerminated)
            break;
        if (mainWindow)
        {
            mainWindow->events.mainLoop();
        }
    }
}

class Win32PlatformWindow : public Window
{
public:
    HWND handle;
    DWORD style;
    bool visible = false;
    Win32PlatformWindow(const WindowDesc& desc)
    {
        DWORD windowExtendedStyle = 0;
        style = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
        if (desc.style == WindowStyle::Default)
        {
            style |= WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_THICKFRAME;
        }

        HINSTANCE instance = (HINSTANCE)GetModuleHandle(0);

        RECT windowRect;
        windowRect.left = 0;
        windowRect.top = 0;
        windowRect.bottom = desc.height;
        windowRect.right = desc.width;
        AdjustWindowRect(&windowRect, style, FALSE);

        handle = CreateWindowExW(
            windowExtendedStyle,
            (LPWSTR)kWindowClassName,
            String(desc.title).toWString().begin(),
            style,
            CW_USEDEFAULT,
            0, // x, y
            windowRect.right,
            windowRect.bottom,
            NULL, // parent
            NULL, // menu
            instance,
            NULL);
        if (handle)
            Win32AppContext::windows[handle] = this;
    }

    ~Win32PlatformWindow() { close(); }

    virtual void setClientSize(uint32_t width, uint32_t height) override
    {
        RECT currentRect;
        GetWindowRect(handle, &currentRect);

        RECT windowRect;
        windowRect.left = currentRect.left;
        windowRect.top = currentRect.top;
        windowRect.bottom = height;
        windowRect.right = width;
        AdjustWindowRect(&windowRect, style, FALSE);

        MoveWindow(
            handle,
            windowRect.left,
            windowRect.top,
            windowRect.right - windowRect.left,
            windowRect.bottom - windowRect.top,
            FALSE);
    }

    virtual Rect getClientRect() override
    {
        RECT currentRect;
        GetClientRect(handle, &currentRect);
        Rect rect;
        rect.x = currentRect.left;
        rect.y = currentRect.top;
        rect.width = currentRect.right - currentRect.left;
        rect.height = currentRect.bottom - currentRect.top;
        return rect;
    }

    virtual void centerScreen() override
    {
        RECT screenRect;
        GetClientRect(GetDesktopWindow(), &screenRect);
        RECT currentRect;
        GetWindowRect(handle, &currentRect);

        auto width = currentRect.right - currentRect.left;
        auto height = currentRect.bottom - currentRect.top;

        auto left = (screenRect.right - width) / 2;
        auto top = (screenRect.bottom - height) / 2;

        MoveWindow(handle, left, top, width, height, FALSE);
    }

    virtual void close() override
    {
        if (handle)
        {
            Win32AppContext::windows.remove(handle);
        }
        DestroyWindow(handle);
        handle = NULL;
    }
    virtual bool getFocused() override { return GetFocus() == handle; }
    virtual bool getVisible() override { return visible; }
    virtual WindowHandle getNativeHandle() override { return WindowHandle::fromHwnd(handle); }
    virtual void setText(Slang::String text) override
    {
        SetWindowText(handle, text.toWString().begin());
    }
    virtual void show() override
    {
        ShowWindow(handle, SW_SHOW);
        visible = true;
    }
    virtual void hide() override
    {
        ShowWindow(handle, SW_HIDE);
        visible = false;
    }
    virtual int getCurrentDpi() override
    {
        int dpi = 96;
        if (Win32AppContext::isWindows81OrGreater && getDpiForMonitor)
        {
            getDpiForMonitor(
                MonitorFromWindow(handle, MONITOR_DEFAULTTOPRIMARY),
                0,
                (UINT*)&dpi,
                (UINT*)&dpi);
            return dpi;
        }
        dpi = GetDeviceCaps(NULL, LOGPIXELSY);
        return dpi;
    }
};

Window* Application::createWindow(const WindowDesc& desc)
{
    return new Win32PlatformWindow(desc);
}


} // namespace platform

#endif
