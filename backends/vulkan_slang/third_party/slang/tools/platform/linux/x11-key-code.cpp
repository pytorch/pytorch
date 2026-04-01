#if defined(SLANG_ENABLE_XLIB)

#include "../window.h"
#include "core/slang-basic.h"

#include <X11/Xlib.h>
#include <X11/keysym.h>

#ifdef None
#undef None
#endif

// Translates X11 key code values to Win32 Virtual Key values

using namespace Slang;

namespace platform
{
Dictionary<int, KeyCode> keyCodeMap;

struct Win32KeyCode
{
    KeyCode vKeyCode;
    int keySym;
};

Win32KeyCode keys[] = {
    {KeyCode::Left, XK_Left},
    {KeyCode::Up, XK_Up},
    {KeyCode::Down, XK_Down},
    {KeyCode::Right, XK_Right},
    {KeyCode::Escape, XK_Escape},
    {KeyCode::Return, XK_Return},
    {KeyCode::Space, XK_space},
    {KeyCode::Shift, XK_Shift_L},
    {KeyCode::Shift, XK_Shift_R},
    {KeyCode::Ctrl, XK_Control_L},
    {KeyCode::Ctrl, XK_Control_R},
    {KeyCode::Alt, XK_Alt_L},
    {KeyCode::Alt, XK_Alt_R},
    {KeyCode::Backspace, XK_BackSpace},
    {KeyCode::Delete, XK_Delete},
    {KeyCode::Home, XK_Home},
    {KeyCode::End, XK_End},
    {KeyCode::PageUp, XK_Page_Up},
    {KeyCode::PageDown, XK_Page_Down},
    {KeyCode::Insert, XK_Insert},
    {KeyCode::Tab, XK_Tab},
    {KeyCode::A, 0x41},
    {KeyCode::B, 0x42},
    {KeyCode::C, 0x43},
    {KeyCode::D, 0x44},
    {KeyCode::E, 0x45},
    {KeyCode::F, 0x46},
    {KeyCode::G, 0x47},
    {KeyCode::H, 0x48},
    {KeyCode::I, 0x49},
    {KeyCode::J, 0x4A},
    {KeyCode::K, 0x4B},
    {KeyCode::L, 0x4C},
    {KeyCode::M, 0x4D},
    {KeyCode::N, 0x4E},
    {KeyCode::O, 0x4F},
    {KeyCode::P, 0x50},
    {KeyCode::Q, 0x51},
    {KeyCode::R, 0x52},
    {KeyCode::S, 0x53},
    {KeyCode::T, 0x54},
    {KeyCode::U, 0x55},
    {KeyCode::V, 0x56},
    {KeyCode::W, 0x57},
    {KeyCode::X, 0x58},
    {KeyCode::Y, 0x59},
    {KeyCode::Z, 0x5A},
    {KeyCode::Semicolon, XK_semicolon},
    {KeyCode::Comma, XK_comma},
    {KeyCode::Dot, XK_period},
    {KeyCode::Slash, XK_slash},
    {KeyCode::Quote, XK_apostrophe},
    {KeyCode::LBracket, XK_bracketleft},
    {KeyCode::RBracket, XK_bracketright},
    {KeyCode::Backslash, XK_backslash},
    {KeyCode::Minus, XK_minus},
    {KeyCode::Plus, XK_equal},
    {KeyCode::Tilde, XK_asciitilde},
    {KeyCode::Key0, 0x30},
    {KeyCode::Key1, 0x31},
    {KeyCode::Key2, 0x32},
    {KeyCode::Key3, 0x33},
    {KeyCode::Key4, 0x34},
    {KeyCode::Key5, 0x35},
    {KeyCode::Key6, 0x36},
    {KeyCode::Key7, 0x37},
    {KeyCode::Key8, 0x38},
    {KeyCode::Key9, 0x39},
    {KeyCode::F1, XK_F1},
    {KeyCode::F2, XK_F2},
    {KeyCode::F3, XK_F3},
    {KeyCode::F4, XK_F4},
    {KeyCode::F5, XK_F5},
    {KeyCode::F6, XK_F6},
    {KeyCode::F7, XK_F7},
    {KeyCode::F8, XK_F8},
    {KeyCode::F9, XK_F9},
    {KeyCode::F10, XK_F10},
    {KeyCode::F11, XK_F11},
    {KeyCode::F12, XK_F12}};

void initKeyCodeTranslationTable(Display* display)
{
    for (auto entry : keys)
    {
        auto systemKeyCode = XKeysymToKeycode(display, entry.keySym);
        keyCodeMap[systemKeyCode] = entry.vKeyCode;
    }
}

void freeKeyCodeTranslationTable()
{
    keyCodeMap = decltype(keyCodeMap)();
}

KeyCode translateKeyCode(int keyCode)
{
    KeyCode result = KeyCode::None;
    keyCodeMap.tryGetValue(keyCode, result);
    return result;
}

int getKeyChar(KeyCode keyCode, int keyState)
{
    bool shift = (keyState & ShiftMask) != 0;
    if (keyCode >= KeyCode::A && keyCode <= KeyCode::Z)
    {
        bool capslock = (keyState & LockMask) != 0;
        bool isCapital = capslock ^ shift;
        if (isCapital)
            return (int)keyCode;
        else
            return (int)keyCode + ('a' - 'A');
    }
    else if (keyCode == KeyCode::Space)
    {
        return ' ';
    }
    else if (keyCode == KeyCode::Return)
    {
        return (int)keyCode;
    }
    else if (keyCode >= KeyCode::Key0 && keyCode <= KeyCode::Key9)
    {
        if (!shift)
            return (int)keyCode;
        else
        {
            switch (keyCode)
            {
            case KeyCode::Key0:
                return ')';
            case KeyCode::Key1:
                return '!';
            case KeyCode::Key2:
                return '@';
            case KeyCode::Key3:
                return '#';
            case KeyCode::Key4:
                return '$';
            case KeyCode::Key5:
                return '%';
            case KeyCode::Key6:
                return '^';
            case KeyCode::Key7:
                return '&';
            case KeyCode::Key8:
                return '*';
            case KeyCode::Key9:
                return '(';
            default:
                return 0;
            }
        }
    }
    if (shift)
    {
        switch (keyCode)
        {
        case KeyCode::Semicolon:
            return ':';
        case KeyCode::Comma:
            return '<';
        case KeyCode::Dot:
            return '>';
        case KeyCode::Slash:
            return '?';
        case KeyCode::Quote:
            return '\"';
        case KeyCode::LBracket:
            return '{';
        case KeyCode::RBracket:
            return '}';
        case KeyCode::Backslash:
            return '|';
        case KeyCode::Minus:
            return '_';
        case KeyCode::Plus:
            return '+';
        case KeyCode::Tilde:
            return '~';
        default:
            return 0;
        }
    }
    else
    {
        switch (keyCode)
        {
        case KeyCode::Semicolon:
            return ';';
        case KeyCode::Comma:
            return ',';
        case KeyCode::Dot:
            return '.';
        case KeyCode::Slash:
            return '/';
        case KeyCode::Quote:
            return '\'';
        case KeyCode::LBracket:
            return '[';
        case KeyCode::RBracket:
            return ']';
        case KeyCode::Backslash:
            return '\\';
        case KeyCode::Minus:
            return '-';
        case KeyCode::Plus:
            return '=';
        case KeyCode::Tilde:
            return '`';
        default:
            return 0;
        }
    }
}
} // namespace platform

#endif
