#ifdef __APPLE__

#include "../window.h"

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

using namespace Slang;
using namespace platform;

namespace platform {
class CocoaPlatformWindow;
static KeyCode keyCodes[256];

class CocoaAppContext
{
public:
    static Window* mainWindow;
    static bool isTerminated;
};

Window* CocoaAppContext::mainWindow;
bool CocoaAppContext::isTerminated = false;
}

@interface WindowDelegate : NSObject <NSWindowDelegate>
{
    CocoaPlatformWindow* window;
}

- (instancetype)initWithPlatformWindow:(CocoaPlatformWindow*)platformWindow;

@end

@interface ContentView : NSView
{
    CocoaPlatformWindow* window;
    NSTrackingArea* trackingArea;
    int mouseX, mouseY;
}

- (instancetype)initWithPlatformWindow:(CocoaPlatformWindow*)platformWindow;

@end

namespace platform {

class CocoaPlatformWindow : public Window
{
public:
    NSWindow* window;
    WindowDelegate* delegate;
    bool shouldClose = false;

    CocoaPlatformWindow(const WindowDesc& desc);
    ~CocoaPlatformWindow();

    virtual void setClientSize(uint32_t width, uint32_t height) override;
    virtual Rect getClientRect() override;
    virtual void centerScreen() override;
    virtual void close() override;
    virtual bool getFocused() override;
    virtual bool getVisible() override;
    virtual WindowHandle getNativeHandle() override;
    virtual void setText(Slang::String text) override;
    virtual void show() override;
    virtual void hide() override;
    virtual int getCurrentDpi() override;
};

void getMousePosition(NSEvent* event, NSView* view, int& x, int& y)
{
    const NSRect contentRect = [view frame];
    const NSPoint pos = [event locationInWindow];
    x = (int)pos.x;
    y = (int)(contentRect.size.height - pos.y);
}

ButtonState::Enum _addButtonState(ButtonState::Enum val, ButtonState::Enum newState)
{
    return (ButtonState::Enum)((int)val | (int)newState);
}

static ButtonState::Enum getModifierState(NSUInteger flags)
{
    ButtonState::Enum result = ButtonState::None;

    if (flags & NSEventModifierFlagShift)
        result = _addButtonState(result, ButtonState::Shift);
    if (flags & NSEventModifierFlagControl)
        result = _addButtonState(result, ButtonState::Control);
    if (flags & NSEventModifierFlagOption)
        result = _addButtonState(result, ButtonState::Alt);
    return result;
}

static KeyCode getKeyCode(NSUInteger keyCode)
{
    return keyCode < 256 ? keyCodes[keyCode] : KeyCode::None;
}

}


@implementation WindowDelegate

- (instancetype)initWithPlatformWindow:(CocoaPlatformWindow*)platformWindow
{
    self = [super init];
    if (self)
    {
        window = platformWindow;
    }
    return self;
}

- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender
{
    return YES;
}

- (BOOL)windowShouldClose:(id)window_
{
    window->shouldClose = true;
    if (CocoaAppContext::mainWindow == window)
        CocoaAppContext::isTerminated = true;
    return YES;
}

- (NSApplicationTerminateReply)applicationShouldTerminate:(NSApplication *)sender
{
    printf("applicationShouldTerminate\n");
    return NSTerminateCancel;
}

- (void)windowDidResize:(NSNotification*)notification
{
    window->events.sizeChanged();
}

- (void)windowDidBecomeKey:(NSNotification*)notification
{
    window->events.focus();
}

- (void)windowDidResignKey:(NSNotification*)notification
{
    window->events.lostFocus();
}

@end

@implementation ContentView

- (instancetype)initWithPlatformWindow:(CocoaPlatformWindow*)platformWindow
{
    self = [super init];
    if (self)
    {
        window = platformWindow;
        mouseX = 0;
        mouseY = 0;
        [self updateTrackingAreas];
    }
    return self;
}

- (void)updateTrackingAreas
{
    if (trackingArea != nil)
    {
        [self removeTrackingArea:trackingArea];
        [trackingArea release];
    }

    const NSTrackingAreaOptions options = NSTrackingMouseEnteredAndExited |
                                          NSTrackingMouseMoved |
                                          NSTrackingActiveInKeyWindow |
                                          NSTrackingEnabledDuringMouseDrag |
                                          NSTrackingCursorUpdate |
                                          NSTrackingInVisibleRect |
                                          NSTrackingAssumeInside;

    trackingArea = [[NSTrackingArea alloc] initWithRect:[self bounds]
                                                options:options
                                                  owner:self
                                               userInfo:nil];

    [self addTrackingArea:trackingArea];
    [super updateTrackingAreas];
}

- (BOOL)isOpaque
{
    return YES;
}

- (BOOL)canBecomeKeyView
{
    return YES;
}

- (BOOL)acceptsFirstResponder
{
    return YES;
}

- (BOOL)acceptsFirstMouse:(NSEvent*)event
{
    return YES;
}

- (BOOL)wantsUpdateLayer
{
    return YES;
}

- (void)mouseDown:(NSEvent*)event
{
    getMousePosition(event, self, mouseX, mouseY);
    ButtonState::Enum buttons = ButtonState::LeftButton;
    buttons = _addButtonState(buttons, getModifierState([event modifierFlags]));
    window->events.mouseDown(MouseEventArgs{mouseX, mouseY, 0, buttons});
}

- (void)mouseUp:(NSEvent*)event
{
    getMousePosition(event, self, mouseX, mouseY);
    ButtonState::Enum buttons = ButtonState::LeftButton;
    buttons = _addButtonState(buttons, getModifierState([event modifierFlags]));
    window->events.mouseUp(MouseEventArgs{mouseX, mouseY, 0, buttons});
}

- (void)rightMouseDown:(NSEvent*)event
{
    getMousePosition(event, self, mouseX, mouseY);
    ButtonState::Enum buttons = ButtonState::RightButton;
    buttons = _addButtonState(buttons, getModifierState([event modifierFlags]));
    window->events.mouseDown(MouseEventArgs{mouseX, mouseY, 0, buttons});
}

- (void)rightMouseUp:(NSEvent*)event
{
    getMousePosition(event, self, mouseX, mouseY);
    ButtonState::Enum buttons = ButtonState::RightButton;
    buttons = _addButtonState(buttons, getModifierState([event modifierFlags]));
    window->events.mouseUp(MouseEventArgs{mouseX, mouseY, 0, buttons});
}

- (void)otherMouseDown:(NSEvent *)event
{
    if ([event buttonNumber] == 2)
    {
        getMousePosition(event, self, mouseX, mouseY);
        ButtonState::Enum buttons = ButtonState::MiddleButton;
        buttons = _addButtonState(buttons, getModifierState([event modifierFlags]));
        window->events.mouseDown(MouseEventArgs{mouseX, mouseY, 0, buttons});
    }
}

- (void)otherMouseUp:(NSEvent *)event
{
    if ([event buttonNumber] == 2)
    {
        getMousePosition(event, self, mouseX, mouseY);
        ButtonState::Enum buttons = ButtonState::MiddleButton;
        buttons = _addButtonState(buttons, getModifierState([event modifierFlags]));
        window->events.mouseUp(MouseEventArgs{mouseX, mouseY, 0, buttons});
    }
}

- (void)mouseDragged:(NSEvent*)event
{
    [self mouseMoved:event];
}

- (void)rightMouseDragged:(NSEvent*)event
{
    [self mouseMoved:event];
}

- (void)otherMouseDragged:(NSEvent*)event
{
    [self mouseMoved:event];
}

- (void)mouseMoved:(NSEvent*)event
{
    getMousePosition(event, self, mouseX, mouseY);
    window->events.mouseMove(MouseEventArgs{mouseX, mouseY, 0, getModifierState([event modifierFlags])});
}

- (void)scrollWheel:(NSEvent *)event
{
    double deltaX = [event scrollingDeltaX];
    double deltaY = [event scrollingDeltaY];
    if ([event hasPreciseScrollingDeltas])
    {
        deltaX *= 0.1;
        deltaY *= 0.1;
    }

    int delta = (int)deltaY;

    window->events.mouseWheel(MouseEventArgs{0, 0, delta, getModifierState([event modifierFlags])});
}

- (void)keyDown:(NSEvent *)event
{
    KeyCode key = getKeyCode([event keyCode]);
    if (key == KeyCode::None)
        return;
    KeyEventArgs keyEventArgs = {key, 0, getModifierState([event modifierFlags]), false};
    window->events.keyDown(keyEventArgs);
    // if (!keyEventArgs.cancelEvent)
    //     [self interpretKeyEvents:@[event]];
}

- (void)keyUp:(NSEvent *)event
{
    KeyCode key = getKeyCode([event keyCode]);
    if (key == KeyCode::None)
        return;
    KeyEventArgs keyEventArgs = {key, 0, getModifierState([event modifierFlags]), false};
    window->events.keyUp(keyEventArgs);
    // if (!keyEventArgs.cancelEvent)
    //     [self interpretKeyEvents:@[event]];
}

- (void)flagsChanged:(NSEvent *)event
{
    KeyCode key = getKeyCode([event keyCode]);
    ButtonState::Enum buttons = getModifierState([event modifierFlags]);
    ButtonState::Enum button = ButtonState::None;
    if (key == KeyCode::Shift)
        button = ButtonState::Shift;
    else if (key == KeyCode::Ctrl)
        button = ButtonState::Control;
    else if (key == KeyCode::Alt)
        button = ButtonState::Alt;
    
    KeyEventArgs keyEventArgs = {key, 0, buttons, false};
    if (button & buttons) {
        window->events.keyDown(keyEventArgs);
    } else {
        window->events.keyUp(keyEventArgs);
    }
}

@end


static NSApplication *_application;


namespace platform {

void Application::init()
{
    _application = [NSApplication sharedApplication];
    [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
    [NSApp activateIgnoringOtherApps:YES];

    // Setup key translation table.
    ::memset(keyCodes, (int)KeyCode::None, sizeof(keyCodes));

    keyCodes[0x1D] = KeyCode::Key0;
    keyCodes[0x12] = KeyCode::Key1;
    keyCodes[0x13] = KeyCode::Key2;
    keyCodes[0x14] = KeyCode::Key3;
    keyCodes[0x15] = KeyCode::Key4;
    keyCodes[0x17] = KeyCode::Key5;
    keyCodes[0x16] = KeyCode::Key6;
    keyCodes[0x1A] = KeyCode::Key7;
    keyCodes[0x1C] = KeyCode::Key8;
    keyCodes[0x19] = KeyCode::Key9;
    keyCodes[0x00] = KeyCode::A;
    keyCodes[0x0B] = KeyCode::B;
    keyCodes[0x08] = KeyCode::C;
    keyCodes[0x02] = KeyCode::D;
    keyCodes[0x0E] = KeyCode::E;
    keyCodes[0x03] = KeyCode::F;
    keyCodes[0x05] = KeyCode::G;
    keyCodes[0x04] = KeyCode::H;
    keyCodes[0x22] = KeyCode::I;
    keyCodes[0x26] = KeyCode::J;
    keyCodes[0x28] = KeyCode::K;
    keyCodes[0x25] = KeyCode::L;
    keyCodes[0x2E] = KeyCode::M;
    keyCodes[0x2D] = KeyCode::N;
    keyCodes[0x1F] = KeyCode::O;
    keyCodes[0x23] = KeyCode::P;
    keyCodes[0x0C] = KeyCode::Q;
    keyCodes[0x0F] = KeyCode::R;
    keyCodes[0x01] = KeyCode::S;
    keyCodes[0x11] = KeyCode::T;
    keyCodes[0x20] = KeyCode::U;
    keyCodes[0x09] = KeyCode::V;
    keyCodes[0x0D] = KeyCode::W;
    keyCodes[0x07] = KeyCode::X;
    keyCodes[0x10] = KeyCode::Y;
    keyCodes[0x06] = KeyCode::Z;

    keyCodes[0x27] = KeyCode::Quote;
    keyCodes[0x2A] = KeyCode::Backslash;
    keyCodes[0x2B] = KeyCode::Comma;
    keyCodes[0x18] = KeyCode::Plus;
    keyCodes[0x32] = KeyCode::Tilde;
    keyCodes[0x21] = KeyCode::LBracket;
    keyCodes[0x1B] = KeyCode::Minus;
    keyCodes[0x2F] = KeyCode::Dot;
    keyCodes[0x1E] = KeyCode::RBracket;
    keyCodes[0x29] = KeyCode::Semicolon;
    keyCodes[0x2C] = KeyCode::Slash;

    keyCodes[0x33] = KeyCode::Backspace;
    keyCodes[0x75] = KeyCode::Delete;
    keyCodes[0x7D] = KeyCode::Down;
    keyCodes[0x77] = KeyCode::End;
    keyCodes[0x24] = KeyCode::Return;
    keyCodes[0x35] = KeyCode::Escape;
    keyCodes[0x7A] = KeyCode::F1;
    keyCodes[0x78] = KeyCode::F2;
    keyCodes[0x63] = KeyCode::F3;
    keyCodes[0x76] = KeyCode::F4;
    keyCodes[0x60] = KeyCode::F5;
    keyCodes[0x61] = KeyCode::F6;
    keyCodes[0x62] = KeyCode::F7;
    keyCodes[0x64] = KeyCode::F8;
    keyCodes[0x65] = KeyCode::F9;
    keyCodes[0x6D] = KeyCode::F10;
    keyCodes[0x67] = KeyCode::F11;
    keyCodes[0x6F] = KeyCode::F12;
    keyCodes[0x73] = KeyCode::Home;
    keyCodes[0x72] = KeyCode::Insert;
    keyCodes[0x7B] = KeyCode::Left;
    keyCodes[0x79] = KeyCode::PageDown;
    keyCodes[0x74] = KeyCode::PageUp;
    keyCodes[0x7C] = KeyCode::Right;
    keyCodes[0x31] = KeyCode::Space;
    keyCodes[0x30] = KeyCode::Tab;
    keyCodes[0x7E] = KeyCode::Up;

    keyCodes[0x38] = KeyCode::Shift;
    keyCodes[0x3B] = KeyCode::Ctrl;
    keyCodes[0x3A] = KeyCode::Alt;
}

void doEventsImpl(bool waitForEvents)
{
    NSEvent *event;
    do {
        event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                   untilDate:waitForEvents ? [NSDate distantFuture] : [NSDate distantPast]
                                      inMode:NSDefaultRunLoopMode
                                     dequeue:YES];
        if (event) {
            [NSApp sendEvent:event];
        }
    } while (!CocoaAppContext::isTerminated && event);
}

void Application::doEvents()
{
    doEventsImpl(false);
}

void Application::quit()
{
    CocoaAppContext::isTerminated = true;
}

void Application::dispose()
{
    CocoaAppContext::mainWindow = nullptr;
}

void Application::run(Window* mainWindow, bool waitForEvents)
{
    if (mainWindow)
    {
        CocoaAppContext::mainWindow = mainWindow;
        mainWindow->show();
    }
    while (!CocoaAppContext::isTerminated)
    {
        doEventsImpl(waitForEvents);
        if (CocoaAppContext::isTerminated)
            break;
        if (mainWindow)
        {
            mainWindow->events.mainLoop();
        }
    }
}

CocoaPlatformWindow::CocoaPlatformWindow(const WindowDesc& desc)
{
    // Create a reference rectangle
    NSRect rect = NSMakeRect(0.0f, 0.0f, desc.width, desc.height);

    // Allocate window
    window = [[NSWindow alloc] initWithContentRect:rect
                                         styleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
                                           backing:NSBackingStoreBuffered
                                             defer:NO];

    const NSWindowCollectionBehavior behavior
        = NSWindowCollectionBehaviorFullScreenPrimary | NSWindowCollectionBehaviorManaged;
    [window setCollectionBehavior:behavior];

    if (desc.style == WindowStyle::Default)
        [window setStyleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable | NSWindowStyleMaskResizable];
    else if (desc.style == WindowStyle::FixedSize)
        [window setStyleMask:NSWindowStyleMaskTitled | NSWindowStyleMaskClosable];

    delegate = [[WindowDelegate alloc] initWithPlatformWindow:this];
    [window setDelegate:delegate];

    NSString* title = [NSString stringWithUTF8String:desc.title];
    [window setTitle:title];

    [window center];
    [window makeKeyAndOrderFront:nil];
}

CocoaPlatformWindow::~CocoaPlatformWindow()
{
    close();
}

void CocoaPlatformWindow::setClientSize(uint32_t width, uint32_t height)
{
    NSSize size = NSMakeSize(width, height);
    [window setContentSize:size];
}

Rect CocoaPlatformWindow::getClientRect()
{
    NSRect rect = [window contentRectForFrameRect:[window frame]];
    return { (int)rect.origin.x, (int)rect.origin.y, (int)rect.size.width, (int)rect.size.height };
}

void CocoaPlatformWindow::centerScreen()
{
    [window center];
}

void CocoaPlatformWindow::close()
{
    [window release];
    [delegate release];

    window = nil;
    delegate = nil;
}

bool CocoaPlatformWindow::getFocused()
{
    return [window isKeyWindow];
}

bool CocoaPlatformWindow::getVisible()
{
    return [window isVisible];
}

WindowHandle CocoaPlatformWindow::getNativeHandle()
{
    return WindowHandle::fromNSWindow(window);
}

void CocoaPlatformWindow::setText(Slang::String text)
{
    NSString* title = [NSString stringWithUTF8String:text.begin()];
    [window setTitle:title];
}

void CocoaPlatformWindow::show()
{
    [window setIsVisible:YES];
}

void CocoaPlatformWindow::hide()
{
    [window setIsVisible:NO];
}

int CocoaPlatformWindow::getCurrentDpi()
{
    // There seems to be no API to get the actual DPI of the screen.
    return 0;
}

Window* Application::createWindow(const WindowDesc& desc) { return new CocoaPlatformWindow(desc); }


} // namespace platform

#endif // __APPLE__
