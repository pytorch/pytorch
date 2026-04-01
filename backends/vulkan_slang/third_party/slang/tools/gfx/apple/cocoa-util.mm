#include "cocoa-util.h"

#import <Cocoa/Cocoa.h>
#import <QuartzCore/CAMetalLayer.h>

namespace gfx {

void CocoaUtil::getNSWindowContentSize(void* nswindow, int* widthOut, int* heightOut)
{
    NSWindow* window = (NSWindow*)nswindow;
    const NSRect contentRect = [window.contentView frame];
    *widthOut = contentRect.size.width;
    *heightOut = contentRect.size.height;
}

void* CocoaUtil::createMetalLayer(void* nswindow)
{
    CAMetalLayer *layer = [CAMetalLayer layer];
    NSWindow* window = (NSWindow*)nswindow;
    window.contentView.layer = layer;
    window.contentView.wantsLayer = YES;
    return layer;
}

void* CocoaUtil::nextDrawable(void* metalLayer)
{
    CAMetalLayer* layer = (CAMetalLayer*)metalLayer;
    return [layer nextDrawable];
}

void CocoaUtil::destroyMetalLayer(void* metalLayer)
{
    CAMetalLayer* layer = (CAMetalLayer*)metalLayer;
    [layer release];
}



}
