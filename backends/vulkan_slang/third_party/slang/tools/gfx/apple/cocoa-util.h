#pragma once

namespace gfx
{

// Utility functions for Cocoa
struct CocoaUtil
{

    static void getNSWindowContentSize(void* nswindow, int* widthOut, int* heightOut);

    static void* createMetalLayer(void* nswindow);
    static void destroyMetalLayer(void* metalLayer);
    static void* nextDrawable(void* metalLayer);
};

} // namespace gfx
