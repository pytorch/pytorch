/*
    Copyright (c) 2005-2018 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.




*/

#import <Foundation/Foundation.h>
#import "OpenGLView.h"

// defined in macvideo.cpp
extern char* window_title;
extern int cocoa_update;
extern int g_sizex, g_sizey;
extern unsigned int *g_pImg;
void on_mouse_func(int x, int y, int k);
void on_key_func(int x);

bool initilized = false;

#if TARGET_OS_IPHONE

#import "OpenGLES/ES2/gl.h"

@implementation OpenGLView

@synthesize timer;
@synthesize imageRect;

- (void)drawRect:(CGRect)start
{
    if (initilized == false) {
        NSLog(@"INITIALIZE");
        timer = [NSTimer scheduledTimerWithTimeInterval:0.03 target:self selector:@selector(update_window) userInfo:nil repeats:YES];
        imageRect = [[UIScreen mainScreen] bounds];
        CGFloat full_height = imageRect.size.height;
        const float ratio=(float)g_sizex/g_sizey;
        imageRect.size.height=imageRect.size.width/ratio;
        imageRect.origin.y=(full_height-imageRect.size.height)/2;
        initilized = true;
    }
    
    CGColorSpaceRef colourSpace = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef dataProvider = CGDataProviderCreateWithData(NULL, g_pImg, 4*g_sizex*g_sizey, NULL);
    
    CGImageRef inputImage = CGImageCreate(g_sizex, g_sizey, 8, 32, g_sizex * 4, colourSpace,(CGBitmapInfo)kCGImageAlphaNoneSkipLast, dataProvider, NULL, NO, kCGRenderingIntentDefault);
    UIImage *image = [UIImage imageWithCGImage:inputImage];
    
    CGDataProviderRelease(dataProvider);
    CGColorSpaceRelease(colourSpace);
    CGImageRelease(inputImage);
    
    [image drawInRect:imageRect];

}

- (void) touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event
{
    CGPoint point = [[touches anyObject] locationInView:self];
    const int x = point.x;
    const int y = point.y;
    if ( (y-imageRect.origin.y) > 0 && y < (imageRect.origin.y + imageRect.size.height ))
    on_mouse_func( x*g_sizex/(imageRect.size.width), (y-imageRect.origin.y)*g_sizey/imageRect.size.height,1);
    [self setNeedsDisplay];
}

-(void) update_window{
    if( cocoa_update ) [self setNeedsDisplay];
}

@end

#elif TARGET_OS_MAC

#import <OpenGL/gl.h>

@implementation OpenGLView

@synthesize timer;

- (void) drawRect:(NSRect)start
{
    if (initilized == false) {
        NSLog(@"INITIALIZE");
        timer = [NSTimer scheduledTimerWithTimeInterval:0.03 target:self selector:@selector(update_window) userInfo:nil repeats:YES];
        initilized = true;
    }
    glWindowPos2i(0, (int)self.visibleRect.size.height);
    glPixelZoom( (float)self.visibleRect.size.width /(float)g_sizex,
                -(float)self.visibleRect.size.height/(float)g_sizey);
    glDrawPixels(g_sizex, g_sizey, GL_BGRA_EXT, GL_UNSIGNED_INT_8_8_8_8_REV, g_pImg);
    glFlush();
}

-(void) update_window{
    if( cocoa_update ) [self setNeedsDisplay:YES]; 
    if( window_title ) [_window setTitle:[NSString stringWithFormat:@"%s", window_title]];
}

-(void) keyDown:(NSEvent *)theEvent{
    on_key_func([theEvent.characters characterAtIndex:0]);
}

-(void) mouseDown:(NSEvent *)theEvent{
    // mouse event for seismic and fractal
    NSPoint point= theEvent.locationInWindow;
    const int x = (int)point.x;
    const int y = (int)point.y;
    NSRect rect = self.visibleRect;
    on_mouse_func(x*g_sizex/(int)rect.size.width,((int)rect.size.height-y)*g_sizey/(int)rect.size.height,1);
    [self setNeedsDisplay:YES];
}

- (BOOL) acceptsFirstResponder
{
    return YES;
}

- (void) rightMouseDown:(NSEvent *)theEvent
{
    return;
}

-(void) viewDidEndLiveResize
{
    NSRect rect = self.visibleRect;
    const int x=(int)rect.size.width;
    const int y=(int)rect.size.height;
    [_window setTitle:[NSString stringWithFormat:@"X=%d Y=%d", x,y]];
}

@end

#endif
