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

#if TARGET_OS_IPHONE

#import <UIKit/UIKit.h>
#import "OpenGLES/ES2/gl.h"

@interface OpenGLView : UIView {
    NSTimer *timer;
    CGRect imageRect;
}

@property (nonatomic, retain) NSTimer *timer;
@property (nonatomic) CGRect imageRect;

- (void) drawRect:(CGRect)rect;
- (void) touchesBegan:(NSSet *)touches withEvent:(UIEvent *)event;

@end

#elif TARGET_OS_MAC

#import <Foundation/Foundation.h>
#import <Cocoa/Cocoa.h>

@interface OpenGLView : NSOpenGLView{
    NSTimer *timer;
}

@property (nonatomic,retain) NSTimer *timer;

- (void) drawRect:(NSRect)start;
- (void) mouseDown:(NSEvent *)theEvent;
- (void) keyDown:(NSEvent *)theEvent;
- (BOOL) acceptsFirstResponder;
- (void) viewDidEndLiveResize;

@end

#endif
