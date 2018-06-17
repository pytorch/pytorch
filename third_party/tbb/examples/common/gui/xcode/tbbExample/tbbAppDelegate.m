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

#import "tbbAppDelegate.h"

#if TARGET_OS_IPHONE

@implementation tbbAppDelegate

- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    return YES;
}

- (void)applicationDidEnterBackground:(UIApplication *)application
{
    exit(EXIT_SUCCESS);
}

@end

#elif TARGET_OS_MAC

@implementation tbbAppDelegate

@synthesize window = _window;

//declared in macvideo.cpp file
extern int g_sizex, g_sizey;

- (void)applicationDidFinishLaunching:(NSNotification *)aNotification
{
    // Insert code here to initialize your application
    NSRect windowSize;
    windowSize.size.height = g_sizey;
    windowSize.size.width = g_sizex;
    windowSize.origin=_window.frame.origin;
    [_window setFrame:windowSize display:YES];

}

- (BOOL) applicationShouldTerminateAfterLastWindowClosed:(NSApplication *) sender
{
    return YES;
}

@end

#endif
