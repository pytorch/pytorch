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

#import <Availability.h>
#import <Foundation/Foundation.h>

#if TARGET_OS_IPHONE

#import <UIKit/UIKit.h>
#import "tbbAppDelegate.h"

void get_screen_resolution(int *x, int *y) {
    // Getting landscape screen resolution in any case
    CGRect imageRect = [[UIScreen mainScreen] bounds];
    *x=imageRect.size.width>imageRect.size.height?imageRect.size.width:imageRect.size.height;
    *y=imageRect.size.width<imageRect.size.height?imageRect.size.width:imageRect.size.height;
    return;
}

int cocoa_main(int argc, char * argv[]) {
    @autoreleasepool {
        return UIApplicationMain(argc, argv, nil, NSStringFromClass([tbbAppDelegate class]));
    }
}

#elif TARGET_OS_MAC

#import <Cocoa/Cocoa.h>

int cocoa_main(int argc, char *argv[])
{
    return NSApplicationMain(argc, (const char **)argv);
}
#endif
