
## PyTorch for iOS

### Cocoapods Developers

PyTorch is now available via Cocoapods, to integrate it to your project, simply add the following line to your `Podfile` and run `pod install`

```ruby
pod 'LibTorch'
```

### Import the library

For Objective-C developers, simply import the umbrella header

```
#import <LibTorch/LibTorch.h>
```

For Swift developers, you need to create an Objective-C class as a bridge to call the C++ APIs. We highly recommend you to follow the [Image Classification](https://github.com/pytorch/ios-demo-app/tree/master/PyTorchDemo) demo where you can find out how C++, Objective-C and Swift work together.

### Disable Bitcode

Since PyTorch is not yet built with bitcode support, you need to disable bitcode for your target by selecting the **Build Settings**, searching for **Enable Bitcode** and set the value to **No**.

## LICENSE

PyTorch is BSD-style licensed, as found in the LICENSE file.
