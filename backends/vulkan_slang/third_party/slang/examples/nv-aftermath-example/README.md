Nsight Aftermath Crash Example
==============================

* Demonstrates use of aftermath API to capture a dump with a GPU crash
* Uses the [obfuscation feature](https://github.com/shader-slang/slang/blob/master/docs/user-guide/a1-03-obfuscation.md)
* Uses an `emit` source map
* Demonstrates use of file system compile products
* Forces a crash via time out, executing a shader that is purposefully slow
* Can be used to capture D3D and Vulkan (change the device type in the sample)
* When enabled GFX is built to use Aftermath it's debug layer 
  * This disables D3D debug layer, as not possible to have both enabled
* NOTE! Will only capture Aftermath DebugInfo with a *debug* build
  * Gfx only enables debugging info (and therefore aftermath) on *debug* builds

This example is *not* enabled by default. Enabling requires requires...
 
* Passing "--enable-aftermath=true" to the command line of `premake`. 
* Having a copy of the [Nsight aftermath SDK](https://developer.nvidia.com/nsight-aftermath) in `external/nv-aftermath` directory.

On windows the following would be reasonable..

```
premake vs2019 --deps=true --enable-aftermath=true
```

Typically D3D12 debug run produces the following files...

* fragment-0.dxil               - Fragment DXIL
* fragment-0.map                - The emit source map, maps locations in the the fragment kernel to the obfuscated source file
* vertex-0.dxil                 - Vertex DXIL
* vertex-0.map                  - The emit source map, maps locations in the vertex kernel to the obfuscated source file
* XXXX-obfuscated.map           - The obfuscated source map. Will be referenced by the other source maps. Maps obfuscated locations to the original source.
* aftermath-dump-X.bin          - The Aftermath crash capture/s
* aftermath-debug-info-X.bin    - The Aftermath debug info/s

Having emit source maps, can be useful as discussed in [the documentation](https://github.com/shader-slang/slang/blob/master/docs/user-guide/a1-03-obfuscation.md#emit-source-maps), but isn't a requirement. If emit source maps are disabled the source maps `fragment-0.map`/`vertex-0.map` will *not* be produced. In this scenario the mapping to the obfuscated source file is embedded into the kernel/s directly. 

A Vulkan run will emit "spv" files, D3D12 will emit "dxil" files and D3D11 will emit "dxbc" files. 

The example source describes how to switch between emit source files, and different devices. 

## Links

* [nsight aftermath](https://developer.nvidia.com/nsight-aftermath)
* [obfuscation](https://github.com/shader-slang/slang/blob/master/docs/user-guide/a1-03-obfuscation.md)
* [source map](https://github.com/source-map/source-map-spec)
