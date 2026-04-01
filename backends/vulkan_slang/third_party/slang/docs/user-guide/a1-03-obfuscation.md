---
layout: user-guide
---

Obfuscation
===========

The Slang obfuscation feature allows developers to distribute shader code in a way where the implementation details are kept secret. For example, let's say a developer has produced a novel way to render and wants to protect that intellectual property. If it is possible to compile all possible uses of the shader code into SPIR-V/DXIL, the developer can ship their product with those binaries without debug information. This is similar to the protection achieved by shipping an executable - a determined person may with a good deal of effort work out how some algorithm in the executable works, but doing so requires a considerable amount of work, and certainly more work than reading the original source code.

If a developer is not able to ship all shader binaries then there is a problem. The developer doesn't want to ship the source code as in doing so it is relatively straightforward to see how it works or even copy the implementation. A developer could provide some level of protection by encrypting the source, but when compilation occurs it will still be necessary to decrypt and so make it available to read. A developer could obfuscate their source before shipping it. In this scenario:

* Requires tooling to do the obfuscation of the source
* Any source on the client that isn't obfuscated needs to be able to call to the obfuscated code
  * Depending on how the obfuscation takes place this could be hard - remapping symbols or obfuscating on the fly on the client
  * If "public" symbols keep their original names they leak information about the implementation
* Obfuscated source provides some protection but not typically as much as a binary format (like an object file without debug information)
* How can you debug, or determine where a crash occurred without the original source? 
* If a failure occurs - how is it possible to report meaningful errors?

Some of these issues are similar to the problems of distributing JavaScript libraries that run on client machines, but which the original authors do not want to directly make available the implementation. Some of the obfuscation solutions used in the JavaScript world are partially applicable to Slang's obfuscation solution, including [source maps](https://github.com/source-map/source-map-spec).

## Obfuscation in Slang

Slang provides an obfuscation feature that addresses these issues. The major parts being

* The ability to compile a module with obfuscation enabled
  * The module is a binary format, that doesn't contain the original names or locations
* The ability to compile regular slang code that can *link* against an obfuscated module
* Code emitted to downstream compilers contain none of the symbols or locations from the original source
* Source map(s) to provide mappings between originating source and obfuscated source produced on the client

Enabling obfuscation can be achieved via the `-obfuscate` option. When using the Slang API the `-obfuscate` option can be passed via `spProcessCommandLineArguments` function or `processCommandLineArguments` method. 

When enabled a few things will happen

* Source locations are scrambled to (blank) lines in an "empty" obfuscation source file.
* A source map is produced mapping from the (blank) lines, to the originating source locations 
* Name hints are stripped.
* If a `slang-module` is being produced, AST information will be stripped.
* The names of symbols are scrambled into hashes

The source Slang emits which is passed down to downstream compilers is obfuscated, and only contains the sections of code necessary for the kernel to compile and function. 

Currently all source that is going to be compiled and linked must all have the `-obfuscate` option enabled to be able to link correctly.

When obfuscation is enabled source locations are scrambled, but Slang will also create a [source map](https://github.com/source-map/source-map-spec), which provides the mapping from the obfuscated locations to the original source. This so called "obfuscated source map" is stored with the module. If compilation produces an error, Slang will automatically use the obfuscated source map to display the error location in the originating source.

If the obfuscated source map isn't available, it will still display a source location if available, but the location will be to the "empty" obfuscated source file. This will appear in diagnostics as "(hex-digits)-obfuscated(line)". With this information and the source map it is possible to output the original source location. Importantly without the obfuscated source map information leakage about the original source is very limited.

It should be noted that the obfuscated source map is of key importance in hiding the information. In the example scenario of protecting intellectual property, a developer should compile the code they wish to protect with `-obfuscate` and distribute *just* the `.slang-module` file to link on the client machine. The source map file should not be distributed onto client machines. 

A developer could use the source map 

* To determine where a problem is occurring by getting the obfuscated error, or crash information. 
* Provide a web service that could provide more meaningful information keyed on the obfuscated location.
  * Such a service could limit what information is returned, but still be meaningful
* A web service could additionally log errors for later analysis with the source map to determine the actual origin.

## Using An Obfuscated Module

To use a `slang-module` with obfuscation requires

* Specifying one or more obfuscated modules via `-r` option
  * Currently there is only support for referencing modules stored in files
* Specifying the `-obfuscate` option

In a non obfuscated module, parts of the AST are serialized. This AST information could be through as broadly analogous to a header in C++. It is enough such that functionality in the module can be semantically checked, and linked with, however it does not, for example, contain the implementations of functions. This means doing a `-r` is roughly equivalent to doing an `import` of the source, without having the source. Any of the types, functions and so forth are available.

With the `-obfuscate` option we strip the AST, in an abundance of caution to try and limit leaking information about the module.

This means that `-r` is *NOT* enough to be able access the functionality of the module. It is necessary to declare the functions and types you wish to use. If a type is used only opaquely - i.e. not accessing its members directly, it is only necessary to declare that the type exists. If fields are accessed directly it is undefined behavior for a definition in one module to be incompatible with the definition in the obfuscated module.

For example, in "module.slang"

```slang
struct Thing
{
    int a; 
    int b;
};

int foo(Thing thing) 
{ 
    return (thing.a + thing.b) - thing.b; 
}
```

In the source that uses this module

```slang
// This is fragile - needs match the definition in "module.slang"
struct Thing
{
    int a;
    int b;
};

int foo(Thing thing);

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    Thing thing;

    int index = (int)dispatchThreadID.x;
        
    thing.a = index;
    thing.b = -index;

    outputBuffer[index] = foo(thing);
}
```

If the type `Thing` is only used opaquely then it would only be necessary to declare that it exists. For example in "module-opaque.slang"

```slang
struct Thing
{
    int a; 
    int b;
};

Thing makeThing(int a, int b)
{
    return {a, b};
}

int foo(Thing thing) 
{ 
    return (thing.a + thing.b) - thing.b; 
}
```

In the source that uses this module

```slang
// We can just declare Thing exists, as its usage is opaque.
struct Thing;
int foo(Thing thing);
Thing makeThing(int a, int b);

RWStructuredBuffer<int> outputBuffer;

[numthreads(4, 1, 1)]
void computeMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    int index = (int)dispatchThreadID.x;
    Thing thing = makeThing(index, -index);
    outputBuffer[index] = foo(thing);
}
```

That this works might seem surprising to users of languages such as C/C++, because in these languages it is necessary to know the layout of `Thing` to be able to create the `thing` variable.  This isn't necessary here though, and this can be very useful for some scenarios.

A future iteration of the feature may include parts of the AST such that an obfuscated slang-module can be used like a regular module. It would be important that what is exposed is clear and under programmer control. By default most of the definitions within a module would typically not be exposed. 
## Accessing Source Maps

During a compilation Slang can produce many different "artifacts". When using the obfuscated source map option to produce a `slang-module` Slang will associate an obfuscated source map providing the mapping to the original source. 

With typical Slang API usage, a compilation takes place and the output is a "blob" that is the output kernel. It is also possible to compile to a container, such as a zip file or a directory. The zip file can contain the kernel as well as source map(s).

For example

```
slangc module-source.slang -o module.zip -g -obfuscate 
```

This will compile "module-source.slang" into SlangIR module (aka `slang-module`) and places the `.slang-module` inside of the zip. As obfuscation is enabled the .zip will also contain the obfuscated source map for the module. 

The `.zip` file can now be used and referenced as a module 

```
slangc source.slang -target dxil -stage compute -entry computeMain -obfuscate -r module.zip
```

Notice here that the `-r` module reference is to the `.zip` file rather than the more usual `.slang-module` that is contained in the zip file. By referencing the module in this way Slang will automatically associate the contained obfuscated source map with the module. It will use that mapping for outputting diagnostics.

It is also worth noticing that in this second compilation, using `module.zip`, we need the `-obfuscate` flag set. If this isn't set linking will not work correctly.

NOTE! As previously discussed, though you should *not* ship the .zip file with the obfuscated source map such that it's available on client machines, as doing so does leak some information about the original source. Not the original source itself, but the names of files and the locations in files. You could ship a .zip to client machines, but make sure the `.map` obfuscated source maps are stripped. Alternatively, and perhaps less riskily, you could ship `.slang-module` files taken from the `.zip` file and then it is clear there is no source map information available.

## Accessing Source Maps without Files

When using the Slang API typically things work through memory, such as accessing a compilation result via a blob. It is possible to access source maps via memory also, but doing so currently requires accessing the result of a compilation as if its a file system. The current API to do this is 

```
ISlangMutableFileSystem* getCompileRequestResultAsFileSystem();
```

This method is currently only available on the `ICompileRequest` and not on the component (aka `IComponentType`) API.

The file system returned is held in memory, and the blob data held in the file system typically shared, so accessing items this way is typically very low overhead. 

The conventions used for the file system representation could best be described as a work in progress, and may change in the future. Internally Slang stores compilation results as a hierarchy of "artifacts". An artifact consists of the main result, plus associated artifacts. An artifact can also be a container which can additionally hold children artifacts. In the current directory structure each artifact is a directory, with the root directory of the `ISlangMutableFileSystem` being the root artifact. 

Given a directory representing an artifact it can contain 2 special directories `children` and `associated`. The `children` directory contains the artifacts that are children of the current directories artifact. Similarly `associated` contains directories for artifacts that are associated with the current artifact.

To give an example, if we compiled a module with obfuscation we might end up with a directory structure like....

```
obfuscated-loc-module.slang-module
associated/
associated/bc65f637-obfuscated/
associated/bc65f637-obfuscated/bc65f637-obfuscated.map
```

The root contains the root artifact `obfuscated-loc-module.slang-module` and the associated directory holds anything associated with that module, in this case there is just one thing associated which is the obfuscated source map. Note all obfuscated source maps have a name ending in `-obfuscated`.

The directory `associated/bc65f637-obfuscated/` is the directory that represents the `bc65f637-obfuscated` artifact, and that just consists of the contained map file.

At the moment the types of files need to be determined by their extensions. A future version will hold a manifest that describes in more detail the content.

## Emit Source Maps

So far we have been mainly discussing "obfuscation" source maps. These maps provide a mapping from output locations to hidden original locations.

It is also possible to generate a source map as part of emitting source to be passed to downstream compilers such as DXC, FXC, GLSLANG, NVRTCC or C++ compilers. This can be achieved via `-line-directive-mode source-map` option. The line directive mode controls how information about the original source is handled when emitting the source. The default mechanism, will add `#line` declarations into the original source. 

Via the API there are a few options to enable emit source maps

```
const char* args[2] = {"-line-directive-mode", "source-map" };
request->processCommandLineArguments(args, 2);

// Or
spProcessCommandLineArguments(request, args, 2);

// Or just setting directly
request->setLineDirectiveMode(SLANG_LINE_DIRECTIVE_MODE_SOURCE_MAP);
 
// Or 
spSetLineDirectiveMode(request, SLANG_LINE_DIRECTIVE_MODE_SOURCE_MAP);
```

The `#line` mechanism is fairly straight forward in that all of the information is including the mapping information is in a single file. A downstream compiler will then embed that information into its debug information. If obfuscation is being used, this will work and the `#line` will actually reference the "made up" "xxx-obfuscated" files.

With the `-line-directive-mode source-map` option no line directives are emitted, but a source map is produced that can map from a location in the emitted source back to its origin. If one of the origins is an obfuscated module this will reference "xxx-obfuscated" files. So in this scenario if you want to do a lookup to a location in the original source you *potentially* have to do two source map lookups.

The first lookup will take you from the emitted source location, as will likely be specified by a debugger, to their origin. Some of the origins might be source that was compiled directly (i.e. not part of an obfuscated module); these files will be named directly. If this leads to a location inside an obfuscated source map, another lookup is needed to get back to the original source location.

Why might you want to use an emit source map rather than use the `#line` mechanism?

* Less source will need to be consumed by the downstream compiler - it can just be emitted as is
* The debugging source locations will be directly the locations within the emitted source
* Source map mapping is accurate from any point in the generated source to any point in the original source
  * The `#line` mechanism is only accurate to a line
* It allows a separation of this information, such that it can be consumed and disposed of as the application requires
* Source maps are a standard, and so can be used in tooling
* Source maps allow for name mapping, mapping a symbol name to the symbol name in the original source
  * This is currently not enabled in Slang, but may be a future addition

Why you might not want to use an emit source map

* The `#line` mechanism doesn't require any special handling, and the mapping back is embedded directly into the emitted source/output binary
* There is more housekeeping in getting keeping and using source maps
* Currently Slang doesn't directly expose a source map processing API directly  
  * We do support source maps in module files, or produced as part of a compilation
  * A developer could use the slang `compiler-core` implementation
  * In the future the project could provide some API support 

## Issues/Future Work

* Support AST emitting in obfuscated modules
* Potentially add API support for source maps
* Add manifest support for artifacts
* Potentially provide a way to interact with artifacts more directly 
* Potentially support for name mapping
* May want to improve the file hierarchy representation
* Provide other ways to ingest modules, such as through memory (currently -r just supports files)
* Provide more support for other kinds of artifacts
  * Diagnostics
  * Meta data (such as bindings used)
  * Reflection
* We use -g to indicate debug information
  * On DXC the debug information is embedded in the DXIL, we allow for pdb to separate, but we currently *don't* strip the PDB from the DXIL
  * If we do strip the PDB, we may need to resign the DXIL
