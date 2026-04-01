Slang SPIR-V Tools
==================

The spir-v tools are needed in the Slang project in order to be able to build glslang. We don't necessarily want/need to build all the spir-v tools - but we do need the files that are generated as part of this process. Those files are then stored in this folder, so that they can just be used without needing to be created as part of the Slang build process.

*glslang* depends on spirv-tools, and specifies an exact version to use, as
such, *spirv-tools* should only be updated along with *glslang*.

*On Linux, you can run the [`external/bump-glslang.sh`](../bump-glslang.sh) script.*

*On any platform, you can follow the below instructions after updating glslang:*

To build spirv-tools we need [cmake](https://cmake.org/download/). On windows we can use cmake with the gui interface. 

Inside the `external/spirv-tools` directory make a directory `build.vs` which is where we are going to generate all the files.

Make sure that there is a suitable version of spirv-headers in `external/spirv-tools/external`. First delete one if its there, and then from the `external/spirv-tools` directory do. 

```
git clone https://github.com/KhronosGroup/SPIRV-Headers.git external/spirv-headers
```

You may need to make sure you have the other dependencies that spirv-tools requires as described on their github main page...

https://github.com/KhronosGroup/SPIRV-Tools

At the time of writing in `external/spirv-tools` the following were needed

```
git clone https://github.com/google/effcee.git external/effcee
git clone https://github.com/google/re2.git external/re2
git clone https://github.com/abseil/abseil-cpp external/abseil_cpp
```

Next run the cmake gui. Set the source path to be `external/spirv-tools` (in the slang directory), and then set the 'where to build binaries' to `external/spirv-tools/build.vs` (or however you named that file. Then click `configure` and once that is done 'generate'. 

Now go into to the `build.vs` directory and open `spirv-tools.sln` with Visual Studio and compile. This will generate many of the files needed, once regular C++/C compilation has started all of the files should have been created. 

Delete all the '.inc' and '.h' files in this directory.
Now copy the files with '.inc' and '.h' extensions from the build.vs directory and copy it into this directory.
