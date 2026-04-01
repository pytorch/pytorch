# Updating external spirv

There are three directories under `external` that are related to SPIR-V:
- external/spirv-headers
- external/spirv-tools
- external/spirv-tools-generated

In order to use the latest or custom SPIR-V, they need to be updated.


## Fork `shader-slang/SPIRV-Tools` repo and update it

Currently Slang uses [shader-slang/SPIRV-Tools](https://github.com/shader-slang/SPIRV-Tools) forked from [KhronosGroup/SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools).
In order for Slang to use the latest changes from `KhronosGroup/SPIRV-Tools`, `shader-slang/SPIRV-Tools` needs to be updated.

1. Fork `shader-slang/SPIRV-Tools` to your personal github organization like `your-name/SPIRV-Tools`.
1. Clone it on your local machine.
   ```
   git clone https://github.com/your-name/SPIRV-Tools.git # replace `your-name` to the actual URL
   ```
1. Fetch from `KhronosGroup/SPIRV-Tools`.
   ```
   git remote add khronos https://github.com/KhronosGroup/SPIRV-Tools.git
   git fetch khronos
   ```
1. Create a branch for a Pull Request.
   ```
   git checkout -b merge/update
   ```
1. Rebase to khronos/main
   ```
   git rebase khronos/main # use ToT
   ```
1. Push to Github.
   ```
   git push origin merge/update
   ```

The steps above will create a branch called `merge/update`. You can use a different name but this document will use the name.


## Modify `.gitmodules` and use the `merge/update` branch

Before creating a Pull Request for `merge/update`, you should test and make sure everything works.

On a Slang repo side, you need to create a branch for the following changes.
```
git clone https://github.com/your-name/slang.git # replace `your-name` to the actual URL
cd slang
git checkout -b update_spirv
```

Open `.gitmodules` and modify the setting to the following,
```
[submodule "external/spirv-tools"]
	path = external/spirv-tools
	url = https://github.com/your-name/SPIRV-Tools.git
[submodule "external/spirv-headers"]
	path = external/spirv-headers
	url = https://github.com/KhronosGroup/SPIRV-Headers.git
```
Note that you need to replace `your-name` with the actual URL from the previous step.

Apply the URL changes with the following commands,
```
git submodule sync
git submodule update --init --recursive

cd spirv-headers
git fetch
git checkout origin/main # use ToT
cd ..

cd external
cd spirv-tools
git fetch
git checkout merge/update # use merger/update branch
```


## Build spirv-tools

A directory, `external/spirv-tools/generated`, holds a set of files generated from spirv-tools directory.
You need to build spirv-tools in order to generate them.

```
cd external
cd spirv-tools
python3.exe utils\git-sync-deps # this step may require you to register your ssh public key to gitlab.khronos.org
cmake.exe . -B build
cmake.exe --build build --config Release
```


## Copy the generated files from `spirv-tools` to `spirv-tools-generated`

Copy some of generated files from `external/spirv-tools/build/` to `external/spirv-tools-generated/`.
The following files are ones you need to copy at the moment, but the list may change in the future.
```
DebugInfo.h
NonSemanticShaderDebugInfo100.h
OpenCLDebugInfo100.h
build-version.inc
core.insts-unified1.inc
debuginfo.insts.inc
enum_string_mapping.inc
extension_enum.inc
generators.inc
glsl.std.450.insts.inc
nonsemantic.clspvreflection.insts.inc
nonsemantic.shader.debuginfo.100.insts.inc
nonsemantic.vkspreflection.insts.inc
opencl.debuginfo.100.insts.inc
opencl.std.insts.inc
operand.kinds-unified1.inc
spv-amd-gcn-shader.insts.inc
spv-amd-shader-ballot.insts.inc
spv-amd-shader-explicit-vertex-parameter.insts.inc
spv-amd-shader-trinary-minmax.insts.inc
```


## Build Slang and run slang-test

There are many ways to build Slang executables. Refer to the [document](https://github.com/shader-slang/slang/blob/master/docs/building.md) for more detail.
For a quick reference, you can build with the following commands,
```
cmake.exe --preset vs2019
cmake.exe --build --preset release
```

After building Slang executables, run `slang-test` to see all tests are passing.
```
set SLANG_RUN_SPIRV_VALIDATION=1
build\Release\bin\slang-test.exe -use-test-server -server-count 8
```

It is often the case that some of tests fail, because of the changes on SPIRV-Header.
You need to properly resolve them before proceed.


## Create A Pull Request on `shader-slang/SPIRV-Tools`

After testing is done, you should create a Pull Request on `shader-slang/SPIRV-Tools` repo.

1. The git-push command will show you a URL for creating a Pull Request like following,
   > https://github.com/your-name/SPIRV-Tools/pull/new/merge/update # replace `your-name` to the actual URL

   Create a Pull Request.
1. Wait for all workflows to pass.
1. Merge the PR and take a note of the commit ID for the next step.

Note that this process will update `shader-slang/SPIRV-Tools` repo, but your merge is not used by `slang` repo yet.


## Create a Pull Request on `shader-slang/slang`

After the PR is merged to `shader-slang/SPIRV-Tools`, `slang` needs to start using it.

On the clone of Slang repo, revert the changes in `.gitmodules` if modified.
```
# revert the change in .gitmodules
git checkout .gitmodules
git submodule sync
git submodule update --init --recursive
```

You need to stage and commit the latest commit IDs of spirv-tools and spirv-headers.
Note that when you want to use a new commit IDs of the submodules, you have to stage with git-add command for the directly of the submodule itself.
```
cd external

# Add changes in spirv-tools-generated
git add spirv-tools-generated

# Add commit ID of spirv-headers
cd spirv-headers
git fetch
git checkout origin/main # Use ToT
cd ..
git add spirv-headers

# Add commit ID of spirv-tools
cd spirv-tools
git fetch
git checkout merge/update # Use merge/update branch
cd ..
git add spirv-tools

# Add more if there are other changes to resolve the test failures.

git commit
git push origin update_spirv
```
Once all changes are pushed to GitHub, you can create a Pull Request on `shader-slang/slang`.
