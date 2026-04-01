# Shader-Slang Open Source Project

## Contribution Guide

Thank you for considering contributing to the Shader-Slang project! We welcome your help to improve and enhance our project. Please take a moment to read through this guide to understand how you can contribute.

This document is designed to guide you in contributing to the project. It is intended to be easy to follow without sending readers to other pages and links. You can simply copy and paste the command lines described in this document.

* Contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant the rights to use your contribution.
* When you submit a pull request, a CLA bot will determine whether you need to sign a CLA. Simply follow the instructions provided.
* Please read and follow the contributor [Code of Conduct](CODE_OF_CONDUCT.md).
* Bug reports and feature requests should be submitted via the GitHub issue tracker.
* Changes should ideally come in as small pull requests on top of master, coming from your own personal fork of the project.
* Large features that will involve multiple contributors or a long development time should be discussed in issues and broken down into smaller pieces that can be implemented and checked in stages.

## Table of Contents
1. [Contribution Process](#contribution-process)
   - [Forking the Repository](#forking-the-repository)
   - [Cloning Your Fork](#cloning-your-fork)
   - [Creating a Branch](#creating-a-branch)
   - [Build Slang from Source](#build-slang-from-source)
   - [Making Changes](#making-changes)
   - [Testing](#testing)
   - [Commit to the Branch](#commit-to-the-branch)
   - [Push to Forked Repository](#push-to-forked-repository)
2. [Pull Request](#pull-request)
   - [Addressing Code Reviews](#addressing-code-reviews)
   - [Labeling Breaking Changes](#labeling-breaking-changes)
   - [Source Code Formatting](#source-code-formatting)
   - [Document Changes](#document-changes)
3. [Code Style](#code-style)
4. [Issue Tracking](#issue-tracking)
5. [Communication](#communication)
6. [License](#license)

## Contribution Process

### Forking the Repository
Navigate to the [Shader-Slang repository](https://github.com/shader-slang/slang).
Click on the "Fork" button in the top right corner to create a copy of the repository in your GitHub account.
This document will assume that the name of your forked repository is "slang".
Make sure your "Actions" are enabled. Visit your forked repository, click on the "Actions" tab, and enable the actions.

### Cloning Your Fork
1. Clone your fork locally, replacing "USER-NAME" in the command below with your actual username.
   ```
   $ git clone --recursive --tags https://github.com/USER-NAME/slang.git
   $ cd slang
   ```

2. Fetch tags by adding the original repository as an upstream.
   It is important to have tags in your forked repository because our workflow/action uses the information for the build process. But the tags are not fetched by default when you fork a repository in GitHub. You need to add the original repository as an upstream and fetch tags manually.
   ```
   $ git remote add upstream https://github.com/shader-slang/slang.git
   $ git fetch --tags upstream
   ```

   You can check whether the tags are fetched properly with the following command.
   ```
   $ git tag -l
   ```

3. Push tags to your forked repository.
   The tags are fetched to your local machine but haven't been pushed to the forked repository yet. You need to push tags to your forked repository with the following command.
   ```
   $ git push --tags origin
   ```

### Creating a Branch
Create a new branch for your contribution:
```
$ git checkout -b feature/your-feature-name
```

### Build Slang from Source
Please follow the instructions on how to [Build Slang from Source](docs/building.md).

For a quick reference, follow the instructions below.

#### Windows
Download and install CMake from [CMake.org/download](https://cmake.org/download).

Run CMake with the following command to generate a Visual Studio 2022 Solution:
```
C:\git\slang> cmake.exe --preset vs2022 # For Visual Studio 2022
C:\git\slang> cmake.exe --preset vs2019 # For Visual Studio 2019
```

Open `build/slang.sln` with Visual Studio IDE and build it for "x64".

Or you can build with the following command:
```
C:\git\slang> cmake.exe --build --preset release
```

On Windows ARM64, prebuilt binaries for LLVM isn't available.
Please build Slang without LLVM dependency by running:

```
cmake.exe --preset vs2022 -DSLANG_SLANG_LLVM_FLAVOR=DISABLE
```

during configuration step.

#### Linux
Install CMake and Ninja.
```
$ sudo apt-get install cmake ninja-build
```
> Warning: Currently the required CMake version is 3.25 or above.

Run CMake with the following command to generate Makefile:
```
$ cmake --preset default
```

Build with the following command:
```
$ cmake --build --preset release
```

#### MacOS
Install Xcode from the App Store.

Install CMake and Ninja; we recommend using [Homebrew](https://brew.sh/) for installing them.
```
$ brew install ninja
$ brew install cmake
```

Run CMake with the following command to generate Makefile:
```
$ cmake --preset default
```

Build with the following command:
```
$ cmake --build --preset release
```

#### Building with a Local Build of slang-llvm
`slang-llvm` is required to run `slang-test` properly.
Follow the instructions below if you wish to build `slang-llvm` locally.
```
$ external/build-llvm.sh --source-dir build/slang-llvm_src --install-prefix build/slang-llvm_install
```

> Note: On Windows you can use `external/build-llvm.ps1` in Powershell.

You need to use the following command to regenerate the Makefile:
```
$ cmake --preset default --fresh -DSLANG_SLANG_LLVM_FLAVOR=USE_SYSTEM_LLVM -DLLVM_DIR=build/slang-llvm_install/lib/cmake/llvm -DClang_DIR=build/slang-llvm_install/lib/cmake/clang
```

Build with the following command:
```
$ cmake --build --preset release
```

#### GitHub REST API Limit
When you execute `cmake --preset`, CMake uses the GitHub REST API, and there is a daily/hourly API limit for each IP address. If you are using an IP address shared by many people, you may hit this limit occasionally. Refer to [Rate limits for the REST API](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api) for more information.

When this happens, you will see a warning message from CMake as follows:
```
CMake Warning at cmake/GitHubRelease.cmake:53 (message):
  If API rate limit is exceeded, Github allows a higher limit when you use
  token.  Try a cmake option -DSLANG_GITHUB_TOKEN=your_token_here
Call Stack (most recent call first):
  cmake/GitHubRelease.cmake:114 (check_release_and_get_latest)
  CMakeLists.txt:141 (get_best_slang_binary_release_url)
```

The limit is higher when you use your personal account with a "personal access token".

To generate a "personal access token" on GitHub, follow steps in [Creating a personal access token (classic)](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic)

Use the generated "token" with a cmake option "-DSLANG_GITHUB_TOKEN=your-token-here".

### Making Changes
Make your changes and ensure to follow our [Design Decisions](docs/design/README.md).

### Testing
Test your changes thoroughly to ensure they do not introduce new issues. This is done by building and running `slang-test` from the repository root directory. For more details about `slang-test`, please refer to the [Documentation on testing](tools/slang-test/README.md).

> Note: `slang-test` is meant to be launched from the root of the repository. It uses a hard-coded directory name "tests/" that is expected to exist in the current working directory.

> Note: One of the options for `slang-test.exe` is `-api`, and it takes an additional keyword to specify which API to test. When the option is `-api all-cpu`, as an example, it means it tests all APIs except CPU. The minus sign (-) after `all` means "exclude," and you can "include" with a plus sign (+) like `-api gl+dx11`.

If you are familiar with Workflows/Actions in GitHub, you can check [Our Workflows](.github/workflows). The "Test Slang" section in [ci.yml](.github/workflows/ci.yml) is where `slang-test` runs.

For a quick reference, follow the instructions below.

#### Windows
1. Download and install VulkanSDK from the [LunarG SDK page](https://www.lunarg.com/vulkan-sdk).
2. Set an environment variable to enable SPIR-V validation in the Slang compiler:
   ```
   C:\git\slang> set SLANG_RUN_SPIRV_VALIDATION=1
   ```
3. Run `slang-test` with multiple threads. This may take 10 minutes or less depending on the performance of your computer.
   ```
   C:\git\slang> build\Release\bin\slang-test.exe -use-test-server -server-count 8
   ```
   > Note: If you increase `-server-count` to more than 16, you may find some of the tests randomly fail. This is a known issue on the graphics driver side.
4. Check whether the tests finished as expected.

#### Linux
1. Install VulkanSDK by following the [Instructions from LunarG](https://vulkan.lunarg.com/doc/view/latest/linux/getting_started_ubuntu.html).
   ```
   $ sudo apt update
   $ sudo apt install vulkan-sdk
   ```
2. Run `slang-test` with multiple threads. This may take 10 minutes or less depending on the performance of your computer.
   ```
   $ ./build/Release/bin/slang-test -use-test-server -server-count 8
   ```
3. Check whether the tests finished as expected.

### Commit to the Branch
Commit your changes to the branch with a descriptive commit message:
```
$ git commit
```

It is important to have a descriptive commit message. Unlike comments inside the source code, the commit messages don't spoil over time because they are tied to specific changes and can be reviewed by many people many years later.

Here is a good example of a commit message:

> Add user authentication feature
> 
> Fixes #1234
> 
> This commit introduces a new user authentication feature. It includes changes to the login page, user database, and session management to provide secure user authentication.

### Push to Forked Repository
Push your branch to your forked repository with the following command:
```
$ git push origin feature/your-feature-name
```

After the changes are pushed to your forked repository, the change needs to be merged to the final destination `shader-slang/slang`.
In order to proceed, you will need to create a "Pull Request," or "PR" for short.

When you push to your forked repository, `git-push` usually prints a URL that allows you to create a PR.

If you missed a chance to use the URL, you can still create a PR from the GitHub webpage.
Go to your forked repository and change the branch name to the one you used for `git-push`.
It will show a message like "This branch is 1 commit ahead of `shader-slang/slang:master`."
You can create a PR by clicking on the message.

## Pull Request
Once a PR is created against `shader-slang/slang:master`, the PR will be merged when the following conditions are met:
1. The PR is reviewed and got approval.
1. All of the workflows pass.

When the conditions above are all met, you will have a chance to rewrite the commit message. Since the Slang repo uses the "squash" strategy for merging, multiple commits in your PR will become one commit. By default, GitHub will concatenate all of the commit messages sequentially, but often it is not readable. Please rewrite the final commit message in a way that people can easily understand what the purpose of the commit is.

There are two cases where the workflow may fail for reasons that are not directly related to the change:
1. "Breaking change" labeling is missing.
1. Source code "Format" needs to be changed.

### Addressing Code Reviews
After your pull request is created, you will receive code reviews from the community within 24 hours.

The PR requires approval from people who have permissions. They will review the changes before approving the pull. During this step, you will get feedback from other people, and they may request you to make some changes.

Follow-up changes that address review comments should be pushed to your pull request branch as additional commits. Any additional commits made to the same branch in your forked repository will show up on the PR as incremental changes.

When your branch is out of sync with top-of-tree, submit a merge commit to keep them in sync. Do not rebase and force push after the PR is created to keep the change history during the review process.

Use these commands to sync your branch:
```
$ git fetch upstream master
$ git merge upstream/master # resolve any conflicts here
$ git submodule update --recursive
```

The Slang repository uses the squash strategy for merging pull requests, which means all your commits will be squashed into one commit by GitHub upon merge.

### Labeling Breaking Changes
All pull requests must be labeled as either `pr: non-breaking` or `pr: breaking change` before it can be merged to the main branch. If you are already a committer, you are expected to label your PR when you create it. If you are not yet a committer, a reviewer will do this for you.

A PR is considered to introduce a breaking change if an existing application that uses Slang may no longer compile or behave the same way with the change. Typical examples of breaking changes include:

- Changes to `slang.h` that modify the Slang API in a way that breaks binary compatibility.
- Changes to the language syntax or semantics that may cause existing Slang code to not compile or produce different run-time results. For example, changing the overload resolution rules.
- Removing or renaming an existing intrinsic from the core module.

### Source Code Formatting
When the PR contains source code changes, one of the workflows will check the formatting of the code.

Code formatting can be automatically fixed on your branch by commenting `/format`; a bot will proceed to open a PR targeting *your* branch. You can merge the generated PR into your branch, and the problem will be resolved.

### Document Changes
When the PR contains document changes for the [Slang User's Guide](https://shader-slang.com/slang/user-guide/), you need to update the table of contents by running a PowerShell script as follows:
```
# Open PowerShell on Windows
cd docs
.\build_toc.ps1

# Add to git commit
git add gfx-user-guide/toc.html
git add user-guide/toc.html
```

Similar to the `/format` bot-command described in the previous section, you can also use `/regenerate-toc` instead.

When the PR is limited to document changes, the build workflows may not start properly. This is because the building process is unnecessary when the PR is limited to document changes. This may lead to a case where some of the required build workflows are stuck waiting to start. When this happens, the committers will manually merge the PR as a workaround, and it will not give you a chance to rewrite the commit message.

## Code Style
Follow our [Coding Conventions](docs/design/coding-conventions.md) to maintain consistency throughout the project.

Here are a few highlights:
1. Indent by four spaces. Don't use tabs except in files that require them (e.g., Makefiles).
1. Don't use the STL containers, iostreams, or the built-in C++ RTTI system.
1. Don't use the C++ variants of C headers (e.g., use `<stdio.h>` instead of `<cstdio>`).
1. Don't use exceptions for non-fatal errors (and even then support a build flag to opt out of exceptions).
1. Types should use UpperCamelCase, values should use lowerCamelCase, and macros should use `SCREAMING_SNAKE_CASE` with a prefix `SLANG_`.
1. Global variables should have a `g` prefix, non-const static class members can have an `s` prefix, constant data (in the sense of static const) should have a `k` prefix, and an `m_` prefix on member variables and a `_` prefix on member functions are allowed.
1. Prefixes based on types (e.g., `p` for pointers) should never be used.
1. In function parameter lists, an `in`, `out`, or `io` prefix can be added to a parameter name to indicate whether a pointer/reference/buffer is intended to be used for input, output, or both input and output.
1. Trailing commas should always be used for array initializer lists.
1. Try to write comments that explain the "why" of your code more than the "what."

## Issue Tracking
We track all our work with GitHub issues. Check the [Issues](https://github.com/shader-slang/slang/issues) for open issues. If you find a bug or want to suggest an enhancement, please open a new issue.

If you're new to the project or looking for a good starting point, consider exploring issues labeled as [Good first bug](https://github.com/shader-slang/slang/issues?q=is%3Aissue+is%3Aopen+label%3AGoodFirstBug). These are beginner-friendly bugs that provide a great entry point for new contributors.

## Communication
Join our [Discussions](https://github.com/shader-slang/slang/discussions).

## License
By contributing to Shader-Slang, you agree that your contributions will be licensed under the Apache License 2.0 with LLVM Exception. The full text of the License can be found in the [LICENSE](LICENSE) file in the root of the repository.
