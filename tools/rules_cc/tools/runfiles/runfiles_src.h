// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Runfiles lookup library for Bazel-built C++ binaries and tests.
//
// USAGE:
// 1.  Depend on this runfiles library from your build rule:
//
//       cc_binary(
//           name = "my_binary",
//           ...
//           deps = ["@rules_cc//cc/private/toolchain/runfiles"],
//       )
//
// 2.  Include the runfiles library.
//
//       #include "tools/cpp/runfiles/runfiles.h"
//
//       using bazel::tools::cpp::runfiles::Runfiles;
//
// 3.  Create a Runfiles object and use rlocation to look up runfile paths:
//
//       int main(int argc, char** argv) {
//         std::string error;
//         std::unique_ptr<Runfiles> runfiles(
//             Runfiles::Create(argv[0], &error));
//
//         // Important:
//         //   If this is a test, use Runfiles::CreateForTest(&error).
//         //   Otherwise, if you don't have the value for argv[0] for whatever
//         //   reason, then use Runfiles::Create(&error).
//
//         if (runfiles == nullptr) {
//           ...  // error handling
//         }
//         std::string path =
//             runfiles->Rlocation("my_workspace/path/to/my/data.txt");
//         ...
//
//      The code above creates a Runfiles object and retrieves a runfile path.
//
//      The Runfiles::Create function uses the runfiles manifest and the
//      runfiles directory from the RUNFILES_MANIFEST_FILE and RUNFILES_DIR
//      environment variables. If not present, the function looks for the
//      manifest and directory near argv[0], the path of the main program.
//
// To start child processes that also need runfiles, you need to set the right
// environment variables for them:
//
//   std::unique_ptr<Runfiles> runfiles(Runfiles::Create(argv[0], &error));
//
//   std::string path = runfiles->Rlocation("path/to/binary"));
//   if (!path.empty()) {
//     ... // create "args" argument vector for execv
//     const auto envvars = runfiles->EnvVars();
//     pid_t child = fork();
//     if (child) {
//       int status;
//       waitpid(child, &status, 0);
//     } else {
//       for (const auto i : envvars) {
//         setenv(i.first.c_str(), i.second.c_str(), 1);
//       }
//       execv(args[0], args);
//     }

#ifndef TOOLS_CPP_RUNFILES_RUNFILES_H_
#define TOOLS_CPP_RUNFILES_RUNFILES_H_ 1

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace bazel {
namespace tools {
namespace cpp {
namespace runfiles {

class Runfiles {
 public:
  virtual ~Runfiles() {}

  // Returns a new `Runfiles` instance.
  //
  // Use this from within `cc_test` rules.
  //
  // Returns nullptr on error. If `error` is provided, the method prints an
  // error message into it.
  //
  // This method looks at the RUNFILES_MANIFEST_FILE and TEST_SRCDIR
  // environment variables.
  static Runfiles* CreateForTest(std::string* error = nullptr);

  // Returns a new `Runfiles` instance.
  //
  // Use this from `cc_binary` or `cc_library` rules. You may pass an empty
  // `argv0` if `argv[0]` from the `main` method is unknown.
  //
  // Returns nullptr on error. If `error` is provided, the method prints an
  // error message into it.
  //
  // This method looks at the RUNFILES_MANIFEST_FILE and RUNFILES_DIR
  // environment variables. If either is empty, the method looks for the
  // manifest or directory using the other environment variable, or using argv0
  // (unless it's empty).
  static Runfiles* Create(const std::string& argv0,
                          std::string* error = nullptr);

  // Returns a new `Runfiles` instance.
  //
  // Use this from any `cc_*` rule if you want to manually specify the paths to
  // the runfiles manifest and/or runfiles directory. You may pass an empty
  // `argv0` if `argv[0]` from the `main` method is unknown.
  //
  // This method is the same as `Create(argv0, error)`, except it uses
  // `runfiles_manifest_file` and `runfiles_dir` as the corresponding
  // environment variable values, instead of looking up the actual environment
  // variables.
  static Runfiles* Create(const std::string& argv0,
                          const std::string& runfiles_manifest_file,
                          const std::string& runfiles_dir,
                          std::string* error = nullptr);

  // Returns the runtime path of a runfile.
  //
  // Runfiles are data-dependencies of Bazel-built binaries and tests.
  //
  // The returned path may not exist. The caller should verify the path's
  // existence.
  //
  // The function may return an empty string if it cannot find a runfile.
  //
  // Args:
  //   path: runfiles-root-relative path of the runfile; must not be empty and
  //     must not contain uplevel references.
  // Returns:
  //   the path to the runfile, which the caller should check for existence, or
  //   an empty string if the method doesn't know about this runfile
  std::string Rlocation(const std::string& path) const;

  // Returns environment variables for subprocesses.
  //
  // The caller should set the returned key-value pairs in the environment of
  // subprocesses, so that those subprocesses can also access runfiles (in case
  // they are also Bazel-built binaries).
  const std::vector<std::pair<std::string, std::string> >& EnvVars() const {
    return envvars_;
  }

 private:
  Runfiles(const std::map<std::string, std::string>&& runfiles_map,
           const std::string&& directory,
           const std::vector<std::pair<std::string, std::string> >&& envvars)
      : runfiles_map_(std::move(runfiles_map)),
        directory_(std::move(directory)),
        envvars_(std::move(envvars)) {}
  Runfiles(const Runfiles&) = delete;
  Runfiles(Runfiles&&) = delete;
  Runfiles& operator=(const Runfiles&) = delete;
  Runfiles& operator=(Runfiles&&) = delete;

  const std::map<std::string, std::string> runfiles_map_;
  const std::string directory_;
  const std::vector<std::pair<std::string, std::string> > envvars_;
};

// The "testing" namespace contains functions that allow unit testing the code.
// Do not use these outside of runfiles_test.cc, they are only part of the
// public API for the benefit of the tests.
// These functions and their interface may change without notice.
namespace testing {

// For testing only.
//
// Computes the path of the runfiles manifest and the runfiles directory.
//
// If the method finds both a valid manifest and valid directory according to
// `is_runfiles_manifest` and `is_runfiles_directory`, then the method sets
// the corresponding values to `out_manifest` and `out_directory` and returns
// true.
//
// If the method only finds a valid manifest or a valid directory, but not
// both, then it sets the corresponding output variable (`out_manifest` or
// `out_directory`) to the value while clearing the other output variable. The
// method still returns true in this case.
//
// If the method cannot find either a valid manifest or valid directory, it
// clears both output variables and returns false.
bool TestOnly_PathsFrom(
    const std::string& argv0, std::string runfiles_manifest_file,
    std::string runfiles_dir,
    std::function<bool(const std::string&)> is_runfiles_manifest,
    std::function<bool(const std::string&)> is_runfiles_directory,
    std::string* out_manifest, std::string* out_directory);

// For testing only.
// Returns true if `path` is an absolute Unix or Windows path.
// For Windows paths, this function does not regard drive-less absolute paths
// (i.e. absolute-on-current-drive, e.g. "\foo\bar") as absolute and returns
// false for these.
bool TestOnly_IsAbsolute(const std::string& path);

}  // namespace testing
}  // namespace runfiles
}  // namespace cpp
}  // namespace tools
}  // namespace bazel

#endif  // TOOLS_CPP_RUNFILES_RUNFILES_H_
