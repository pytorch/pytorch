// Copyright 2020 The Bazel Authors. All rights reserved.
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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>  // NOLINT
#include <unordered_set>

using std::ifstream;
using std::regex;
using std::string;
using std::unordered_set;

string getBasename(const string &path) {
  // Assumes we're on an OS with "/" as the path separator
  auto idx = path.find_last_of("/");
  if (idx == string::npos) {
    return path;
  }
  return path.substr(idx + 1);
}

// Returns 0 if there are no duplicate basenames in the object files (both via
// -filelist as well as shell args), 1 otherwise
int main(int argc, const char *argv[]) {
  unordered_set<string> basenames;
  const regex libRegex = regex(".*\\.a$");
  const regex noArgFlags =
      regex("-static|-s|-a|-c|-L|-T|-D|-no_warning_for_no_symbols");
  const regex singleArgFlags = regex("-arch_only|-syslibroot|-o");
  // Set i to 1 to skip executable path
  for (int i = 1; argv[i] != nullptr; i++) {
    const string arg = argv[i];
    if (arg == "-filelist") {
      ifstream list(argv[i + 1]);
      for (string line; getline(list, line);) {
        const string basename = getBasename(line);
        const auto pair = basenames.insert(basename);
        if (!pair.second) {
          return EXIT_FAILURE;
        }
      }
      list.close();
      i++;
    } else if (regex_match(arg, noArgFlags)) {
    } else if (regex_match(arg, singleArgFlags)) {
      i++;
    } else if (arg[0] == '-') {
      return EXIT_FAILURE;
      // Unrecognized flag, let the wrapper deal with it
    } else if (regex_match(arg, libRegex)) {
      // Archive inputs can remain untouched, as they come from other targets.
    } else {
      const string basename = getBasename(arg);
      const auto pair = basenames.insert(basename);
      if (!pair.second) {
        return EXIT_FAILURE;
      }
    }
  }
  return EXIT_SUCCESS;
}
