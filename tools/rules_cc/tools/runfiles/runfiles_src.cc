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

// The "srcs_for_embedded_tools" rule in the same package sets the line below to
// include runfiles.h from the correct path. Do not modify the line below.
#include "tools/runfiles/runfiles_src.h"

#ifdef _WIN32
#include <windows.h>
#else  // not _WIN32
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif  // _WIN32

#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <memory>
#endif  // _WIN32

namespace bazel {
namespace tools {
namespace cpp {
namespace runfiles {

using std::function;
using std::map;
using std::pair;
using std::string;
using std::vector;

namespace {

bool starts_with(const string& s, const char* prefix) {
  if (!prefix || !*prefix) {
    return true;
  }
  if (s.empty()) {
    return false;
  }
  return s.find(prefix) == 0;
}

bool contains(const string& s, const char* substr) {
  if (!substr || !*substr) {
    return true;
  }
  if (s.empty()) {
    return false;
  }
  return s.find(substr) != string::npos;
}

bool ends_with(const string& s, const string& suffix) {
  if (suffix.empty()) {
    return true;
  }
  if (s.empty()) {
    return false;
  }
  return s.rfind(suffix) == s.size() - suffix.size();
}

bool IsReadableFile(const string& path) {
  return std::ifstream(path).is_open();
}

bool IsDirectory(const string& path) {
#ifdef _WIN32
  DWORD attrs = GetFileAttributesA(path.c_str());
  return (attrs != INVALID_FILE_ATTRIBUTES) &&
         (attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
  struct stat buf;
  return stat(path.c_str(), &buf) == 0 && S_ISDIR(buf.st_mode);
#endif
}

bool PathsFrom(const std::string& argv0, std::string runfiles_manifest_file,
               std::string runfiles_dir, std::string* out_manifest,
               std::string* out_directory);

bool PathsFrom(const std::string& argv0, std::string runfiles_manifest_file,
               std::string runfiles_dir,
               std::function<bool(const std::string&)> is_runfiles_manifest,
               std::function<bool(const std::string&)> is_runfiles_directory,
               std::string* out_manifest, std::string* out_directory);

bool ParseManifest(const string& path, map<string, string>* result,
                   string* error);

}  // namespace

Runfiles* Runfiles::Create(const string& argv0,
                           const string& runfiles_manifest_file,
                           const string& runfiles_dir, string* error) {
  string manifest, directory;
  if (!PathsFrom(argv0, runfiles_manifest_file, runfiles_dir, &manifest,
                 &directory)) {
    if (error) {
      std::ostringstream err;
      err << "ERROR: " << __FILE__ << "(" << __LINE__
          << "): cannot find runfiles (argv0=\"" << argv0 << "\")";
      *error = err.str();
    }
    return nullptr;
  }

  const vector<pair<string, string> > envvars = {
      {"RUNFILES_MANIFEST_FILE", manifest},
      {"RUNFILES_DIR", directory},
      // TODO(laszlocsomor): remove JAVA_RUNFILES once the Java launcher can
      // pick up RUNFILES_DIR.
      {"JAVA_RUNFILES", directory}};

  map<string, string> runfiles;
  if (!manifest.empty()) {
    if (!ParseManifest(manifest, &runfiles, error)) {
      return nullptr;
    }
  }

  return new Runfiles(std::move(runfiles), std::move(directory),
                      std::move(envvars));
}

bool IsAbsolute(const string& path) {
  if (path.empty()) {
    return false;
  }
  char c = path.front();
  return (c == '/' && (path.size() < 2 || path[1] != '/')) ||
         (path.size() >= 3 &&
          ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) &&
          path[1] == ':' && (path[2] == '\\' || path[2] == '/'));
}

string GetEnv(const string& key) {
#ifdef _WIN32
  DWORD size = ::GetEnvironmentVariableA(key.c_str(), NULL, 0);
  if (size == 0) {
    return string();  // unset or empty envvar
  }
  std::unique_ptr<char[]> value(new char[size]);
  ::GetEnvironmentVariableA(key.c_str(), value.get(), size);
  return value.get();
#else
  char* result = getenv(key.c_str());
  return (result == NULL) ? string() : string(result);
#endif
}

string Runfiles::Rlocation(const string& path) const {
  if (path.empty() || starts_with(path, "../") || contains(path, "/..") ||
      starts_with(path, "./") || contains(path, "/./") ||
      ends_with(path, "/.") || contains(path, "//")) {
    return string();
  }
  if (IsAbsolute(path)) {
    return path;
  }
  const auto value = runfiles_map_.find(path);
  if (value != runfiles_map_.end()) {
    return value->second;
  }
  if (!directory_.empty()) {
    return directory_ + "/" + path;
  }
  return "";
}

namespace {

bool ParseManifest(const string& path, map<string, string>* result,
                   string* error) {
  std::ifstream stm(path);
  if (!stm.is_open()) {
    if (error) {
      std::ostringstream err;
      err << "ERROR: " << __FILE__ << "(" << __LINE__
          << "): cannot open runfiles manifest \"" << path << "\"";
      *error = err.str();
    }
    return false;
  }
  string line;
  std::getline(stm, line);
  size_t line_count = 1;
  while (!line.empty()) {
    string::size_type idx = line.find_first_of(' ');
    if (idx == string::npos) {
      if (error) {
        std::ostringstream err;
        err << "ERROR: " << __FILE__ << "(" << __LINE__
            << "): bad runfiles manifest entry in \"" << path << "\" line #"
            << line_count << ": \"" << line << "\"";
        *error = err.str();
      }
      return false;
    }
    (*result)[line.substr(0, idx)] = line.substr(idx + 1);
    std::getline(stm, line);
    ++line_count;
  }
  return true;
}

}  // namespace

namespace testing {

bool TestOnly_PathsFrom(const string& argv0, string mf, string dir,
                        function<bool(const string&)> is_runfiles_manifest,
                        function<bool(const string&)> is_runfiles_directory,
                        string* out_manifest, string* out_directory) {
  return PathsFrom(argv0, mf, dir, is_runfiles_manifest, is_runfiles_directory,
                   out_manifest, out_directory);
}

bool TestOnly_IsAbsolute(const string& path) { return IsAbsolute(path); }

}  // namespace testing

Runfiles* Runfiles::Create(const string& argv0, string* error) {
  return Runfiles::Create(argv0, GetEnv("RUNFILES_MANIFEST_FILE"),
                          GetEnv("RUNFILES_DIR"), error);
}

Runfiles* Runfiles::CreateForTest(std::string* error) {
  return Runfiles::Create(std::string(), GetEnv("RUNFILES_MANIFEST_FILE"),
                          GetEnv("TEST_SRCDIR"), error);
}

namespace {

bool PathsFrom(const string& argv0, string mf, string dir, string* out_manifest,
               string* out_directory) {
  return PathsFrom(
      argv0, mf, dir, [](const string& path) { return IsReadableFile(path); },
      [](const string& path) { return IsDirectory(path); }, out_manifest,
      out_directory);
}

bool PathsFrom(const string& argv0, string mf, string dir,
               function<bool(const string&)> is_runfiles_manifest,
               function<bool(const string&)> is_runfiles_directory,
               string* out_manifest, string* out_directory) {
  out_manifest->clear();
  out_directory->clear();

  bool mfValid = is_runfiles_manifest(mf);
  bool dirValid = is_runfiles_directory(dir);

  if (!argv0.empty() && !mfValid && !dirValid) {
    mf = argv0 + ".runfiles/MANIFEST";
    dir = argv0 + ".runfiles";
    mfValid = is_runfiles_manifest(mf);
    dirValid = is_runfiles_directory(dir);
    if (!mfValid) {
      mf = argv0 + ".runfiles_manifest";
      mfValid = is_runfiles_manifest(mf);
    }
  }

  if (!mfValid && !dirValid) {
    return false;
  }

  if (!mfValid) {
    mf = dir + "/MANIFEST";
    mfValid = is_runfiles_manifest(mf);
    if (!mfValid) {
      mf = dir + "_manifest";
      mfValid = is_runfiles_manifest(mf);
    }
  }

  if (!dirValid &&
      (ends_with(mf, ".runfiles_manifest") || ends_with(mf, "/MANIFEST"))) {
    static const size_t kSubstrLen = 9;  // "_manifest" or "/MANIFEST"
    dir = mf.substr(0, mf.size() - kSubstrLen);
    dirValid = is_runfiles_directory(dir);
  }

  if (mfValid) {
    *out_manifest = mf;
  }

  if (dirValid) {
    *out_directory = dir;
  }

  return true;
}

}  // namespace

}  // namespace runfiles
}  // namespace cpp
}  // namespace tools
}  // namespace bazel
