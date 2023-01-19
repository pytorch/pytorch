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

#include "tools/runfiles/runfiles_src.h"
#ifdef _WIN32
#include <windows.h>
#endif  // _WIN32

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#define RUNFILES_TEST_TOSTRING_HELPER(x) #x
#define RUNFILES_TEST_TOSTRING(x) RUNFILES_TEST_TOSTRING_HELPER(x)
#define LINE_AS_STRING() RUNFILES_TEST_TOSTRING(__LINE__)

namespace bazel {
namespace tools {
namespace cpp {
namespace runfiles {
namespace {

using bazel::tools::cpp::runfiles::testing::TestOnly_IsAbsolute;
using bazel::tools::cpp::runfiles::testing::TestOnly_PathsFrom;
using std::cerr;
using std::endl;
using std::function;
using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

class RunfilesTest : public ::testing::Test {
 protected:
  // Create a temporary file that is deleted with the destructor.
  class MockFile {
   public:
    // Create an empty file with the given name under $TEST_TMPDIR.
    static MockFile* Create(const string& name);

    // Create a file with the given name and contents under $TEST_TMPDIR.
    // The method ensures to create all parent directories, so `name` is allowed
    // to contain directory components.
    static MockFile* Create(const string& name, const vector<string>& lines);

    ~MockFile();
    const string& Path() const { return path_; }

    string DirName() const {
      string::size_type pos = path_.find_last_of('/');
      return pos == string::npos ? "" : path_.substr(0, pos);
    }

   private:
    MockFile(const string& path) : path_(path) {}
    MockFile(const MockFile&) = delete;
    MockFile(MockFile&&) = delete;
    MockFile& operator=(const MockFile&) = delete;
    MockFile& operator=(MockFile&&) = delete;

    const string path_;
  };

  void AssertEnvvars(const Runfiles& runfiles,
                     const string& expected_manifest_file,
                     const string& expected_directory);

  static string GetTemp();
};

void RunfilesTest::AssertEnvvars(const Runfiles& runfiles,
                                 const string& expected_manifest_file,
                                 const string& expected_directory) {
  vector<pair<string, string> > expected = {
      {"RUNFILES_MANIFEST_FILE", expected_manifest_file},
      {"RUNFILES_DIR", expected_directory},
      {"JAVA_RUNFILES", expected_directory}};
  ASSERT_EQ(runfiles.EnvVars(), expected);
}

string RunfilesTest::GetTemp() {
#ifdef _WIN32
  DWORD size = ::GetEnvironmentVariableA("TEST_TMPDIR", NULL, 0);
  if (size == 0) {
    return string();  // unset or empty envvar
  }
  unique_ptr<char[]> value(new char[size]);
  ::GetEnvironmentVariableA("TEST_TMPDIR", value.get(), size);
  return value.get();
#else
  char* result = getenv("TEST_TMPDIR");
  return result != NULL ? string(result) : string();
#endif
}

RunfilesTest::MockFile* RunfilesTest::MockFile::Create(const string& name) {
  return Create(name, vector<string>());
}

RunfilesTest::MockFile* RunfilesTest::MockFile::Create(
    const string& name, const vector<string>& lines) {
  if (name.find("..") != string::npos || TestOnly_IsAbsolute(name)) {
    cerr << "WARNING: " << __FILE__ << "(" << __LINE__ << "): bad name: \""
         << name << "\"" << endl;
    return nullptr;
  }

  string tmp(RunfilesTest::GetTemp());
  if (tmp.empty()) {
    cerr << "WARNING: " << __FILE__ << "(" << __LINE__
         << "): $TEST_TMPDIR is empty" << endl;
    return nullptr;
  }
  string path(tmp + "/" + name);

  string::size_type i = 0;
#ifdef _WIN32
  while ((i = name.find_first_of("/\\", i + 1)) != string::npos) {
    string d = tmp + "\\" + name.substr(0, i);
    if (!CreateDirectoryA(d.c_str(), NULL)) {
      cerr << "ERROR: " << __FILE__ << "(" << __LINE__
           << "): failed to create directory \"" << d << "\"" << endl;
      return nullptr;
    }
  }
#else
  while ((i = name.find_first_of('/', i + 1)) != string::npos) {
    string d = tmp + "/" + name.substr(0, i);
    if (mkdir(d.c_str(), 0777)) {
      cerr << "ERROR: " << __FILE__ << "(" << __LINE__
           << "): failed to create directory \"" << d << "\"" << endl;
      return nullptr;
    }
  }
#endif

  std::ofstream stm(path);
  for (auto i : lines) {
    stm << i << std::endl;
  }
  return new MockFile(path);
}

RunfilesTest::MockFile::~MockFile() { std::remove(path_.c_str()); }

TEST_F(RunfilesTest, CreatesManifestBasedRunfilesFromManifestNextToBinary) {
  unique_ptr<MockFile> mf(MockFile::Create(
      "foo" LINE_AS_STRING() ".runfiles_manifest", {"a/b c/d"}));
  EXPECT_TRUE(mf != nullptr);
  string argv0(mf->Path().substr(
      0, mf->Path().size() - string(".runfiles_manifest").size()));

  string error;
  unique_ptr<Runfiles> r(Runfiles::Create(argv0, "", "", &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(r->Rlocation("a/b"), "c/d");
  // We know it's manifest-based because it returns empty string for unknown
  // paths.
  EXPECT_EQ(r->Rlocation("unknown"), "");
  AssertEnvvars(*r, mf->Path(), "");
}

TEST_F(RunfilesTest,
       CreatesManifestBasedRunfilesFromManifestInRunfilesDirectory) {
  unique_ptr<MockFile> mf(MockFile::Create(
      "foo" LINE_AS_STRING() ".runfiles/MANIFEST", {"a/b c/d"}));
  EXPECT_TRUE(mf != nullptr);
  string argv0(mf->Path().substr(
      0, mf->Path().size() - string(".runfiles/MANIFEST").size()));

  string error;
  unique_ptr<Runfiles> r(Runfiles::Create(argv0, "", "", &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(r->Rlocation("a/b"), "c/d");
  EXPECT_EQ(r->Rlocation("foo"), argv0 + ".runfiles/foo");
  AssertEnvvars(*r, mf->Path(), argv0 + ".runfiles");
}

TEST_F(RunfilesTest, CreatesManifestBasedRunfilesFromEnvvar) {
  unique_ptr<MockFile> mf(MockFile::Create(
      "foo" LINE_AS_STRING() ".runfiles_manifest", {"a/b c/d"}));
  EXPECT_TRUE(mf != nullptr);

  string error;
  unique_ptr<Runfiles> r(Runfiles::Create("ignore-argv0", mf->Path(),
                                          "non-existent-runfiles_dir", &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(r->Rlocation("a/b"), "c/d");
  // We know it's manifest-based because it returns empty string for unknown
  // paths.
  EXPECT_EQ(r->Rlocation("unknown"), "");
  AssertEnvvars(*r, mf->Path(), "");
}

TEST_F(RunfilesTest, CannotCreateManifestBasedRunfilesDueToBadManifest) {
  unique_ptr<MockFile> mf(MockFile::Create(
      "foo" LINE_AS_STRING() ".runfiles_manifest", {"a b", "nospace"}));
  EXPECT_TRUE(mf != nullptr);

  string error;
  unique_ptr<Runfiles> r(
      Runfiles::Create("ignore-argv0", mf->Path(), "", &error));
  ASSERT_EQ(r, nullptr);
  EXPECT_NE(error.find("bad runfiles manifest entry"), string::npos);
  EXPECT_NE(error.find("line #2: \"nospace\""), string::npos);
}

TEST_F(RunfilesTest, ManifestBasedRunfilesRlocationAndEnvVars) {
  unique_ptr<MockFile> mf(MockFile::Create(
      "foo" LINE_AS_STRING() ".runfiles_manifest", {"a/b c/d"}));
  EXPECT_TRUE(mf != nullptr);

  string error;
  unique_ptr<Runfiles> r(
      Runfiles::Create("ignore-argv0", mf->Path(), "", &error));

  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(r->Rlocation("a/b"), "c/d");
  EXPECT_EQ(r->Rlocation("c/d"), "");
  EXPECT_EQ(r->Rlocation(""), "");
  EXPECT_EQ(r->Rlocation("foo"), "");
  EXPECT_EQ(r->Rlocation("foo/"), "");
  EXPECT_EQ(r->Rlocation("foo/bar"), "");
  EXPECT_EQ(r->Rlocation("../foo"), "");
  EXPECT_EQ(r->Rlocation("foo/.."), "");
  EXPECT_EQ(r->Rlocation("foo/../bar"), "");
  EXPECT_EQ(r->Rlocation("./foo"), "");
  EXPECT_EQ(r->Rlocation("foo/."), "");
  EXPECT_EQ(r->Rlocation("foo/./bar"), "");
  EXPECT_EQ(r->Rlocation("//foo"), "");
  EXPECT_EQ(r->Rlocation("foo//"), "");
  EXPECT_EQ(r->Rlocation("foo//bar"), "");
  EXPECT_EQ(r->Rlocation("/Foo"), "/Foo");
  EXPECT_EQ(r->Rlocation("c:/Foo"), "c:/Foo");
  EXPECT_EQ(r->Rlocation("c:\\Foo"), "c:\\Foo");
}

TEST_F(RunfilesTest, DirectoryBasedRunfilesRlocationAndEnvVars) {
  unique_ptr<MockFile> dummy(
      MockFile::Create("foo" LINE_AS_STRING() ".runfiles/dummy", {"a/b c/d"}));
  EXPECT_TRUE(dummy != nullptr);
  string dir = dummy->DirName();

  string error;
  unique_ptr<Runfiles> r(Runfiles::Create("ignore-argv0", "", dir, &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  EXPECT_EQ(r->Rlocation("a/b"), dir + "/a/b");
  EXPECT_EQ(r->Rlocation("c/d"), dir + "/c/d");
  EXPECT_EQ(r->Rlocation(""), "");
  EXPECT_EQ(r->Rlocation("foo"), dir + "/foo");
  EXPECT_EQ(r->Rlocation("foo/"), dir + "/foo/");
  EXPECT_EQ(r->Rlocation("foo/bar"), dir + "/foo/bar");
  EXPECT_EQ(r->Rlocation("../foo"), "");
  EXPECT_EQ(r->Rlocation("foo/.."), "");
  EXPECT_EQ(r->Rlocation("foo/../bar"), "");
  EXPECT_EQ(r->Rlocation("./foo"), "");
  EXPECT_EQ(r->Rlocation("foo/."), "");
  EXPECT_EQ(r->Rlocation("foo/./bar"), "");
  EXPECT_EQ(r->Rlocation("//foo"), "");
  EXPECT_EQ(r->Rlocation("foo//"), "");
  EXPECT_EQ(r->Rlocation("foo//bar"), "");
  EXPECT_EQ(r->Rlocation("/Foo"), "/Foo");
  EXPECT_EQ(r->Rlocation("c:/Foo"), "c:/Foo");
  EXPECT_EQ(r->Rlocation("c:\\Foo"), "c:\\Foo");
  AssertEnvvars(*r, "", dir);
}

TEST_F(RunfilesTest, ManifestAndDirectoryBasedRunfilesRlocationAndEnvVars) {
  unique_ptr<MockFile> mf(MockFile::Create(
      "foo" LINE_AS_STRING() ".runfiles/MANIFEST", {"a/b c/d"}));
  EXPECT_TRUE(mf != nullptr);
  string dir = mf->DirName();

  string error;
  unique_ptr<Runfiles> r(
      Runfiles::Create("ignore-argv0", mf->Path(), "", &error));

  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());
  EXPECT_EQ(r->Rlocation("a/b"), "c/d");
  EXPECT_EQ(r->Rlocation("c/d"), dir + "/c/d");
  EXPECT_EQ(r->Rlocation(""), "");
  EXPECT_EQ(r->Rlocation("foo"), dir + "/foo");
  EXPECT_EQ(r->Rlocation("foo/"), dir + "/foo/");
  EXPECT_EQ(r->Rlocation("foo/bar"), dir + "/foo/bar");
  EXPECT_EQ(r->Rlocation("../foo"), "");
  EXPECT_EQ(r->Rlocation("foo/.."), "");
  EXPECT_EQ(r->Rlocation("foo/../bar"), "");
  EXPECT_EQ(r->Rlocation("./foo"), "");
  EXPECT_EQ(r->Rlocation("foo/."), "");
  EXPECT_EQ(r->Rlocation("foo/./bar"), "");
  EXPECT_EQ(r->Rlocation("//foo"), "");
  EXPECT_EQ(r->Rlocation("foo//"), "");
  EXPECT_EQ(r->Rlocation("foo//bar"), "");
  EXPECT_EQ(r->Rlocation("/Foo"), "/Foo");
  EXPECT_EQ(r->Rlocation("c:/Foo"), "c:/Foo");
  EXPECT_EQ(r->Rlocation("c:\\Foo"), "c:\\Foo");
  AssertEnvvars(*r, mf->Path(), dir);
}

TEST_F(RunfilesTest, ManifestBasedRunfilesEnvVars) {
  unique_ptr<MockFile> mf(
      MockFile::Create(string("foo" LINE_AS_STRING() ".runfiles_manifest")));
  EXPECT_TRUE(mf != nullptr);

  string error;
  unique_ptr<Runfiles> r(
      Runfiles::Create("ignore-argv0", mf->Path(), "", &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  AssertEnvvars(*r, mf->Path(), "");
}

TEST_F(RunfilesTest, CreatesDirectoryBasedRunfilesFromDirectoryNextToBinary) {
  // We create a directory as a side-effect of creating a mock file.
  unique_ptr<MockFile> mf(
      MockFile::Create(string("foo" LINE_AS_STRING() ".runfiles/dummy")));
  string argv0(mf->Path().substr(
      0, mf->Path().size() - string(".runfiles/dummy").size()));

  string error;
  unique_ptr<Runfiles> r(Runfiles::Create(argv0, "", "", &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  EXPECT_EQ(r->Rlocation("a/b"), argv0 + ".runfiles/a/b");
  // We know it's directory-based because it returns some result for unknown
  // paths.
  EXPECT_EQ(r->Rlocation("unknown"), argv0 + ".runfiles/unknown");
  AssertEnvvars(*r, "", argv0 + ".runfiles");
}

TEST_F(RunfilesTest, CreatesDirectoryBasedRunfilesFromEnvvar) {
  // We create a directory as a side-effect of creating a mock file.
  unique_ptr<MockFile> mf(
      MockFile::Create(string("foo" LINE_AS_STRING() ".runfiles/dummy")));
  string dir = mf->DirName();

  string error;
  unique_ptr<Runfiles> r(Runfiles::Create("ignore-argv0", "", dir, &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  EXPECT_EQ(r->Rlocation("a/b"), dir + "/a/b");
  EXPECT_EQ(r->Rlocation("foo"), dir + "/foo");
  EXPECT_EQ(r->Rlocation("/Foo"), "/Foo");
  EXPECT_EQ(r->Rlocation("c:/Foo"), "c:/Foo");
  EXPECT_EQ(r->Rlocation("c:\\Foo"), "c:\\Foo");
  AssertEnvvars(*r, "", dir);
}

TEST_F(RunfilesTest, FailsToCreateAnyRunfilesBecauseEnvvarsAreNotDefined) {
  unique_ptr<MockFile> mf(
      MockFile::Create(string("foo" LINE_AS_STRING() ".runfiles/MANIFEST")));
  EXPECT_TRUE(mf != nullptr);

  string error;
  unique_ptr<Runfiles> r(
      Runfiles::Create("ignore-argv0", mf->Path(), "whatever", &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  // We create a directory as a side-effect of creating a mock file.
  mf.reset(MockFile::Create(string("foo" LINE_AS_STRING() ".runfiles/dummy")));
  r.reset(Runfiles::Create("ignore-argv0", "", mf->DirName(), &error));
  ASSERT_NE(r, nullptr);
  EXPECT_TRUE(error.empty());

  r.reset(Runfiles::Create("ignore-argv0", "", "", &error));
  ASSERT_EQ(r, nullptr);
  EXPECT_NE(error.find("cannot find runfiles"), string::npos);
}

TEST_F(RunfilesTest, MockFileTest) {
  {
    unique_ptr<MockFile> mf(
        MockFile::Create(string("foo" LINE_AS_STRING() "/..")));
    EXPECT_TRUE(mf == nullptr);
  }

  {
    unique_ptr<MockFile> mf(MockFile::Create(string("/Foo" LINE_AS_STRING())));
    EXPECT_TRUE(mf == nullptr);
  }

  {
    unique_ptr<MockFile> mf(
        MockFile::Create(string("C:/Foo" LINE_AS_STRING())));
    EXPECT_TRUE(mf == nullptr);
  }

  string path;
  {
    unique_ptr<MockFile> mf(
        MockFile::Create(string("foo" LINE_AS_STRING() "/bar1/qux")));
    EXPECT_TRUE(mf != nullptr);
    path = mf->Path();

    std::ifstream stm(path);
    EXPECT_TRUE(stm.good());
    string actual;
    stm >> actual;
    EXPECT_TRUE(actual.empty());
  }
  {
    std::ifstream stm(path);
    EXPECT_FALSE(stm.good());
  }

  {
    unique_ptr<MockFile> mf(MockFile::Create(
        string("foo" LINE_AS_STRING() "/bar2/qux"), vector<string>()));
    EXPECT_TRUE(mf != nullptr);
    path = mf->Path();

    std::ifstream stm(path);
    EXPECT_TRUE(stm.good());
    string actual;
    stm >> actual;
    EXPECT_TRUE(actual.empty());
  }
  {
    std::ifstream stm(path);
    EXPECT_FALSE(stm.good());
  }

  {
    unique_ptr<MockFile> mf(
        MockFile::Create(string("foo" LINE_AS_STRING() "/bar3/qux"),
                         {"hello world", "you are beautiful"}));
    EXPECT_TRUE(mf != nullptr);
    path = mf->Path();

    std::ifstream stm(path);
    EXPECT_TRUE(stm.good());
    string actual;
    std::getline(stm, actual);
    EXPECT_EQ("hello world", actual);
    std::getline(stm, actual);
    EXPECT_EQ("you are beautiful", actual);
    std::getline(stm, actual);
    EXPECT_EQ("", actual);
  }
  {
    std::ifstream stm(path);
    EXPECT_FALSE(stm.good());
  }
}

TEST_F(RunfilesTest, IsAbsolute) {
  EXPECT_FALSE(TestOnly_IsAbsolute("foo"));
  EXPECT_FALSE(TestOnly_IsAbsolute("foo/bar"));
  EXPECT_FALSE(TestOnly_IsAbsolute("\\foo"));
  EXPECT_TRUE(TestOnly_IsAbsolute("c:\\foo"));
  EXPECT_TRUE(TestOnly_IsAbsolute("c:/foo"));
  EXPECT_TRUE(TestOnly_IsAbsolute("/foo"));
  EXPECT_TRUE(TestOnly_IsAbsolute("x:\\foo"));
  EXPECT_FALSE(TestOnly_IsAbsolute("::\\foo"));
  EXPECT_FALSE(TestOnly_IsAbsolute("x\\foo"));
  EXPECT_FALSE(TestOnly_IsAbsolute("x:"));
  EXPECT_TRUE(TestOnly_IsAbsolute("x:\\"));
}

TEST_F(RunfilesTest, PathsFromEnvVars) {
  string mf, dir;

  // Both envvars have a valid value.
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return path == "mock1/MANIFEST"; },
      [](const string& path) { return path == "mock2"; }, &mf, &dir));
  EXPECT_EQ(mf, "mock1/MANIFEST");
  EXPECT_EQ(dir, "mock2");

  // RUNFILES_MANIFEST_FILE is invalid but RUNFILES_DIR is good and there's a
  // runfiles manifest in the runfiles directory.
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return path == "mock2/MANIFEST"; },
      [](const string& path) { return path == "mock2"; }, &mf, &dir));
  EXPECT_EQ(mf, "mock2/MANIFEST");
  EXPECT_EQ(dir, "mock2");

  // RUNFILES_MANIFEST_FILE is invalid but RUNFILES_DIR is good, but there's no
  // runfiles manifest in the runfiles directory.
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return false; },
      [](const string& path) { return path == "mock2"; }, &mf, &dir));
  EXPECT_EQ(mf, "");
  EXPECT_EQ(dir, "mock2");

  // RUNFILES_DIR is invalid but RUNFILES_MANIFEST_FILE is good, and it is in
  // a valid-looking runfiles directory.
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return path == "mock1/MANIFEST"; },
      [](const string& path) { return path == "mock1"; }, &mf, &dir));
  EXPECT_EQ(mf, "mock1/MANIFEST");
  EXPECT_EQ(dir, "mock1");

  // RUNFILES_DIR is invalid but RUNFILES_MANIFEST_FILE is good, but it is not
  // in any valid-looking runfiles directory.
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return path == "mock1/MANIFEST"; },
      [](const string& path) { return false; }, &mf, &dir));
  EXPECT_EQ(mf, "mock1/MANIFEST");
  EXPECT_EQ(dir, "");

  // Both envvars are invalid, but there's a manifest in a runfiles directory
  // next to argv0, however there's no other content in the runfiles directory.
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return path == "argv0.runfiles/MANIFEST"; },
      [](const string& path) { return false; }, &mf, &dir));
  EXPECT_EQ(mf, "argv0.runfiles/MANIFEST");
  EXPECT_EQ(dir, "");

  // Both envvars are invalid, but there's a manifest next to argv0. There's
  // no runfiles tree anywhere.
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return path == "argv0.runfiles_manifest"; },
      [](const string& path) { return false; }, &mf, &dir));
  EXPECT_EQ(mf, "argv0.runfiles_manifest");
  EXPECT_EQ(dir, "");

  // Both envvars are invalid, but there's a valid manifest next to argv0, and a
  // valid runfiles directory (without a manifest in it).
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return path == "argv0.runfiles_manifest"; },
      [](const string& path) { return path == "argv0.runfiles"; }, &mf, &dir));
  EXPECT_EQ(mf, "argv0.runfiles_manifest");
  EXPECT_EQ(dir, "argv0.runfiles");

  // Both envvars are invalid, but there's a valid runfiles directory next to
  // argv0, though no manifest in it.
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return false; },
      [](const string& path) { return path == "argv0.runfiles"; }, &mf, &dir));
  EXPECT_EQ(mf, "");
  EXPECT_EQ(dir, "argv0.runfiles");

  // Both envvars are invalid, but there's a valid runfiles directory next to
  // argv0 with a valid manifest in it.
  EXPECT_TRUE(TestOnly_PathsFrom(
      "argv0", "mock1/MANIFEST", "mock2",
      [](const string& path) { return path == "argv0.runfiles/MANIFEST"; },
      [](const string& path) { return path == "argv0.runfiles"; }, &mf, &dir));
  EXPECT_EQ(mf, "argv0.runfiles/MANIFEST");
  EXPECT_EQ(dir, "argv0.runfiles");
}

}  // namespace
}  // namespace runfiles
}  // namespace cpp
}  // namespace tools
}  // namespace bazel
