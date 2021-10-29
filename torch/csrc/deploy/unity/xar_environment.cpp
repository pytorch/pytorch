#include <dirent.h>
#include <dlfcn.h>
#include <fmt/format.h>
#include <sys/stat.h>
#include <torch/csrc/deploy/unity/xar_environment.h>

namespace torch {
namespace deploy {

XarEnvironment::XarEnvironment(std::string pythonAppDir)
    : pythonAppDir_(std::move(pythonAppDir)),
      pythonAppRoot_(pythonAppDir_ + "/python_app_root") {
  setupPythonApp();
  preloadSharedLibraries();
}

// NOLINTNEXTLINE(modernize-use-equals-default)
XarEnvironment::~XarEnvironment() {
  // We should delete the pythonAppDir_ here. However if we did that, the
  // next time we run the executable, we will get issue to load shared
  // libraries since the path we add to LD_LIBRARY_PATH does not exist yet.
  // Also the pythonAppDir_ will anyway be re-created the next time we run the
  // executable.
  //
  // Keep the teardown step a noop for now.
}

void XarEnvironment::configureInterpreter(Interpreter* interp) {
  auto I = interp->acquireSession();
  I.global("sys", "path").attr("append")({pythonAppRoot_});
}

/*
 * I can not use std::filesystem since that's added in C++17 and clang-tidy
 * seems using a older version of C++ and can not find it.
 *
 * Create a small utility to check the existence of a directory instead.
 */
bool _dirExists(const std::string& dirPath) {
  DIR* dir = opendir(dirPath.c_str());
  if (dir) {
    closedir(dir);
    return true;
  } else {
    return false;
  }
}

extern "C" char _binary_python_app_start[];
extern "C" char _binary_python_app_end[];

void XarEnvironment::setupPythonApp() {
  TORCH_CHECK(
      !alreadySetupPythonApp_,
      "Already setup the python application. It should only been done once!");

  auto pythonAppPkgSize = _binary_python_app_end - _binary_python_app_start;
  LOG(INFO) << "Embedded binary size " << pythonAppPkgSize;

  /*
   * NOTE, we need set /tmp/torch_deploy_python_app/python_app_root as
   * LD_LIBRARY_PATH when running this program. Another prerequisite is the path
   * must exist before running this program. Otherwise the dynamic loader will
   * remove the path and we get an empty LD_LIBRARY_PATH! That will cause
   * dynamic library not found issue.
   */
  TORCH_CHECK(
      _dirExists(pythonAppRoot_),
      fmt::format(
          "The python app root {} must exist before running the binary. Otherwise there will be issues to find dynamic libraries. Please create the directory and rerun",
          pythonAppRoot_));

  /*
   * NOTE: we remove the pythonAppDir_ below. Anything under it will be gone.
   * Normally the directory just contains potential stuff left over from the
   * past runs. It should be pretty safe to discard them.
   */
  std::string rmCmd = fmt::format("rm -rf {}", pythonAppDir_);
  TORCH_CHECK(system(rmCmd.c_str()) == 0, "Fail to remove the directory.");

  // recreate the directory
  auto r = mkdir(pythonAppDir_.c_str(), 0777);
  TORCH_CHECK(r == 0, "Failed to create directory: ", strerror(errno));

  std::string pythonAppArchive = std::string(pythonAppDir_) + "/python_app.xar";
  auto fp = fopen(pythonAppArchive.c_str(), "wb");
  TORCH_CHECK(fp != nullptr, "Fail to create file: ", strerror(errno));
  auto written = fwrite(_binary_python_app_start, 1, pythonAppPkgSize, fp);
  TORCH_CHECK(written == pythonAppPkgSize, "Expected written == size");
  fclose(fp);

  std::string extractCommand = fmt::format(
      "unsquashfs -o 4096 -d {} {}", pythonAppRoot_, pythonAppArchive);
  r = system(extractCommand.c_str());
  TORCH_CHECK(r == 0, "Fail to extract the python package");

  alreadySetupPythonApp_ = true;
}

void XarEnvironment::preloadSharedLibraries() {
  // preload the following libraries since the CustomLoader has some limitations
  // 1. CustomLoader can not find the correct order to loader them
  // 2. CustomLoader use RTLD_LOCAL so the symbol defined in one lib can not be
  // used by another
  std::array<const char*, 3> preloadList = {
      "libmkl_core.so", "libmkl_intel_thread.so", nullptr};
  for (int i = 0; preloadList[i]; ++i) {
    TORCH_CHECK(
        dlopen(preloadList[i], RTLD_GLOBAL | RTLD_LAZY) != nullptr,
        "Fail to open the shared library ",
        preloadList[i],
        ": ",
        dlerror());
  }
}

} // namespace deploy
} // namespace torch
