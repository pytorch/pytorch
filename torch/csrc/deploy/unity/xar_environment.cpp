#include <dirent.h>
#include <dlfcn.h>
#include <fmt/format.h>
#include <sys/stat.h>
#include <torch/csrc/deploy/Exception.h>
#include <torch/csrc/deploy/elf_file.h>
#include <torch/csrc/deploy/unity/xar_environment.h>

namespace torch {
namespace deploy {

XarEnvironment::XarEnvironment(std::string exePath, std::string pythonAppDir)
    : exePath_(std::move(exePath)),
      pythonAppDir_(std::move(pythonAppDir)),
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

bool _fileExists(const std::string& filePath) {
  FILE* fp = fopen(filePath.c_str(), "rb");
  if (fp) {
    fclose(fp);
    return true;
  } else {
    return false;
  }
}

void XarEnvironment::setupPythonApp() {
  MULTIPY_CHECK(
      !alreadySetupPythonApp_,
      "Already setup the python application. It should only been done once!");

  // must match the section name specified in unity.bzl
  constexpr const char* SECTION_NAME = ".torch_deploy_payload.unity";
  ElfFile elfFile(exePath_.c_str());
  auto payloadSection = elfFile.findSection(SECTION_NAME);
  MULTIPY_CHECK(
      payloadSection != multipy::nullopt, "Missing the payload section");
  const char* pythonAppPkgStart = payloadSection->start;
  auto pythonAppPkgSize = payloadSection->len;
  LOG(INFO) << "Embedded binary size " << pythonAppPkgSize;

  /*
   * [NOTE about LD_LIBRARY_PATH]
   * Some python applications uses python extensions that depends on shared
   * libraries in the XAR file. E.g., scipy depends on MKL libraries shipped
   * with the XAR. For those cases, we need ensure 2 things before running the
   * executable:
   * 1, make sure the path /tmp/torch_deploy_python_app/python_app_root exists.
   * 2, add /tmp/torch_deploy_python_app/python_app_root to the LD_LIBRRY_PATH.
   *
   * If either condition is not met, we fail to load the dependent shared
   * libraries in the XAR file.
   *
   * There are simple cases though. If the use case only relies on the libraries
   * built into torch::deploy like torch, numpy, pyyaml etc., or if the
   * extensions used does not rely on extra shared libraries in the XAR file,
   * then neither of the prerequisites need to be met.
   *
   * We used to fatal if the path is not preexisted. But to make (stress)
   * unit-test and other simple uses cases easier, we change it to a warning. If
   * you encounter shared library not found issue, it's likely that your use
   * case are the aforementioned complex case, make sure the two prerequisites
   * are met and run again.
   */
  LOG_IF(WARNING, !_dirExists(pythonAppRoot_))
      << "The python app root " << pythonAppRoot_ << " does not exists before "
      << " running the executable. If you encounter shared libraries not found "
      << " issue, try create the directory and run the executable again. Check "
      << "the note in the code for more details";

  /*
   * NOTE: we remove the pythonAppDir_ below. Anything under it will be gone.
   * Normally the directory just contains potential stuff left over from the
   * past runs. It should be pretty safe to discard them.
   */
  std::string rmCmd = fmt::format("rm -rf {}", pythonAppDir_);
  MULTIPY_CHECK(system(rmCmd.c_str()) == 0, "Fail to remove the directory.");

  // recreate the directory
  auto r = mkdir(pythonAppDir_.c_str(), 0777);
  MULTIPY_CHECK(r == 0, "Failed to create directory: " + strerror(errno));

  std::string pythonAppArchive = std::string(pythonAppDir_) + "/python_app.xar";
  auto fp = fopen(pythonAppArchive.c_str(), "wb");
  MULTIPY_CHECK(fp != nullptr, "Fail to create file: " + strerror(errno));
  auto written = fwrite(pythonAppPkgStart, 1, pythonAppPkgSize, fp);
  MULTIPY_CHECK(written == pythonAppPkgSize, "Expected written == size");
  fclose(fp);

  std::string extractCommand = fmt::format(
      "unsquashfs -o 4096 -d {} {}", pythonAppRoot_, pythonAppArchive);
  r = system(extractCommand.c_str());
  MULTIPY_CHECK(
      r == 0,
      "Fail to extract the python package" + std::to_string(r) +
          extractCommand.c_str());

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
    // only preload the library if it exists in pythonAppRoot_
    auto path = pythonAppRoot_ + "/" + preloadList[i];
    if (!_fileExists(path)) {
      LOG(INFO) << "The preload library " << preloadList[i]
                << " does not exist in the python app root, skip loading it";
      continue;
    }
    MULTIPY_CHECK(
        dlopen(preloadList[i], RTLD_GLOBAL | RTLD_LAZY) != nullptr,
        "Fail to open the shared library " + preloadList[i] + ": " + dlerror());
  }
}

} // namespace deploy
} // namespace torch
