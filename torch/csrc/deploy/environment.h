#pragma once
#include <fmt/format.h>
#include <torch/csrc/deploy/elf_file.h>
#include <fstream>
#include <string>

namespace torch {
namespace deploy {

class Interpreter;

/*
 * An environment is the concept to decribe the circumstances in which a
 * torch::deploy interpreter runs. In can be an xar file embedded in the binary,
 * a filesystem path for the installed libraries etc.
 */
class Environment {
  std::vector<std::string> extraPythonPaths_;
  // all zipped python libraries will be written
  // under this directory
  std::string extraPythonLibrariesDir_;
  void setupZippedPythonModules(const std::string& pythonAppDir) {
#ifdef FBCODE_CAFFE2
    std::string execPath;
    std::ifstream("/proc/self/cmdline") >> execPath;
    ElfFile elfFile(execPath.c_str());
    // load the zipped torch modules
    constexpr const char* ZIPPED_TORCH_NAME = ".torch_python_modules";
    auto zippedTorchSection = elfFile.findSection(ZIPPED_TORCH_NAME);
    TORCH_CHECK(
        zippedTorchSection.has_value(), "Missing the zipped torch section");
    const char* zippedTorchStart = zippedTorchSection->start;
    auto zippedTorchSize = zippedTorchSection->len;

    std::string zipArchive =
        std::string(pythonAppDir) + "/torch_python_modules.zip";
    auto zippedFile = fopen(zipArchive.c_str(), "wb");
    TORCH_CHECK(
        zippedFile != nullptr, "Fail to create file: ", strerror(errno));
    fwrite(zippedTorchStart, 1, zippedTorchSize, zippedFile);
    fclose(zippedFile);

    extraPythonPaths_.push_back(zipArchive);
#endif
    extraPythonLibrariesDir_ = pythonAppDir;
  }

 public:
  explicit Environment() {
    char tempDirName[] = "/tmp/torch_deploy_zipXXXXXX";
    char* tempDirectory = mkdtemp(tempDirName);
    setupZippedPythonModules(tempDirectory);
  }
  explicit Environment(const std::string& pythonAppDir) {
    setupZippedPythonModules(pythonAppDir);
  }
  virtual ~Environment() {
    auto rmCmd = fmt::format("rm -rf {}", extraPythonLibrariesDir_);
    system(rmCmd.c_str());
  }
  virtual void configureInterpreter(Interpreter* interp) = 0;
  virtual const std::vector<std::string>& getExtraPythonPaths() {
    return extraPythonPaths_;
  }
};

} // namespace deploy
} // namespace torch
