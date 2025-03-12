#include "torch/csrc/nativert/executor/DelegateExecutor.h"

#include <unistd.h>

#include "c10/util/Logging.h"

#include "torch/csrc/nativert/common/FileUtil.h"
#include "torch/csrc/nativert/common/String.h"

namespace torch::nativert {

std::string extractToTemporaryFolder(
    std::shared_ptr<caffe2::serialize::PyTorchStreamReader> packageReader,
    const std::string& targetPath) {
  char outputDir[] = "/tmp/model_XXXXXX";
  char* tempdir = mkdtemp(outputDir);
  TORCH_CHECK(
      tempdir != nullptr,
      "error creating temporary directory for compiled model. errno: ",
      errno);

  std::vector<std::string> allRecords = packageReader->getAllRecords();

  for (const auto& path : allRecords) {
    if (!starts_with(path, targetPath) || ends_with(path, "/")) {
      continue;
    }

    TORCH_CHECK(
        packageReader->hasRecord(path), path, " not present in model package");
    auto [dataPointer, dataSize] = packageReader->getRecord(path);

    std::string fileName = path.substr(path.rfind('/') + 1);
    std::string extractedFilename = std::string(outputDir) + "/" + fileName;

    VLOG(1) << "Extracting " << extractedFilename
            << " from archive path: " << path << " size: " << dataSize;

    File extracted(extractedFilename, O_CREAT | O_WRONLY, 0640);
    const auto bytesWritten =
        writeFull(extracted.fd(), dataPointer.get(), dataSize);
    TORCH_CHECK(
        bytesWritten != -1,
        "failure copying from archive path ",
        path,
        " to temporary file");
  }

  return std::string(outputDir);
}

} // namespace torch::nativert
