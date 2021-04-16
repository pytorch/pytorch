#include <torch/csrc/jit/mobile/import.h>

#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/compilation_unit.h>
#include <torch/csrc/jit/mobile/import.h>
#include <torch/csrc/jit/mobile/observer.h>
#include <torch/csrc/jit/runtime/instruction.h>
#include <torch/csrc/jit/serialization/export.h>
#include <torch/csrc/jit/serialization/import_export_constants.h>
#include <torch/csrc/jit/serialization/unpickler.h>
#include <torch/custom_class.h>

#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace c10 {
// std::string serializeType(const Type &t);
TypePtr parseType(const std::string& pythonStr);
} // namespace c10

namespace torch {
namespace jit {
using caffe2::serialize::IStreamAdapter;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::ReadAdapterInterface;




// Forward declare so that _backport_for_mobile() overloads can
// call this method directly.
void _backport_for_mobile_impl(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options);

void _backport_for_mobile(
    std::istream& in,
    c10::optional<at::Device> device) {
  ExtraFilesMap extra_files;
  return _backport_for_mobile(in, device, extra_files);
}

void _backport_for_mobile(
    const std::string& input_file_name,
    const std::string& output_file_name,
    const uint64_t version,
    c10::optional<at::Device> device) {
  ExtraFilesMap extra_files;
  return _backport_for_mobile(input_file_name, output_file_name, version, device, extra_files);
}

void _backport_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device) {
  ExtraFilesMap extra_files;
  return _backport_for_mobile(std::move(rai), device, extra_files);
}

void _backport_for_mobile(
    std::istream& in,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  std::unique_ptr<IStreamAdapter> rai = std::make_unique<IStreamAdapter>(&in);
  auto module = _backport_for_mobile(std::move(rai), device, extra_files);
  return module;
}

void _backport_for_mobile(
    const std::string& input_file_name,
    const std::string& output_file_name,
    const uint64_t version,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(input_file_name);
  _backport_for_mobile(std::move(rai), device, extra_files);
}

void _backport_for_mobile(
    const std::string& filename,
    c10::optional<at::Device> device,
    ExtraFilesMap& extra_files,
    uint64_t module_load_options) {
  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
  auto module = _backport_for_mobile_impl(
      std::move(rai), device, extra_files, module_load_options);
  return module;
}

void _backport_for_mobile(
    std::unique_ptr<ReadAdapterInterface> rai,
    c10::optional<c10::Device> device,
    ExtraFilesMap& extra_files) {
  std::cout << "place holder" << std::endl;
//  auto module = _backport_for_mobile_impl(
//      std::move(rai), device, extra_files, _default_mobile_module_load_options);
//  return module;
}

//mobile::Module _backport_for_mobile_impl(
//    std::unique_ptr<ReadAdapterInterface> rai,
//    c10::optional<c10::Device> device,
//    ExtraFilesMap& extra_files,
//    uint64_t module_load_options) {
//  auto observer = torch::observerConfig().getModuleObserver();
//  auto instance_key = std::rand();
//  if (observer) {
//    observer->onEnterLoadModel(instance_key);
//  }
//  const size_t model_size = rai != nullptr ? rai->size() : 0;
//  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
//  BytecodeDeserializer deserializer(std::move(reader), module_load_options);
//  try {
//    mobile::Module result = deserializer.deserialize(device, extra_files);
//    std::unordered_map<std::string, std::string> copied_metadata =
//        result.metadata();
//    if (result.metadata().find("model_name") == result.metadata().end()) {
//      copied_metadata["model_name"] = result.name();
//    }
//    copied_metadata["model_size"] = c10::guts::to_string(model_size);
//    if (observer) {
//      observer->onExitLoadModel(instance_key, copied_metadata);
//    }
//    return result;
//  } catch (c10::Error& error) {
//    if (observer) {
//      observer->onFailLoadModel(
//          instance_key,
//          error.what(),
//          deserializer.deserializeMetadata(std::move(device)));
//    }
//    TORCH_RETHROW(error);
//  } catch (...) {
//    auto currentException = std::current_exception();
//    try {
//      if (!currentException) {
//        TORCH_CHECK(false, "Unknown exception");
//      } else {
//        try {
//          std::rethrow_exception(currentException);
//        } catch (const std::exception& e) {
//          TORCH_CHECK(false, e.what());
//        }
//      }
//    } catch (c10::Error& error) {
//      if (observer) {
//        observer->onFailLoadModel(
//            instance_key,
//            error.what(),
//            deserializer.deserializeMetadata(std::move(device)));
//      }
//      TORCH_RETHROW(error);
//    }
//  }
//}
//
//void _load_extra_only_for_mobile(
//    const std::string& filename,
//    c10::optional<at::Device> device,
//    ExtraFilesMap& extra_files) {
//  std::unique_ptr<FileAdapter> rai = std::make_unique<FileAdapter>(filename);
//  auto observer = torch::observerConfig().getModuleObserver();
//  auto instance_key = std::rand();
//  if (observer) {
//    observer->onEnterLoadModel(instance_key);
//  }
//  auto reader = torch::make_unique<PyTorchStreamReader>(std::move(rai));
//  BytecodeDeserializer deserializer(std::move(reader));
//  deserializer.deserialize_only_extra(device, extra_files);
//}

} // namespace jit
} // namespace torch
