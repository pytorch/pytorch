#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>

#include <sstream>

#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/import.h>
#include <torch/csrc/jit/import_source.h>
#include <torch/torch.h>

namespace torch {
namespace jit {
using namespace script;

void testSaveExtraFilesHook() {
  // no secrets
  {
    std::stringstream ss;
    {
      Module m("__torch__.m");
      ExtraFilesMap extra;
      extra["metadata.json"] = "abc";
      m.save(ss, extra);
    }
    ss.seekg(0);
    {
      ExtraFilesMap extra;
      extra["metadata.json"] = "";
      extra["secret.json"] = "";
      jit::load(ss, c10::nullopt, extra);
      ASSERT_EQ(extra["metadata.json"], "abc");
      ASSERT_EQ(extra["secret.json"], "");
    }
  }
  // some secret
  {
    std::stringstream ss;
    {
      SetExportModuleExtraFilesHook([](const Module&) -> ExtraFilesMap {
        return {{"secret.json", "topsecret"}};
      });
      Module m("__torch__.m");
      ExtraFilesMap extra;
      extra["metadata.json"] = "abc";
      m.save(ss, extra);
      SetExportModuleExtraFilesHook(nullptr);
    }
    ss.seekg(0);
    {
      ExtraFilesMap extra;
      extra["metadata.json"] = "";
      extra["secret.json"] = "";
      jit::load(ss, c10::nullopt, extra);
      ASSERT_EQ(extra["metadata.json"], "abc");
      ASSERT_EQ(extra["secret.json"], "topsecret");
    }
  }
}

static const auto pretty_printed = R"JIT(
op_version_set = 1000
def foo(x: Tensor,
    y: Tensor) -> Tensor:
  _0 = torch.add(torch.mul(x, 2), y, alpha=1)
  return _0
)JIT";

void testImportTooNew() {
  Module m("__torch__.m");
  const std::vector<at::Tensor> constant_table;
  auto src = std::make_shared<Source>(pretty_printed);
  SourceImporter si(m.class_compilation_unit(), &constant_table, nullptr);
  ASSERT_ANY_THROW(si.LEGACY_import_methods(m, src));
}

} // namespace jit
} // namespace torch
