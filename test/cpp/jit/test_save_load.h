#pragma once

#include <test/cpp/jit/test_base.h>
#include <test/cpp/jit/test_utils.h>

#include <sstream>

#include <torch/csrc/jit/export.h>
#include <torch/csrc/jit/import.h>
#include <torch/torch.h>

namespace torch {
namespace jit {
namespace script {

void testSaveExtraFilesHook() {
  // no secrets
  {
    std::stringstream ss;
    {
      Module m("m");
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
      Module m("m");
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

} // namespace script
} // namespace jit
} // namespace torch
