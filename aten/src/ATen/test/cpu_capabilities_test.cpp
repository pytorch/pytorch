#include <gtest/gtest.h>

#include <ATen/cpu/Utils.h>
#include <torch/headeronly/cpu/Utils.h>

#include <cstdint>
#include <string>
#include <type_traits>
#include <variant>

TEST(CPUCapabilitiesTest, HeaderOnlyValuesArePresentInATenValues) {
  const auto headeronly_capabilities =
      torch::headeronly::cpu::headeronly_get_cpu_capabilities();
  const auto aten_capabilities = at::cpu::get_cpu_capabilities();

  for (const auto& [key, value] : headeronly_capabilities) {
    const auto aten_entry = aten_capabilities.find(key);
    // If the key is missing, report it but keep going to check the rest of the keys.
    EXPECT_NE(aten_entry, aten_capabilities.end()) << key;
    if (aten_entry == aten_capabilities.end()) {
      continue;
    }

    // Need to convert from IValue to the expected type.
    std::visit(
        [&](const auto& headeronly_value) {
          using Value = std::decay_t<decltype(headeronly_value)>;
          if constexpr (std::is_same_v<Value, bool>) {
            ASSERT_TRUE(aten_entry->second.isBool()) << key;
            EXPECT_EQ(aten_entry->second.toBool(), headeronly_value) << key;
          } else if constexpr (std::is_same_v<Value, std::int64_t>) {
            ASSERT_TRUE(aten_entry->second.isInt()) << key;
            EXPECT_EQ(aten_entry->second.toInt(), headeronly_value) << key;
          } else if constexpr (std::is_same_v<Value, std::string>) {
            ASSERT_TRUE(aten_entry->second.isString()) << key;
            EXPECT_EQ(aten_entry->second.toStringRef(), headeronly_value)
                << key;
          }
        },
        value);
  }
}
