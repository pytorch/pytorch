#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/Vitals.h>
#include <c10/util/irange.h>
#include <cstdlib>

using namespace at::vitals;
using ::testing::HasSubstr;

TEST(Vitals, Basic) {
  std::stringstream buffer;

  std::streambuf* sbuf = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());
  {
#ifdef _WIN32
    _putenv("TORCH_VITAL=1");
#else
    setenv("TORCH_VITAL", "1", 1);
#endif
    TORCH_VITAL_DEFINE(Testing);
    TORCH_VITAL(Testing, Attribute0) << 1;
    TORCH_VITAL(Testing, Attribute1) << "1";
    TORCH_VITAL(Testing, Attribute2) << 1.0f;
    TORCH_VITAL(Testing, Attribute3) << 1.0;
    auto t = at::ones({1, 1});
    TORCH_VITAL(Testing, Attribute4) << t;
  }
  std::cout.rdbuf(sbuf);

  auto s = buffer.str();
  ASSERT_THAT(s, HasSubstr("Testing.Attribute0\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute1\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute2\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute3\t\t 1"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute4\t\t  1"));
}

TEST(Vitals, MultiString) {
  std::stringstream buffer;

  std::streambuf* sbuf = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());
  {
#ifdef _WIN32
    _putenv("TORCH_VITAL=1");
#else
    setenv("TORCH_VITAL", "1", 1);
#endif
    TORCH_VITAL_DEFINE(Testing);
    TORCH_VITAL(Testing, Attribute0) << 1 << " of " << 2;
    TORCH_VITAL(Testing, Attribute1) << 1;
    TORCH_VITAL(Testing, Attribute1) << " of ";
    TORCH_VITAL(Testing, Attribute1) << 2;
  }
  std::cout.rdbuf(sbuf);

  auto s = buffer.str();
  ASSERT_THAT(s, HasSubstr("Testing.Attribute0\t\t 1 of 2"));
  ASSERT_THAT(s, HasSubstr("Testing.Attribute1\t\t 1 of 2"));
}

TEST(Vitals, OnAndOff) {
  for (const auto i : c10::irange(2)) {
    std::stringstream buffer;

    std::streambuf* sbuf = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
    {
#ifdef _WIN32
      if (i) {
        _putenv("TORCH_VITAL=1");
      } else {
        _putenv("TORCH_VITAL=0");
      }
#else
      setenv("TORCH_VITAL", i ? "1" : "", 1);
#endif
      TORCH_VITAL_DEFINE(Testing);
      TORCH_VITAL(Testing, Attribute0) << 1;
    }
    std::cout.rdbuf(sbuf);

    auto s = buffer.str();
    auto f = s.find("Testing.Attribute0\t\t 1");
    if (i) {
      ASSERT_TRUE(f != std::string::npos);
    } else {
      ASSERT_TRUE(f == std::string::npos);
    }
  }
}

TEST(Vitals, APIVitals) {
  std::stringstream buffer;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  bool rvalue;
  std::streambuf* sbuf = std::cout.rdbuf();
  std::cout.rdbuf(buffer.rdbuf());
  {
#ifdef _WIN32
    _putenv("TORCH_VITAL=1");
#else
    setenv("TORCH_VITAL", "1", 1);
#endif
    APIVitals api_vitals;
    rvalue = api_vitals.setVital("TestingSetVital", "TestAttr", "TestValue");
  }
  std::cout.rdbuf(sbuf);

  auto s = buffer.str();
  ASSERT_TRUE(rvalue);
  ASSERT_THAT(s, HasSubstr("TestingSetVital.TestAttr\t\t TestValue"));
}
