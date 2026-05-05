#include <gtest/gtest.h>
#include <torch/torch.h>
#include <iostream>
#include <sstream>

TEST(PrintSciModeTest, ToggleScientificNotation) {
  auto t = torch::tensor({0.00001, 100000.0});

  // Test with scientific notation enabled
  torch::set_printoption_sci_mode(true);
  std::ostringstream oss1;
  oss1 << t;
  auto out1 = oss1.str();
  std::cout << "With sci_mode=true: '" << out1 << "'" << std::endl;
  EXPECT_TRUE(
      out1.find("e-") != std::string::npos ||
      out1.find("e+") != std::string::npos);

  // Test with scientific notation disabled
  torch::set_printoption_sci_mode(false);
  std::ostringstream oss2;
  oss2 << t;
  auto out2 = oss2.str();
  std::cout << "With sci_mode=false: '" << out2 << "'" << std::endl;
  EXPECT_TRUE(
      out2.find("e-") == std::string::npos &&
      out2.find("e+") == std::string::npos);
}
