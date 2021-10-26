#include <torch/csrc/deploy/unity/unity.h>

namespace torch {
namespace deploy {

int doMain(int argc, char** argv) {
  Unity unity(2);
  unity.runMainModule();
  return 0;
}

} // namespace deploy
} // namespace torch

int main(int argc, char** argv) {
  return torch::deploy::doMain(argc, argv);
}
