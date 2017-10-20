#include "ATen/ATen.h"

using namespace at;

void assertEqualTensorList(TensorList t1, TensorList t2) {
  assert(t1.size() == t2.size());
  for (size_t i = 0; i < t1.size(); ++i) {
    assert(t1[ i ].equal(t2[ i ]));
  }
}

int main() {
  Type & T = CPU(kFloat);

  auto t = T.randn({3, 3});
  // split
  {
    // test method, type, namespace give same result
    auto splitMethod = t.split(1, 0);
    auto splitType = T.split(t, 1, 0);
    auto splitNs = at::split(t, 1, 0);
    assertEqualTensorList(splitMethod, splitType);
    assertEqualTensorList(splitMethod, splitNs);

    // test rebuilding with cat
    assert(at::cat(splitMethod, 0).equal(t));
  }

  {
    // test method, type, namespace give same result
    auto chunkMethod = t.chunk(3, 0);
    auto chunkType = T.chunk(t, 3, 0);
    auto chunkNs = at::chunk(t, 3, 0);
    assertEqualTensorList(chunkMethod, chunkType);
    assertEqualTensorList(chunkMethod, chunkNs);

    // test rebuilding with cat
    assert(at::cat(chunkMethod, 0).equal(t));
  }

  return 0;
}
