#pragma once

namespace tlib {

class Type;
struct Tensor {
  virtual Type & type() const = 0;
  virtual ~Tensor() {}
};

}
