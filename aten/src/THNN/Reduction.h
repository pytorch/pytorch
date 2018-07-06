#ifndef REDUCE_H
#define REDUCE_H

namespace Reduction {

// NB: Keep this in sync with Reduction class in torch/nn/modules/functional.py
// These constants control the reduction behavior of loss functions.
// Ideally, this would be a scoped enum, but jit doesn't support that
enum Reduction {
  None,             // Do not reduce
  ElementwiseMean,  // Sum losses and take mean over each individually computed loss element
  Sum,              // Sum losses
  END
};
}

#endif
