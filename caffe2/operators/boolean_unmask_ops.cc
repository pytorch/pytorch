#include "caffe2/operators/boolean_unmask_ops.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/tensor.h"

namespace caffe2 {

template <>
bool BooleanUnmaskOp<CPUContext>::RunOnDevice() {
  int maskSize = Input(0).numel();
  int numMasks = InputSize() / 2;
  auto& valueMeta = Input(1).dtype();

  auto* valuesOut = Output(0);
  valuesOut->Resize(maskSize);
  auto* valuesOutPtr = (char*)valuesOut->raw_mutable_data(valueMeta);

  std::vector<int> nextValueIndices(numMasks, 0);
  for (int maskOffset = 0; maskOffset < maskSize; ++maskOffset) {
    bool maskFound = false;
    for (int maskIndex = 0; maskIndex < numMasks; ++maskIndex) {
      auto& mask = Input(maskIndex * 2);
      CAFFE_ENFORCE_EQ(mask.dim(), 1);
      CAFFE_ENFORCE_EQ(mask.numel(), maskSize);
      const auto* maskPtr = mask.template data<bool>();

      auto& values = Input(maskIndex * 2 + 1);
      CAFFE_ENFORCE_EQ(values.dim(), 1);
      const auto* valuesPtr = (char*)values.raw_data();

      if (maskPtr[maskOffset]) {
        auto& valueIndex = nextValueIndices[maskIndex];
        CAFFE_ENFORCE_LT(valueIndex, values.numel());
        auto* src = valuesPtr + (valueIndex++) * valueMeta.itemsize();
        auto* dst = valuesOutPtr + maskOffset * valueMeta.itemsize();
        std::copy(src, src + valueMeta.itemsize(), dst);
        maskFound = true;
        break;
      }
    }
    CAFFE_ENFORCE(
        maskFound, "All masks have False at position ", maskOffset, ".");
  }
  // check all indices match value length
  for (int i = 0; i < numMasks; ++i) {
    auto& values = Input(i * 2 + 1);
    CAFFE_ENFORCE_EQ(
        values.numel(),
        nextValueIndices[i],
        "The number of true at mask ",
        i,
        " does not match the corresponding value size.");
  }
  return true;
}

REGISTER_CPU_OPERATOR(BooleanUnmask, BooleanUnmaskOp<CPUContext>);

OPERATOR_SCHEMA(BooleanUnmask)
    .NumInputs([](int n) { return n > 0 && n % 2 == 0; })
    .NumOutputs(1)
    .SetDoc(R"DOC(
Given a series of masks and values, reconstruct values together according to masks. A comprehensive example:
```
mask1   = True, False, True, False, False
values1 = 1.0, 3.0
mask2   = False, True, False, False, False
values2 = 2.0
mask3   = False, False, False, True, True
values3 = 4.0, 5.0
```

Reconstruct by:

```
output = net.BooleanUnmask([mask1, values1, mask2, values2, mask3, values3], ["output"])
output = 1.0, 2.0, 3.0, 4.0, 5.0
```

Note that for all mask positions, there must be at least one True. This is not allowed:

```
mask1   = True, False
values1 = 1.0
mask2   = False, False
values2 =

output = net.BooleanUnmask([mask1, values1, mask2, values2], ["output"])
```

If there are multiple True values for a field, we accept the first value, and no longer expect a value for that location:

```
mask1   = True, False
values1 = 1.0
mask2   = True, True
values2 = 2.0

output = net.BooleanUnmask([mask1, values1, mask2, values2], ["output"])
output = 1.0, 2.0
```

*** Note that we alternate `data` and `mask` inputs

Github Links:
- https://github.com/pytorch/pytorch/blob/main/caffe2/operators/boolean_unmask_ops.cc

<details>

<summary> <b>Example</b> </summary>

**Code**

```

workspace.ResetWorkspace()

op = core.CreateOperator(
    "BooleanUnmask",
    ["mask1", "data1", "mask2", "data2"],
    ["unmasked_data"]
)

workspace.FeedBlob("mask1", np.array([True,False,False,True,True,False]))
workspace.FeedBlob("data1", np.array([1,4,5]))
workspace.FeedBlob("mask2", np.array([False,True,True,False,False,True]))
workspace.FeedBlob("data2", np.array([2,3,6]))

print("data1:", workspace.FetchBlob("data1"))
print("mask1:", workspace.FetchBlob("mask1"))
print("data2:", workspace.FetchBlob("data2"))
print("mask2:", workspace.FetchBlob("mask2"))
workspace.RunOperatorOnce(op)
print("unmasked_data:", workspace.FetchBlob("unmasked_data"))

```

**Result**

```

data1: [1 4 5]
mask1: [ True False False  True  True False]
data2: [2 3 6]
mask2: [False  True  True False False  True]
unmasked_data: [1 2 3 4 5 6]

```

</details>
)DOC")
    .Input(0,"data","(*Tensor*): 1D input tensor(s)")
    .Input(1,"mask","(*Tensor`<bool>`*): 1D boolean mask tensor(s)")
    .Output(0, "unmasked_data", "(*Tensor*): 1D tensor of same type as `data` input that contains the unmasked input tensor");

NO_GRADIENT(BooleanUnmask)
}
