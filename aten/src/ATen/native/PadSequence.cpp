#include <ATen/ATen.h>

namespace at {
namespace native {
  Tensor pad_sequence(TensorList sequences, bool batch_first, double padding_value) {
    const auto sequences_size = sequences.size();
    TORCH_CHECK(sequences_size > 0, "received an empty list of sequences");
    IntArrayRef max_size = sequences[0].sizes();
    IntArrayRef trailing_dims = max_size.slice(1);
    int64_t max_len = std::max_element(
      sequences.begin(),
      sequences.end(),
      [](const Tensor &a, const Tensor &b) {
        return a.size(0) < b.size(0);
      }
    )->size(0);

    DimVector out_dims;
    if (batch_first) {
      out_dims = {sequences_size, max_len};
    } else {
      out_dims = {max_len, sequences_size};
    }
    out_dims.insert(out_dims.end(), trailing_dims.begin(), trailing_dims.end());
    
    Tensor out = at::full(out_dims, padding_value, sequences[0].options());
    for (auto i = 0; i < sequences_size; i++) {
      const Tensor currseq = sequences[i];
      const int64_t length_i = currseq.size(0);
      // use index notation to prevent duplicate references to the tensor
      if (batch_first) {
        out.select(0, i).narrow(0, 0, length_i).copy_(currseq);
      } else {
	out.narrow(0, 0, length_i).select(1, i).copy_(currseq);
      }
    }
    return out;
  }
} // at
} // native
