#ifndef THNN_POOLING_SHAPE_H
#define THNN_POOLING_SHAPE_H

template<typename T>
static inline T pooling_output_shape_pad_lr(
    T inputSize, T kernelSize, T pad_l, T pad_r, T stride, T dilation,
    bool ceil_mode
  ) {
    T outputSize = ((inputSize + pad_l + pad_r - dilation * (kernelSize - 1)
        - 1 + (ceil_mode ? stride - 1 : 0)) / stride + 1);
    if (pad_l) {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride >= inputSize + pad_l)
          --outputSize;
    }
    return outputSize;
}

template<typename T>
static inline T pooling_output_shape(
      T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    return pooling_output_shape_pad_lr(
        inputSize, kernelSize, pad, pad, stride, dilation, ceil_mode);
}

#endif
