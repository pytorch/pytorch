#ifndef THNN_POOLING_SHAPE_H
#define THNN_POOLING_SHAPE_H

template<typename T>
static inline T pooling_output_shape(
        T inputSize, T kernelSize, T pad, T stride, T dilation, bool ceil_mode) {
    T outputSize = ((inputSize + 2 * pad - dilation * (kernelSize - 1) - 1 + (ceil_mode ? stride - 1 : 0)) / stride + 1);
    if (pad) {
        // ensure that the last pooling starts inside the image
        // needed to avoid problems in ceil mode
        if ((outputSize - 1) * stride >= inputSize + pad)
          --outputSize;
    }
    return outputSize;
}

#endif
