#include <tuple>
#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"

namespace at {
namespace native {

template <typename scalar_t>
Tensor MaxUnpooling2d_forward_out_cpu_(
    Tensor& output,
    const Tensor& input,
    const Tensor& indices,
    int64_t outputHeight,
    int64_t outputWidth) {
  int64_t numBatch = 1;
  int64_t dimc = 0;
  int64_t dimh = 1;
  int64_t dimw = 2;
  if(input.ndimension() == 4) {
    numBatch = input.size(0);
    dimc++;
    dimh++;
    dimw++;
  }
  auto numChannels = input.size(dimc);
  auto inputHeight = input.size(dimh);
  auto inputWidth = input.size(dimw);

  auto* rawInput = input.data<scalar_t>();
  auto* rawIndices = indices.data<int64_t>();
  auto* rawOutput = output.data<scalar_t>();

  for (auto n = 0; n < numBatch; n++) {
    auto nOutputOffset = n * numChannels * outputWidth * outputHeight;
    auto nInputOffset = n * numChannels * inputWidth * inputHeight;
    int k;
    int has_error = 0;
    int error_index = 0;
#pragma omp parallel for private(k)
    for (k = 0; k < numChannels; k++) {
      auto finalOutputOffset = nOutputOffset + k * outputWidth * outputHeight;
      auto finalInputOffset = nInputOffset + k * inputWidth * inputHeight;
      auto* output_p_k = rawOutput + finalOutputOffset;
      auto* input_p_k = rawInput + finalInputOffset;
      auto* ind_p_k = rawIndices + finalInputOffset;

      int maxp;
      for (auto i = 0; i < inputHeight; i++) {
        for (auto j = 0; j < inputWidth; j++) {
          maxp = ind_p_k[i * inputWidth + j];
          if (maxp < 0 || maxp >= outputWidth * outputHeight) {
#pragma omp critical
            {
              has_error = 1;
              error_index = maxp;
            }
          } else {
            output_p_k[maxp] = input_p_k[i * inputWidth + j];
          }
        }
      }
    }
    if (has_error) {
      AT_ERROR(
          "Found an invalid max index %ld (output volumes are of size %dx%d)",
          error_index,
          outputHeight,
          outputWidth);
    }
  }
  return output;
}

Tensor& MaxUnpooling2d_forward_out_cpu(
    Tensor& output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  AT_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");
  AT_CHECK(
      self.ndimension() == 4,
      "Input to MaxUnpooling2d should be a NCHW Tensor");
  AT_CHECK(
      self.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");

  // is_empty check
  AT_CHECK(self.numel() > 0,
    "Input must be non-empty");

  auto outputHeight = output_size[0];
  auto outputWidth = output_size[1];


  int64_t numBatch = 1;
  int64_t numChannels;
  if(self.ndimension() == 4) {
    numBatch = self.size(0);
    numChannels = self.size(1);
  }
  else {
    numChannels = self.size(0);
  }

  auto self_contiguous = self.contiguous();
  auto indices_contiguous = indices.contiguous();

  if(self_contiguous.ndimension() == 3)
  {
    output.resize_({numChannels, outputHeight, outputWidth});
  }
  else {
    output.resize_({numBatch, numChannels, outputHeight, outputWidth});
  }
  output.zero_();

  AT_DISPATCH_FLOATING_TYPES(
      self.type(), "MaxUnpooling2d_forward_out_cpu_", ([&] {
        MaxUnpooling2d_forward_out_cpu_<scalar_t>(
            output,
            self_contiguous,
            indices_contiguous,
            output_size[0],
            output_size[1]);
      }));
  return output;
};

Tensor MaxUnpooling2d_forward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  AT_CHECK(
      (self.ndimension() == 4 || self.ndimension() == 5),
      "Input to MaxUnpooling2d should be a 4d or 5d Tensor",
      self.sizes());
  AT_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");

  auto output = at::zeros(
      {self.size(0), self.size(1), output_size[0], output_size[1]},
      self.options());
  MaxUnpooling2d_forward_out_cpu(output, self, indices, output_size);
  return output;
<<<<<<< HEAD
=======
};

Tensor MaxUnpooling2d_backward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntList output_size) {
  AT_ERROR("not implemented");
>>>>>>> Add maxunpooling forward functions in CUDA
}

template <typename scalar_t>
Tensor MaxUnpooling3d_forward_out_cpu_(
    Tensor& output,
    const Tensor& input,
    const Tensor& indices,
    int64_t oT,
    int64_t oW,
    int64_t oH,
    int64_t dT,
    int64_t dW,
    int64_t dH,
    int64_t pT,
    int64_t pW,
    int64_t pH) {
  auto nBatch = 1;
  auto dimw = 3;
  auto dimh = 2;
  auto dimt = 1;

  if(input.ndimension() == 5) {
    nBatch = input.size(0);
    dimw++;
    dimh++;
    dimt++;
  }

  auto nSlices = input.size(dimt-1);
  auto iT = input.size(dimt);
  auto iH = input.size(dimh);
  auto iW = input.size(dimw);

  auto* input_data = input.data<scalar_t>();
  auto* output_data = output.data<scalar_t>();
  auto* indices_data = indices.data<int64_t>();

  for (auto p = 0; p < nBatch; p++) {
    auto inputOffset = p * nSlices * iT * iW * iH;
    auto outputOffset = p * nSlices * oT * oW * oH;
    int k;
    int has_error = 0;
    int error_index = 0;
#pragma omp parallel for private(k)
    for (k = 0; k < nSlices; k++) {
      auto finalInputOffset = inputOffset + k * iT * iW * iH;
      auto finalOutputOffset = outputOffset + k * oT * oW * oH;

      auto* output_p_k = output_data + finalOutputOffset;
      auto* input_p_k = input_data + finalInputOffset;
      auto* ind_p_k = indices_data + finalInputOffset;
      int maxp;
      for (auto t = 0; t < iT; t++) {
        for (auto i = 0; i < iH; i++) {
          for (auto j = 0; j < iW; j++) {
            auto index = t * iH * iW + i * iW + j;
            maxp = ind_p_k[index];
            if (maxp < 0 || maxp >= oT * oW * oH) {
#pragma omp critical
              {
                has_error = 1;
                error_index = maxp;
              }
            } else {
              output_p_k[maxp] = input_p_k[index];
            }
          }
        }
      }
      if (has_error) {
        AT_ERROR(
            "found an invalid max index %ld (output volumes are of size %dx%dx%d)",
            error_index,
            oT,
            oH,
            oW);
      }
    }
  }
  return output;
}

void MaxUnpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding,
    bool check_grad) {
  // is_empty check
  AT_CHECK(input.numel() > 0,
    "Input must be non-empty");
  AT_CHECK((input.ndimension() == 4 || input.ndimension() == 5),
    "Input must be 4d or 5d tensor");
  AT_CHECK(input.sizes() == indices.sizes());
  AT_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "stride should be never greater than zero, but got stride: ",
      stride);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  if (input.ndimension() == 5)
  {
    dimw++;
    dimh++;
    dimt++;
    dimn++;
  }

  int nslices = input.size(dimn);
  if (check_grad) {
    if (output_size[0] != gradOutput.size(dimt) ||
        output_size[1] != gradOutput.size(dimh) ||
        output_size[2] != gradOutput.size(dimw)) {
      AT_ERROR(
          "Inconsistent gradOutput size. output_size: ,",
          output_size,
          ". gradOutput: ",
          gradOutput);
    }
    AT_CHECK(gradOutput.ndimension() == input.ndimension() && gradOutput.size(dimn) == nslices);
  }
}

at::Tensor& MaxUnpooling3d_forward_out_cpu(
    Tensor& output,
    const Tensor& self,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding) {
  AT_CHECK(
      (output.ndimension() == 4 || output.ndimension() == 5),
      "Output to MaxUnpooling2d should be a 4d or 5d Tensor",
      output.sizes());
  AT_CHECK(
      (self.ndimension() == 4 || self.ndimension() == 5),
      "Input to MaxUnpooling2d should be a 4d or 5d Tensor",
      self.sizes());
  AT_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size");
  AT_CHECK(
      stride.size() == 3,
      "There should be exactly three elements (depth, height, width) in stide");
  AT_CHECK(
      padding.size() == 3,
      "There should be exactly three elements (depth, height, width) in padding");
  AT_CHECK(
      self.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");
  MaxUnpooling3d_shape_check(
      self, at::empty({}), indices, output_size, stride, padding, false);

  if(self.ndimension() == 5)
  {
    output.resize_({self.size(0),
             self.size(1),
             output_size[0],
             output_size[1],
             output_size[2]});
  }
  else {
    output.resize_({self.size(0),
             output_size[0],
             output_size[1],
             output_size[2]});
  }
  output.zero_();

  AT_DISPATCH_FLOATING_TYPES(
      self.type(), "MaxUnpooling3d_forward_out_cpu_", ([&] {
        MaxUnpooling3d_forward_out_cpu_<scalar_t>(
            output,
            self.contiguous(),
            indices.contiguous(),
            output_size[0],
            output_size[1],
            output_size[2],
            stride[0],
            stride[1],
            stride[2],
            padding[0],
            padding[1],
            padding[2]);
      }));
  return output;
}

Tensor MaxUnpooling3d_forward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntList output_size,
    IntList stride,
    IntList padding) {
  AT_CHECK(
      self.ndimension() == 5,
      "Input to MaxUnpooling2d should be a NCDHW Tensor",
      self.sizes());
  AT_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size");
  auto output = at::zeros(
      {self.size(0),
       self.size(1),
       output_size[0],
       output_size[1],
       output_size[2]},
      self.options());
  MaxUnpooling3d_forward_out_cpu(
      output, self, indices, output_size, stride, padding);
  return output;
}
} // namespace native
} // namespace at
