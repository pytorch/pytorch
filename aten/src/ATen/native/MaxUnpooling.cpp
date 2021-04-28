#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>
#include <tuple>

namespace at {
namespace native {

template <typename scalar_t>
Tensor max_unpooling2d_forward_out_cpu_frame(
    Tensor& output,
    const Tensor& input,
    const Tensor& indices,
    int64_t oheight,
    int64_t owidth) {
  int64_t numBatch = 1;
  int64_t dimc = 0;
  int64_t dimh = 1;
  int64_t dimw = 2;
  if (input.ndimension() == 4) {
    numBatch = input.size(0);
    dimc++;
    dimh++;
    dimw++;
  }
  int64_t numChannels = input.size(dimc);
  int64_t inputHeight = input.size(dimh);
  int64_t inputWidth = input.size(dimw);

  auto* rawInput = input.data_ptr<scalar_t>();
  auto* rawIndices = indices.data_ptr<int64_t>();
  auto* rawOutput = output.data_ptr<scalar_t>();

  at::internal::lazy_init_num_threads();

  for (int64_t n = 0; n < numBatch; n++) {
    int64_t nOutputOffset = n * numChannels * owidth * oheight;
    int64_t nInputOffset = n * numChannels * inputWidth * inputHeight;
    int64_t k = 0;
    bool has_error = false;
    int64_t error_index = 0;
#pragma omp parallel for private(k)
    for (k = 0; k < numChannels; k++) {
      int64_t finalOutputOffset = nOutputOffset + k * owidth * oheight;
      int64_t finalInputOffset = nInputOffset + k * inputWidth * inputHeight;
      scalar_t* output_p_k = rawOutput + finalOutputOffset;
      scalar_t* input_p_k = rawInput + finalInputOffset;
      int64_t* ind_p_k = rawIndices + finalInputOffset;

      int64_t maxp;
      for (int64_t i = 0; i < inputHeight; i++) {
        for (int64_t j = 0; j < inputWidth; j++) {
          maxp = ind_p_k[i * inputWidth + j];
          if (maxp < 0 || maxp >= owidth * oheight) {
#pragma omp critical
            {
              has_error = true;
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
          "Found an invalid max index: ",
          error_index,
          " (output volumes are of size ",
          oheight,
          "x",
          owidth);
    }
  }
  return output;
}

Tensor& max_unpooling2d_forward_out_cpu(const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    Tensor& output) {
  auto oheight = output_size[0];
  auto owidth = output_size[1];
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  TORCH_CHECK(
      output_size.size() == 2,
      "There should be exactly two elements (height, width) in output_size");
  TORCH_CHECK(
      (self_.ndimension() == 3 || self_.ndimension() == 4),
      "Input to max_unpooling2d should be a 3d or 4d Tensor");
  TORCH_CHECK(
      self_.sizes() == indices_.sizes(),
      "Shape of indices should match shape of input");

  TORCH_CHECK(self_.numel() > 0, "Input must be non-empty");

  auto self = self_.contiguous();
  auto indices = indices_.contiguous();

  if (self.ndimension() == 3) {
    int64_t numChannels = self.size(0);
    output.resize_({numChannels, oheight, owidth});
  } else {
    int64_t numBatch = self.size(0);
    int64_t numChannels = self.size(1);
    output.resize_({numBatch, numChannels, oheight, owidth});
  }
  output.zero_();

  AT_DISPATCH_FLOATING_TYPES(
      self.scalar_type(), "max_unpooling2d_forward_out_cpu_frame", ([&] {
        max_unpooling2d_forward_out_cpu_frame<scalar_t>(
            output, self, indices, oheight, owidth);
      }));
  return output;
};

Tensor max_unpooling2d_forward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto output = at::empty({0}, self.options());
  at::native::max_unpooling2d_forward_out_cpu(self, indices, output_size, output);
  return output;
}

template <typename scalar_t>
Tensor max_unpooling3d_forward_out_cpu_frame(
    Tensor& output,
    const Tensor& input,
    const Tensor& indices,
    int64_t oT,
    int64_t oH,
    int64_t oW) {
  int64_t nBatch = 1;
  int64_t dimw = 3;
  int64_t dimh = 2;
  int64_t dimt = 1;

  if (input.ndimension() == 5) {
    nBatch = input.size(0);
    dimw++;
    dimh++;
    dimt++;
  }

  int64_t nSlices = input.size(dimt - 1);
  int64_t iT = input.size(dimt);
  int64_t iH = input.size(dimh);
  int64_t iW = input.size(dimw);

  scalar_t* input_data = input.data_ptr<scalar_t>();
  scalar_t* output_data = output.data_ptr<scalar_t>();
  int64_t* indices_data = indices.data_ptr<int64_t>();

  at::internal::lazy_init_num_threads();

  for (int64_t p = 0; p < nBatch; p++) {
    int64_t inputOffset = p * nSlices * iT * iW * iH;
    int64_t outputOffset = p * nSlices * oT * oW * oH;
    int64_t k = 0;
    bool has_error = false;
    int error_index = 0;
#pragma omp parallel for private(k)
    for (k = 0; k < nSlices; k++) {
      int64_t finalInputOffset = inputOffset + k * iT * iW * iH;
      int64_t finalOutputOffset = outputOffset + k * oT * oW * oH;

      scalar_t* output_p_k = output_data + finalOutputOffset;
      scalar_t* input_p_k = input_data + finalInputOffset;
      int64_t* ind_p_k = indices_data + finalInputOffset;
      int maxp;
      for (int64_t t = 0; t < iT; t++) {
        for (int64_t i = 0; i < iH; i++) {
          for (int64_t j = 0; j < iW; j++) {
            int64_t index = t * iH * iW + i * iW + j;
            maxp = ind_p_k[index];
            if (maxp < 0 || maxp >= oT * oW * oH) {
#pragma omp critical
              {
                has_error = true;
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
            "found an invalid max index ",
            error_index,
            " (output volumes are of size ",
            oT,
            "x",
            oH,
            "x",
            oW);
      }
    }
  }
  return output;
}

static void max_unpooling3d_shape_check(
    const Tensor& input,
    const Tensor& gradOutput,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];
  TORCH_CHECK(
      indices.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  TORCH_CHECK(
      (input.ndimension() == 4 || input.ndimension() == 5),
      "Input to max_unpooling3d should be a 4d or 5d Tensor",
      input.sizes());
  TORCH_CHECK(
      output_size.size() == 3,
      "There should be exactly three elements (depth, height, width) in output_size");
  TORCH_CHECK(
      stride.size() == 3,
      "There should be exactly three elements (depth, height, width) in stride");
  TORCH_CHECK(
      padding.size() == 3,
      "There should be exactly three elements (depth, height, width) in padding");
  TORCH_CHECK(
      input.sizes() == indices.sizes(),
      "Shape of indices should match shape of input");

  TORCH_CHECK(input.numel() > 0, "Input must be non-empty");

  TORCH_CHECK(
      stride[0] > 0 && stride[1] > 0 && stride[2] > 0,
      "strides should be greater than zero, but got stride: ",
      stride);

  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int dimn = 0;

  if (input.ndimension() == 5) {
    dimw++;
    dimh++;
    dimt++;
    dimn++;
  }

  int nslices = input.size(dimn);

  if (gradOutput.defined()) {
    if (oT != gradOutput.size(dimt) || oH != gradOutput.size(dimh) ||
        oW != gradOutput.size(dimw)) {
      AT_ERROR(
          "Inconsistent gradOutput size. oT= ",
          oT,
          ", oH= ",
          oH,
          ", oW= ",
          oW,
          ". gradOutput: ",
          gradOutput.size(dimt),
          "x",
          gradOutput.size(dimh),
          "x",
          gradOutput.size(dimw));
    }
    TORCH_CHECK(
        gradOutput.ndimension() == input.ndimension() &&
            gradOutput.size(dimn) == nslices,
        "gradOutput and input Tensors should have same number of dimensions and also the same number of channels/slices");
  }
}

Tensor& max_unpooling3d_forward_out_cpu(const Tensor& self_,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& output) {
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  int64_t oT = output_size[0];
  int64_t oH = output_size[1];
  int64_t oW = output_size[2];

  auto self = self_.contiguous();
  auto indices = indices_.contiguous();

  max_unpooling3d_shape_check(
      self_, Tensor(), indices_, output_size, stride, padding);

  if (self_.ndimension() == 5) {
    output.resize_({self.size(0), self.size(1), oT, oH, oW});
  } else {
    output.resize_({self.size(0), oT, oH, oW});
  }
  output.zero_();

  AT_DISPATCH_FLOATING_TYPES(
      self.scalar_type(), "max_unpooling3d_forward_out_cpu_frame", ([&] {
        max_unpooling3d_forward_out_cpu_frame<scalar_t>(
            output,
            self,
            indices,
            oT,
            oH,
            oW);
      }));
  return output;
}

Tensor max_unpooling3d_forward_cpu(
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto output = at::empty({0}, self.options());
  at::native::max_unpooling3d_forward_out_cpu(
      self, indices, output_size, stride, padding, output);
  return output;
}

template <typename scalar_t>
static void max_unpooling2d_backward_out_cpu_frame(
    scalar_t* gradInput_p,
    scalar_t* gradOutput_p,
    int64_t* ind_p,
    int64_t nslices,
    int64_t iheight,
    int64_t iwidth,
    int64_t oheight,
    int64_t owidth) {
  bool has_error = false;
  int64_t error_index = 0;
  int64_t k = 0;

  at::internal::lazy_init_num_threads();
#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++) {
    scalar_t* gradInput_p_k = gradInput_p + k * iwidth * iheight;
    scalar_t* gradOutput_p_k = gradOutput_p + k * owidth * oheight;
    int64_t* ind_p_k = ind_p + k * iwidth * iheight;

    int64_t i, j;
    int64_t maxp;

    for (i = 0; i < iheight; i++) {
      for (j = 0; j < iwidth; j++) {
        maxp = ind_p_k[i * iwidth + j]; /* retrieve position of max */
        if (maxp < 0 || maxp >= owidth * oheight) {
#pragma omp critical
          {
            has_error = true;
            error_index = maxp;
          }
        }
        gradInput_p_k[i * iwidth + j] =
            gradOutput_p_k[maxp]; /* update gradient */
      }
    }
  }
  if (has_error) {
    AT_ERROR(
        "invalid max index ",
        error_index,
        ", owidth= ",
        owidth,
        ", oheight= ",
        oheight);
  }
}

Tensor& max_unpooling2d_backward_out_cpu(const Tensor& grad_output_,
    const Tensor& self,
    const Tensor& indices_,
    IntArrayRef output_size,
    Tensor& grad_input) {
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  int64_t oheight = output_size[0];
  int64_t owidth = output_size[1];
  int dimw = 2;
  int dimh = 1;
  int nbatch = 1;
  int nslices;
  int iheight;
  int iwidth;
  TORCH_CHECK(
      indices_.scalar_type() == at::ScalarType::Long,
      "elements in indices should be type int64");
  TORCH_CHECK(
      self.sizes() == indices_.sizes(), "Input shape must match indices shape");

  TORCH_CHECK(output_size.size() == 2, "Output size must be 2");

  /* get contiguous gradOutput and indices */
  auto grad_output = grad_output_.contiguous();
  auto indices = indices_.contiguous();

  /* resize */
  grad_input.resize_as_(self);
  grad_input.zero_();

  if (self.ndimension() == 4) {
    nbatch = self.size(0);
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = self.size(dimh - 1);
  iheight = self.size(dimh);
  iwidth = self.size(dimw);

  if (owidth != grad_output.size(dimw) || oheight != grad_output.size(dimh)) {
    AT_ERROR(
        "Inconsistent gradOutput size. output height = ",
        oheight,
        ", output width = ",
        owidth,
        ", gradOutput: ",
        grad_output.size(dimh),
        "x",
        grad_output.size(dimw));
  }
  AT_DISPATCH_FLOATING_TYPES(
      self.scalar_type(), "max_unpooling2d_backward_out_cpu_frame", ([&] {
        int p;
        for (p = 0; p < nbatch; p++) {
          auto inputOffset = p * nslices * iheight * iwidth;
          auto outputOffset = p * nslices * oheight * owidth;
          max_unpooling2d_backward_out_cpu_frame<scalar_t>(
              grad_input.data_ptr<scalar_t>() + inputOffset,
              grad_output.data_ptr<scalar_t>() + outputOffset,
              indices.data_ptr<int64_t>() + inputOffset,
              nslices,
              iheight,
              iwidth,
              oheight,
              owidth);
        }
      }));
  return grad_input;
}

Tensor max_unpooling2d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size) {
  auto grad_input = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::max_unpooling2d_backward_out_cpu(
      grad_output, self, indices, output_size, grad_input);
  return grad_input;
}

template <typename scalar_t>
static void max_unpooling3d_backward_out_cpu_frame(
    scalar_t* gradInput_p,
    scalar_t* gradOutput_p,
    int64_t* ind_p,
    int64_t nslices,
    int64_t iT,
    int64_t iH,
    int64_t iW,
    int64_t oT,
    int64_t oH,
    int64_t oW) {
  int64_t k = 0;
  bool has_error = false;
  int error_index = 0;

  at::internal::lazy_init_num_threads();

#pragma omp parallel for private(k)
  for (k = 0; k < nslices; k++) {
    scalar_t* gradInput_p_k = gradInput_p + k * iT * iH * iW;
    scalar_t* gradOutput_p_k = gradOutput_p + k * oT * oH * oW;
    int64_t* ind_p_k = ind_p + k * iT * iH * iW;

    int64_t t, i, j, index;
    int64_t maxp;
    for (t = 0; t < iT; t++) {
      for (i = 0; i < iH; i++) {
        for (j = 0; j < iW; j++) {
          index = t * iH * iW + i * iW + j;
          maxp = ind_p_k[index]; /* retrieve position of max */
          if (maxp < 0 || maxp >= oT * oH * oW) {
#pragma omp critical
            {
              has_error = true;
              error_index = maxp;
            }
          }
          gradInput_p_k[index] = gradOutput_p_k[maxp]; /* update gradient */
        }
      }
    }
  }
  if (has_error) {
    AT_ERROR(
        "invalid max index ",
        error_index,
        ", oT= ",
        oT,
        ", oW= ",
        oW,
        ",oH= ",
        oH);
  }
}

Tensor& max_unpooling3d_backward_out_cpu(const Tensor& grad_output_,
    const Tensor& self,
    const Tensor& indices_,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding,
    Tensor& grad_input) {
  TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
  auto oT = output_size[0];
  auto oH = output_size[1];
  auto oW = output_size[2];
  int dimw = 3;
  int dimh = 2;
  int dimt = 1;
  int nbatch = 1;
  int nslices;
  int iT;
  int iH;
  int iW;

  max_unpooling3d_shape_check(
      self, grad_output_, indices_, output_size, stride, padding);

  // TODO (from THNN): check gradOutput shape
  /* get contiguous gradOutput */
  auto grad_output = grad_output_.contiguous();
  auto indices = indices_.contiguous();

  /* resize */
  grad_input.resize_as_(self);
  grad_input.zero_();
  if (self.ndimension() == 5) {
    nbatch = self.size(0);
    dimt++;
    dimw++;
    dimh++;
  }

  /* sizes */
  nslices = self.size(dimt - 1);
  iT = self.size(dimt);
  iH = self.size(dimh);
  iW = self.size(dimw);

  /* backprop */
  AT_DISPATCH_FLOATING_TYPES(
      self.scalar_type(), "max_unpooling3d_backward_out_cpu_frame", ([&] {
        int p;
        for (p = 0; p < nbatch; p++) {
          int inputOffset = p * nslices * iT * iH * iW;
          int outputOffset = p * nslices * oT * oH * oW;
          max_unpooling3d_backward_out_cpu_frame<scalar_t>(
              grad_input.data_ptr<scalar_t>() + inputOffset,
              grad_output.data_ptr<scalar_t>() + outputOffset,
              indices.data_ptr<int64_t>() + inputOffset,
              nslices,
              iT,
              iH,
              iW,
              oT,
              oH,
              oW);
        }
      }));
  return grad_input;
}

Tensor max_unpooling3d_backward_cpu(
    const Tensor& grad_output,
    const Tensor& self,
    const Tensor& indices,
    IntArrayRef output_size,
    IntArrayRef stride,
    IntArrayRef padding) {
  auto grad_input = at::empty_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  at::native::max_unpooling3d_backward_out_cpu(
      grad_output, self, indices, output_size, stride, padding, grad_input);
  return grad_input;
}
} // namespace native
} // namespace at
