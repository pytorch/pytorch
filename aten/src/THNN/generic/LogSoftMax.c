#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LogSoftMax.c"
#else

#ifdef _MSC_VER
  #define LOG_SOFTMAX_SIZE_TYPE int64_t
  #define LOG_SOFTMAX_CAST_TYPE (int64_t)
#else
  #define LOG_SOFTMAX_SIZE_TYPE uint64_t
  #define LOG_SOFTMAX_CAST_TYPE
#endif

void THNN_(LogSoftMax_updateOutput)(
          THNNState *state,
          THTensor *input,
          THTensor *output,
          int64_t dim) {
  THArgCheck(dim >= 0 && dim < input->nDimension, 4,
	     "dim out of range (got %d, but input has %d dims)", dim, input->nDimension);

  uint64_t outer_size = 1;
  uint64_t dim_size = input->size[dim];
  uint64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= input->size[i];
  for (int64_t i = dim + 1; i < input->nDimension; ++i)
    inner_size *= input->size[i];

  input = THTensor_(newContiguous)(input);
  THTensor_(resizeAs)(output, input);

  real *input_data_base  = THTensor_(data)(input);
  real *output_data_base = THTensor_(data)(output);

  uint64_t dim_stride = inner_size;
  uint64_t outer_stride = dim_size * dim_stride;

  LOG_SOFTMAX_SIZE_TYPE i, d;

#pragma omp parallel for private(i, d)
  for (i = 0; i < LOG_SOFTMAX_CAST_TYPE (outer_size * inner_size); i++)
  {
    uint64_t outer_idx = i / inner_size;
    uint64_t inner_idx = i % inner_size;
    real *input_data  = input_data_base  + outer_idx * outer_stride + inner_idx;
    real *output_data = output_data_base + outer_idx * outer_stride + inner_idx;

    real max_input = -THInf;
    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      max_input = THMax(max_input, input_data[d * dim_stride]);

    accreal logsum = 0;
    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      logsum += exp(input_data[d * dim_stride] - max_input);
    logsum = max_input + log(logsum);

    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      output_data[d * dim_stride] = input_data[d * dim_stride] - logsum;
  }

  THTensor_(free)(input);
}

void THNN_(LogSoftMax_updateGradInput)(
          THNNState *state,
          THTensor *input,
          THTensor *gradOutput,
          THTensor *gradInput,
          THTensor *output,
          int64_t dim)
{
  THNN_CHECK_SHAPE(output, gradOutput);
  THArgCheck(dim >= 0 && dim < output->nDimension, 6,
	     "dim out of range (got %d, but input has %d dims)", dim, output->nDimension);

  uint64_t outer_size = 1;
  uint64_t dim_size = output->size[dim];
  uint64_t inner_size = 1;
  for (int64_t i = 0; i < dim; ++i)
    outer_size *= output->size[i];
  for (int64_t i = dim + 1; i < output->nDimension; ++i)
    inner_size *= output->size[i];

  gradOutput = THTensor_(newContiguous)(gradOutput);
  output = THTensor_(newContiguous)(output);
  THTensor_(resizeAs)(gradInput, output);

  real *gradInput_data_base  = THTensor_(data)(gradInput);
  real *output_data_base     = THTensor_(data)(output);
  real *gradOutput_data_base = THTensor_(data)(gradOutput);

  uint64_t dim_stride = inner_size;
  uint64_t outer_stride = dim_size * dim_stride;

  LOG_SOFTMAX_SIZE_TYPE i, d;

#pragma omp parallel for private(i, d)
  for (i = 0; i < LOG_SOFTMAX_CAST_TYPE (outer_size * inner_size); i++)
  {
    uint64_t outer_idx = i / inner_size;
    uint64_t inner_idx = i % inner_size;
    real *gradInput_data  = gradInput_data_base  + outer_idx * outer_stride + inner_idx;
    real *output_data     = output_data_base     + outer_idx * outer_stride + inner_idx;
    real *gradOutput_data = gradOutput_data_base + outer_idx * outer_stride + inner_idx;

    accreal sum = 0;
    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      sum += gradOutput_data[d * dim_stride];

    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      gradInput_data[d * dim_stride] = gradOutput_data[d * dim_stride] - exp(output_data[d * dim_stride]) * sum;
  }

  THTensor_(free)(gradOutput);
  THTensor_(free)(output);
}

#endif
