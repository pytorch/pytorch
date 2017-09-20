#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/LookupTable.c"
#else

static void THNN_(LookupTable_resetCount)(
          THInteger_t *count_data,
          THIndexTensor *input)
{
  ptrdiff_t i;
  THIndex_t *input_data = THIndexTensor_(data)(input);
  ptrdiff_t numel = THIndexTensor_(nElement)(input);

  for (i = 0; i<numel; i++)
  {
    int64_t k = input_data[i] - TH_INDEX_BASE;
    count_data[k] = 0;
  }
  for (i = 0; i<numel; i++)
  {
    int64_t k = input_data[i] - TH_INDEX_BASE;
    count_data[k]++;
  }
}

void THNN_(LookupTable_accGradParameters)(
          THNNState *state,
          THIndexTensor *input,
          THTensor *gradOutput,
          THTensor *gradWeight,
          THIntegerTensor *count,
          THTensor *sorted,
          THIndexTensor *indices,
          bool scaleGradByFreq,
          int paddingValue,
          accreal ascale)
{
  real scale = TH_CONVERT_ACCREAL_TO_REAL(ascale);
  ptrdiff_t i;
  THInteger_t *count_data = NULL;

  if (scaleGradByFreq)
  {
    THIntegerTensor_(resize1d)(count, gradWeight->size[0]);
    count_data = THIntegerTensor_(data)(count);
  }

  if (!THTensor_(isContiguous)(gradWeight))
    THError("gradWeight must be contiguous");
  if (!THIndexTensor_(isContiguous)(input))
    THError("input must be contiguous");
  if (THIndexTensor_(nDimension)(input) != 1 && THIndexTensor_(nDimension)(input) != 2) {
    THDescBuff s1 = THIndexTensor_(sizeDesc)(input);
    THError("input must be a vector or matrix, but is of shape: %s", s1.str);
  }

  THIndex_t *input_data = THIndexTensor_(data)(input);
  ptrdiff_t numel = THIndexTensor_(nElement)(input);
  int64_t numw = THTensor_(size)(gradWeight, 0);

  // check that inputs are all within range
  for (i=0; i<numel; i++)
    if (input_data[i] < TH_INDEX_BASE || input_data[i] >= numw + TH_INDEX_BASE) {
      THError("inputs need to be in the range %ld <= input < %ld, "
	      "but got input of value: %ld", TH_INDEX_BASE, (numw + TH_INDEX_BASE),
	      input_data[i]);
    }

  gradOutput = THTensor_(newContiguous)(gradOutput);

  real *gw = THTensor_(data)(gradWeight);
  real *go = THTensor_(data)(gradOutput);
  int64_t stride = THTensor_(stride)(gradWeight, 0);

  if (count_data)
    THNN_(LookupTable_resetCount)(count_data, input);

#ifdef _OPENMP
  if (numel > 1000)
  {
    // The strategy is to parallelize over sections of the vocabulary, so that
    // thread 1 handles updates to gradWeight[0..nVocab/nThreads]. Every thread
    // has to traverse the entire input, but the dominating factor is the axpy
    // BLAS call.
    #pragma omp parallel private(i)
    {
      int tid = omp_get_thread_num();
      int nthreads = omp_get_num_threads();

      int64_t start = tid * (numw/nthreads + 1);
      int64_t end = start + (numw/nthreads + 1);
      for (i=0; i<numel; i++)
      {
        if (input_data[i] != paddingValue)
        {
            int64_t k = input_data[i] - TH_INDEX_BASE;
            if (k >= start && k < end)
            {
                real scale_ = scale;
                if (count_data) scale_ /= count_data[k];
                THBlas_(axpy)(stride, scale_, go + i*stride, 1, gw + k*stride, 1);
            }
        }
      }
    }

    THTensor_(free)(gradOutput);
    return;
  }
#endif

  for (i=0; i<numel; i++)
  {
    if (input_data[i] != paddingValue)
    {
        int64_t k = input_data[i] - TH_INDEX_BASE;
        real scale_ = scale;
        if (count_data) scale_ /= count_data[k];
        THBlas_(axpy)(stride, scale_, go + i*stride, 1, gw + k*stride, 1);
     }
  }

  THTensor_(free)(gradOutput);
}

/*
 * Keep the norm of weight smaller than maxNorm
 */

static void THNN_(LookupTable_renormRow)(
          real *row_data,
          int64_t stride,
          real maxNorm,
          real normType)
{
  real norm = 0;
  real new_norm;
  int64_t j;
  for (j=0; j<stride; j++)
  {
    if (normType == 1) {
      norm += fabs(row_data[j]);
    } else if (normType == 2) {
      norm += row_data[j] * row_data[j];
    } else {
      norm += pow(fabs(row_data[j]), normType);
    }
  }
  norm = pow(norm, 1.0 / normType);
  if (norm > maxNorm)
  {
    new_norm = maxNorm / (norm + 1e-7);
    for (j=0; j<stride; j++) {
      row_data[j] *= new_norm;
    }
  }
}

static int THNN_(compare_THIndex)(const void* a, const void* b)
{
   return *(const THIndex_t*)a < *(const THIndex_t*)b ? -1 : 1;
}

void THNN_(LookupTable_renorm)(
          THNNState *state,
          THIndexTensor *idx,
          THTensor *weight,
          accreal maxNorm_,
          accreal normType_)
{
  real maxNorm = TH_CONVERT_ACCREAL_TO_REAL(maxNorm_);
  real normType = TH_CONVERT_ACCREAL_TO_REAL(normType_);
  if (!THTensor_(isContiguous)(weight))
    THError("weight must be contiguous");
  if (!THIndexTensor_(isContiguous)(idx))
    THError("input must be contiguous");
  if (THIndexTensor_(nDimension)(idx) != 1)
    THError("idx must be a vector");
  if (normType <= 0)
    THError("non-positive-norm not supported");

  ptrdiff_t i;
  THIndex_t *row_idx = THIndexTensor_(data)(idx);
  ptrdiff_t numel = THIndexTensor_(nElement)(idx);

  int64_t numw = THTensor_(size)(weight, 0);
  int64_t stride = THTensor_(stride)(weight, 0);
  real *gw = THTensor_(data)(weight);
  for (i=0; i<numel; i++) {
    if (row_idx[i] < TH_INDEX_BASE || row_idx[i] >= numw + TH_INDEX_BASE) {
      THError("input need to be in the range %ld <= input < %ld, "
	      "but got input of value: %ld", TH_INDEX_BASE, (numw + TH_INDEX_BASE),
	      row_idx[i]);
    }
  }
  // get unique indices
  qsort(row_idx, numel, sizeof(THIndex_t), THNN_(compare_THIndex));
  ptrdiff_t ptr = 0;
  for (i=0; i<numel; i++)
    if (i == 0 || row_idx[i] != row_idx[i-1])
      row_idx[ptr++] = row_idx[i];
  numel = ptr;

#ifdef _OPENMP
  if (numel > 1000)
  {
    // The strategy is to parallelize over the rows that appear in
    // row_idx, so that thread 1 handles the rows in row_idx[0..numel/nThreads].
    // This distributes the work evenly to each thread.
    #pragma omp parallel for private(i)
    for (i=0; i<numel; i++)
    {
      int64_t k = row_idx[i] - TH_INDEX_BASE;
      THNN_(LookupTable_renormRow)(gw + k*stride, stride, maxNorm, normType);
    }
    return;
  }
#endif
  for (i=0; i<numel; i++)
  {
    int64_t k = row_idx[i] - TH_INDEX_BASE;
    THNN_(LookupTable_renormRow)(gw + k*stride, stride, maxNorm, normType);
  }
}

#endif
