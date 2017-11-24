#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/FusedRNNKernel.cu"
#else
#include <cstdarg>

#include "../common.h"

#define DATATYPE TensorUtils<THCTensor>::DataType

//factor will be 3 for GRU and 4 for LSTM
void THNN_(FusedRNNAssertSizes)(THCState *state, int factor, int count, ...)
{
  va_list list;
  va_start(list, count);
  THCTensor *input = va_arg(list, THCTensor*);
  THCTensor *hidden = va_arg(list, THCTensor*);
  THArgCheck(THCTensor_(nElement)(state, input) ==
             THCTensor_(nElement)(state, hidden),
             3, "Input and Hidden tensor sizes should be the same.");

  THAssertMsg(TensorUtils<THCTensor>::getDims(state, input) <= MAX_CUTORCH_DIMS,
              "Tensor dimension is too large.");

  THAssertMsg(TensorUtils<THCTensor>::getDims(state, hidden) <= MAX_CUTORCH_DIMS,
              "Tensor dimension is too large.");

  for (int arg=2; arg < count; ++arg){
    THCTensor *tens = va_arg(list, THCTensor*);
    THArgCheck(THCTensor_(nElement)(state, input) ==
               THCTensor_(nElement)(state, tens)*factor,
               3, "A pointwise tensor was not the right size, should have 1/%u the elements of input/hidden tensor.", arg, factor);
    THAssertMsg(TensorUtils<THCTensor>::getDims(state, tens) <= MAX_CUTORCH_DIMS,
         "Tensor dimension is too large.");
  }

  va_end(list);
}

int THNN_(minIndexType)(THCState *state, int count, ...)
{
  va_list list;
  va_start(list, count);

  THCTensor* tens = va_arg(list, THCTensor*);
  int startDim = TensorUtils<THCTensor>::getDims(state, tens);
  bool canCollapse = THCTensor_(isContiguous)(state,tens);

  for (int arg=1; arg < count; ++arg){
    tens = va_arg(list, THCTensor*);
    canCollapse = canCollapse && THCTensor_(isContiguous)(state, tens);
    if(TensorUtils<THCTensor>::getDims(state, tens) != startDim){
      va_end(list);
      return -1;
    }
  }
  va_end(list);
  if(canCollapse) return -2;
  return startDim;
}

bool THNN_(canUse32BitIndexMath)(THCState *state, int count, ...)
{
  va_list list;
  va_start(list, count);

  for (int arg=0; arg < count; ++arg){
    THCTensor *tens = va_arg(list, THCTensor*);
    if (!TensorUtils<THCTensor>::canUse32BitIndexMath(state, tens)){
      va_end(list);
      return false;
    }
  }
  va_end(list);
  return true;
}

#define DEVICE_LINEAR_GET(D_TENSOR, INDEX)                              \
  D_TENSOR.data[IndexToOffset<T, IndexType, Dims>::get(INDEX, D_TENSOR)]

#define H2F(input) __half2float(input)
#define F2H(input) __float2half(input)

template <typename T, typename IndexType, int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THNN_(GRUForward)(TensorInfo<T, IndexType> Input,
            TensorInfo<T, IndexType> Hidden,
            TensorInfo<T, IndexType> Bias1,
            TensorInfo<T, IndexType> Bias2,
            TensorInfo<T, IndexType> _hx,
            TensorInfo<T, IndexType> _hy,
            TensorInfo<T, IndexType> storage,
            IndexType hsz,
            IndexType totalElements)
{
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x)
    {

      IndexType offset = (linearIndex/hsz)*3*hsz+linearIndex%hsz;

      T ir = DEVICE_LINEAR_GET(Input, offset+0*hsz);
      T ii = DEVICE_LINEAR_GET(Input, offset+1*hsz);
      T in = DEVICE_LINEAR_GET(Input, offset+2*hsz);
      T hr = DEVICE_LINEAR_GET(Hidden,offset+0*hsz);
      T hi = DEVICE_LINEAR_GET(Hidden,offset+1*hsz);
      T hn = DEVICE_LINEAR_GET(Hidden,  offset+2*hsz);

      T hx = DEVICE_LINEAR_GET(_hx, linearIndex);
      T* hy = &DEVICE_LINEAR_GET(_hy, linearIndex);

      bool has_bias = (Bias1.data != NULL);

      T b1r, b1i, b1n, b2r, b2i, b2n;

      if(has_bias){
        b1r = DEVICE_LINEAR_GET(Bias1, linearIndex%hsz+0*hsz);
        b1i = DEVICE_LINEAR_GET(Bias1, linearIndex%hsz+1*hsz);
        b1n = DEVICE_LINEAR_GET(Bias1, linearIndex%hsz+2*hsz);

        b2r = DEVICE_LINEAR_GET(Bias2, linearIndex%hsz+0*hsz);
        b2i = DEVICE_LINEAR_GET(Bias2, linearIndex%hsz+1*hsz);
        b2n = DEVICE_LINEAR_GET(Bias2, linearIndex%hsz+2*hsz);
      }else{
#ifndef THC_REAL_IS_HALF
        b1r = 0.0; b1i = 0.0; b1n = 0.0;
        b2r = 0.0; b2i = 0.0; b2n = 0.0;
#else
        b1r = F2H(0.0); b1i = F2H(0.0); b1n = F2H(0.0);
        b2r = F2H(0.0); b2i = F2H(0.0); b2n = F2H(0.0);
#endif
      }


      offset = (linearIndex/hsz)*5*hsz+linearIndex%hsz;
#ifndef THC_REAL_IS_HALF

      T rg, ig, ng;

      rg = ir + hr + b1r + b2r;
      ig = ii + hi + b1i + b2i;

      TensorSigmoidOp<real>()(&rg, &rg);
      TensorSigmoidOp<real>()(&ig, &ig);
      ng = in + b1n + rg * (hn + b2n);
      ng = THCNumerics<T>::tanh(ng);
      *hy = ng + ig * (hx - ng);

      //SAVE FOR BACKWARDS
      DEVICE_LINEAR_GET(storage, offset+0*hsz) = rg;
      DEVICE_LINEAR_GET(storage, offset+1*hsz) = ig;
      DEVICE_LINEAR_GET(storage, offset+2*hsz) = ng;
      DEVICE_LINEAR_GET(storage, offset+3*hsz) = hx;
      DEVICE_LINEAR_GET(storage, offset+4*hsz) = hn + b2n;

#else

      float rg, ig, ng;

      rg = H2F(ir) + H2F(hr) + H2F(b1r) + H2F(b2r);
      ig = H2F(ii) + H2F(hi) + H2F(b1i) + H2F(b2i);

      TensorSigmoidOp<float>()(&rg, &rg);
      TensorSigmoidOp<float>()(&ig, &ig);
      ng = H2F(in) + H2F(b1n) + rg*( H2F(hn)+H2F(b2n) );
      ng = THCNumerics<float>::tanh(ng);
      *hy = F2H( ng + ig * ( H2F(hx)-ng ) );

      //SAVE FOR BACKWARDS
      DEVICE_LINEAR_GET(storage, offset+0*hsz) = F2H(rg);
      DEVICE_LINEAR_GET(storage, offset+1*hsz) = F2H(ig);
      DEVICE_LINEAR_GET(storage, offset+2*hsz) = F2H(ng);
      DEVICE_LINEAR_GET(storage, offset+3*hsz) = hx;
      DEVICE_LINEAR_GET(storage, offset+4*hsz) = F2H(H2F(hn) + H2F(b2n));
#endif
    }
}

template <typename T, typename IndexType, int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
THNN_(GRUBackward)(TensorInfo<T, IndexType> gradInInput,
             TensorInfo<T, IndexType> gradInHidden,
             TensorInfo<T, IndexType> gradOutput,
             TensorInfo<T, IndexType> gradInputHx,
             TensorInfo<T, IndexType> storage,
             IndexType hsz,
             IndexType totalElements)
{
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType offset = (linearIndex/hsz)*5*hsz+linearIndex%hsz;

    T rg = DEVICE_LINEAR_GET(storage, offset+0*hsz);
    T ig = DEVICE_LINEAR_GET(storage, offset+1*hsz);
    T ng = DEVICE_LINEAR_GET(storage, offset+2*hsz);
    T hx = DEVICE_LINEAR_GET(storage, offset+3*hsz);
    T hn = DEVICE_LINEAR_GET(storage, offset+4*hsz);

    T go = DEVICE_LINEAR_GET(gradOutput, linearIndex);

    offset = (linearIndex/hsz)*3*hsz+linearIndex%hsz;

#ifndef THC_REAL_IS_HALF
    T gig = go*(hx-ng)*(1-ig)*(ig);
    T ghx = go*(ig);
    T gin = go*(1-ig)*(1-ng*ng);
    T ghn = gin *rg;
    T grg = gin*hn*(1-rg)*rg;

    DEVICE_LINEAR_GET(gradInputHx, linearIndex) = ghx;

    DEVICE_LINEAR_GET(gradInInput, offset+0*hsz) = grg;
    DEVICE_LINEAR_GET(gradInInput, offset+1*hsz) = gig;
    DEVICE_LINEAR_GET(gradInInput, offset+2*hsz) = gin;

    DEVICE_LINEAR_GET(gradInHidden, offset+0*hsz) = grg;
    DEVICE_LINEAR_GET(gradInHidden, offset+1*hsz) = gig;
    DEVICE_LINEAR_GET(gradInHidden, offset+2*hsz) = ghn;
#else
    float gig = H2F(go)*( H2F(hx)-H2F(ng) )*( 1-H2F(ig) )*H2F(ig);
    float ghx = H2F(go)*H2F(ig);
    float gin = H2F(go)*( 1-H2F(ig) )*( 1-H2F(ng)*H2F(ng) );
    float ghn = H2F(gin) * H2F(rg);
    float grg = H2F(gin)*H2F(hn)*( 1-H2F(rg) )*H2F(rg);

    DEVICE_LINEAR_GET(gradInInput, offset+0*hsz) = F2H(grg);
    DEVICE_LINEAR_GET(gradInInput, offset+1*hsz) = F2H(gig);
    DEVICE_LINEAR_GET(gradInInput, offset+2*hsz) = F2H(gin);

    DEVICE_LINEAR_GET(gradInHidden, offset+0*hsz) = F2H(grg);
    DEVICE_LINEAR_GET(gradInHidden, offset+1*hsz) = F2H(gig);
    DEVICE_LINEAR_GET(gradInHidden, offset+2*hsz) = F2H(ghn);
    DEVICE_LINEAR_GET(gradInputHx, linearIndex) = F2H(ghx);

#endif
  }
}

template <typename T, typename IndexType, int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
  THNN_(LSTMForward)(TensorInfo<T, IndexType> input,
            TensorInfo<T, IndexType> hidden,
            TensorInfo<T, IndexType> bias1,
            TensorInfo<T, IndexType> bias2,
            TensorInfo<T, IndexType> _cx,
            TensorInfo<T, IndexType> _hy,
            TensorInfo<T, IndexType> _cy,
            IndexType hsz,
            IndexType totalElements)
{

    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x)
    {

      IndexType offset = (linearIndex/hsz)*4*hsz+linearIndex%hsz;

      T* iig = &DEVICE_LINEAR_GET(input, offset+0*hsz);
      T* ifg = &DEVICE_LINEAR_GET(input, offset+1*hsz);
      T* icg = &DEVICE_LINEAR_GET(input, offset+2*hsz);
      T* iog = &DEVICE_LINEAR_GET(input, offset+3*hsz);

      T hig = DEVICE_LINEAR_GET(hidden, offset+0*hsz);
      T hfg = DEVICE_LINEAR_GET(hidden, offset+1*hsz);
      T hcg = DEVICE_LINEAR_GET(hidden,  offset+2*hsz);
      T hog = DEVICE_LINEAR_GET(hidden,  offset+3*hsz);

      T cx = DEVICE_LINEAR_GET(_cx, linearIndex);

      T* hy = &DEVICE_LINEAR_GET(_hy, linearIndex);
      T* cy = &DEVICE_LINEAR_GET(_cy, linearIndex);

      bool has_bias = (bias1.data != NULL);

      T b1i, b1f, b1c, b1o;
      T b2i, b2f, b2c, b2o;

      if(has_bias){
    b1i = DEVICE_LINEAR_GET(bias1, linearIndex%hsz+0*hsz);
    b1f = DEVICE_LINEAR_GET(bias1, linearIndex%hsz+1*hsz);
    b1c = DEVICE_LINEAR_GET(bias1, linearIndex%hsz+2*hsz);
    b1o = DEVICE_LINEAR_GET(bias1, linearIndex%hsz+3*hsz);

    b2i = DEVICE_LINEAR_GET(bias2, linearIndex%hsz+0*hsz);
    b2f = DEVICE_LINEAR_GET(bias2, linearIndex%hsz+1*hsz);
    b2c = DEVICE_LINEAR_GET(bias2, linearIndex%hsz+2*hsz);
    b2o = DEVICE_LINEAR_GET(bias2, linearIndex%hsz+3*hsz);

      }else{
#ifndef THC_REAL_IS_HALF
    b1i = 0.0; b1f = 0.0; b1c = 0.0; b1o = 0.0;
    b2i = 0.0; b2f = 0.0; b2c = 0.0; b2o = 0.0;
#else
    b1i = F2H(0.0); b1f = F2H(0.0); b1c = F2H(0.0); b1o = F2H(0.0);
    b2i = F2H(0.0); b2f = F2H(0.0); b2c = F2H(0.0); b2o = F2H(0.0);
#endif
      }

#ifndef THC_REAL_IS_HALF
      T ig, fg, cg, og;

      ig = *iig + hig + b1i + b2i;
      fg = *ifg + hfg + b1f + b2f;
      cg = *icg + hcg + b1c + b2c;
      og = *iog + hog + b1o + b2o;

      TensorSigmoidOp<real>()(&ig, &ig);
      TensorSigmoidOp<real>()(&fg, &fg);
      cg = THCNumerics<T>::tanh(cg);
      TensorSigmoidOp<real>()(&og, &og);

      *cy = (fg * cx) + (ig * cg);
      *hy = og * THCNumerics<T>::tanh(*cy);

      *iig = ig;
      *ifg = fg;
      *icg = cg;
      *iog = og;
#else
      float ig, fg, cg, og;
      float f_hy, f_cy;

      ig = H2F(*iig) + H2F(hig) + H2F(b1i) + H2F(b2i);
      fg = H2F(*ifg) + H2F(hfg) + H2F(b1f) + H2F(b2f);
      cg = H2F(*icg) + H2F(hcg) + H2F(b1c) + H2F(b2c);
      og = H2F(*iog) + H2F(hog) + H2F(b1o) + H2F(b2o);

      TensorSigmoidOp<float>()(&ig, &ig);
      TensorSigmoidOp<float>()(&fg, &fg);
      cg = THCNumerics<float>::tanh(cg);
      TensorSigmoidOp<float>()(&og, &og);

      f_cy = (fg * H2F(cx) ) + (ig * cg);
      f_hy = og * THCNumerics<float>::tanh(f_cy);

      *hy = F2H(f_hy);
      *cy = F2H(f_cy);

      //SAVE FOR BACKWARDS
      //Also need cy and cx but can be saved easily in python
      *iig = F2H(ig);
      *ifg = F2H(fg);
      *icg = F2H(cg);
      *iog = F2H(og);
#endif
    }
}

template <typename T, typename IndexType, int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
  THNN_(LSTMBackward)(TensorInfo<T, IndexType> storage,
              TensorInfo<T, IndexType> gradInGates,
              TensorInfo<T, IndexType> _cx,
              TensorInfo<T, IndexType> _cy,
              TensorInfo<T, IndexType> gradoutput,
              TensorInfo<T, IndexType> gradoutputcell,
              TensorInfo<T, IndexType> gradInputCx,
              IndexType hsz,
              IndexType totalElements)
{
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType offset = (linearIndex/hsz)*4*hsz+linearIndex%hsz;

    T ig = DEVICE_LINEAR_GET(storage, offset+0*hsz);
    T fg = DEVICE_LINEAR_GET(storage, offset+1*hsz);
    T cg = DEVICE_LINEAR_GET(storage, offset+2*hsz);
    T og = DEVICE_LINEAR_GET(storage, offset+3*hsz);

    T* ih = &DEVICE_LINEAR_GET(gradInGates, offset+0*hsz);
    T* fh = &DEVICE_LINEAR_GET(gradInGates, offset+1*hsz);
    T* ch = &DEVICE_LINEAR_GET(gradInGates, offset+2*hsz);
    T* oh = &DEVICE_LINEAR_GET(gradInGates, offset+3*hsz);

    //will return hidden grads here
    T cx = DEVICE_LINEAR_GET(_cx, linearIndex);
    T cy = DEVICE_LINEAR_GET(_cy, linearIndex);

    T* gi = &DEVICE_LINEAR_GET(gradInputCx, linearIndex);

    T go = DEVICE_LINEAR_GET(gradoutput, linearIndex);
    T goc= DEVICE_LINEAR_GET(gradoutputcell, linearIndex);
#ifndef THC_REAL_IS_HALF
    T gcx = THCNumerics<T>::tanh(cy);

    T gog = go * gcx;
    gcx = go * og * ( 1 - gcx*gcx) + goc;

    T gig = gcx * cg;
    T gfg = gcx * cx;
    T gcg = gcx * ig;

    gcx = gcx * fg;

    gig = gig * (1-ig) * ig;
    gfg = gfg * (1-fg) * fg;
    gcg = gcg * (1-cg*cg);
    gog = gog * (1-og) * og;

    *ih = gig;
    *fh = gfg;
    *ch = gcg;
    *oh = gog;

    *gi = gcx;
#else
    float gcx = THCNumerics<float>::tanh(H2F(cy));
    float gog = H2F(go) * gcx;
    gcx = H2F(go) * H2F(og) * ( 1 - gcx*gcx) + H2F(goc);

    float gcg = gcx * H2F(fg);
    float gfg = gcx * H2F(cg);
    float gig = gcx * H2F(cx);

    gog = gog * ( (1-H2F(og))*H2F(og) );
    gcg = gcg * (1-H2F(cg)*H2F(cg));
    gfg = gfg * ( (1-H2F(fg))*H2F(fg) );
    gig = gig * ( (1-H2F(ig))*H2F(ig) );

    *ih = F2H(gig);
    *fh = F2H(gfg);
    *ch = F2H(gcg);
    *oh = F2H(gog);

    *gi = F2H(gcx);
#endif
  }
}


// ************ START Create function calls ********** //
#define FILL_FUNCTION(ITYPE, DIM, FUNCTION) FUNCTION(ITYPE, DIM)

#define FILL_DIM(ITYPE, DIM, FUNCTION)          \
  switch (DIM) {                                \
  case -2:                                      \
    FILL_FUNCTION(ITYPE, -2, FUNCTION);         \
    break;                                      \
  case 1:                                       \
    FILL_FUNCTION(ITYPE, 1, FUNCTION);          \
    break;                                      \
  case 2:                                       \
    FILL_FUNCTION(ITYPE, 2, FUNCTION);          \
    break;                                      \
  default:                                      \
    FILL_FUNCTION(ITYPE, -1, FUNCTION);         \
    break;                                      \
  }

#define LSTM_FORWARD(ITYPE, DIM) THNN_(LSTMForward)             \
  <DATATYPE, ITYPE, DIM>                                        \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>        \
  (inputI, hiddenI,                                             \
   bias1I, bias2I, cxI, hyI, cyI,                               \
   hid_size, totalElements);

#define LSTM_BACKWARD(ITYPE, DIM) THNN_(LSTMBackward)           \
  <DATATYPE, ITYPE, DIM>                                        \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>        \
  (storageI, gradingatesI, cxI, cyI,                            \
   gradoutI, gradoutcI, gradincxI,                              \
   hid_size, totalElements);

#define GRU_FORWARD(ITYPE, DIM) THNN_(GRUForward)<DATATYPE, ITYPE, DIM> \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>                \
  (inputI, hiddenI, bias1I, bias2I, hxI, hyI, storageI,                 \
   hid_size, totalElements);

#define GRU_BACKWARD(ITYPE, DIM) THNN_(GRUBackward)                     \
  <DATATYPE, ITYPE, DIM>                                                \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>                \
  (gradininputI, gradinhiddenI, gradoutI, gradinhxI, storageI,                        \
   hid_size, totalElements);

// ************ END Create actual function calls ************ //

template<typename INDTYPE>
void THNN_(LSTM_forw_ind_wrap)(
   THCState *state,
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *bias1,
   THCTensor *bias2,
   THCTensor *cx,
   THCTensor *hy,
   THCTensor *cy)
{
  bool has_bias = (bias1!=NULL);

  int maxDim;
  if(has_bias){
    THCUNN_assertSameGPU(state, 7, input, hidden, bias1, bias2, hy, cy, cx);
    maxDim = THNN_(minIndexType)
      (state, 7, input, hidden, bias1, bias2, hy, cy, cx);
  }else{
    THCUNN_assertSameGPU(state, 5, input, hidden, hy, cy, cx);
    maxDim = THNN_(minIndexType)
      (state, 5, input, hidden, hy, cy, cx);
  }

  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, cx);

  const dim3 block = getApplyBlock();
  dim3 grid;
  THAssertMsg(getApplyGrid(state, totalElements, grid),
          "Could not get grid size for pointwise apply.");

  TensorInfo<DATATYPE, INDTYPE> inputI =
    getTensorInfo<THCTensor, INDTYPE>(state, input);
  TensorInfo<DATATYPE, INDTYPE> hiddenI =
    getTensorInfo<THCTensor, INDTYPE>(state, hidden);
  TensorInfo<DATATYPE, INDTYPE> cxI =
    getTensorInfo<THCTensor, INDTYPE>(state, cx);
  TensorInfo<DATATYPE, INDTYPE> hyI =
    getTensorInfo<THCTensor, INDTYPE>(state, hy);
  TensorInfo<DATATYPE, INDTYPE> cyI =
    getTensorInfo<THCTensor, INDTYPE>(state, cy);

  INDTYPE hid_size = cxI.sizes[cxI.dims-1];
  if(has_bias){
    THAssertMsg( hid_size*4 == THCTensor_(nElement)(state, bias1) &&
                 hid_size*4 == THCTensor_(nElement)(state, bias2),
                 "Bias in pointwise operation is an incorrect size, must be 4 x feature size.");
  }

  if(maxDim == -2){
    inputI.collapseDims();
    hiddenI.collapseDims();
    cxI.collapseDims();
    hyI.collapseDims();
    cyI.collapseDims();
  }

  INDTYPE zero[1] = {0};
  TensorInfo<DATATYPE, INDTYPE> nullinfo =
    TensorInfo<DATATYPE, INDTYPE>(NULL, 1, zero, zero);
  TensorInfo<DATATYPE, INDTYPE> bias1I = nullinfo;
  TensorInfo<DATATYPE, INDTYPE> bias2I = nullinfo;

  if(has_bias){
    bias1I = getTensorInfo<THCTensor, INDTYPE>(state, bias1);
    bias2I = getTensorInfo<THCTensor, INDTYPE>(state, bias2);
    if(maxDim == -2){
      bias1I.collapseDims();
      bias2I.collapseDims();
    }
  }

  FILL_DIM(INDTYPE, maxDim, LSTM_FORWARD);

}
void THNN_(LSTMFused_updateOutput)(
   THCState *state,
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *bias1,
   THCTensor *bias2,
   THCTensor *cx,
   THCTensor *hy,
   THCTensor *cy)
{
  THCTensor_(resizeAs)(state, hy, cx);
  THCTensor_(resizeAs)(state, cy, cx);
  THNN_(FusedRNNAssertSizes)(state, 4, 5, input, hidden, hy, cy, cx);

  bool has_bias = (bias1!=NULL);
  bool canUse32bi;
  if(has_bias){
    canUse32bi = THNN_(canUse32BitIndexMath)
      (state, 7, input, hidden, bias1, bias2, hy, cy, cx);
  }else{
    canUse32bi = THNN_(canUse32BitIndexMath)
      (state, 5, input, hidden, hy, cy, cx);
  }

  if(canUse32bi){
    THNN_(LSTM_forw_ind_wrap)<uint32_t>
      (state, input, hidden, bias1, bias2, cx, hy, cy);
  }else{
    THNN_(LSTM_forw_ind_wrap)<uint64_t>
      (state, input, hidden, bias1, bias2, cx, hy, cy);
  }
    THCudaCheck(cudaGetLastError());
}

template<typename INDTYPE>
void THNN_(LSTM_back_ind_wrap)(
   THCState *state,
   THCTensor *storage,
   THCTensor *gradInGates,
   THCTensor *cx,
   THCTensor *cy,
   THCTensor *gradOutput,
   THCTensor *gradOutputCell,
   THCTensor *gradInputCx)
{
  int maxDim = THNN_(minIndexType)
    (state, 7, storage, gradInGates, cx, cy,
     gradOutput, gradOutputCell, gradInputCx);
  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, gradOutput);

  const dim3 block = getApplyBlock();
  dim3 grid;
  THAssertMsg(getApplyGrid(state, totalElements, grid),
              "Could not get grid size for pointwise apply");

  TensorInfo<DATATYPE, INDTYPE> storageI =
    getTensorInfo<THCTensor, INDTYPE>(state, storage);
  TensorInfo<DATATYPE, INDTYPE> gradingatesI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInGates);
  TensorInfo<DATATYPE, INDTYPE> cxI =
    getTensorInfo<THCTensor, INDTYPE>(state, cx);
  TensorInfo<DATATYPE, INDTYPE> cyI =
    getTensorInfo<THCTensor, INDTYPE>(state, cy);
  TensorInfo<DATATYPE, INDTYPE> gradoutI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradOutput);
  TensorInfo<DATATYPE, INDTYPE> gradoutcI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradOutputCell);
  TensorInfo<DATATYPE, INDTYPE> gradincxI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInputCx);

  INDTYPE hid_size = gradoutI.sizes[gradoutI.dims-1];

  if(maxDim == -2){
    storageI.collapseDims();
    gradingatesI.collapseDims();
    cxI.collapseDims();
    cyI.collapseDims();
    gradoutI.collapseDims();
    gradoutcI.collapseDims();
    gradincxI.collapseDims();
  }
  FILL_DIM(INDTYPE, maxDim, LSTM_BACKWARD);

}

void THNN_(LSTMFused_updateGradInput)(
   THCState *state,
   THCTensor *storage,
   THCTensor *gradInGates,
   THCTensor *cx,
   THCTensor *cy,
   THCTensor *gradOutput,
   THCTensor *gradOutputCell,
   THCTensor *gradInputCx)
{
  THCTensor_(resizeAs)(state, gradInputCx, gradOutput);
  THCUNN_assertSameGPU(state, 7, storage, gradInGates, cx, cy,
               gradOutput, gradOutputCell, gradInputCx);
  THNN_(FusedRNNAssertSizes)
    (state, 4, 7, storage, gradInGates, cx, cy,
     gradOutput, gradOutputCell, gradInputCx);

  bool canUse32bi = THNN_(canUse32BitIndexMath)
    (state, 7, storage, gradInGates, cx, cy,
     gradOutput, gradOutputCell, gradInputCx);

  if(canUse32bi){
    THNN_(LSTM_back_ind_wrap)<uint32_t>
      (state, storage, gradInGates, cx, cy,
       gradOutput, gradOutputCell, gradInputCx);
  }else{
    THNN_(LSTM_back_ind_wrap)<uint64_t>
      (state, storage, gradInGates, cx, cy,
       gradOutput, gradOutputCell, gradInputCx);
  }
  THCudaCheck(cudaGetLastError());
}

template<typename INDTYPE>
void THNN_(GRU_forw_ind_wrap)(
   THCState *state,
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *bias1,
   THCTensor *bias2,
   THCTensor *hx,
   THCTensor *hy,
   THCTensor *storage)
{
  bool has_bias = (bias1!=NULL);
  int maxDim;

  if(has_bias){
    THCUNN_assertSameGPU
      (state, 7, input, hidden, hx, hy, bias1, bias2, storage);
    maxDim = THNN_(minIndexType)
      (state, 7, input, hidden, hx, hy, bias1, bias2, storage);
  }else{
    THCUNN_assertSameGPU
      (state, 5, input, hidden, hx, hy, storage);
    maxDim = THNN_(minIndexType)
      (state, 5, input, hidden, hx, hy, storage);
  }

  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, hx);

  const dim3 block = getApplyBlock();
  dim3 grid;
  THAssertMsg(getApplyGrid(state, totalElements, grid),
              "Could not get grid size for pointwise apply.");

  TensorInfo<DATATYPE, INDTYPE> inputI =
    getTensorInfo<THCTensor, INDTYPE>(state, input);
  TensorInfo<DATATYPE, INDTYPE> hiddenI =
    getTensorInfo<THCTensor, INDTYPE>(state, hidden);
  TensorInfo<DATATYPE, INDTYPE> hxI =
    getTensorInfo<THCTensor, INDTYPE>(state, hx);
  TensorInfo<DATATYPE, INDTYPE> hyI =
    getTensorInfo<THCTensor, INDTYPE>(state, hy);
  TensorInfo<DATATYPE, INDTYPE> storageI =
    getTensorInfo<THCTensor, INDTYPE>(state, storage);

  INDTYPE hid_size = hxI.sizes[hxI.dims-1];
  if(has_bias){
    THAssertMsg( hid_size*3 == THCTensor_(nElement)(state, bias1) &&
                 hid_size*3 == THCTensor_(nElement)(state, bias2),
                 "Bias in pointwise operation is an incorrect size, must be 3 x feature size.");
  }

  if(maxDim == -2){
    inputI.collapseDims();
    hiddenI.collapseDims();
    hyI.collapseDims();
    hxI.collapseDims();
    storageI.collapseDims();
  }

  INDTYPE zero[1] = {0};
  TensorInfo<DATATYPE, INDTYPE> nullinfo =
    TensorInfo<DATATYPE, INDTYPE>(NULL, 1, zero, zero);
  TensorInfo<DATATYPE, INDTYPE> bias1I = nullinfo;
  TensorInfo<DATATYPE, INDTYPE> bias2I = nullinfo;

  if(has_bias){
    bias1I = getTensorInfo<THCTensor, INDTYPE>(state, bias1);
    bias2I = getTensorInfo<THCTensor, INDTYPE>(state, bias2);
    if(maxDim == -2){
      bias1I.collapseDims();
      bias2I.collapseDims();
    }
  }

  FILL_DIM(INDTYPE, maxDim, GRU_FORWARD);

}

void THNN_(GRUFused_updateOutput)(
   THCState *state,
   THCTensor *input,
   THCTensor *hidden,
   THCTensor *bias1,
   THCTensor *bias2,
   THCTensor *hx,
   THCTensor *hy,
   THCTensor *storage)
{
  THCTensor_(resizeAs)(state, hy, hx);
  THNN_(FusedRNNAssertSizes)(state, 3, 4, input, hidden, hx, hy);
  THArgCheck(THCTensor_(nElement)(state, storage) ==
             THCTensor_(nElement)(state, hx)*5,
             3, "Storage tensor for fused kernel was not sized correctly.");


  bool has_bias = (bias1!=NULL);
  bool canUse32bi;

  if(has_bias){
    canUse32bi = THNN_(canUse32BitIndexMath)
      (state, 7, input, hidden, hx, hy, bias1, bias2, storage);
  }else{
    canUse32bi = THNN_(canUse32BitIndexMath)
      (state, 5, input, hidden, hx, hy, storage);
  }

  if(canUse32bi){
    THNN_(GRU_forw_ind_wrap)<uint32_t>
      (state, input, hidden, bias1, bias2, hx, hy, storage);
  }else{
    THNN_(GRU_forw_ind_wrap)<uint64_t>
      (state, input, hidden, bias1, bias2, hx, hy, storage);
  }

  THCudaCheck(cudaGetLastError());
}

template<typename INDTYPE>
void THNN_(GRU_back_ind_wrap)(
   THCState *state,
   THCTensor *gradInInput,
   THCTensor *gradInHidden,
   THCTensor *gradOutput,
   THCTensor *gradInputHx,
   THCTensor *storage)
{

  int maxDim = THNN_(minIndexType)(state, 5, gradInInput, gradInHidden, gradOutput,
                                   gradInputHx, storage);
  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, gradOutput);

  const dim3 block = getApplyBlock();
  dim3 grid;
  THAssertMsg(getApplyGrid(state, totalElements, grid),
          "Could not get grid size for pointwise apply");

  TensorInfo<DATATYPE, INDTYPE> gradininputI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInInput);
  TensorInfo<DATATYPE, INDTYPE> gradinhiddenI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInHidden);
  TensorInfo<DATATYPE, INDTYPE> gradoutI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradOutput);
  TensorInfo<DATATYPE, INDTYPE> gradinhxI =
    getTensorInfo<THCTensor, INDTYPE>(state, gradInputHx);
  TensorInfo<DATATYPE, INDTYPE> storageI =
    getTensorInfo<THCTensor, INDTYPE>(state, storage);

  INDTYPE hid_size = gradoutI.sizes[gradoutI.dims-1];

  if(maxDim == -2){
    gradininputI.collapseDims();
    gradinhiddenI.collapseDims();
    gradoutI.collapseDims();
    gradinhxI.collapseDims();
    storageI.collapseDims();
  }
  FILL_DIM(INDTYPE, maxDim, GRU_BACKWARD);
}

void THNN_(GRUFused_updateGradInput)(
   THCState *state,
   THCTensor *gradInInput,
   THCTensor *gradInHidden,
   THCTensor *gradOutput,
   THCTensor *gradInputHx,
   THCTensor *storage)
{
  THCTensor_(resizeAs)(state, gradInputHx, gradOutput);
  THCUNN_assertSameGPU(state, 5, gradInInput, gradInHidden, gradOutput, gradInputHx, storage);
  THNN_(FusedRNNAssertSizes)(state, 3, 4, gradInInput, gradInHidden, gradOutput, gradInputHx);
  bool canUse32bi = THNN_(canUse32BitIndexMath)(state, 5, gradInInput, gradInHidden,
                                                gradOutput, gradInputHx, storage);
  if(canUse32bi){
    THNN_(GRU_back_ind_wrap)<uint32_t>
      (state, gradInInput, gradInHidden, gradOutput, gradInputHx, storage);
  }else{
    THNN_(GRU_back_ind_wrap)<uint64_t>
      (state, gradInInput, gradInHidden, gradOutput, gradInputHx, storage);
  }

  THCudaCheck(cudaGetLastError());
}

//Clean up compiler namespace
#undef DEVICE_LINEAR_GET
#undef H2F
#undef F2H
#undef EXPAND_FUNCTION
#undef EXPAND_DIM
#undef EXPAND_TYPE
#undef FILL_TYPES_FORWARD
#undef FILL_FORWARD
#undef FILL_TYPES_BACKWARD
#undef FILL_BACKWARD

#endif
