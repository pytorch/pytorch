#ifndef THC_GENERIC_FILE
#define THC_GENERIC_FILE "generic/FusedRNNKernel.cu"
#else
#include <cstdarg>

#include "../common.h"

#define DATATYPE TensorUtils<THCTensor>::DataType

//factor will be 3 for GRU and 4 for LSTM
void THNN_(FusedRNNAssertSizes)(THCState *state, int factor, int count ...)
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
  
  int maxDim = -2;
  for (int arg=0; arg < count; ++arg){
    THCTensor* tens = va_arg(list, THCTensor*);
    if(THCTensor_(isContiguous)(state, tens)) continue;
    int tensdims = TensorUtils<THCTensor>::getDims(state, tens);
    maxDim = (( tensdims> maxDim) ? tensdims : maxDim);
  }
  
  va_end(list);
  return maxDim;
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


#define DEVICE_LINEAR_GET(D_TENSOR, INDEX)				\
  D_TENSOR.data[IndexToOffset<T, IndexType, Dims>::get(INDEX, D_TENSOR)]

#define H2F(input) __half2float(input)
#define F2H(input) __float2half(input)

template <typename T,
	  typename IndexType,
	  int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
  THNN_(GRUForward)(TensorInfo<T, IndexType> Input,
		    TensorInfo<T, IndexType> Hidden,
		    TensorInfo<T, IndexType> Bias1,
		    TensorInfo<T, IndexType> Bias2,
		    TensorInfo<T, IndexType> Hx,
		    TensorInfo<T, IndexType> Output,
		    IndexType hsz,
		    IndexType totalElements)
{
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x)
    {

      IndexType offset = (linearIndex/hsz)*3*hsz+linearIndex%hsz;

      T* ir = &DEVICE_LINEAR_GET(Input, offset+0*hsz);
      T* ii = &DEVICE_LINEAR_GET(Input, offset+1*hsz);
      T* in = &DEVICE_LINEAR_GET(Input, offset+2*hsz);

      T* hr = &DEVICE_LINEAR_GET(Hidden,offset+0*hsz);
      T* hi = &DEVICE_LINEAR_GET(Hidden,offset+1*hsz);
      T hn = DEVICE_LINEAR_GET(Hidden,  offset+2*hsz);

      T hx = DEVICE_LINEAR_GET(Hx, linearIndex);

      T b1r = DEVICE_LINEAR_GET(Bias1, linearIndex%hsz+0*hsz);
      T b1i = DEVICE_LINEAR_GET(Bias1, linearIndex%hsz+1*hsz);
      T b1n = DEVICE_LINEAR_GET(Bias1, linearIndex%hsz+2*hsz);

      T b2r = DEVICE_LINEAR_GET(Bias2, linearIndex%hsz+0*hsz);
      T b2i = DEVICE_LINEAR_GET(Bias2, linearIndex%hsz+1*hsz);
      T b2n = DEVICE_LINEAR_GET(Bias2, linearIndex%hsz+2*hsz);

      T* out = &DEVICE_LINEAR_GET(Output, linearIndex);

#ifndef THC_REAL_IS_HALF
      
      T rg, ig, ng;

      rg = *ir + *hr + b1r + b2r;
      ig = *ii + *hi + b1i + b2i;
      
      TensorSigmoidOp<real>()(&rg, &rg);
      TensorSigmoidOp<real>()(&ig, &ig);
      ng = *in + b1n + rg * (hn + b2n);
      ng = THCNumerics<T>::tanh(ng);
      *out = ng + ig * (hx - ng);

      //SAVE FOR BACKWARDS
      *ir = rg;
      *ii = ig;
      *in = ng;
      *hr = hx;
      *hi = hn + b2n;
#else
      
      float rg, ig, ng;

      rg = H2F(*ir) + H2F(*hr) + H2F(b1r) + H2F(b2r);
      ig = H2F(*ii) + H2F(*hi) + H2F(b1i) + H2F(b2i);
      
      TensorSigmoidOp<float>()(&rg, &rg);
      TensorSigmoidOp<float>()(&ig, &ig);
      ng = H2F(*in) + H2F(b1n) + rg*( H2F(hn)+H2F(b2n) );
      ng = THCNumerics<float>::tanh(ng);
      *out = F2H( ng + ig * ( H2F(hx)-ng ) );

      //SAVE FOR BACKWARDS
      *ir = F2H(rg);
      *ii = F2H(ig);
      *in = F2H(ng);
      *hr = hx;
      *hi = F2H( H2F(hn) + H2F(b2n) );

#endif
    }
}

template <typename T,
	  typename IndexType,
	  int Dims>
#if __CUDA_ARCH__ >= 350
__launch_bounds__(32 * 16, 4)
#endif
__global__ void
  THNN_(GRUBackward)(TensorInfo<T, IndexType> input,
		     TensorInfo<T, IndexType> hidden,
		     TensorInfo<T, IndexType> gradoutput,
		     TensorInfo<T, IndexType> gradinput,
		     IndexType hsz,
		     IndexType totalElements)
{
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < totalElements;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType offset = (linearIndex/hsz)*3*hsz+linearIndex%hsz;;

    //will return input grads here
    T* rg = &DEVICE_LINEAR_GET(input, offset+0*hsz);
    T* ig = &DEVICE_LINEAR_GET(input, offset+1*hsz);
    T* ng = &DEVICE_LINEAR_GET(input, offset+2*hsz);
    //will return hidden grads here
    T* hx = &DEVICE_LINEAR_GET(hidden, offset+0*hsz);
    T* hn = &DEVICE_LINEAR_GET(hidden, offset+1*hsz);
    T* oghn=&DEVICE_LINEAR_GET(hidden, offset+2*hsz);
    
    T* gi = &DEVICE_LINEAR_GET(gradinput, linearIndex);
    
    T* go = &DEVICE_LINEAR_GET(gradoutput, linearIndex);
    
#ifndef THC_REAL_IS_HALF
    
    T gig = (*go)*(*hx-*ng)*( 1-(*ig) )*(*ig);
    T ghx = (*go)*(*ig);
    T gin = (*go)*(1-*ig)*( 1-(*ng)*(*ng) );
    T ghn = (gin) * (*rg);
    T grg = (gin)*(*hn)*( 1-(*rg) )*(*rg);

    *gi = ghx;
    
    *rg = grg;
    *ig = gig;
    *ng = gin;
    
    *hx = grg;
    *hn = gig;
    *oghn = ghn;
    
#else
    float gig = H2F(*go)*( H2F(*hx)-H2F(*ng) )*( 1-H2F(*ig) )*H2F(*ig);
    float ghx = H2F(*go)*H2F(*ig);
    float gin = H2F(*go)*( 1-H2F(*ig) )*( 1-H2F(*ng)*H2F(*ng) );
    float ghn = H2F(gin) * H2F(*rg);
    float grg = H2F(gin)*H2F(*hn)*( 1-H2F(*rg) )*H2F(*rg);

    *gi = F2H(ghx);
    
    *rg = F2H(grg);
    *ig = F2H(gig);
    *ng = F2H(gin);
    
    *hx = F2H(grg);
    *hn = F2H(gig);
    *oghn = F2H(ghn);

#endif
      
  }
}


// *********** START Generate specializations *************** //
#define EXPAND_FUNCTION(ITYPE, DIM)					\
  template __global__ void THNN_(GRUForward)<DATATYPE, ITYPE, DIM>	\
    (TensorInfo<DATATYPE, ITYPE> inputI,				\
     TensorInfo<DATATYPE, ITYPE> hiddenI,				\
     TensorInfo<DATATYPE, ITYPE> bias1I,				\
     TensorInfo<DATATYPE, ITYPE> bias2I,				\
     TensorInfo<DATATYPE, ITYPE> prevHI,				\
     TensorInfo<DATATYPE, ITYPE> outputI,				\
     ITYPE hsz,								\
     ITYPE totalElements);						\
									\
  template void __global__ THNN_(GRUBackward)<DATATYPE, ITYPE, DIM>	\
    (TensorInfo<DATATYPE, ITYPE> input,					\
     TensorInfo<DATATYPE, ITYPE> hidden,				\
     TensorInfo<DATATYPE, ITYPE> gradoutput,				\
     TensorInfo<DATATYPE, ITYPE> gradinput,				\
     ITYPE hsz,								\
     ITYPE totalElements);						\
  
#define EXPAND_DIM(ITYPE)				\
  EXPAND_FUNCTION(ITYPE, -2)				\
  EXPAND_FUNCTION(ITYPE, -1)                     	\
  EXPAND_FUNCTION(ITYPE, 1)                      	\
  EXPAND_FUNCTION(ITYPE, 2)                      	\

#define EXPAND_TYPE                                     \
  EXPAND_DIM(unsigned int)				\
  EXPAND_DIM(unsigned long)				\

EXPAND_TYPE

// ************ END generating specializations ************** //

// ************ START Create actual function calls ********** //
#define FILL_TYPES_FORWARD(ITYPE, DIM)  THNN_(GRUForward)<DATATYPE, ITYPE, DIM> \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>		\
  (inputI, hiddenI, bias1I, bias2I, prevHI, outputI, hid_size, totalElements); \
  
#define FILL_FORWARD(ITYPE, DIM)		\
  switch (DIM) {				\
  case -2:					\
    FILL_TYPES_FORWARD(ITYPE, -2);		\
    break;					\
  case 1:					\
    FILL_TYPES_FORWARD(ITYPE, 1);		\
    break;					\
  case 2:					\
    FILL_TYPES_FORWARD(ITYPE, 2);		\
    break;					\
  default:					\
    FILL_TYPES_FORWARD(ITYPE, -1);		\
    break;					\
  }

#define FILL_TYPES_BACKWARD(ITYPE, DIM)  THNN_(GRUBackward)<DATATYPE, ITYPE, DIM> \
  <<<grid, block, 0, THCState_getCurrentStream(state)>>>		\
  (inputI, hiddenI, gradoutI, gradinI, hid_size, totalElements);	
  
#define FILL_BACKWARD(ITYPE, DIM)		\
  switch (DIM) {				\
  case -2:					\
    FILL_TYPES_BACKWARD(ITYPE, -2);		\
    break;					\
  case 1:					\
    FILL_TYPES_BACKWARD(ITYPE, 1);		\
    break;					\
  case 2:					\
    FILL_TYPES_BACKWARD(ITYPE, 2);		\
    break;					\
  default:					\
    FILL_TYPES_BACKWARD(ITYPE, -1);		\
    break;					\
  }

// ************ END Create actual function calls ************ //

void THNN_(GRUFused_updateOutput)(
          THCState *state,
          THCTensor *input,
	  THCTensor *hidden,
	  THCTensor *bias1,
	  THCTensor *bias2,
	  THCTensor *prevH,
	  THCTensor *output)
{
  THCTensor_(resizeAs)(state, output, prevH);
  THNN_(FusedRNNAssertSizes)(state, 3, 4, input, hidden, prevH, output);
  THCUNN_assertSameGPU(state, 6, input, hidden, prevH, output, bias1, bias2);
  bool canUse32bi = THNN_(canUse32BitIndexMath)(state, 6, input, hidden, prevH, output, bias1, bias2);
  int maxDim = THNN_(minIndexType)(state, 6, input, hidden, prevH, output, bias1, bias2);

  const dim3 block = getApplyBlock();
  //const dim3 block(32, 32);
  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, prevH);

  dim3 grid;
  TensorInfo<DATATYPE, unsigned long> tmphi =
    getTensorInfo<THCTensor, unsigned long>(state, prevH);
  unsigned long tmp_hid = tmphi.sizes[tmphi.dims-1];

  THAssertMsg( tmp_hid*3 == THCTensor_(nElement)(state, bias1) &&
	       tmp_hid*3 == THCTensor_(nElement)(state, bias2),
	       "Bias in pointwise operation is an incorrect size, must be 3 x feature size.");
  
  THAssertMsg(getApplyGrid(state, totalElements, grid),
	      "Could not get grid size for pointwise apply");
  if(canUse32bi){
    TensorInfo<DATATYPE, unsigned int> inputI =
      getTensorInfo<THCTensor, unsigned int>(state, input);
    TensorInfo<DATATYPE, unsigned int> hiddenI =
      getTensorInfo<THCTensor, unsigned int>(state, hidden);
    TensorInfo<DATATYPE, unsigned int> outputI =
      getTensorInfo<THCTensor, unsigned int>(state, output);
    TensorInfo<DATATYPE, unsigned int> prevHI =
      getTensorInfo<THCTensor, unsigned int>(state, prevH);
    TensorInfo<DATATYPE, unsigned int> bias1I =
      getTensorInfo<THCTensor, unsigned int>(state, bias1);
    TensorInfo<DATATYPE, unsigned int> bias2I =
      getTensorInfo<THCTensor, unsigned int>(state, bias2);


    unsigned int hid_size = prevHI.sizes[prevHI.dims-1];

    inputI.collapseDims();
    hiddenI.collapseDims();
    outputI.collapseDims();
    prevHI.collapseDims();
    bias1I.collapseDims();
    bias2I.collapseDims();
    
    FILL_FORWARD(unsigned int, maxDim);
      
  }else{

    TensorInfo<DATATYPE, unsigned long> inputI =
      getTensorInfo<THCTensor, unsigned long>(state, input);
    TensorInfo<DATATYPE, unsigned long> hiddenI =
      getTensorInfo<THCTensor, unsigned long>(state, hidden);
    TensorInfo<DATATYPE, unsigned long> outputI =
      getTensorInfo<THCTensor, unsigned long>(state, output);
    TensorInfo<DATATYPE, unsigned long> prevHI =
      getTensorInfo<THCTensor, unsigned long>(state, prevH);
    TensorInfo<DATATYPE, unsigned long> bias1I =
      getTensorInfo<THCTensor, unsigned long>(state, bias1);
    TensorInfo<DATATYPE, unsigned long> bias2I =
      getTensorInfo<THCTensor, unsigned long>(state, bias2);

    unsigned long hid_size = prevHI.sizes[prevHI.dims-1];

    inputI.collapseDims();
    hiddenI.collapseDims();
    outputI.collapseDims();
    prevHI.collapseDims();
    bias1I.collapseDims();
    bias2I.collapseDims();

    FILL_FORWARD(unsigned long, maxDim);
  }

  THCudaCheck(cudaGetLastError());
}

void THNN_(GRUFused_updateGradInput)(
          THCState *state,
          THCTensor *input,
          THCTensor *hidden,
          THCTensor *gradOutput,
          THCTensor *gradInput)
{
  THCTensor_(resizeAs)(state, gradInput, gradOutput);
  THCUNN_assertSameGPU(state, 4, input, hidden, gradOutput, gradInput);
  THNN_(FusedRNNAssertSizes)(state, 3, 4, input, hidden, gradOutput, gradInput);
  bool canUse32bi = THNN_(canUse32BitIndexMath)(state, 4, input, hidden, gradOutput, gradInput);
  int maxDim = THNN_(minIndexType)(state, 4, input, hidden, gradOutput, gradInput);

  const dim3 block = getApplyBlock();

  ptrdiff_t totalElements = TensorUtils<THCTensor>::getNumElements(state, gradOutput);

  dim3 grid;
  
  THAssertMsg(getApplyGrid(state, totalElements, grid),
	      "Could not get grid size for pointwise apply");

  if(canUse32bi){
    TensorInfo<DATATYPE, unsigned int> inputI =
      getTensorInfo<THCTensor, unsigned int>(state, input);
    TensorInfo<DATATYPE, unsigned int> hiddenI =
      getTensorInfo<THCTensor, unsigned int>(state, hidden);
    TensorInfo<DATATYPE, unsigned int> gradoutI =
      getTensorInfo<THCTensor, unsigned int>(state, gradOutput);
    TensorInfo<DATATYPE, unsigned int> gradinI =
      getTensorInfo<THCTensor, unsigned int>(state, gradInput);

    unsigned int hid_size = gradoutI.sizes[gradoutI.dims-1];

    inputI.collapseDims();
    hiddenI.collapseDims();
    gradoutI.collapseDims();
    gradinI.collapseDims();

    FILL_BACKWARD(unsigned int, maxDim);
      
  }else{
    TensorInfo<DATATYPE, unsigned long> inputI =
      getTensorInfo<THCTensor, unsigned long>(state, input);
    TensorInfo<DATATYPE, unsigned long> hiddenI =
      getTensorInfo<THCTensor, unsigned long>(state, hidden);
    TensorInfo<DATATYPE, unsigned long> gradoutI =
      getTensorInfo<THCTensor, unsigned long>(state, gradOutput);
    TensorInfo<DATATYPE, unsigned long> gradinI =
      getTensorInfo<THCTensor, unsigned long>(state, gradInput);

    unsigned long hid_size = gradoutI.sizes[gradoutI.dims-1];
    
    inputI.collapseDims();
    hiddenI.collapseDims();
    gradoutI.collapseDims();
    gradinI.collapseDims();

    FILL_BACKWARD(unsigned long, maxDim);

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
