#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorMath.cpp"
#else

using namespace thd;
using namespace rpc;
using namespace master;

void THDTensor_(gather)(THDTensor *self, THDTensor *src, int dim, THDLongTensor *index) {
  THArgCheck(dim < self->nDimension, 2, "Index dimension is out of bounds");
  THArgCheck(THDLongTensor_nDimension(index) == self->nDimension, 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(src->nDimension == self->nDimension, 4,
             "Input tensor must have same dimensions as output tensor");

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorGather,
      self,
      src,
      dim,
      index
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(scatter)(THDTensor *self, int dim, THDLongTensor *index, THDTensor *src) {
  THArgCheck(dim < self->nDimension, 2, "Index dimension is out of bounds");
  THArgCheck(THDLongTensor_nDimension(index) == self->nDimension, 3,
             "Index tensor must have same dimensions as output tensor");
  THArgCheck(src->nDimension == self->nDimension, 4,
             "Input tensor must have same dimensions as output tensor");

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorScatter,
      self,
      dim,
      index,
      src
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(scatterFill)(THDTensor *self, int dim, THDLongTensor *index, real val) {
  THArgCheck(dim < self->nDimension, 2, "Index dimension is out of bounds");
  THArgCheck(THDLongTensor_nDimension(index) == self->nDimension, 3,
             "Index tensor must have same dimensions as output tensor");

  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorScatterFill,
      self,
      dim,
      index,
      val
    ),
    THDState::s_current_worker
  );
}

THD_API void THDTensor_(scatterAdd)(THDTensor *self, int dim, THDLongTensor *index,
                                 THDTensor *src) {
  THError("scatterAdd not implemented");
}

void THDTensor_(max)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, int dimension, int keepdim) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THDLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMax, self, indices_, src, dimension, keepdim),
    THDState::s_current_worker
  );

  if (!keepdim) {
    THDTensor_(_squeeze1d)(self, self, dimension);
    THDLongTensor__squeeze1d(indices_, indices_, dimension);
  }
}

void THDTensor_(min)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, int dimension, int keepdim) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 2, "dimension %d out of range",
      dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THDLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMin, self, indices_, src, dimension, keepdim),
    THDState::s_current_worker
  );

  if (!keepdim) {
    THDTensor_(_squeeze1d)(self, self, dimension);
    THDLongTensor__squeeze1d(indices_, indices_, dimension);
  }
}

void THDTensor_(kthvalue)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, int64_t k, int dimension, int keepdim) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 3, "dimension out of range");
  THArgCheck(k > 0 && k <= src->size[dimension], 2, "selected index out of range");

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THDLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorKthvalue, self, indices_, src, k, dimension, keepdim),
    THDState::s_current_worker
  );

  if (!keepdim) {
    THDTensor_(_squeeze1d)(self, self, dimension);
    THDLongTensor__squeeze1d(indices_, indices_, dimension);
  }
}

void THDTensor_(mode)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, int dimension, int keepdim) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 3, "dimension out of range");

  THLongStorage *dim = THDTensor_(newSizeOf)(src);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(self, dim, NULL);
  THDLongTensor_resize(indices_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMode, self, indices_, src, dimension, keepdim),
    THDState::s_current_worker
  );

  if (!keepdim) {
    THDTensor_(_squeeze1d)(self, self, dimension);
    THDLongTensor__squeeze1d(indices_, indices_, dimension);
  }
}

void THDTensor_(median)(THDTensor *self, THDLongTensor *indices_, THDTensor *src, int dimension, int keepdim) {
  THArgCheck(dimension >= 0 && dimension < src->nDimension, 3, "dimension out of range");

  int64_t t_size_dim = src->size[dimension];
  int64_t k = (t_size_dim - 1) >> 1; /* take middle or one-before-middle element */

  THDTensor_(kthvalue)(self, indices_, src, k + 1, dimension, keepdim);
}


void THDTensor_(fill)(THDTensor *tensor, real value) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorFill,
      tensor,
      value
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(zero)(THDTensor *r) {
  THDTensor_(fill)(r, 0);
}

void THDTensor_(maskedFill)(THDTensor *tensor, THDByteTensor *mask, real value) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorMaskedFill,
      tensor,
      mask,
      value
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(maskedCopy)(THDTensor *tensor, THDByteTensor *mask, THDTensor* src) {
  if (THDTensor_(nElement)(tensor) != THDByteTensor_nElement(mask))
    THError("Number of elements of destination tensor != Number of elements in mask");
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorMaskedCopy,
      tensor,
      mask,
      src
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(maskedSelect)(THDTensor *tensor, THDTensor* src, THDByteTensor *mask) {
  ptrdiff_t numel = THDByteTensor_sumall(mask);
  THDTensor_(resize1d)(tensor, numel);
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorMaskedSelect,
      tensor,
      src,
      mask
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(nonzero)(THDLongTensor *subscript, THDTensor *tensor) {
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorNonzero,
      subscript,
      tensor
    ),
    THDState::s_current_worker
  );
  int64_t numel = receiveValueFromWorker<int64_t>(tensor->storage->node_id);
  THDLongTensor__resize2d(subscript, numel, tensor->nDimension);
}

void THDTensor_(indexSelect)(THDTensor *tensor, THDTensor *src, int dim, THDLongTensor *index) {
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim %d is out of bounds of tensor",
             dim + TH_INDEX_BASE);
  THArgCheck(src->nDimension > 0, 2, "Source tensor is empty");
  THLongStorage *newSize = THLongStorage_newWithSize(src->nDimension);
  THLongStorage_rawCopy(newSize, src->size);
  THDTensor_(resize)(tensor, newSize, NULL);
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorIndexSelect,
      tensor,
      src,
      dim,
      index
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(indexCopy)(THDTensor *tensor, int dim, THDLongTensor *index, THDTensor *src) {
  ptrdiff_t numel = THDLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim %d is out of bounds of tensor",
             dim + TH_INDEX_BASE);
  THArgCheck(numel == src->size[dim], 4, "Number of indices should be equal to source:size(dim)");
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorIndexCopy,
      tensor,
      dim,
      index,
      src
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(indexAdd)(THDTensor *tensor, int dim, THDLongTensor *index, THDTensor *src) {
  ptrdiff_t numel = THDLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < src->nDimension, 4, "Indexing dim %d is out of bounds of tensor",
             dim + TH_INDEX_BASE);
  THArgCheck(numel == src->size[dim], 4, "Number of indices should be equal to source:size(dim)");
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorIndexAdd,
      tensor,
      dim,
      index,
      src
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(indexFill)(THDTensor *tensor, int dim, THDLongTensor *index, real val) {
  ptrdiff_t numel = THDLongTensor_nElement(index);
  THArgCheck(index->nDimension == 1, 3, "Index is supposed to be a vector");
  THArgCheck(dim < tensor->nDimension, 4, "Indexing dim %d is out of bounds of tensor",
             dim + TH_INDEX_BASE);
  masterCommandChannel->sendMessage(
    packMessage(
      Functions::tensorIndexFill,
      tensor,
      dim,
      index,
      val
    ),
    THDState::s_current_worker
  );
}

void THDTensor_(zeros)(THDTensor *tensor, THLongStorage *size) {
  THDTensor_(resize)(tensor, size, nullptr);
  THDTensor_(zero)(tensor);
}

void THDTensor_(ones)(THDTensor *tensor, THLongStorage *size) {
  THDTensor_(resize)(tensor, size, nullptr);
  THDTensor_(fill)(tensor, 1);
}

ptrdiff_t THDTensor_(numel)(THDTensor *t) {
  return THDTensor_(nElement)(t);
}

void THDTensor_(diag)(THDTensor *r_, THDTensor *t, int k) {
  THArgCheck(THDTensor_(nDimension)(t) == 1 || THDTensor_(nDimension)(t) == 2,
      1, "matrix or a vector expected");

  if (THDTensor_(nDimension)(t) == 1) {
    int64_t t_size = THDTensor_(size)(t, 0);
    int64_t sz = t_size + (k >= 0 ? k : -k);

    THDTensor_(resize2d)(r_, sz, sz);
    THDTensor_(zero)(r_);
  } else {
    int64_t sz;
    if (k >= 0)
      sz = std::min(THDTensor_(size)(t, 0), THDTensor_(size)(t, 1)-k);
    else
      sz = std::min(THDTensor_(size)(t, 0)+k, THDTensor_(size)(t, 1));
    THDTensor_(resize1d)(r_, sz);
  }
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorDiag, r_, t, k),
    THDState::s_current_worker
  );
}

void THDTensor_(eye)(THDTensor *r, int64_t n, int64_t m) {
  THArgCheck(n > 0, 1, "invalid argument");

  if (m <= 0)
    m = n;

  THDTensor_(resize2d)(r, n, m);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorEye, r, n, m),
    THDState::s_current_worker
  );
}


void THDTensor_(range)(THDTensor *r_, accreal xmin,
                              accreal xmax, accreal step) {
  THArgCheck(step > 0 || step < 0, 3, "step must be a non-null number");
  THArgCheck(((step > 0) && (xmax >= xmin)) || ((step < 0) && (xmax <= xmin)),
              2, "upper bound and larger bound incoherent with step sign");

  ptrdiff_t size = static_cast<ptrdiff_t>((((xmax - xmin) / step) + 1));

  if (THDTensor_(nElement)(r_) != size)
    THDTensor_(resize1d)(r_, size);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRange, r_, xmin, xmax, step),
    THDState::s_current_worker
  );
}

void THDTensor_(randperm)(THDTensor *r_, THDGenerator *_generator, int64_t n) {
  THArgCheck(n > 0, 1, "must be strictly positive");
  THDTensor_(resize1d)(r_, n);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRange, r_, _generator, n),
    THDState::s_current_worker
  );
}

void THDTensor_(reshape)(THDTensor *r_, THDTensor *t, THLongStorage *size) {
  THDTensor_(resize)(r_, size, NULL);
  THDTensor_(copy)(r_, t);
}

void THDTensor_(sort)(THDTensor *rt_, THDLongTensor *ri_,
                             THDTensor *t, int dimension,
                             int descendingOrder) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t),
      2, "invalid dimension %d", dimension + TH_INDEX_BASE);

  THDTensor_(resizeAs)(rt_, t);
  THDTensor_(copy)(rt_, t);

  {
    THLongStorage *size = THDTensor_(newSizeOf)(t);
    THDLongTensor_resize(ri_, size, NULL);
    THLongStorage_free(size);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorSort, rt_, ri_, t, dimension, descendingOrder),
    THDState::s_current_worker
  );
}

void THDTensor_(topk)(THDTensor *rt_, THDLongTensor *ri_,
                      THDTensor *t, int64_t k, int dim,
                      int dir, int sorted) {
  int numDims = THDTensor_(nDimension)(t);
  THArgCheck(dim >= 0 && dim < numDims, 3, "dim not in range");

  int64_t sliceSize = THDTensor_(size)(t, dim);
  THArgCheck(k > 0 && k <= sliceSize, 2, "k not in range for dimension");

  THLongStorage *topKSize = THDTensor_(newSizeOf)(t);
  THLongStorage_set(topKSize, dim, k);
  THDTensor_(resize)(rt_, topKSize, NULL);
  THDLongTensor_resize(ri_, topKSize, NULL);
  THLongStorage_free(topKSize);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTopk, rt_, ri_, t, k, dim, dir, sorted),
    THDState::s_current_worker
  );
}

void THDTensor_(tril)(THDTensor *r_, THDTensor *t, int64_t k) {
  THArgCheck(THDTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  THDTensor_(resizeAs)(r_, t);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTril, r_, t, k),
    THDState::s_current_worker
  );
}

void THDTensor_(triu)(THDTensor *r_, THDTensor *t, int64_t k) {
  THArgCheck(THDTensor_(nDimension)(t) == 2, 1, "expected a matrix");

  THDTensor_(resizeAs)(r_, t);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorTriu, r_, t, k),
    THDState::s_current_worker
  );
}
void THDTensor_(cat)(THDTensor *r_, THDTensor *ta, THDTensor *tb, int dimension) {
  THDTensor* inputs[2];
  inputs[0] = ta;
  inputs[1] = tb;
  THDTensor_(catArray)(r_, inputs, 2, dimension);
}

void THDTensor_(catArray)(THDTensor *result, THDTensor **inputs,
                          int numInputs, int dimension) {
  THLongStorage *size;
  int64_t offset;
  int ndim = dimension + 1;
  int ldimension = dimension;
  bool allEmpty = true;
  for (int i = 0; i < numInputs; i++)
    ndim = std::max(ndim, inputs[i]->nDimension);

  if (dimension == -2)
    ldimension = ndim ? (ndim - 1) : 0;

  THArgCheck(numInputs > 0, 3, "invalid number of inputs %d", numInputs);
  THArgCheck(ldimension >= 0, 4, "invalid dimension %d", dimension + TH_INDEX_BASE);

  size = THLongStorage_newWithSize(ndim);

  for (int i = 0; i < ndim; i++) {
    int64_t dimSize = i < inputs[0]->nDimension ?
                   inputs[0]->size[i] :
                   std::min(inputs[0]->nDimension, 1);
    if (i == ldimension) {
      for (int j = 1; j < numInputs; j++) {
        dimSize += i < inputs[j]->nDimension ?
                   inputs[j]->size[i] :
                   std::min(inputs[j]->nDimension, 1);
      }
    } else {
      for (int j = 1; j < numInputs; j++) {
        int64_t sz = i < inputs[j]->nDimension ?
                  inputs[j]->size[i] :
                  std::min(inputs[j]->nDimension, 1);
        if (dimSize != sz && dimSize && sz) {
          THLongStorage_free(size);
          THError("inconsistent tensor sizes");
        } else if (!dimSize) {
          dimSize = sz;
        }
      }
    }
    allEmpty = allEmpty && !dimSize;
    THLongStorage_set(size, i, dimSize);
  }

  if (!allEmpty) {
    THDTensor_(resize)(result, size, NULL);
    std::vector<THDTensor*> inputs_vec(inputs, inputs + numInputs);

    // There's no need to send numInputs,
    // since sending inputs_vec does this implicitly
    masterCommandChannel->sendMessage(
      packMessage(Functions::tensorCatArray, result, inputs_vec, dimension),
      THDState::s_current_worker
    );
  }

  THLongStorage_free(size);
}

int THDTensor_(equal)(THDTensor *ta, THDTensor *tb) {
  if (!THDTensor_(isSameSizeAs)(ta, tb))
    return 0;
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorEqual, ta, tb),
    THDState::s_current_worker
  );
  return receiveValueFromWorker<int>(ta->storage->node_id);
}

void THDTensor_(tpow)(THDTensor *r_, real value, THDTensor *t) {
  THDTensor_(resizeAs)(r_, t);
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorEqual, r_, t, value),
    THDState::s_current_worker
  );
}

#define TENSOR_IMPLEMENT_LOGICAL(NAME,UPPNAME)                                \
  void THDTensor_(NAME##Value)(THDByteTensor *r_, THDTensor* t, real value) { \
    THDByteTensor__resize(r_, t->nDimension, t->size, NULL);                  \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME##Value, r_, t, value),           \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \
  void THDTensor_(NAME##ValueT)(THDTensor* r_, THDTensor* t, real value)  {   \
    THDTensor_(_resize)(r_, t->nDimension, t->size, NULL);                    \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME##ValueT, r_, t, value),          \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \
  void THDTensor_(NAME##Tensor)(THDByteTensor *r_, THDTensor *ta, THDTensor *tb) { \
    THDByteTensor__resize(r_, ta->nDimension, ta->size, NULL);                \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME##Tensor, r_, ta, tb),            \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \
  void THDTensor_(NAME##TensorT)(THDTensor *r_, THDTensor *ta, THDTensor *tb) { \
    THDTensor_(_resize)(r_, ta->nDimension, ta->size, NULL);                  \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME##TensorT, r_, ta, tb),           \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \


TENSOR_IMPLEMENT_LOGICAL(lt,Lt)
TENSOR_IMPLEMENT_LOGICAL(gt,Lt)
TENSOR_IMPLEMENT_LOGICAL(le,Le)
TENSOR_IMPLEMENT_LOGICAL(ge,Ge)
TENSOR_IMPLEMENT_LOGICAL(eq,Eq)
TENSOR_IMPLEMENT_LOGICAL(ne,Ne)

#undef TENSOR_IMPLEMENT_LOGICAL

#define TENSOR_IMPLEMENT_POINTWISE_FUNCTION(NAME, UPPNAME)    \
  void THDTensor_(NAME)(THDTensor *r_, THDTensor *t) {        \
    THDTensor_(resizeAs)(r_, t);                              \
    masterCommandChannel->sendMessage(                        \
      packMessage(Functions::tensor##UPPNAME, r_, t),         \
      THDState::s_current_worker                              \
    );                                                        \
  }                                                           \

#define TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION(NAME, UPPNAME)              \
  void THDTensor_(NAME)(THDTensor *r_, THDTensor *t, real value) {            \
    THDTensor_(resizeAs)(r_, t);                                              \
    masterCommandChannel->sendMessage(                                        \
      packMessage(Functions::tensor##UPPNAME, r_, t, value),                  \
      THDState::s_current_worker                                              \
    );                                                                        \
  }                                                                           \


#if defined(TH_REAL_IS_LONG) || defined(TH_REAL_IS_INT) ||\
    defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(abs,Abs)
#endif

#if defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(sigmoid,Sigmoid)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(log,Log)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(log10,Log10)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(log1p,Log1p)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(log2, Log2)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(exp,Exp)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(expm1,Expm1)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(cos,Cos)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(acos,Acos)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(cosh,Cosh)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(sin,Sin)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(asin,Asin)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(sinh,Sinh)

TENSOR_IMPLEMENT_POINTWISE_FUNCTION(tan,Tan)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(atan,Atan)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(tanh,Tanh)
TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION(pow,Pow)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(sqrt,Sqrt)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(rsqrt,Rsqrt)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(ceil,Ceil)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(floor,Floor)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(round,Round)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(trunc,Trunc)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(frac,Frac)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(neg,Neg)
TENSOR_IMPLEMENT_POINTWISE_FUNCTION(cinv,Cinv)

#undef TENSOR_IMPLEMENT_POINTWISE_VALUE_FUNCTION
#undef TENSOR_IMPLEMENT_POINTWISE_FUNCTION

void THDTensor_(atan2)(THDTensor *r_, THDTensor *tx, THDTensor *ty) {
  THDTensor_(resizeAs)(r_, tx);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorAtan2, r_, tx, ty),
    THDState::s_current_worker
  );
}

void THDTensor_(lerp)(THDTensor *r_, THDTensor *a, THDTensor *b, real weight) {
  THArgCheck(THDTensor_(nElement)(a) == THDTensor_(nElement)(b), 2,
             "sizes do not match");
  THDTensor_(resizeAs)(r_, a);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLerp, r_, a, b, weight),
    THDState::s_current_worker
  );
}

void THDTensor_(mean)(THDTensor *r_, THDTensor *t, int dimension, int keepdim) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t), 2,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMean, r_, t, dimension, keepdim),
    THDState::s_current_worker
  );

  if (!keepdim) {
    THDTensor_(_squeeze1d)(r_, r_, dimension);
  }
}

void THDTensor_(std)(THDTensor *r_, THDTensor *t, int dimension, int biased, int keepdim) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t), 3,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorStd, r_, t, dimension, biased, keepdim),
    THDState::s_current_worker
  );

  if (!keepdim) {
    THDTensor_(_squeeze1d)(r_, r_, dimension);
  }
}

void THDTensor_(var)(THDTensor *r_, THDTensor *t, int dimension, int biased, int keepdim) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t), 3,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorVar, r_, t, dimension, biased, keepdim),
    THDState::s_current_worker
  );

  if (!keepdim) {
    THDTensor_(_squeeze1d)(r_, r_, dimension);
  }
}

void THDTensor_(norm)(THDTensor *r_, THDTensor *t, real value, int dimension, int keepdim) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(t), 3,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THLongStorage *dim = THDTensor_(newSizeOf)(t);
  THLongStorage_set(dim, dimension, 1);
  THDTensor_(resize)(r_, dim, NULL);
  THLongStorage_free(dim);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorNorm, r_, t, dimension, keepdim, value),
    THDState::s_current_worker
  );

  if (!keepdim) {
    THDTensor_(_squeeze1d)(r_, r_, dimension);
  }
}

accreal THDTensor_(normall)(THDTensor *tensor, real value) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorNormall, tensor, value),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

void THDTensor_(renorm)(THDTensor *res, THDTensor *src, real value,
                        int dimension, real maxnorm) {
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(src), 3,
             "invalid dimension %d", dimension + TH_INDEX_BASE);
  THArgCheck(value > 0, 2, "non-positive-norm not supported");
  THArgCheck(THDTensor_(nDimension)(src) > 1, 1,
             "need at least 2 dimensions, got %d dimensions",
             THDTensor_(nDimension)(src));

  THDTensor_(resizeAs)(res, src);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRenorm, res, src, dimension, value, maxnorm),
    THDState::s_current_worker
  );
}

accreal THDTensor_(dist)(THDTensor *tensor, THDTensor *src, real value) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorDist, tensor, src, value),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

accreal THDTensor_(meanall)(THDTensor *tensor) {
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorMeanall, tensor),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

accreal THDTensor_(varall)(THDTensor *tensor, int biased) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorVarall, tensor, biased),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

accreal THDTensor_(stdall)(THDTensor *tensor, int biased) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorStdall, tensor, biased),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<accreal>(THDState::s_current_worker);
}

void THDTensor_(linspace)(THDTensor *r_, real a, real b, int64_t n) {
  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");

  if (THDTensor_(nElement)(r_) != n) {
    THDTensor_(resize1d)(r_, n);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLinspace, r_, n, a, b),
    THDState::s_current_worker
  );
}

void THDTensor_(logspace)(THDTensor *r_, real a, real b, int64_t n) {
  THArgCheck(n > 1 || (n == 1 && (a == b)), 3, "invalid number of points");

  if (THDTensor_(nElement)(r_) != n) {
    THDTensor_(resize1d)(r_, n);
  }

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLogspace, r_, n, a, b),
    THDState::s_current_worker
  );
}

void THDTensor_(rand)(THDTensor *r_, THDGenerator *_generator,
                      THLongStorage *size) {
  THDTensor_(resize)(r_, size, NULL);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRand, r_, _generator, size),
    THDState::s_current_worker
  );
}

void THDTensor_(randn)(THDTensor *r_, THDGenerator *_generator,
                       THLongStorage *size) {
  THDTensor_(resize)(r_, size, NULL);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorRandn, r_, _generator, size),
    THDState::s_current_worker
  );
}

void THDTensor_(histc)(THDTensor *hist, THDTensor *tensor, int64_t nbins,
                       real minvalue, real maxvalue) {
  THDTensor_(resize1d)(hist, nbins);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorHistc, hist, tensor, nbins, minvalue, maxvalue),
    THDState::s_current_worker
  );
}

void THDTensor_(bhistc)(THDTensor *hist, THDTensor *tensor, int64_t nbins,
                        real minvalue, real maxvalue) {
  THArgCheck(THDTensor_(nDimension)(tensor) < 3, 2,
             "invalid dimension %d, the input must be a 2d tensor",
             THDTensor_(nDimension)(tensor));

  int dimension = 1;
  THArgCheck(dimension >= 0 && dimension < THDTensor_(nDimension)(tensor), 2,
             "invalid dimension %d", dimension + TH_INDEX_BASE);

  THDTensor_(resize2d)(hist, tensor->size[0], nbins);

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorBhistc, hist, tensor, nbins, minvalue, maxvalue),
    THDState::s_current_worker
  );
}

#endif // defined(TH_REAL_IS_DOUBLE) || defined(TH_REAL_IS_FLOAT)

#if defined(TH_REAL_IS_BYTE)

int THDTensor_(logicalAnd)(THDTensor *tensor) {
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLogicalAnd, tensor),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<int>(THDState::s_current_worker);
}

int THDTensor_(logicalAny)(THDTensor *tensor) {
  THArgCheck(tensor->nDimension > 0, 1, "empty Tensor");

  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorLogicalAny, tensor),
    THDState::s_current_worker
  );

  return receiveValueFromWorker<int>(THDState::s_current_worker);
}

#endif // defined(TH_REAL_IS_BYTE)

THD_API void THDTensor_(clshift)(THDTensor *r_, THDTensor *t, THDTensor *src) {
  THError("clshift not implemented");
}

THD_API void THDTensor_(crshift)(THDTensor *r_, THDTensor *t, THDTensor *src) {
  THError("crshift not implemented");
}

THD_API void THDTensor_(cbitand)(THDTensor *r_, THDTensor *t, THDTensor *src) {
  THError("cbitand not implemented");
}

THD_API void THDTensor_(cbitor)(THDTensor *r_, THDTensor *t, THDTensor *src) {
  THError("cbitor not implemented");
}

THD_API void THDTensor_(cbitxor)(THDTensor *r_, THDTensor *t, THDTensor *src) {
  THError("cbitxor not implemented");
}

THD_API void THDTensor_(lshift)(THDTensor *r_, THDTensor *t, real value) {
  THError("lshift not implemented");
}

THD_API void THDTensor_(rshift)(THDTensor *r_, THDTensor *t, real value) {
  THError("rshift not implemented");
}

THD_API void THDTensor_(bitand)(THDTensor *r_, THDTensor *t, real value) {
  THError("bitand not implemented");
}

THD_API void THDTensor_(bitor)(THDTensor *r_, THDTensor *t, real value) {
  THError("bitor not implemented");
}

THD_API void THDTensor_(bitxor)(THDTensor *r_, THDTensor *t, real value) {
  THError("bitxor not implemented");
}

#endif // TH_GENERIC_FILE
