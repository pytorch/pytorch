#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/generate_proposals_op.h"
#include "caffe2/utils/math.h"

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace caffe2 {

namespace {
const float __device__ BBOX_XFORM_CLIP_DEFAULT =
    4.135166556742356; // log(1000.0 / 16.0);
const float __device__ INVALID_SCORE = -123456.0f;

struct ImInfo {
  float height;
  float width;
  float scale;
};

__host__ __device__ inline float clamp(float x, float a, float b) {
  return x < b ? x > a ? x : a : b;
}
__host__ __device__ inline float max(float a, float b) {
  return a < b ? b : a;
}
__host__ __device__ inline float min(float a, float b) {
  return a > b ? b : a;
}

struct BBox {
  float coords[4];
  __host__ __device__ inline float getWidth() const {
    return coords[2] - coords[0];
  }
  __host__ __device__ inline float getHeight() const {
    return coords[3] - coords[1];
  }
  __host__ __device__ inline float area() const {
    return getWidth() * getHeight();
  }
  __host__ __device__ inline float left() const {
    return coords[0];
  }
  __host__ __device__ inline float top() const {
    return coords[1];
  }
  __host__ __device__ inline BBox& clampToWH(float w, float h) {
    coords[0] = clamp(coords[0], 0, w);
    coords[1] = clamp(coords[1], 0, h);
    coords[2] = clamp(coords[2], 0, w);
    coords[3] = clamp(coords[3], 0, h);
    return *this;
  }
  __device__ BBox(float cx, float cy, float w, float h)
      : coords{cx - .5f * w, cy - .5f * h, cx + .5f * w, cy + .5f * h} {}
  __host__ __device__ float operator[](unsigned idx) const {
    return coords[idx];
  }
  // Returns intersection/union area ration with other box
  __host__ __device__ float overlapRatio(const BBox& other) const {
    float x1 = max(coords[0], other[0]);
    float y1 = max(coords[1], other[1]);
    float x2 = min(coords[2], other[2]);
    float y2 = min(coords[3], other[3]);
    float width = max(x2 - x1, 0.f);
    float height = max(y2 - y1, 0.f);
    float interS = width * height;
    return interS / (area() + other.area() - interS);
  }
};

struct TransformConfig {
  unsigned W;
  unsigned H;
  unsigned A;
  float stride;
  float minSize;
  BBox* bboxOut;
  float* scoresOut;
};

__global__ void transformBBoxes(
    TransformConfig c,
    const BBox* anchors,
    const ImInfo* info,
    const float* delta,
    const float* scores) {
  const unsigned WH = c.W * c.H;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= WH * c.A)
    return;
  unsigned xyOffs = idx / c.A;
  unsigned a = idx % c.A;
  int x = xyOffs % c.W;
  int y = xyOffs / c.W;

  float wid = 1.0f + anchors[a].getWidth();
  float hei = 1.0f + anchors[a].getHeight();
  float ctr_x = anchors[a].left() + x * c.stride + .5 * wid;
  float ctr_y = anchors[a].top() + y * c.stride + .5 * hei;
  unsigned offs = xyOffs + 4 * WH * a;
  float dx = delta[offs];
  float dy = delta[offs + WH];
  float dw = min(delta[offs + 2 * WH], BBOX_XFORM_CLIP_DEFAULT);
  float dh = min(delta[offs + 3 * WH], BBOX_XFORM_CLIP_DEFAULT);
  float pred_ctr_x = dx * wid + ctr_x;
  float pred_ctr_y = dy * hei + ctr_y;
  float pred_w = exp(dw) * wid;
  float pred_h = exp(dh) * hei;
  BBox box(pred_ctr_x, pred_ctr_y, pred_w, pred_h);
  box.clampToWH(info->width - 1, info->height - 1);
  c.bboxOut[idx] = box;
  c.scoresOut[idx] = min(box.getWidth(), box.getHeight()) > c.minSize
      ? scores[xyOffs + WH * a]
      : INVALID_SCORE;
}

__device__ void setOutputBox(float* ptr, float imgNo, const BBox& box) {
  ptr[0] = imgNo;
  ptr[1] = box[0];
  ptr[2] = box[1];
  ptr[3] = box[2];
  ptr[4] = box[3];
}

struct NMSConfig {
  int* indexes;
  const BBox* bboxes;
  const float* scores;
  float* outputBoxes;
  float* outputScores;
  int* offsets; // 0 - input offset, 1 - output offset
  unsigned inputSize;
  unsigned outputSize;
  float threshold;
  float imgNo;
};

// A single thread kernel
__global__ void advanceInputOffset(
    int* pOffset,
    const int* indexes,
    const float* scores,
    unsigned size) {
  int offset = *pOffset + 1;
  while (offset < size &&
         (indexes[offset] < 0 || scores[offset] == INVALID_SCORE))
    offset++;
  *pOffset = offset;
}

__global__ void NMSStep(NMSConfig config) {
  int offset = config.offsets[0];
  auto& outOffset = config.offsets[1];
  if (offset >= config.inputSize || outOffset >= config.outputSize)
    return;
  unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx + offset >= config.inputSize)
    return;
  const BBox& selectedBox = config.bboxes[config.indexes[offset]];
  if (idx == 0) {
    // Thread 0 copies best prediction out
    config.outputScores[outOffset] = config.scores[offset];
    setOutputBox(config.outputBoxes + 5 * outOffset, config.imgNo, selectedBox);
    outOffset++;
    return;
  }
  auto& boxIdx = config.indexes[idx + offset];
  if (boxIdx < 0)
    return;
  if (config.scores[idx + offset] == INVALID_SCORE) {
    boxIdx = -1;
  } else if (
      selectedBox.overlapRatio(config.bboxes[boxIdx]) > config.threshold) {
    boxIdx = -1;
  }
}
} // namespace

template <>
bool GenerateProposalsOp<CUDAContext>::RunOnDevice() {
  const auto& scores = Input(0);
  const auto& bbox_deltas = Input(1);
  const auto& im_info_tensor = Input(2);
  const auto& anchors = Input(3);
  auto* out_rois = Output(0);
  auto* out_rois_probs = Output(1);

  CAFFE_ENFORCE_EQ(
      scores.ndim(), 4, "Scores must be in form  (img_count, A, H, W)");
  CAFFE_ENFORCE(scores.IsType<float>(), scores.meta().name());
  CAFFE_ENFORCE_EQ(anchors.ndim(), 2, "Anchors must be in form (A, D)");
  CAFFE_ENFORCE_EQ(anchors.dim(1), 4, "Anchors must be in form (A, 4)");

  offsets_.Resize(2);
  math::Set<int, CUDAContext>(1, -1, offsets_.mutable_data<int>(), &context_);
  math::Set<int, CUDAContext>(
      1, 0, offsets_.mutable_data<int>() + 1, &context_);

  TransformConfig config;
  const auto num_images = scores.dim(0);
  config.A = scores.dim(1);
  config.H = scores.dim(2);
  config.W = scores.dim(3);
  config.stride = args_.feat_stride_;
  config.minSize = args_.rpn_min_size_;
  const auto K = config.W * config.H;
  const auto totalBoxNum = config.A * K;
  static_assert(
      sizeof(BBox) == sizeof(float) * 4, "BoundingBox is just 4 floats");
  transformedBBoxes_.Resize(totalBoxNum * 4);
  transposedScores_.Resize(totalBoxNum);
  indicies_.Resize(totalBoxNum);

  auto devIndSize = std::min(totalBoxNum, (unsigned)args_.rpn_post_nms_topN_);

  config.bboxOut =
      reinterpret_cast<BBox*>(transformedBBoxes_.mutable_data<float>());
  config.scoresOut = transposedScores_.mutable_data<float>();
  transformBBoxes<<<
      CAFFE_GET_BLOCKS(totalBoxNum),
      CAFFE_CUDA_NUM_THREADS,
      0,
      context_.cuda_stream()>>>(
      config,
      reinterpret_cast<const BBox*>(anchors.data<float>()),
      reinterpret_cast<const ImInfo*>(im_info_tensor.data<float>()),
      bbox_deltas.data<float>(),
      scores.data<float>());

  // Sort weights and indicies on GPU
  auto* indDevPtr = indicies_.mutable_data<int>();
  auto* scoresDevPtr = transposedScores_.mutable_data<float>();
  thrust::sequence(
      thrust::cuda::par.on(context_.cuda_stream()),
      indDevPtr,
      indDevPtr + totalBoxNum);

  thrust::sort_by_key(
      thrust::cuda::par.on(context_.cuda_stream()),
      scoresDevPtr,
      scoresDevPtr + totalBoxNum,
      indDevPtr,
      thrust::greater<float>());
  // Copy output, suppressing overlapping boxes
  out_rois->Resize(devIndSize, 5);
  out_rois_probs->Resize(devIndSize);

  NMSConfig nmsConfig;
  nmsConfig.imgNo = 0;
  nmsConfig.offsets = offsets_.mutable_data<int>();
  nmsConfig.inputSize =
      std::min((unsigned)args_.rpn_pre_nms_topN_, totalBoxNum);
  nmsConfig.bboxes = config.bboxOut;
  nmsConfig.scores = scoresDevPtr;
  nmsConfig.indexes = indDevPtr;
  nmsConfig.outputBoxes = out_rois->mutable_data<float>();
  nmsConfig.outputScores = out_rois_probs->mutable_data<float>();
  nmsConfig.outputSize = devIndSize;
  nmsConfig.threshold = args_.rpn_nms_thresh_;
  for (unsigned cnt(0); cnt < devIndSize; ++cnt) {
    advanceInputOffset<<<1, 1, 0, context_.cuda_stream()>>>(
        nmsConfig.offsets,
        nmsConfig.indexes,
        nmsConfig.scores,
        nmsConfig.inputSize);
    NMSStep<<<
        CAFFE_GET_BLOCKS(nmsConfig.inputSize),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(nmsConfig);
  }
  int proposedBoxes = -1;
  CUDA_ENFORCE(cudaMemcpyAsync(
      &proposedBoxes,
      nmsConfig.offsets + 1,
      sizeof(int),
      cudaMemcpyDeviceToHost,
      context_.cuda_stream()));
  context_.FinishDeviceComputation();
  out_rois_probs->Resize(proposedBoxes);
  out_rois->Resize(proposedBoxes, 5);

  return true;
}

REGISTER_CUDA_OPERATOR(GenerateProposals, GenerateProposalsOp<CUDAContext>);
} // namespace caffe2
