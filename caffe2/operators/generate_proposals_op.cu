#include <cub/cub.cuh>
#include "caffe2/core/context.h"
#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/generate_proposals_op.h"
#include "caffe2/operators/generate_proposals_op_util_boxes.h" // BBOX_XFORM_CLIP_DEFAULT
#include "caffe2/operators/generate_proposals_op_util_nms.h"
#include "caffe2/operators/generate_proposals_op_util_nms_gpu.h"

namespace caffe2 {
namespace {
__global__ void GeneratePreNMSUprightBoxesKernel(
    const int* d_sorted_scores_keys,
    const int nboxes_to_generate,
    const float* d_bbox_deltas,
    const float4* d_anchors,
    const int H,
    const int W,
    const int K, // K = H*W
    const int A,
    const int KA, // KA = K*A
    const float feat_stride,
    const float min_size,
    const float* d_img_info_vec,
    const int num_images,
    const float bbox_xform_clip,
    const bool correct_transform,
    float4* d_out_boxes,
    const int prenms_nboxes, // leading dimension of out_boxes
    float* d_inout_scores,
    char* d_boxes_keep_flags) {
  CUDA_2D_KERNEL_LOOP(ibox, nboxes_to_generate, image_index, num_images) {
    // box_conv_index : # of the same box, but indexed in
    // the scores from the conv layer, of shape (A,H,W)
    // the num_images dimension was already removed
    // box_conv_index = a*K + h*W + w
    const int box_conv_index = d_sorted_scores_keys[image_index * KA + ibox];

    // We want to decompose box_conv_index in (a,h,w)
    // such as box_conv_index = a*K + h*W + w
    // (avoiding modulos in the process)
    int remaining = box_conv_index;
    const int dA = K; // stride of A
    const int a = remaining / dA;
    remaining -= a * dA;
    const int dH = W; // stride of H
    const int h = remaining / dH;
    remaining -= h * dH;
    const int w = remaining; // dW = 1

    // Loading the anchor a
    // float is a struct with float x,y,z,w
    const float4 anchor = d_anchors[a];
    // x1,y1,x2,y2 :coordinates of anchor a, shifted for position (h,w)
    const float shift_w = feat_stride * w;
    float x1 = shift_w + anchor.x;
    float x2 = shift_w + anchor.z;
    const float shift_h = feat_stride * h;
    float y1 = shift_h + anchor.y;
    float y2 = shift_h + anchor.w;

    // TODO use fast math when possible

    // Deltas for that box
    // Deltas of shape (num_images,4*A,K)
    // We're going to compute 4 scattered reads
    // better than the alternative, ie transposing the complete deltas
    // array first
    int deltas_idx = image_index * (KA * 4) + a * 4 * K + h * W + w;
    const float dx = d_bbox_deltas[deltas_idx];
    // Stride of K between each dimension
    deltas_idx += K;
    const float dy = d_bbox_deltas[deltas_idx];
    deltas_idx += K;
    float dw = d_bbox_deltas[deltas_idx];
    deltas_idx += K;
    float dh = d_bbox_deltas[deltas_idx];

    // Upper bound on dw,dh
    dw = fmin(dw, bbox_xform_clip);
    dh = fmin(dh, bbox_xform_clip);

    // Applying the deltas
    float width = x2 - x1 + 1.0f;
    const float ctr_x = x1 + 0.5f * width;
    const float pred_ctr_x = ctr_x + width * dx; // TODO fuse madd
    const float pred_w = width * expf(dw);
    x1 = pred_ctr_x - 0.5f * pred_w;
    x2 = pred_ctr_x + 0.5f * pred_w;

    float height = y2 - y1 + 1.0f;
    const float ctr_y = y1 + 0.5f * height;
    const float pred_ctr_y = ctr_y + height * dy;
    const float pred_h = height * expf(dh);
    y1 = pred_ctr_y - 0.5f * pred_h;
    y2 = pred_ctr_y + 0.5f * pred_h;

    if (correct_transform) {
      x2 -= 1.0f;
      y2 -= 1.0f;
    }

    // Clipping box to image
    const float img_height = d_img_info_vec[3 * image_index + 0];
    const float img_width = d_img_info_vec[3 * image_index + 1];
    const float min_size_scaled =
        min_size * d_img_info_vec[3 * image_index + 2];
    x1 = fmax(fmin(x1, img_width - 1.0f), 0.0f);
    y1 = fmax(fmin(y1, img_height - 1.0f), 0.0f);
    x2 = fmax(fmin(x2, img_width - 1.0f), 0.0f);
    y2 = fmax(fmin(y2, img_height - 1.0f), 0.0f);

    // Filter boxes
    // Removing boxes with one dim < min_size
    // (center of box is in image, because of previous step)
    width = x2 - x1 + 1.0f; // may have changed
    height = y2 - y1 + 1.0f;
    bool keep_box = fmin(width, height) >= min_size_scaled;

    // We are not deleting the box right now even if !keep_box
    // we want to keep the relative order of the elements stable
    // we'll do it in such a way later
    // d_boxes_keep_flags size: (num_images,prenms_nboxes)
    // d_out_boxes size: (num_images,prenms_nboxes)
    const int out_index = image_index * prenms_nboxes + ibox;
    d_boxes_keep_flags[out_index] = keep_box;
    d_out_boxes[out_index] = {x1, y1, x2, y2};

    // d_inout_scores size: (num_images,KA)
    if (!keep_box)
      d_inout_scores[image_index * KA + ibox] = FLT_MIN; // for NMS
  }
}

__global__ void WriteOutput(
    const float4* d_image_boxes,
    const float* d_image_scores,
    const int* d_image_boxes_keep_list,
    const int nboxes,
    const int image_index,
    float* d_image_out_rois,
    float* d_image_out_rois_probs) {
  CUDA_1D_KERNEL_LOOP(i, nboxes) {
    const int ibox = d_image_boxes_keep_list[i];
    const float4 box = d_image_boxes[ibox];
    const float score = d_image_scores[ibox];
    // Scattered memory accesses
    // postnms_nboxes is small anyway
    d_image_out_rois_probs[i] = score;
    const int base_idx = 5 * i;
    d_image_out_rois[base_idx + 0] = image_index;
    d_image_out_rois[base_idx + 1] = box.x;
    d_image_out_rois[base_idx + 2] = box.y;
    d_image_out_rois[base_idx + 3] = box.z;
    d_image_out_rois[base_idx + 4] = box.w;
  }
}

__global__ void InitializeDataKernel(
    const int num_images,
    const int KA,
    int* d_image_offsets,
    int* d_boxes_keys_iota) {
  CUDA_2D_KERNEL_LOOP(box_idx, KA, img_idx, num_images) {
    d_boxes_keys_iota[img_idx * KA + box_idx] = box_idx;

    // One 1D line sets the 1D data
    if (box_idx == 0) {
      d_image_offsets[img_idx] = KA * img_idx;
      // One thread sets the last+1 offset
      if (img_idx == 0)
        d_image_offsets[num_images] = KA * num_images;
    }
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

  CAFFE_ENFORCE_EQ(scores.ndim(), 4, scores.ndim());
  CAFFE_ENFORCE(scores.template IsType<float>(), scores.meta().name());

  const auto num_images = scores.dim(0);
  const auto A = scores.dim(1);
  const auto H = scores.dim(2);
  const auto W = scores.dim(3);
  const auto box_dim_conv = anchors.dim(1);

  CAFFE_ENFORCE(box_dim_conv == 4); // only upright boxes in GPU version for now

  constexpr int box_dim = 4;
  const int K = H * W;
  const int conv_layer_nboxes = K * A;
  // Getting data members ready

  // We'll sort the scores
  // we want to remember their original indexes,
  // ie their indexes in the tensor of shape (num_images,A,K)
  // from the conv layer
  // each row of d_conv_layer_indexes is at first initialized to 1..A*K
  dev_conv_layer_indexes_.Resize(num_images, conv_layer_nboxes);
  int* d_conv_layer_indexes =
      dev_conv_layer_indexes_.template mutable_data<int>();

  // d_image_offset[i] = i*K*A for i from 1 to num_images+1
  // Used by the segmented sort to only sort scores within one image
  dev_image_offset_.Resize(num_images + 1);
  int* d_image_offset = dev_image_offset_.template mutable_data<int>();

  // The following calls to CUB primitives do nothing
  // (because the first arg is nullptr)
  // except setting cub_*_temp_storage_bytes
  size_t cub_sort_temp_storage_bytes = 0;
  float* flt_ptr = nullptr;
  int* int_ptr = nullptr;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      nullptr,
      cub_sort_temp_storage_bytes,
      flt_ptr,
      flt_ptr,
      int_ptr,
      int_ptr,
      num_images * conv_layer_nboxes,
      num_images,
      int_ptr,
      int_ptr,
      0,
      8 * sizeof(float), // sort all bits
      context_.cuda_stream());

  // Allocate temporary storage for CUB
  dev_cub_sort_buffer_.Resize(cub_sort_temp_storage_bytes);
  void* d_cub_sort_temp_storage =
      dev_cub_sort_buffer_.template mutable_data<char>();

  size_t cub_select_temp_storage_bytes = 0;
  char* char_ptr = nullptr;
  cub::DeviceSelect::Flagged(
      nullptr,
      cub_select_temp_storage_bytes,
      flt_ptr,
      char_ptr,
      flt_ptr,
      int_ptr,
      K * A,
      context_.cuda_stream());

  // Allocate temporary storage for CUB
  dev_cub_select_buffer_.Resize(cub_select_temp_storage_bytes);
  void* d_cub_select_temp_storage =
      dev_cub_select_buffer_.template mutable_data<char>();

  // Initialize :
  // - each row of dev_conv_layer_indexes to 1..K*A
  // - each d_nboxes to 0
  // - d_image_offset[i] = K*A*i for i 1..num_images+1
  // 2D grid
  InitializeDataKernel<<<
      (CAFFE_GET_BLOCKS(A * K), num_images),
      CAFFE_CUDA_NUM_THREADS, // blockDim.y == 1
      0,
      context_.cuda_stream()>>>(
      num_images, conv_layer_nboxes, d_image_offset, d_conv_layer_indexes);

  // Sorting input scores
  dev_sorted_conv_layer_indexes_.Resize(num_images, conv_layer_nboxes);
  dev_sorted_scores_.Resize(num_images, conv_layer_nboxes);
  const float* d_in_scores = scores.data<float>();
  int* d_sorted_conv_layer_indexes =
      dev_sorted_conv_layer_indexes_.template mutable_data<int>();
  float* d_sorted_scores = dev_sorted_scores_.template mutable_data<float>();
  ;
  cub::DeviceSegmentedRadixSort::SortPairsDescending(
      d_cub_sort_temp_storage,
      cub_sort_temp_storage_bytes,
      d_in_scores,
      d_sorted_scores,
      d_conv_layer_indexes,
      d_sorted_conv_layer_indexes,
      num_images * conv_layer_nboxes,
      num_images,
      d_image_offset,
      d_image_offset + 1,
      0,
      8 * sizeof(float), // sort all bits
      context_.cuda_stream());

  // Keeping only the topN pre_nms
  const int nboxes_to_generate = std::min(conv_layer_nboxes, rpn_pre_nms_topN_);

  // Generating the boxes associated to the topN pre_nms scores
  dev_boxes_.Resize(num_images, box_dim * nboxes_to_generate);
  dev_boxes_keep_flags_.Resize(num_images, nboxes_to_generate);
  const float* d_bbox_deltas = bbox_deltas.data<float>();
  const float* d_anchors = anchors.data<float>();
  const float* d_im_info_vec = im_info_tensor.data<float>();
  float4* d_boxes =
      reinterpret_cast<float4*>(dev_boxes_.template mutable_data<float>());
  ;
  char* d_boxes_keep_flags =
      dev_boxes_keep_flags_.template mutable_data<char>();

  GeneratePreNMSUprightBoxesKernel<<<
      (CAFFE_GET_BLOCKS(nboxes_to_generate), num_images),
      CAFFE_CUDA_NUM_THREADS, // blockDim.y == 1
      0,
      context_.cuda_stream()>>>(
      d_sorted_conv_layer_indexes,
      nboxes_to_generate,
      d_bbox_deltas,
      reinterpret_cast<const float4*>(d_anchors),
      H,
      W,
      K,
      A,
      K * A,
      feat_stride_,
      rpn_min_size_,
      d_im_info_vec,
      num_images,
      utils::BBOX_XFORM_CLIP_DEFAULT,
      correct_transform_coords_,
      d_boxes,
      nboxes_to_generate,
      d_sorted_scores,
      d_boxes_keep_flags);
  const int nboxes_generated = nboxes_to_generate;
  dev_image_prenms_boxes_.Resize(box_dim * nboxes_generated);
  float4* d_image_prenms_boxes = reinterpret_cast<float4*>(
      dev_image_prenms_boxes_.template mutable_data<float>());
  dev_image_prenms_scores_.Resize(nboxes_generated);
  float* d_image_prenms_scores =
      dev_image_prenms_scores_.template mutable_data<float>();
  dev_image_boxes_keep_list_.Resize(nboxes_generated);
  int* d_image_boxes_keep_list =
      dev_image_boxes_keep_list_.template mutable_data<int>();

  const int max_postnms_nboxes = std::min(nboxes_generated, rpn_post_nms_topN_);
  dev_postnms_rois_.Resize(5 * num_images * max_postnms_nboxes);
  dev_postnms_rois_probs_.Resize(num_images * max_postnms_nboxes);
  float* d_postnms_rois = dev_postnms_rois_.template mutable_data<float>();
  float* d_postnms_rois_probs =
      dev_postnms_rois_probs_.template mutable_data<float>();

  dev_prenms_nboxes_.Resize(num_images);
  host_prenms_nboxes_.Resize(num_images);
  int* d_prenms_nboxes = dev_prenms_nboxes_.template mutable_data<int>();
  int* h_prenms_nboxes = host_prenms_nboxes_.template mutable_data<int>();

  int nrois_in_output = 0;
  for (int image_index = 0; image_index < num_images; ++image_index) {
    // Sub matrices for current image
    const float4* d_image_boxes = &d_boxes[image_index * nboxes_generated];
    const float* d_image_sorted_scores = &d_sorted_scores[image_index * K * A];
    char* d_image_boxes_keep_flags =
        &d_boxes_keep_flags[image_index * nboxes_generated];

    float* d_image_postnms_rois = &d_postnms_rois[5 * nrois_in_output];
    float* d_image_postnms_rois_probs = &d_postnms_rois_probs[nrois_in_output];

    // Moving valid boxes (ie the ones with d_boxes_keep_flags[ibox] == true)
    // to the output tensors

    cub::DeviceSelect::Flagged(
        d_cub_select_temp_storage,
        cub_select_temp_storage_bytes,
        d_image_boxes,
        d_image_boxes_keep_flags,
        d_image_prenms_boxes,
        d_prenms_nboxes,
        nboxes_generated,
        context_.cuda_stream());

    cub::DeviceSelect::Flagged(
        d_cub_select_temp_storage,
        cub_select_temp_storage_bytes,
        d_image_sorted_scores,
        d_image_boxes_keep_flags,
        d_image_prenms_scores,
        d_prenms_nboxes,
        nboxes_generated,
        context_.cuda_stream());

    host_prenms_nboxes_.CopyFrom(dev_prenms_nboxes_);

    // We know prenms_boxes <= topN_prenms, because nboxes_generated <=
    // topN_prenms Calling NMS on the generated boxes
    const int prenms_nboxes = *h_prenms_nboxes;
    int nkeep;
    utils::nms_gpu_upright(
        reinterpret_cast<const float*>(d_image_prenms_boxes),
        prenms_nboxes,
        rpn_nms_thresh_,
        d_image_boxes_keep_list,
        &nkeep,
        dev_nms_mask_,
        host_nms_mask_,
        &context_);

    // All operations done after previous sort were keeping the relative order
    // of the elements the elements are still sorted keep topN <=> truncate the
    // array
    const int postnms_nboxes = std::min(nkeep, rpn_post_nms_topN_);

    // Moving the out boxes to the output tensors,
    // adding the image_index dimension on the fly
    WriteOutput<<<
        CAFFE_GET_BLOCKS(postnms_nboxes),
        CAFFE_CUDA_NUM_THREADS,
        0,
        context_.cuda_stream()>>>(
        d_image_prenms_boxes,
        d_image_prenms_scores,
        d_image_boxes_keep_list,
        postnms_nboxes,
        image_index,
        d_image_postnms_rois,
        d_image_postnms_rois_probs);

    nrois_in_output += postnms_nboxes;
  }

  // Using a buffer because we cannot call ShrinkTo
  out_rois->Resize(nrois_in_output, 5);
  out_rois_probs->Resize(nrois_in_output);
  float* d_out_rois = out_rois->template mutable_data<float>();
  float* d_out_rois_probs = out_rois_probs->template mutable_data<float>();

  CUDA_CHECK(cudaMemcpyAsync(
      d_out_rois,
      d_postnms_rois,
      nrois_in_output * 5 * sizeof(float),
      cudaMemcpyDeviceToDevice,
      context_.cuda_stream()));
  CUDA_CHECK(cudaMemcpyAsync(
      d_out_rois_probs,
      d_postnms_rois_probs,
      nrois_in_output * sizeof(float),
      cudaMemcpyDeviceToDevice,
      context_.cuda_stream()));

  return true;
}

REGISTER_CUDA_OPERATOR(GenerateProposals, GenerateProposalsOp<CUDAContext>);
} // namespace caffe2
