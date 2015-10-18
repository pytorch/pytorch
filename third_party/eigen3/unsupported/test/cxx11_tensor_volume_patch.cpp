#include "main.h"

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;

static void test_single_voxel_patch()
{
  Tensor<float, 5> tensor(4,2,3,5,7);
  tensor.setRandom();
  Tensor<float, 5, RowMajor> tensor_row_major = tensor.swap_layout();

  Tensor<float, 6> single_voxel_patch;
  single_voxel_patch = tensor.extract_volume_patches(1, 1, 1);
  VERIFY_IS_EQUAL(single_voxel_patch.dimension(0), 4);
  VERIFY_IS_EQUAL(single_voxel_patch.dimension(1), 1);
  VERIFY_IS_EQUAL(single_voxel_patch.dimension(2), 1);
  VERIFY_IS_EQUAL(single_voxel_patch.dimension(3), 1);
  VERIFY_IS_EQUAL(single_voxel_patch.dimension(4), 2 * 3 * 5);
  VERIFY_IS_EQUAL(single_voxel_patch.dimension(5), 7);

  Tensor<float, 6, RowMajor> single_voxel_patch_row_major;
  single_voxel_patch_row_major = tensor_row_major.extract_volume_patches(1, 1, 1);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(0), 7);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(1), 2 * 3 * 5);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(2), 1);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(3), 1);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(4), 1);
  VERIFY_IS_EQUAL(single_voxel_patch_row_major.dimension(5), 4);

  for (int i = 0; i < tensor.size(); ++i) {
    VERIFY_IS_EQUAL(tensor.data()[i], single_voxel_patch.data()[i]);
    VERIFY_IS_EQUAL(tensor_row_major.data()[i], single_voxel_patch_row_major.data()[i]);
    VERIFY_IS_EQUAL(tensor.data()[i], tensor_row_major.data()[i]);
  }
}


static void test_entire_volume_patch()
{
  const int depth = 4;
  const int patch_z = 2;
  const int patch_y = 3;
  const int patch_x = 5;
  const int batch = 7;

  Tensor<float, 5> tensor(depth, patch_z, patch_y, patch_x, batch);
  tensor.setRandom();
  Tensor<float, 5, RowMajor> tensor_row_major = tensor.swap_layout();

  Tensor<float, 6> entire_volume_patch;
  entire_volume_patch = tensor.extract_volume_patches(patch_z, patch_y, patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch.dimension(0), depth);
  VERIFY_IS_EQUAL(entire_volume_patch.dimension(1), patch_z);
  VERIFY_IS_EQUAL(entire_volume_patch.dimension(2), patch_y);
  VERIFY_IS_EQUAL(entire_volume_patch.dimension(3), patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch.dimension(4), patch_z * patch_y * patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch.dimension(5), batch);

  Tensor<float, 6, RowMajor> entire_volume_patch_row_major;
  entire_volume_patch_row_major = tensor_row_major.extract_volume_patches(patch_z, patch_y, patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(0), batch);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(1), patch_z * patch_y * patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(2), patch_x);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(3), patch_y);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(4), patch_z);
  VERIFY_IS_EQUAL(entire_volume_patch_row_major.dimension(5), depth);

  const int dz = patch_z - 1;
  const int dy = patch_y - 1;
  const int dx = patch_x - 1;

  const int forward_pad_z = dz - dz / 2;
  const int forward_pad_y = dy - dy / 2;
  const int forward_pad_x = dx - dx / 2;

  for (int pz = 0; pz < patch_z; pz++) {
    for (int py = 0; py < patch_y; py++) {
      for (int px = 0; px < patch_x; px++) {
        const int patchId = pz + patch_z * (py + px * patch_y);
        for (int z = 0; z < patch_z; z++) {
          for (int y = 0; y < patch_y; y++) {
            for (int x = 0; x < patch_x; x++) {
              for (int b = 0; b < batch; b++) {
                for (int d = 0; d < depth; d++) {
                  float expected = 0.0f;
                  float expected_row_major = 0.0f;
                  const int eff_z = z - forward_pad_z + pz;
                  const int eff_y = y - forward_pad_y + py;
                  const int eff_x = x - forward_pad_x + px;
                  if (eff_z >= 0 && eff_y >= 0 && eff_x >= 0 &&
                      eff_z < patch_z && eff_y < patch_y && eff_x < patch_x) {
                    expected = tensor(d, eff_z, eff_y, eff_x, b);
                    expected_row_major = tensor_row_major(b, eff_x, eff_y, eff_z, d);
                  }
                  VERIFY_IS_EQUAL(entire_volume_patch(d, z, y, x, patchId, b), expected);
                  VERIFY_IS_EQUAL(entire_volume_patch_row_major(b, patchId, x, y, z, d), expected_row_major);
                }
              }
            }
          }
        }
      }
    }
  }
}

void test_cxx11_tensor_volume_patch()
{
  CALL_SUBTEST(test_single_voxel_patch());
  CALL_SUBTEST(test_entire_volume_patch());
}
