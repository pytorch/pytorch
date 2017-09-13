// Copyright 2004-present Facebook. All Rights Reserved.

#include "caffe2/proto/caffe2.pb.h"

namespace caffe2 {
void testOpenGL();
void compareModelsForOpenGL(std::string name,
                            const NetDef& initNet,
                            NetDef predictNet,
                            int width,
                            int height,
                            int channel,
                            std::string input_type,
                            std::string input_order);

void compareBatchedToTiledModels(std::string name,
                                 const NetDef& initNet,
                                 NetDef predictNet,
                                 int width,
                                 int height,
                                 int channel,
                                 std::string input_type,
                                 std::string input_order);

int runModelBenchmarks(caffe2::NetDef& init_net,
                       caffe2::NetDef& predict_net,
                       int warm_up_runs,
                       int main_runs,
                       int channel,
                       int height,
                       int width,
                       std::string input_type,
                       std::string input_order,
                       std::string engine,
                       bool run_individual = false,
                       bool use_texture_input = false,
                       bool use_tiling = false,
                       bool run_fusion = true);

typedef enum {
  AveragePool,
  MaxPool,
  Conv,
  ConvTranspose,
  ConvPRelu,
  ConvTransposePRelu,
  ConvRelu,
  ConvTransposeRelu
} PoolOp;

void testOpenGLConv(int N,
                    int C,
                    int H,
                    int W,
                    int K, // output_channels
                    int kernel_h,
                    int kernel_w,
                    int pad,
                    int stride,
                    PoolOp poolOp,
                    float error,
                    bool random_input = true,
                    int input_batch_size = 1,
                    int output_batch_size = 1,
                    int input_tile_x = 1,
                    int input_tile_y = 1,
                    bool tiling = false);

} // namespace caffe2
