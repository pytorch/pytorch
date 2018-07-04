
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
                       bool run_individual    = false,
                       bool use_texture_input = false,
                       bool use_tiling        = false,
                       bool run_fusion        = true);
} // namespace caffe2
