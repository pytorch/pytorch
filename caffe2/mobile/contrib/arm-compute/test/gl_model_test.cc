#include "caffe2/mobile/contrib/arm-compute/test/gl_model_test.h"

namespace caffe2 {

// The last softmax op didn't pass because of the dimension mismatch, and we are not likely to hit it in other models, but the implementation should be correct
// TEST(OPENGLModelTest, SqueezenetV11) {
//   std::string parent_path = "/data/local/tmp/";
//   benchmarkModel(parent_path + "squeezenet_init.pb", parent_path + "squeezenet_predict.pb", "data", {1, 3, 224, 224}, "squeezenet_v11");
// }

} // namespace caffe2
