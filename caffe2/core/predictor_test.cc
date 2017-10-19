/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <google/protobuf/text_format.h>
#include "caffe2/core/context.h"
#include "caffe2/core/operator.h"
#include "caffe2/core/predictor.h"
#include "caffe2/core/tensor.h"
#include "caffe2/utils/math.h"

#include <gtest/gtest.h>

namespace caffe2 {

namespace {

const char* predictSpec = R"DOC(
        name: "predict"
        type: "dag"
        external_input: "data"
        external_input: "W"
        external_input: "b"
        external_output: "y"
        op {
          input: "data"
          input: "W"
          input: "b"
          output: "y"
          type: "FC"
        }
)DOC";

const char* initSpec = R"DOC(
        name: "init"
        type: "dag"
        op {
          type: "ConstantFill"
          output: "W"
          arg {
            name: "shape"
            ints: 10
            ints: 4
          }
          arg {
            name: "value"
            f: 2.0
          }
        }
        op {
          type: "ConstantFill"
          output: "b"
          arg {
            name: "shape"
            ints: 10
          }
          arg {
            name: "value"
            f: 2.0
          }
        }

)DOC";

std::unique_ptr<Blob> randomTensor(
    const std::vector<TIndex>& dims,
    CPUContext* ctx) {
  auto blob = make_unique<Blob>();
  auto* t = blob->GetMutable<TensorCPU>();
  t->Resize(dims);
  math::RandUniform<float, CPUContext>(
      t->size(), -1.0, 1.0, t->template mutable_data<float>(), ctx);
  return blob;
}

NetDef parseNetDef(const std::string& value) {
  NetDef def;
  CAFFE_ENFORCE(
      google::protobuf::TextFormat::ParseFromString(value, &def),
      "Failed to parse NetDef with value: ",
      value);
  return def;
};
}

class PredictorTest : public testing::Test {
 public:
  void SetUp() override {
    DeviceOption op;
    op.set_random_seed(1701);
    ctx_ = caffe2::make_unique<CPUContext>(op);
    NetDef init, run;
    p_ = caffe2::make_unique<Predictor>(
        parseNetDef(initSpec), parseNetDef(predictSpec));
  }

  std::unique_ptr<CPUContext> ctx_;
  std::unique_ptr<Predictor> p_;
};

TEST_F(PredictorTest, SimpleBatchSized) {
  auto inputData = randomTensor({1, 4}, ctx_.get());
  Predictor::TensorVector input{inputData->template GetMutable<TensorCPU>()};
  Predictor::TensorVector output;
  p_->run(input, &output);
  EXPECT_EQ(output.size(), 1);
  EXPECT_TRUE(output.front()->dims().size() == 2);
  EXPECT_TRUE(output.front()->dim(0) == 1);
  EXPECT_TRUE(output.front()->dim(1) == 10);
  EXPECT_NEAR(output.front()->data<float>()[4], 0.1209, 1E-4);
}

TEST_F(PredictorTest, SimpleBatchSizedMapInput) {
  auto inputData = randomTensor({1, 4}, ctx_.get());
  Predictor::TensorMap input{
      {"data", inputData->template GetMutable<TensorCPU>()}};
  Predictor::TensorVector output;
  p_->run_map(input, &output);
  EXPECT_EQ(output.size(), 1);
  EXPECT_TRUE(output.front()->dims().size() == 2);
  EXPECT_TRUE(output.front()->dim(0) == 1);
  EXPECT_TRUE(output.front()->dim(1) == 10);
  EXPECT_NEAR(output.front()->data<float>()[4], 0.1209, 1E-4);
}
} // namespace caffe2
