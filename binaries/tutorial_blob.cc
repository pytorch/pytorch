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

#include "caffe2/core/blob.h"
#include "caffe2/core/init.h"
#include "caffe2/core/tensor.h"
#include "caffe2/core/logging.h"

// We will be lazy and just use the whole namespace.
using namespace caffe2;


int main(int argc, char** argv) {
  caffe2::GlobalInit(&argc, &argv);
  caffe2::ShowLogInfoToStderr();

  LOG(INFO) <<
      "This script corresponds to the Blob part of the Caffe2 C++ "
      "tutorial.";

  LOG(INFO) << "Let's create a blob myblob.";

  Blob myblob;

  LOG(INFO) << "Let's set it to int and set the value to 10.";

  int* myint = myblob.GetMutable<int>();
  *myint = 10;

  LOG(INFO)
      << "Is the blob type int? "
      << myblob.IsType<int>();

  LOG(INFO)
      << "Is the blob type float? "
      << myblob.IsType<float>();
               
  const int& myint_const = myblob.Get<int>();
  LOG(INFO)
      << "The value of the int number stored in the blob is: "
      << myint_const;

  LOG(INFO)
      << "Let's try to get a float pointer. This will trigger an exception.";

  try {
    const float& myfloat = myblob.Get<float>();
    LOG(FATAL) << "This line should never happen.";
  } catch (std::exception& e) {
    LOG(INFO)
        << "As expected, we got an exception. Its content says: "
        << e.what();
  }

  LOG(INFO) <<
      "However, we can change the content type (and destroy the old "
      "content) by calling GetMutable. Let's change it to double.";

  double* mydouble = myblob.GetMutable<double>();
  *mydouble = 3.14;

  LOG(INFO) << "The new content is: " << myblob.Get<double>();

  LOG(INFO) <<
      "If we have a pre-created object, we can use Reset() to transfer the "
      "object to a blob.";

  std::string* pvec = new std::string();
  myblob.Reset(pvec); // no need to release pvec, myblob takes ownership.
  
  LOG(INFO) << "Is the blob now of type string? "
            << myblob.IsType<std::string>();

  LOG(INFO) << "This concludes the blob tutorial.";
  return 0;
}
