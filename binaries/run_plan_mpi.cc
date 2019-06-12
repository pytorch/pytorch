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

#include <mpi.h>

#include "c10/util/Flags.h"
#include "caffe2/core/init.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/proto/caffe2_pb.h"
#include "caffe2/utils/proto_utils.h"

C10_DEFINE_string(plan, "", "The given path to the plan protobuffer.");

int main(int argc, char** argv) {
  c10::SetUsageMessage("Runs a caffe2 plan that has MPI operators in it.");
  int mpi_ret;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_ret);
  if (mpi_ret != MPI_THREAD_MULTIPLE &&
      mpi_ret != MPI_THREAD_SERIALIZED) {
    std::cerr << "Caffe2 MPI requires the underlying MPI to support the "
                 "MPI_THREAD_SERIALIZED or MPI_THREAD_MULTIPLE mode.\n";
    return 1;
  }
  caffe2::GlobalInit(&argc, &argv);
  LOG(INFO) << "Loading plan: " << FLAGS_plan;
  caffe2::PlanDef plan_def;
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_plan, &plan_def));
  std::unique_ptr<caffe2::Workspace> workspace(new caffe2::Workspace());
  workspace->RunPlan(plan_def);

  // This is to allow us to use memory leak checks.
  caffe2::ShutdownProtobufLibrary();
  MPI_Finalize();
  return 0;
}
