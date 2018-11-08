#include "context_intel_fpga.h"

#include "caffe2/core/allocator.h"
#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/tensor.h"
using namespace std;

/*********************************
 * Intel FPGA Stratix 10 Speficic OpenCL implementation
 *********************************/
constexpr auto FPGA_ENGINE = "FPGA";
constexpr auto INTEL_DEVNAME = "Intel(R)";
constexpr auto NUM_QUEUES = 4 + 1;
constexpr auto FPGA_PROGRAM_PATH = "FPGA_MMM_AOCX";
constexpr auto min_matrix_size = 512;

namespace caffe2 {

const std::vector<std::string> kernel_names = {"loadA",
                                               "loadB",
                                               "store",
                                               "extra"};

FPGAContextSingleton::FPGAContextSingleton(const string& devname, int n_queues)
    : BaseSingletonContext() {
  engine = FPGA_ENGINE;
  auto platform_id = 0;
  auto platforms = std::vector<cl::Platform>();
  cl::Platform::get(&platforms);
  bool found = false;
  for (platform_id = 0; platform_id < platforms.size(); platform_id++) {
    std::string platName = platforms[platform_id].getInfo<CL_PLATFORM_VENDOR>();
    VLOG(1) << "Discovered " << platName;
    if (platName.find(devname) != string::npos) {
      found = true;
      break;
    }
  }
  if (!found) {
    CAFFE_THROW("Cannot find device that matches " + devname);
  }
  platform = platforms[platform_id];

  devices = std::vector<cl::Device>();
  platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
  const auto device_id = 0;
  if (devices.size() == 0 || device_id >= devices.size()) {
    CAFFE_THROW("Cannot find OpenCL compatible device.");
  }
  device = devices[device_id];

  auto devfullname = device.getInfo<CL_DEVICE_NAME>();
  auto devvendor = device.getInfo<CL_DEVICE_VENDOR>();
  auto devcomputeunits = device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
  auto devglobmem = device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>();
  auto devmaxalloc = device.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>();

  LOG(INFO) << "Looking for device " << devname << " and " << n_queues
            << " queues";
  LOG(INFO) << "Device Name: " << devfullname;
  LOG(INFO) << "Device Vendor: " << devvendor;
  LOG(INFO) << "Device Compute Units: " << devcomputeunits;
  LOG(INFO) << "Device Global Memory: " << devglobmem / (1024 * 1024) << " MB";
  LOG(INFO) << "Device Memory Allocation Size: " << devmaxalloc / (1024 * 1024)
            << " MB";

  context = cl::Context({device});
  for (int i = 0; i < n_queues; i++) {
    queues.push_back(cl::CommandQueue(context, device));
  }
  LoadProgram();
}

void FPGAContextSingleton::LoadProgram() {
  std::string AOCX_FILE = "/home/hyz/altera/mm/bin/mmm.aocx";
  auto fpga_path = getenv(FPGA_PROGRAM_PATH);
  if (fpga_path != nullptr) {
    AOCX_FILE = fpga_path;
  }
  LOG(INFO) << "Using program: " << AOCX_FILE;

  std::ifstream fp;
  fp.open(AOCX_FILE.c_str(), std::ifstream::in | std::ifstream::binary);
  CAFFE_ENFORCE(fp.is_open());
  fp.seekg(0, fp.end);
  int len = fp.tellg();
  fp.seekg(0, fp.beg);
  char* binary = new char[len];
  fp.read(binary, len);
  fp.close();

  cl::Program::Binaries matmul_prog;
  matmul_prog.push_back(std::make_pair(binary, len));

  program_catalog["matmul"] = cl::Program(context, {device}, matmul_prog);
  program_catalog["matmul"].build({device});

  auto log =
      program_catalog["matmul"].getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);

  LOG(INFO) << "Program compilation: " << log;

  for (int i = 0; i < kernel_names.size(); i++) {
    kernel_catalog[kernel_names[i]] =
        cl::Kernel(program_catalog["matmul"], kernel_names[i].c_str());
  }
  LOG(INFO) << "Loaded " << kernel_names.size() << " kernels";
}

FPGAContextSingleton& FPGAContextSingleton::getInstance() {
  static FPGAContextSingleton* instance;
  if (instance == nullptr) {
    instance = new FPGAContextSingleton(INTEL_DEVNAME, NUM_QUEUES);
  }
  return *instance;
}

// Create a cl::Buffer with the contents transposed
cl::Buffer*
FPGAContextSingleton::transposeBuffer(cl::Buffer* buff, int h, int w) {
  bfloat16* arr = readBuffer(buff, h, w);
  bfloat16* arrT = new bfloat16[h * w];
  assert(arr);
  assert(arrT);
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      arrT[j * h + i] = arr[i * w + j];
    }
  }

  auto output = OpenCLContext::New(h * w * sizeof(float)).get();
  writeBuffer(arrT, h, w, static_cast<cl::Buffer*>(output));
  delete[] arr;
  delete[] arrT;

  return static_cast<cl::Buffer*>(output);
}

void FPGAContextSingleton::copyBuffer(
    cl::Buffer* src,
    cl::Buffer* dst,
    size_t nbytes) {
  queues.back().enqueueCopyBuffer(*src, *dst, 0, 0, nbytes);
  queues.back().flush();
}

bfloat16* FPGAContextSingleton::readBuffer(cl::Buffer* buff, int h, int w) {
  bfloat16* tmp = new bfloat16[h * w];
  assert(tmp);
  queues.begin()->enqueueReadBuffer(
      *(cl::Buffer*)buff, true, 0, h * w * sizeof(bfloat16), tmp);
  return tmp;
}

std::mutex& FPGAContextSingleton::mutex() {
  static std::mutex m;
  return m;
}

void FPGAContextSingleton::printBuffer(
    cl::Buffer* buff,
    int h,
    int w,
    int ph,
    int pw) {
  bfloat16* tmp = readBuffer(buff, h, w);
  bfp_converter x;
  x.bfp[0] = 0;
  for (int i = 0; i < ph; i++) {
    for (int j = 0; j < pw; j++) {
      x.bfp[1] = tmp[i * w + j];
      cout << x.fp32 << " ";
    }
    cout << endl;
  }
  delete[] tmp;
}

void FPGAContextSingleton::writeBuffer(
    const bfloat16* data,
    const int h,
    const int w,
    cl::Buffer* buff) {
  queues.back().enqueueWriteBuffer(
      *buff, true, 0, h * w * sizeof(bfloat16), data);
}

cl::Buffer* FPGAContextSingleton::createIdentity(int n) {
  bfloat16* tmp = new bfloat16[n * n];
  memset(tmp, 0, sizeof(bfloat16) * n * n);
  for (int i = 0; i < n; i++) {
    tmp[i * n + i] = 0x3f80; // 1.0 in bfloat16
  }
  cl::Buffer* ret =
      static_cast<cl::Buffer*>(OpenCLContext::New(n * n * sizeof(float)).get());
  writeBuffer(tmp, n, n, ret);
  delete[] tmp;

  return ret;
}

// Allocates a buffer compatible with the tile size of the FPGA
// and copies the data into it
// Extends a matrix of A(ha,wa) to A(newha, newwa)
// where newha is extended from ha and
//       newwa is extended from wa
// If the input buffer should be interpreted as
// A(wa, ha) and the extended to A(newwa, newha)
// where newwa is extended from wa
//       newha is extended from ha
cl::Buffer* FPGAContextSingleton::copyToTileSizeBuffer(
    const float* A,
    const bool trans,
    const int ha,
    const int wa,
    int* newha,
    int* newwa) {
  *newha = min_matrix_size, *newwa = min_matrix_size;
  while (*newha < ha) {
    *newha *= 2;
  }
  while (*newwa < wa) {
    *newwa *= 2;
  }

  CAFFE_ENFORCE(ha != *newha || wa != *newwa);
  VLOG(1) << "copy to " << *newha << " " << *newwa;
  // Copy from host
  // Rearrange to bigger matrix
  // Copy to device
  bfloat16* tmp = readBuffer((cl::Buffer*)A, ha, wa);
  bfloat16* tmp_expanded = new bfloat16[*newha * *newwa];
  assert(tmp_expanded);
  memset(tmp_expanded, 0, *newha * *newwa * sizeof(bfloat16));

  if (!trans) {
    for (int i = 0; i < ha; i++) {
      memcpy(tmp_expanded + i * *newwa, tmp + i * wa, wa * sizeof(bfloat16));
    }
  } else {
    for (int i = 0; i < wa; i++) {
      memcpy(tmp_expanded + i * *newha, tmp + i * ha, ha * sizeof(bfloat16));
    }
  }

  auto memhdl = OpenCLContext::New(*newha * *newwa * sizeof(float));
  auto newA = static_cast<cl::Buffer*>(memhdl.get());
  VLOG(1) << "created new buffer " << *newha * *newwa;
  writeBuffer(tmp_expanded, *newwa, *newha, newA);

  delete[] tmp_expanded;
  delete[] tmp;

  return newA;
}

// Copy data from tile sized array into user specified size matrix, the rest
// of the data is discarded
void FPGAContextSingleton::copyFromTileSizeBuffer(
    cl::Buffer* src,
    int srch,
    int srcw,
    const float* dst,
    int dsth,
    int dstw) {
  // Copy from host
  // Rearrange to smaller matrix

  VLOG(1) << "copy from (" << srch << "," << srcw << ") to (" << dsth << ","
          << dstw << ")";

  bfloat16* tmp = readBuffer(src, srcw, dsth);
  bfloat16* tmp_towrite = new bfloat16[dstw * dsth];
  assert(tmp_towrite);
  for (int i = 0; i < dsth; i++) {
    memcpy(tmp_towrite + i * dstw, tmp + i * srcw, dstw * sizeof(bfloat16));
  }

  writeBuffer(tmp_towrite, dsth, dstw, (cl::Buffer*)dst);

  delete[] tmp_towrite;
  delete[] tmp;
}

/*
  C = A * op(B)    where A is of size (ha * wa) and op(B) is of size(wa, wb)
                   C will be of size (ha * wb)
  op can be nothing or a transposition
*/
bool FPGAContextSingleton::MatMul(
    const bool TransA,
    const bool TransB,
    const float* A,
    const float* dA,
    const float* B,
    float* C,
    const int ha,
    const int wa,
    const int wb,
    const bool reluA,
    const bool revreluA) {
  CAFFE_ENFORCE_EQ(TransA, false); // Not supported yet

  // either relu or revrelu or none
  CAFFE_ENFORCE_EQ(reluA && revreluA, false);
  VLOG(1) << ha << " " << wa << " " << wb << " " << TransA << " " << TransB;

  // LOG(INFO) << "A";
  // printBuffer((cl::Buffer*)A, ha, wa, ha, wa);
  // if (revreluA) {
  //   LOG(INFO) << "dA";
  //   printBuffer((cl::Buffer*)dA, ha, wa, ha, wa);
  // }
  // LOG(INFO) << "B";
  // printBuffer((cl::Buffer*)B, wa, wb, wa, wb);

  const float* inputA = A;
  const float* inputdA = dA;
  auto input_ha = ha;
  auto input_wa = wa;
  if (ha < min_matrix_size || ha & (ha - 1) || wa < min_matrix_size ||
      wa & (wa - 1)) {
    inputA = (const float*)copyToTileSizeBuffer(
        A, false, ha, wa, &input_ha, &input_wa);
    if (inputdA) {
      inputdA = (const float*)copyToTileSizeBuffer(
          dA, false, ha, wa, &input_ha, &input_wa);
    }
    VLOG(1) << "A resized to " << input_ha << " " << input_wa;
  }

  const float* inputB = B;
  auto input_hb = wa;
  auto input_wb = wb;
  if (input_hb < min_matrix_size || input_hb & (input_hb - 1) ||
      wb < min_matrix_size || wb & (wb - 1)) {
    inputB = (const float*)copyToTileSizeBuffer(
        B, TransB, wa, wb, &input_hb, &input_wb);
    VLOG(1) << "B resized to " << input_hb << " " << input_wb;
  }

  float* outputC = C;
  auto output_hc = ha;
  auto output_wc = wb;
  if (ha < min_matrix_size || ha & (ha - 1) || wb < min_matrix_size ||
      wb & (wb - 1)) {
    outputC =
        (float*)copyToTileSizeBuffer(C, false, ha, wb, &output_hc, &output_wc);
    VLOG(1) << "C resized to " << output_hc << " " << output_wc;
  }

  auto kernelA = kernel_catalog["loadA"];
  unsigned short matrix_cols = input_wa / DOT_PROD_VECTOR_SIZE; // no stride
  unsigned short matrix_row_stride = matrix_cols; // no stride
  unsigned matrix_block_stride = matrix_row_stride * MAT_A_BLOCK_HEIGHT;
  unsigned matrix_size = matrix_row_stride * input_ha;
  unsigned char other_matrix_col_blocks = input_wb / MAT_B_BLOCK_WIDTH;
  unsigned iterations = matrix_size * other_matrix_col_blocks +
      2 * VECTORS_AT_A_TIME *
          MAT_A_BLOCK_HEIGHT; // 2 extra blocks to flush data
  unsigned char relu = reluA, revrelu = revreluA;
  unsigned char disableA = 0;
  VLOG(1) << "loadA: (" << ha << ", " << wa << ") (" << input_ha << ", "
          << input_wa << ") " << matrix_cols << " " << matrix_row_stride << " "
          << matrix_block_stride << " " << matrix_size << " "
          << (int)other_matrix_col_blocks << " " << iterations << " "
          << (int)relu << " " << (int)revrelu << " " << (int)disableA;

  if (revreluA) {
    CAFFE_ENFORCE_EQ(kernelA.setArg(0, *(cl::Buffer*)inputdA), CL_SUCCESS);
  } else {
    CAFFE_ENFORCE_EQ(kernelA.setArg(0, *(cl::Buffer*)inputA), CL_SUCCESS);
  }
  CAFFE_ENFORCE_EQ(kernelA.setArg(1, *(cl::Buffer*)inputA), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(2, matrix_cols), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(3, matrix_row_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(4, matrix_block_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(5, matrix_size), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(6, other_matrix_col_blocks), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(7, iterations), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(8, relu), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(9, revrelu), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(10, disableA), CL_SUCCESS);

  auto kernelB = kernel_catalog["loadB"];
  unsigned char transposed = 0;
  if (TransB) {
    transposed = 1;
  }
  matrix_cols = transposed ? (input_wa / DOT_PROD_VECTOR_SIZE)
                           : (input_wb / DOT_PROD_VECTOR_SIZE);
  matrix_row_stride = matrix_cols; // no stride
  matrix_block_stride = matrix_row_stride * MAT_B_BLOCK_WIDTH;
  matrix_size = matrix_row_stride * (transposed ? input_wb : input_wa);
  unsigned char row_blocks = transposed
      ? matrix_size / matrix_block_stride
      : (matrix_cols / (COLUMNS_INTERLEAVED * PE_COLS / VEC));
  unsigned char col_blocks = transposed ? (matrix_cols / VECTORS_AT_A_TIME)
                                        : (matrix_size / matrix_block_stride);
  other_matrix_col_blocks = input_ha / MAT_A_BLOCK_WIDTH;
  iterations = matrix_size * other_matrix_col_blocks +
      2 * VECTORS_AT_A_TIME * MAT_B_BLOCK_WIDTH; // 2 extra blocks to flush data
  revrelu = 0;
  unsigned char disableB = 0;

  VLOG(1) << "loadB: (" << wa << ", " << wb << ") (" << input_hb << ", "
          << input_wb << ") " << matrix_row_stride << " " << matrix_block_stride
          << " " << matrix_size << " " << (int)row_blocks << " "
          << (int)col_blocks << " " << (int)other_matrix_col_blocks << " "
          << iterations << " " << (int)transposed << " " << (int)revrelu << " "
          << (int)disableB;

  CAFFE_ENFORCE_EQ(kernelB.setArg(0, *(cl::Buffer*)inputB), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(1, *(cl::Buffer*)inputB), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(2, matrix_row_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(3, matrix_block_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(4, matrix_size), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(5, row_blocks), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(6, col_blocks), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(7, other_matrix_col_blocks), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(8, iterations), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(9, transposed), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(10, revrelu), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(11, disableB), CL_SUCCESS);

  auto kernelC = kernel_catalog["store"];
  matrix_cols = output_wc;
  matrix_row_stride = matrix_cols; // no stride
  matrix_block_stride = matrix_row_stride * ROWS_INTERLEAVED * PE_ROWS;
  iterations = output_hc * output_wc / PE_COLS + ACCUM_SHIFT_REG_SIZE * PE_ROWS;
  unsigned char sendtranspose = 0;
  VLOG(1) << "storec: " << matrix_cols << " " << matrix_row_stride << " "
          << matrix_block_stride << " " << (int)sendtranspose << " "
          << iterations;

  CAFFE_ENFORCE_EQ(kernelC.setArg(0, *(cl::Buffer*)outputC), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(1, matrix_cols), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(2, matrix_row_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(3, matrix_block_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(4, sendtranspose), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(5, iterations), CL_SUCCESS);

  const std::vector<std::string> kernel_names = {"loadA", "loadB", "store"};

  int n = 3;
  for (int i = 0; i < n; i++) {
    auto ret = queues[i].enqueueTask(kernel_catalog[kernel_names[i]]);
    CAFFE_ENFORCE_EQ(ret, CL_SUCCESS, "error at kernel", i);
    VLOG(1) << "enqueued event " << i << " " << kernel_names[i];
  }

  for (int i = 0; i < n; i++) {
    queues[i].flush();
  }
  for (int i = 0; i < n; i++) {
    queues[i].finish();
  }

  if (output_hc != ha || output_wc != wb) {
    copyFromTileSizeBuffer(
        (cl::Buffer*)outputC, output_hc, output_wc, C, ha, wb);
    delete (cl::Buffer*)outputC;
  }
  if (inputA != A) {
    delete (cl::Buffer*)inputA;
  }
  if (inputdA != dA) {
    delete (cl::Buffer*)inputdA;
  }
  if (inputB != B) {
    delete (cl::Buffer*)inputB;
  }
  VLOG(1) << "FPGA execution ended";
  // LOG(INFO) << "C";
  // printBuffer((cl::Buffer*)C, ha, wb, ha, wb);

  return true;
}

bool FPGAContextSingleton::MatMulAccum(
    const bool TransA,
    const bool TransB,
    const float* A,
    const float* B,
    float* C,
    const int ha,
    const int wa,
    const int wb) {
  CAFFE_ENFORCE_EQ(TransA, false); // not implemented

  const float* inputA = A;
  auto input_ha = ha;
  auto input_wa = wa;
  if (ha < min_matrix_size || ha & (ha - 1) || wa < min_matrix_size ||
      wa & (wa - 1)) {
    inputA = (const float*)copyToTileSizeBuffer(
        A, false, ha, wa, &input_ha, &input_wa);
    VLOG(1) << "A resized to " << input_ha << " " << input_wa;
  }

  const float* inputB = B;
  auto input_hb = wa;
  auto input_wb = wb;
  if (input_hb < min_matrix_size || input_hb & (input_hb - 1) ||
      wb < min_matrix_size || wb & (wb - 1)) {
    inputB = (const float*)copyToTileSizeBuffer(
        B, TransB, wa, wb, &input_hb, &input_wb);
    VLOG(1) << "B resized to " << input_hb << " " << input_wb;
  }

  float* outputC = C;
  auto output_hc = ha;
  auto output_wc = wb;
  if (ha < min_matrix_size || ha & (ha - 1) || wb < min_matrix_size ||
      wb & (wb - 1)) {
    outputC =
        (float*)copyToTileSizeBuffer(C, false, ha, wb, &output_hc, &output_wc);
    VLOG(1) << "C resized to " << output_hc << " " << output_wc;
  }

  cl::Buffer* intermediateC = static_cast<cl::Buffer*>(
      OpenCLContext::New(output_hc * output_wc * sizeof(float)).get());

  auto kernelA = kernel_catalog["loadA"];
  unsigned short matrix_cols = input_wa / DOT_PROD_VECTOR_SIZE; // no stride
  unsigned short matrix_row_stride = matrix_cols; // no stride
  unsigned matrix_block_stride = matrix_row_stride * MAT_A_BLOCK_HEIGHT;
  unsigned matrix_size = matrix_row_stride * input_ha;
  unsigned char other_matrix_col_blocks = input_wb / MAT_B_BLOCK_WIDTH;
  unsigned iterations = matrix_size * other_matrix_col_blocks +
      2 * VECTORS_AT_A_TIME *
          MAT_A_BLOCK_HEIGHT; // 2 extra blocks to flush data
  unsigned char relu = 0, revrelu = 0;
  unsigned char disableA = 0;
  VLOG(1) << "loadA: (" << ha << ", " << wa << ") (" << input_ha << ", "
          << input_wa << ") " << matrix_cols << " " << matrix_row_stride << " "
          << matrix_block_stride << " " << matrix_size << " "
          << (int)other_matrix_col_blocks << " " << iterations << " "
          << (int)relu << " " << (int)revrelu << " " << (int)disableA;

  CAFFE_ENFORCE_EQ(kernelA.setArg(0, *(cl::Buffer*)inputA), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(1, *(cl::Buffer*)inputA), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(2, matrix_cols), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(3, matrix_row_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(4, matrix_block_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(5, matrix_size), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(6, other_matrix_col_blocks), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(7, iterations), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(8, relu), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(9, revrelu), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelA.setArg(10, disableA), CL_SUCCESS);

  auto kernelB = kernel_catalog["loadB"];
  unsigned char transposed = 0;
  if (TransB) {
    transposed = 1;
  }
  matrix_cols = transposed ? (input_wa / DOT_PROD_VECTOR_SIZE)
                           : (input_wb / DOT_PROD_VECTOR_SIZE);
  matrix_row_stride = matrix_cols; // no stride
  matrix_block_stride = matrix_row_stride * MAT_B_BLOCK_WIDTH;
  matrix_size = matrix_row_stride * (transposed ? input_wb : input_wa);
  unsigned char row_blocks = transposed
      ? matrix_size / matrix_block_stride
      : (matrix_cols / (COLUMNS_INTERLEAVED * PE_COLS / VEC));
  unsigned char col_blocks = transposed ? (matrix_cols / VECTORS_AT_A_TIME)
                                        : (matrix_size / matrix_block_stride);
  other_matrix_col_blocks = input_ha / MAT_A_BLOCK_WIDTH;
  iterations = matrix_size * other_matrix_col_blocks +
      2 * VECTORS_AT_A_TIME * MAT_B_BLOCK_WIDTH; // 2 extra blocks to flush data
  revrelu = 0;
  unsigned char disableB = 0;

  VLOG(1) << "loadB: (" << wa << ", " << wb << ") (" << input_hb << ", "
          << input_wb << ") " << matrix_row_stride << " " << matrix_block_stride
          << " " << matrix_size << " " << (int)row_blocks << " "
          << (int)col_blocks << " " << (int)other_matrix_col_blocks << " "
          << iterations << " " << (int)transposed << " " << (int)revrelu << " "
          << (int)disableB;

  CAFFE_ENFORCE_EQ(kernelB.setArg(0, *(cl::Buffer*)inputB), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(1, *(cl::Buffer*)inputB), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(2, matrix_row_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(3, matrix_block_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(4, matrix_size), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(5, row_blocks), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(6, col_blocks), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(7, other_matrix_col_blocks), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(8, iterations), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(9, transposed), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(10, revrelu), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelB.setArg(11, disableB), CL_SUCCESS);

  auto kernelC = kernel_catalog["store"];
  matrix_cols = output_wc;
  matrix_row_stride = matrix_cols; // no stride
  matrix_block_stride = matrix_row_stride * ROWS_INTERLEAVED * PE_ROWS;
  iterations = output_hc * output_wc / PE_COLS + ACCUM_SHIFT_REG_SIZE * PE_ROWS;
  unsigned char sendtranspose = 1;
  VLOG(1) << "storec: " << matrix_cols << " " << matrix_row_stride << " "
          << matrix_block_stride << " " << (int)sendtranspose << " "
          << iterations;

  // intermediateC not needed
  CAFFE_ENFORCE_EQ(kernelC.setArg(0, *(cl::Buffer*)intermediateC), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(1, matrix_cols), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(2, matrix_row_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(3, matrix_block_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(4, sendtranspose), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelC.setArg(5, iterations), CL_SUCCESS);

  auto kernelExtra = kernel_catalog["extra"];

  cl::Buffer* H =
      static_cast<cl::Buffer*>(OpenCLContext::New(sizeof(bfloat16)).get());
  float a = 1.0;
  float d = 0.0;
  float e = 0.0;
  matrix_cols = output_wc;
  matrix_row_stride = output_wc;
  matrix_block_stride = matrix_row_stride * ROWS_INTERLEAVED * PE_ROWS;
  unsigned char matrix_block_rows = output_hc / MAT_C_BLOCK_HEIGHT;
  matrix_size = output_wc * output_hc;
  iterations = matrix_size / PE_COLS;
  unsigned char mode = 0;

  // C = Trans(Trans(A* B)+ Trans(C))
  cl::Buffer* transC =
      transposeBuffer((cl::Buffer*)outputC, output_hc, output_wc);
  VLOG(1) << "extra: " << a << " " << d << " " << e << " " << matrix_cols << " "
          << matrix_row_stride << " " << matrix_block_stride << " "
          << (int)matrix_block_rows << " " << matrix_size << " " << iterations
          << " " << (int)mode;

  CAFFE_ENFORCE_EQ(kernelExtra.setArg(0, *(cl::Buffer*)transC), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(1, *(cl::Buffer*)H), CL_SUCCESS); // dummy
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(2, a), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(3, d), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(4, e), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(5, matrix_cols), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(6, matrix_row_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(7, matrix_block_stride), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(8, matrix_block_rows), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(9, matrix_size), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(10, iterations), CL_SUCCESS);
  CAFFE_ENFORCE_EQ(kernelExtra.setArg(11, mode), CL_SUCCESS);

  const std::vector<std::string> kernel_names = {
      "loadA", "loadB", "store", "extra"};

  int n = 4;
  for (int i = 0; i < n; i++) {
    auto ret = queues[i].enqueueTask(kernel_catalog[kernel_names[i]]);
    CAFFE_ENFORCE_EQ(ret, CL_SUCCESS, "error at kernel", i);
    VLOG(1) << "enqueued event " << i << " " << kernel_names[i];
  }

  for (int i = 0; i < n - 1; i++) {
    VLOG(1) << "flushing " << i;
    queues[i].flush();
  }
  for (int i = 0; i < n; i++) {
    VLOG(1) << "finishing " << i;
    queues[i].finish();
  }
  VLOG(1) << "finished";

  auto C2 = transposeBuffer((cl::Buffer*)transC, output_wc, output_hc);
  copyBuffer(
      C2, (cl::Buffer*)outputC, output_wc * output_hc * sizeof(bfloat16));

  delete (cl::Buffer*)C2;
  delete (cl::Buffer*)transC;
  delete (cl::Buffer*)intermediateC;
  delete H;

  if (output_hc != ha || output_wc != wb) {
    VLOG(1) << "copy buffers back " << outputC << " " << C;
    copyFromTileSizeBuffer(
        (cl::Buffer*)outputC, output_hc, output_wc, C, ha, wb);
    delete (cl::Buffer*)outputC;
  }
  if (inputA != A) {
    delete (cl::Buffer*)inputA;
  }
  if (inputB != B) {
    delete (cl::Buffer*)inputB;
  }
  VLOG(1) << "FPGA execution ended";
  // printBuffer((cl::Buffer *)C, ha, wb, ha, wb);

  return true;
}

bool FPGAContextSingleton::MatVecMul(
    const bool TransA,
    const float* A,
    const float* x,
    float* y,
    const int ha,
    const int wa) {
  if (TransA) {
    cl::Buffer* A_T = transposeBuffer((cl::Buffer*)A, ha, wa);
    MatMul(
        false,
        false,
        (const float*)A_T,
        nullptr,
        x,
        y,
        wa,
        ha,
        1,
        false,
        false);
    delete A_T;
  } else {
    MatMul(false, false, A, nullptr, x, y, ha, wa, 1, false, false);
  }

  return true;
}

bool FPGAContextSingleton::ReLU(const float* A, float* A_r, int ha, int wa) {
  // Create Identity
  cl::Buffer* I = createIdentity(wa);
  MatMul(false, false, A, NULL, (float*)I, A_r, ha, wa, wa, true, false);
  delete I;

  return true;
}

bool FPGAContextSingleton::ReLUGrad(
    const float* A,
    const float* dA,
    float* C,
    const int ha,
    const int wa) {
  cl::Buffer* I = createIdentity(wa);
  MatMul(false, false, A, dA, (const float*)I, C, ha, wa, wa, false, true);
  delete I;

  return true;
}

} // namespace caffe2
