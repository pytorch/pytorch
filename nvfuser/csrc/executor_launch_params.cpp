#include <executor_launch_params.h>

#include <ATen/cuda/CUDAContext.h>

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

void LaunchParams::assertValid() {
  TORCH_INTERNAL_ASSERT(
      bdimx() * bdimy() * bdimz() > 0 &&
          bdimx() * bdimy() * bdimz() <=
              (int64_t)at::cuda::getCurrentDeviceProperties()
                  ->maxThreadsPerMultiProcessor,
      "Selected invalid number of threads for cuda: ",
      bdimx() * bdimy() * bdimz());
  TORCH_INTERNAL_ASSERT(
      gdimx() > 0 && gdimx() < (std::int64_t(1) << 32) - 1,
      "Invalid number of blocks in x direction: ",
      gdimx());
  TORCH_INTERNAL_ASSERT(
      gdimy() > 0 && gdimy() <= 65535,
      "Invalid number of blocks in y direction: ",
      gdimy());
  TORCH_INTERNAL_ASSERT(
      gdimz() > 0 && gdimz() <= 65535,
      "Invalid number of blocks in z direction: ",
      gdimz());
}

void LaunchParams::bind(int64_t val, ParallelType p_type) {
  switch (p_type) {
    case ParallelType::TIDx:
      checkAndSet(val, bdimx_, "blockDim.x");
      break;
    case ParallelType::BIDx:
      checkAndSet(val, gdimx_, "gridDim.x");
      break;
    case ParallelType::TIDy:
      checkAndSet(val, bdimy_, "blockDim.y");
      break;
    case ParallelType::BIDy:
      checkAndSet(val, gdimy_, "gridDim.y");
      break;
    case ParallelType::TIDz:
      checkAndSet(val, bdimz_, "blockdim.z");
      break;
    case ParallelType::BIDz:
      checkAndSet(val, gdimz_, "gridDim.z");
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to bind invalid parallel type in launch config: ",
          p_type);
  }
  assertValid();
}

int64_t LaunchParams::getDim(ParallelType p_type) const {
  switch (p_type) {
    case ParallelType::TIDx:
      return bdimx();
    case ParallelType::BIDx:
      return gdimx();
    case ParallelType::TIDy:
      return bdimy();
    case ParallelType::BIDy:
      return gdimy();
    case ParallelType::TIDz:
      return bdimz();
    case ParallelType::BIDz:
      return gdimz();
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to get with invalid parallel type in launch config: ",
          p_type);
  }
}

bool LaunchParams::hasDim(ParallelType p_type) const {
  return getRawVal(p_type) != UNINITIALIZED_VAL;
}

const int64_t& LaunchParams::getRawVal(ParallelType p_type) const {
  switch (p_type) {
    case ParallelType::TIDx:
      return bdimx_;
    case ParallelType::BIDx:
      return gdimx_;
    case ParallelType::TIDy:
      return bdimy_;
    case ParallelType::BIDy:
      return gdimy_;
    case ParallelType::TIDz:
      return bdimz_;
    case ParallelType::BIDz:
      return gdimz_;
    default:
      TORCH_INTERNAL_ASSERT(
          false,
          "Tried to get with invalid parallel type in launch config: ",
          p_type);
  }
}

bool LaunchParams::operator==(const LaunchParams& other) const {
  return gdimx_ == other.gdimx_ && gdimy_ == other.gdimy_ &&
      bdimx_ == other.bdimx_ && bdimy_ == other.bdimy_ && smem_ == other.smem_;
}

void LaunchParams::print() const {
  std::cout << toString();
}

std::string LaunchParams::toString() const {
  std::stringstream ss;
  ss << "Launch Parameters: "
     << "BlockDim.x = " << (bdimx_ == UNINITIALIZED_VAL ? -1 : bdimx_) << ", "
     << "BlockDim.y = " << (bdimy_ == UNINITIALIZED_VAL ? -1 : bdimy_) << ", "
     << "BlockDim.z = " << (bdimz_ == UNINITIALIZED_VAL ? -1 : bdimz_) << ", "
     << "GridDim.x = " << (gdimx_ == UNINITIALIZED_VAL ? -1 : gdimx_) << ", "
     << "GridDim.y = " << (gdimy_ == UNINITIALIZED_VAL ? -1 : gdimy_) << ", "
     << "GridDim.z = " << (gdimz_ == UNINITIALIZED_VAL ? -1 : gdimz_) << ", "
     << "Smem Size = " << smem() << "\n";
  return ss.str();
}

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
