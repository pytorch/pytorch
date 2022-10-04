#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/csrc/cuda/memory_snapshot.h>
#include <torch/csrc/jit/serialization/pickler.h>
namespace torch {
namespace cuda {

using torch::jit::Pickler;
using c10::IValue;
using c10::Dict;

using c10::cuda::CUDACachingAllocator::BlockInfo;
using c10::cuda::CUDACachingAllocator::History;
using c10::cuda::CUDACachingAllocator::SegmentInfo;

namespace {
  std::unique_ptr<c10::cuda::CUDACachingAllocator::Context> blank_context() {
    // in the future the C++-only version of context gathering could include C++ or torchscript frames.
    return std::make_unique<c10::cuda::CUDACachingAllocator::Context>();
  }
  std::vector<char> write_pickle(const IValue& v) {
    std::vector<char> result;
    {
      auto writer = [&](const char* data, size_t size) {
        result.insert(result.end(), data, data + size);
      };
      Pickler pickler(writer, nullptr, nullptr, nullptr, nullptr, false);
      pickler.protocol();
      pickler.pushIValue(v);
      pickler.stop();
    }
    return result;
  }
  Dict<IValue, IValue> new_dict() {
    return Dict<IValue,IValue>(c10::AnyType::get(), c10::AnyType::get());
  }
  c10::List<IValue> new_list() {
    return List<IValue>(c10::AnyType::get());
  }
}
  void _record_memory_history(bool enabled) {
    c10::cuda::CUDACachingAllocator::setContextRecorder(enabled ? blank_context : nullptr);
  }

  std::vector<char> _memory_snapshot_pickled() {
    IValue device_s = "device";
    IValue address_s = "address";
    IValue total_size_s = "total_size";
    IValue allocated_size_s = "allocated_size";
    IValue active_size_s = "active_size";
    IValue stream_s = "stream";
    IValue segment_type_s = "segment_type";
    IValue large_s = "large";
    IValue small_s = "small";
    IValue size_s = "size";
    IValue state_s = "state";
    IValue active_allocated_s = "active_allocated";
    IValue active_pending_free_s = "active_pending_free";
    IValue inactive_s = "inactive";
    IValue addr_s = "addr";
    IValue real_size_s = "real_size";
    IValue filename_s = "filename";
    IValue name_s = "name";
    IValue line_s = "line";
    IValue frames_s = "frames";
    IValue history_s = "history";
    IValue blocks_s = "blocks";

    auto empty_frames = new_list();

    const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
      auto segmentDict = new_dict();
      segmentDict.insert(device_s, segmentInfo.device);
      segmentDict.insert(address_s, segmentInfo.address);
      segmentDict.insert(total_size_s, segmentInfo.total_size);
      segmentDict.insert(allocated_size_s, segmentInfo.allocated_size);
      segmentDict.insert(active_size_s, segmentInfo.active_size);
      segmentDict.insert(stream_s, int64_t(segmentInfo.stream));
      segmentDict.insert(segment_type_s, (segmentInfo.is_large ? large_s : small_s));

      auto blocks = new_list();
      for (const auto& blockInfo : segmentInfo.blocks) {
        auto blockDict = new_dict();
        blockDict.insert(size_s, blockInfo.size);
        blockDict.insert(state_s,
            (blockInfo.allocated
                ? active_allocated_s
                : (blockInfo.active ? active_pending_free_s : inactive_s)));
        if (blockInfo.history) {
          auto history = new_list();
          History* h = blockInfo.history;
          while (h) {
            auto history_entry = new_dict();
            history_entry.insert(addr_s, (int64_t)h->addr);
            history_entry.insert(real_size_s, (int64_t) h->real_size);
            if (h->context) {
              history_entry.insert(frames_s, empty_frames);
            }
            h = h->next.get();
            history.push_back(std::move(history_entry));
          }
          blockDict.insert(history_s, std::move(history));
        }
        blocks.push_back(blockDict);
      }
      segmentDict.insert(blocks_s, blocks);

      return segmentDict;
    };

    const std::vector<SegmentInfo>& snapshot =
        c10::cuda::CUDACachingAllocator::snapshot();

    auto result = new_list();
    for (const auto& segmentInfo : snapshot) {
      result.push_back(segmentInfoToDict(segmentInfo));
    }

    return write_pickle(result);
 }
}
} // namespace torch
