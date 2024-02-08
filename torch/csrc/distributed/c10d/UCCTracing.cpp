#ifdef USE_C10D_UCC

#include <torch/csrc/distributed/c10d/UCCTracing.hpp>
#include <torch/csrc/distributed/c10d/UCCUtils.hpp>

#include <torch/csrc/distributed/c10d/ParamCommsUtils.hpp>

#include <sys/stat.h>
#include <cstdlib>
#include <ctime>
#include <fstream>

namespace c10d {

void ProcessGroupUCCLogger::initCommsTracer() {
  trace_generator = std::make_shared<CommTraceLogger>();
  initialized_CommTraceLogger = true;
}

void ProcessGroupUCCLogger::flushComms(int rank, int world_size) {
  if (!initialized_CommTraceLogger ||
      trace_generator->getCommsTrace().empty()) {
    return;
  }

  std::string dirname = c10::str("ProcessGroupUCC_trace_np", world_size);
  time_t now_ = time(0);
  std::tm* ltm = localtime(&now_);
  if (ltm) {
    dirname += c10::str(
        "_", (1 + ltm->tm_mon), "_", ltm->tm_mday, "_", (1900 + ltm->tm_year));
  }

  std::string fullpath = "/tmp/" + dirname;
  char* user_path = std::getenv("TORCH_UCC_COMMS_TRACE_OUTPUT_DIR");
  if (user_path) {
    fullpath = user_path;
  }
  std::string trace_filename = c10::str(fullpath, "/rank", rank, ".json");
  std::ofstream _outfile;
  if (!_outfile.is_open()) {
    if (!mkdir(fullpath.c_str(), 0777)) {
      LOG(INFO) << getLogPrefix() << "[INFO] failed to mkdir " << fullpath;
    } else if (errno != EEXIST) {
      return;
    }
    _outfile.open(trace_filename, std::ofstream::out | std::ofstream::trunc);
  }
  // flush the traced comms
  if (_outfile.is_open()) {
    _outfile << "[" << c10::Join(",", trace_generator->getCommsTrace())
             << "\n]";
    _outfile.flush();
    _outfile.close();
  }
}

/* unused */
void CommTraceLogger::setCurBlock(const std::string& name) {
  curBlocks_.push_back(
      c10::str("\"", name, "\"")); // add quote marks for JSON format
}

/* unused */
void CommTraceLogger::popBlock() {
  // TODO: remove specific name
  curBlocks_.pop_back();
}

void CommTraceLogger::recordOptionalInfo(int root) {
  curRoot_ = root;
}

void CommTraceLogger::recordOptionalInfo(
    const std::vector<int64_t>& outputSplitSizes,
    const std::vector<int64_t>& inputSplitSizes) {
  curOutSplitSizes_ = outputSplitSizes;
  curInSplitSizes_ = inputSplitSizes;
}

void CommTraceLogger::recordComms(
    const std::string& commName,
    const uintptr_t workReq,
    const int rank,
    const int world_size,
    const std::vector<at::Tensor>& inputTensors,
    const std::vector<at::Tensor>& outputTensors) {
  auto inNelems = (!inputTensors.empty()) ? inputTensors[0].numel() : 0;
  auto outNelems = (!outputTensors.empty()) ? outputTensors[0].numel() : 0;
  auto dtype =
      (!outputTensors.empty()) ? outputTensors[0].scalar_type() : at::kByte;
  auto devType = (!outputTensors.empty()) ? outputTensors[0].device().type()
                                          : c10::DeviceType::CPU;
  auto now = std::chrono::system_clock::now();
  static auto startTS = now;
  int64_t time_since_begin =
      std::chrono::duration_cast<std::chrono::nanoseconds>(now - startTS)
          .count();

  // TODO: get markers from torch profiler if enabled

  // common fields for all operations
  std::string cur_trace_ = c10::str(
      "\n\t\t\"markers\": [",
      curBlocks_,
      "]",
      ",\n\t\t\"startTime_ns\": ",
      time_since_begin,
      ",\n\t\t\"comms\": \"",
      commName,
      "\"",
      ",\n\t\t\"req\": ",
      workReq,
      ",\n\t\t\"seqnum\": ",
      seqnum,
      ",\n\t\t\"world_size\": ",
      world_size);

  if (inNelems > 0 || outNelems > 0) {
    // for most collectives - append msg sizes, data type, device type
    cur_trace_ = c10::str(
        cur_trace_,
        ",\n\t\t\"in_msg_size\": ",
        inNelems,
        ",\n\t\t\"out_msg_size\": ",
        outNelems,
        ",\n\t\t\"dtype\": \"",
        at::toString(dtype),
        "\",\n\t\t\"devType\": \"",
        c10::DeviceTypeName(devType),
        "\"");
  }
  if (curRoot_ != -1) {
    // append root rank if applicable, e.g., broadcast, gather, scatter
    cur_trace_ = c10::str(cur_trace_, ",\n\t\t\"root\": ", curRoot_);
  }
  if (!curInSplitSizes_.empty() || !curOutSplitSizes_.empty()) {
    // append input and output splits if applicable, e.g., ALLTOALL_BASE
    cur_trace_ = c10::str(
        cur_trace_,
        ",\n\t\t\"in_split\": [",
        c10::Join(",", curInSplitSizes_),
        "]"
        ",\n\t\t\"out_split\": [",
        c10::Join(",", curOutSplitSizes_),
        "]");
  }
  comms_trace_.push_back(c10::str("\n\t{", cur_trace_, "\n\t}"));

  // record the trace to kineto trace if applicable
  RECORD_PARAM_COMMS(
      static_cast<int64_t>(seqnum), // seq
      0, // process group ptr
      rank,
      commName.c_str(),
      inNelems,
      outNelems,
      dtype,
      curInSplitSizes_,
      curOutSplitSizes_,
      -1,
      -1,
      world_size);

  ++seqnum;

  // reset optional field
  curRoot_ = -1;
  curInSplitSizes_ = {};
  curOutSplitSizes_ = {};
}

} // namespace c10d

#endif // USE_C10D_UCC
