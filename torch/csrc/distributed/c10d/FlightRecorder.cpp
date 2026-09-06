#include <c10/util/FileSystem.h>
#include <torch/csrc/distributed/c10d/FlightRecorderDetail.hpp>
#include <fstream>

namespace c10d {

void DebugInfoWriter::write(const std::string& trace) {
  std::string filename = filename_;
  if (enable_dynamic_filename_) {
    LOG(INFO) << "Writing Flight Recorder debug info to a dynamic file name";
    filename = c10::str(getCvarString({"TORCH_FR_DUMP_TEMP_FILE"}, ""));
  } else {
    LOG(INFO) << "Writing Flight Recorder debug info to a static file name";
  }
  // Open a file for writing. The ios::binary flag is used to write data as
  // binary.
  std::ofstream file(filename, std::ios::binary);

  // Check if the file was opened successfully.
  if (!file.is_open()) {
    LOG(ERROR) << "Error opening file for writing Flight Recorder debug info: "
               << filename;
    return;
  }

  if (!file.write(trace.data(), static_cast<std::streamsize>(trace.size()))) {
    const auto bad = file.bad();
    LOG(ERROR) << "Error writing Flight Recorder debug info to file: "
               << filename << " bad bit: " << bad;
    return;
  }

  // Flush the buffer to ensure data is written to the file
  file.flush();
  if (file.bad()) {
    LOG(ERROR) << "Error flushing Flight Recorder debug info: " << filename;
    return;
  }

  LOG(INFO) << "Finished writing Flight Recorder debug info to " << filename;
}

DebugInfoWriter& DebugInfoWriter::getWriter(int rank) {
  if (writer_ == nullptr) {
// Attempt to write to running user's HOME directory cache folder - if it
// exists.
#ifdef _WIN32
    const char* cacheHome = nullptr;
#else
    // Uses XDG_CACHE_HOME if it's set
    const char* cacheHome = std::getenv("XDG_CACHE_HOME");
#endif
    std::string cacheRoot;
    if (cacheHome) {
      cacheRoot = cacheHome;
    } else {
      cacheRoot = getCvarString({"HOME"}, "/tmp") + "/.cache";
    }
    auto cacheDirPath = std::filesystem::path(cacheRoot + "/torch");
    // Create the .cache directory if it doesn't exist
    c10::filesystem::create_directories(cacheDirPath);
    auto defaultLocation = cacheDirPath / "comm_lib_trace_rank_";

    // For internal bc compatibility, we keep the old the ENV check.
    std::string fileNamePrefix = getCvarString(
        {"TORCH_FR_DUMP_TEMP_FILE", "TORCH_NCCL_DEBUG_INFO_TEMP_FILE"},
        defaultLocation.string().c_str());
    bool useDynamicFileName =
        getCvarBool({"TORCH_FR_DUMP_DYNAMIC_FILE_NAME"}, false);
    // Using std::unique_ptr here to auto-delete the writer object
    // when the pointer itself is destroyed.
    std::unique_ptr<DebugInfoWriter> writerPtr(
        new DebugInfoWriter(fileNamePrefix, rank, useDynamicFileName));
    DebugInfoWriter::registerWriter(std::move(writerPtr));
  }
  return *writer_;
}

void DebugInfoWriter::registerWriter(std::unique_ptr<DebugInfoWriter> writer) {
  if (hasWriterRegistered_.load()) {
    TORCH_WARN_ONCE(
        "DebugInfoWriter has already been registered, and since we need the writer to stay "
        "outside ProcessGroup, user needs to ensure that this extra registration is indeed needed. "
        "And we will only use the last registered writer.");
  }
  hasWriterRegistered_.store(true);
  writer_ = std::move(writer);
}

std::unique_ptr<DebugInfoWriter> DebugInfoWriter::writer_ = nullptr;
std::atomic<bool> DebugInfoWriter::hasWriterRegistered_(false);

template <>
float getDurationFromEvent<c10::Event>(
    c10::Event& startEvent,
    c10::Event& endEvent) {
  TORCH_CHECK(
      endEvent.query(),
      "getDurationFromEvent can only be called after the end event has completed.");
  return static_cast<float>(startEvent.elapsedTime(endEvent));
}

// For any third party library that uses the flight recorder, if one wants to
// use an Event type other than c10::Event, one also needs to registers here to
// avoid linking errors.
template struct FlightRecorder<c10::Event>;

FlightRecorder<c10::Event>* getFlightRecorder(const std::string& backend) {
  if (backend.empty() || backend == kDefaultFRBackend) {
    return FlightRecorder<c10::Event>::get();
  }
  static std::mutex mutex;
  static std::unordered_map<std::string, FlightRecorder<c10::Event>*> registry;
  std::lock_guard<std::mutex> lock(mutex);
  auto& instance = registry[backend];
  if (instance == nullptr) {
    // Intentionally leaked on exit, like FlightRecorder::get(): entries hold
    // Python state that may already have been destructed.
    instance = new FlightRecorder<c10::Event>();
  }
  return instance;
}

bool recordsFlightRecorderNatively(const std::string& backend) {
  return backend == "gloo" || backend == "nccl" || backend == "xccl";
}

std::string dump_fr_trace(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive,
    const std::string& backend) {
  return getFlightRecorder(backend)->dump(
      std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>{},
      includeCollectives,
      includeStackTraces,
      onlyActive);
}

std::string dump_fr_trace_json(
    bool includeCollectives,
    bool onlyActive,
    const std::string& backend) {
  return getFlightRecorder(backend)->dump_json(
      std::unordered_map<
          std::string,
          std::unordered_map<std::string, std::string>>{},
      includeCollectives,
      onlyActive);
}

void dump_fr_trace_file(
    int rank,
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive,
    const std::string& backend) {
  // Serialize writes so concurrent dumps cannot interleave into the same
  // file, but take the trace before locking: dump() can block on device APIs
  // and must not hold up an unrelated caller.
  static std::mutex writeDebugInfoMutex;
  auto trace = dump_fr_trace(
      includeCollectives, includeStackTraces, onlyActive, backend);
  std::lock_guard<std::mutex> lock(writeDebugInfoMutex);
  DebugInfoWriter& writer = DebugInfoWriter::getWriter(rank);
  LOG(INFO) << "Dumping Flight Recorder trace to " << writer.getWriterTarget();
  writer.write(trace);
}

bool try_dump_fr_trace_file(
    bool includeCollectives,
    bool includeStackTraces,
    bool onlyActive,
    const std::string& backend) {
  auto* recorder = getFlightRecorder(backend);
  int rank = recorder->getRank();
  if (!recorder->enabled_ || rank < 0) {
    return false;
  }
  dump_fr_trace_file(
      rank, includeCollectives, includeStackTraces, onlyActive, backend);
  return true;
}

void reset_fr_trace(const std::string& backend) {
  getFlightRecorder(backend)->reset_all();
}
} // namespace c10d
