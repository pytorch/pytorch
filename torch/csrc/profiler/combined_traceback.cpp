#include <torch/csrc/profiler/combined_traceback.h>
#include <torch/csrc/utils/cpp_stacktraces.h>

namespace torch {

static std::atomic<CapturedTraceback::Python*> python_support_ = nullptr;

std::shared_ptr<CapturedTraceback> CapturedTraceback::gather(
    bool python,
    bool script,
    bool cpp) {
  auto r = std::make_shared<CapturedTraceback>();
  if (python) {
    auto p = python_support_.load();
    while (p && r->frames_.empty()) {
      r->frames_ = p->gather();
      r->python_ = p;
      p = p->next_;
    }
  }
  if (script) {
    r->script_frames_ = torch::jit::currentCallstack();
  }
  if (cpp) {
    r->cpp_frames_ = unwind::unwind();
  }
  return r;
}

int CapturedTraceback::traversePython(visitproc visit, void* arg) {
  TORCH_INTERNAL_ASSERT(python_);
  return python_->traverse(frames_, visit, arg);
}

int CapturedTraceback::clearPython() {
  TORCH_INTERNAL_ASSERT(python_);
  return python_->clear(frames_);
}

CapturedTraceback::~CapturedTraceback() {
  if (!frames_.empty()) {
    TORCH_INTERNAL_ASSERT(python_);
    python_->release(frames_);
  }
}

struct PyFrameHash {
  std::size_t operator()(const CapturedTraceback::PyFrame& f) const {
    return std::hash<void*>()(f.code) ^ std::hash<int>()(f.lasti);
  }
};

struct PyFrameEq {
  std::size_t operator()(
      const CapturedTraceback::PyFrame& lhs,
      const CapturedTraceback::PyFrame& rhs) const {
    return lhs.code == rhs.code && lhs.lasti == rhs.lasti;
  }
};

SymbolizedTracebacks symbolize(
    const std::vector<CapturedTraceback*>& to_symbolize) {
  SymbolizedTracebacks r;

  std::unordered_map<void*, size_t> ip_to_frame_offset;
  std::unordered_map<CapturedTraceback::PyFrame, size_t, PyFrameHash, PyFrameEq>
      py_to_frame_offset;
  std::vector<void*> all_cpp_ips;

  // dedup and collect any C++ frames that need symbols for
  for (const auto& e : to_symbolize) {
    for (void* f : e->cpp_frames_) {
      if (!ip_to_frame_offset.count(f)) {
        ip_to_frame_offset[f] = all_cpp_ips.size();
        all_cpp_ips.push_back(f);
      }
    }
  }
  // gather symbol names for C++ frames
  if (!all_cpp_ips.empty()) {
    r.all_frames = unwind::symbolize(all_cpp_ips, torch::get_symbolize_mode());
  }

  // batch symbolization requests so we dedup frame objects
  // however, we might have to request from different python interpreters
  // make sure we flush requests before switching interpreters;
  CapturedTraceback::Python* cur_python = nullptr;
  std::vector<CapturedTraceback::PyFrame> cur_py_frames;
  size_t py_frames_size_ = 0;

  for (const auto& e : to_symbolize) {
    if (e->python_) {
      if (cur_python != e->python_ && !cur_py_frames.empty()) {
        if (cur_python) {
          // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
          cur_python->appendSymbolized(cur_py_frames, r);
        }
        cur_py_frames.clear();
      }
      cur_python = e->python_;
      for (const auto& f : e->frames_) {
        if (!py_to_frame_offset.count(f)) {
          py_to_frame_offset[f] = py_frames_size_++;
          cur_py_frames.push_back(f);
        }
      }
    }
  }
  if (!cur_py_frames.empty()) {
    if (cur_python) {
      // NOLINTNEXTLINE(clang-analyzer-core.CallAndMessage)
      cur_python->appendSymbolized(cur_py_frames, r);
    }
    cur_py_frames.clear();
  }
  std::vector<std::vector<uint64_t>> python_frame_fragments =
      std::move(r.tracebacks);
  r.tracebacks = {};

  for (const auto& sc : to_symbolize) {
    r.tracebacks.emplace_back();
    auto py_it = sc->frames_.begin();
    auto py_end = sc->frames_.end();

    bool jit_appended = false;

    auto append_python = [&](const CapturedTraceback::PyFrame& f) {
      const auto& fragment =
          python_frame_fragments.at(py_to_frame_offset.at(f));
      r.tracebacks.back().insert(
          r.tracebacks.back().end(), fragment.begin(), fragment.end());
    };

    auto append_jit = [&]() {
      if (jit_appended) {
        return;
      }
      jit_appended = true;
      for (const auto& f : sc->script_frames_) {
        unwind::Frame frame;
        frame.funcname =
            f.filename; // sic: torchscript puts funcname in filename field
        auto flc = f.range.file_line_col();
        if (flc) {
          size_t col = 0;
          std::tie(frame.filename, frame.lineno, col) = *flc;
        } else {
          frame.filename = "??";
          frame.lineno = 0;
        }
        r.tracebacks.back().push_back(r.all_frames.size());
        r.all_frames.emplace_back(std::move(frame));
      }
    };

    for (void* f : sc->cpp_frames_) {
      uint64_t cpp_frame = ip_to_frame_offset.at(f);
      const unwind::Frame& uf = r.all_frames.at(cpp_frame);
      if (uf.funcname.find("PyEval_EvalFrame") != std::string::npos) {
        if (py_it != py_end) {
          append_python(*py_it++);
        }
      } else if (
          uf.funcname.rfind("torch::jit::InterpreterStateImpl::run", 0) !=
          std::string::npos) {
        append_jit();
      }
      r.tracebacks.back().push_back(cpp_frame);
    }

    // add frames if we otherwise haven't seen the C++ frame indicating where
    // it should go
    append_jit();

    for (; py_it != py_end; ++py_it) {
      append_python(*py_it);
    }

    // Gather all user defined frames
    for (const auto& f : sc->user_defined_frames_) {
      r.tracebacks.back().push_back(r.all_frames.size());
      r.all_frames.emplace_back(f);
    }
  }
  return r;
}

void CapturedTraceback::addPythonUnwinder(CapturedTraceback::Python* p) {
  CapturedTraceback::Python* old_unwinder = python_support_.load();
  do {
    p->next_ = old_unwinder;
  } while (!python_support_.compare_exchange_strong(old_unwinder, p));
}

} // namespace torch
