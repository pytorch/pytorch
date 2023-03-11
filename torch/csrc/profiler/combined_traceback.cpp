#include <torch/csrc/profiler/combined_traceback.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

namespace torch {

// Locking:
// We need to free PyCodeObjects when ~StackContext runs, but
// CUDACachingAllocator may hold its device lock when ~StackContext runs.

// Because the thread calling the allocator _may_ hold the GIL,
// attempting to lock the GIL in ~StackContext can deadlock:
// T0: GIL Lock -> Call Allocator    ->| Waiting Device Lock
// T1: Call Allocator -> Device Lock ->| Waiting GIL Lock
// Instead the destructor defers freeing stack frames by putting them in
// to_free_frames. We still need a lock to manage this vector, but
// we can ensure an overall lock ordering of GIL -> device_lock ->
// to_free_frames_mutex because ::gather is called outside of the device lock.

static std::mutex to_free_frames_mutex;
static std::vector<CapturedTraceback::PyFrame> to_free_frames;

struct PythonTraceback : public CapturedTraceback::Python {
    std::vector<CapturedTraceback::PyFrame> gather() override {
      std::vector<CapturedTraceback::PyFrame> frames;
      py::gil_scoped_acquire acquire;
      {
        std::lock_guard lock(to_free_frames_mutex);
        for (CapturedTraceback::PyFrame f : to_free_frames) {
          Py_XDECREF(f.code);
        }
        to_free_frames.clear();
      }
      PyFrameObject* f = PyEval_GetFrame();
      Py_XINCREF(f);
      while (f) {
        frames.emplace_back(
            CapturedTraceback::PyFrame{PyFrame_GetCode(f), PyFrame_GetLasti(f)});
        auto f_back = PyFrame_GetBack(f);
        Py_XDECREF(f);
        f = f_back;
      }
      return frames;
    }
    void release(std::vector<CapturedTraceback::PyFrame>& frames) override {
      std::lock_guard lock(to_free_frames_mutex);
      to_free_frames.insert(to_free_frames.end(), frames.begin(), frames.end());
    }
    void appendSymbolized(const std::vector<CapturedTraceback::PyFrame>& to_symbolize, SymbolizedTracebacks& result) override {
      auto torch = py::module::import("torch");
      py::str line_s = "line";
      py::str name_s = "name";
      py::str filename_s = "filename";

      py::object stack_frames_for_code;
      if (py::hasattr(torch, "_inductor")) {
        py::object inductor = torch.attr("_inductor");
        if (py::hasattr(inductor, "codecache")) {
          stack_frames_for_code = inductor.attr("codecache")
                                      .attr("PyCodeCache")
                                      .attr("stack_frames_for_code");
        }
      }
      for (const auto & f : to_symbolize) {
        py::handle filename = f.code->co_filename;
        py::handle funcname = f.code->co_name;
        auto lineno = PyCode_Addr2Line(f.code, f.lasti);
        result.tracebacks.emplace_back();
        result.tracebacks.back().push_back(result.all_frames.size());
        result.all_frames.emplace_back(unwind::Frame {py::cast<std::string>(filename), py::cast<std::string>(funcname), (uint64_t)lineno});
        // find all the additional frames associated with inductor generated
        // code
        if (stack_frames_for_code.ptr()) {
          py::object extra = stack_frames_for_code(filename, lineno);
          if (!extra.is_none()) {
            for (py::handle h : extra) {
              result.tracebacks.back().push_back(result.all_frames.size());
              result.all_frames.emplace_back(unwind::Frame {py::cast<std::string>(h[filename_s]), py::cast<std::string>(h[name_s]), py::cast<uint64_t>(h[line_s])});
            }
          }
        }
      }
    }
};

std::vector<py::object> py_symbolize(std::vector<CapturedTraceback*>& to_symbolize) {
  // we dedup repeated to_symbolize objects to prevent
  // creating a bunch of duplicated frame objects
  std::unordered_map<CapturedTraceback*, uint64_t> cached_frames;
  std::vector<CapturedTraceback*> unique_frames;
  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      cached_frames.insert({sc, unique_frames.size()});
      unique_frames.push_back(sc);
    }
  }
  auto s = symbolize(unique_frames);

  py::str line_s = "line";
  py::str name_s = "name";
  py::str filename_s = "filename";
  std::vector<py::dict> all_frames;
  for (const auto & f : s.all_frames) {
    py::dict d;
    d[name_s] = f.funcname;
    d[filename_s] = f.filename;
    d[line_s] = f.lineno;
    all_frames.emplace_back(std::move(d));
  }

  std::vector<py::object> py_unique_frames;
  for (const auto & t : s.tracebacks) {
    py::list l;
    for (const auto & e : t) {
      l.append(all_frames.at(e));
    }
    py_unique_frames.push_back(std::move(l));
  }

  std::vector<py::object> result;
  for (const auto& sc : to_symbolize) {
    result.push_back(py_unique_frames.at(cached_frames.at(sc)));
  }
  return result;
}

static std::atomic<CapturedTraceback::Python*> python_support_ = new PythonTraceback();



std::shared_ptr<CapturedTraceback> CapturedTraceback::gather(
    bool python,
    bool script,
    bool cpp) {
  auto r = std::make_shared<CapturedTraceback>();
  if (python) {
    auto p = python_support_.load();
    while (p && r->frames_.size() == 0) {
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

CapturedTraceback::~CapturedTraceback() {
  if (frames_.size() > 0) {
    TORCH_INTERNAL_ASSERT(python_);
    python_->release(frames_);
  }
}

struct PyFrameHash {
    std::size_t operator() (const CapturedTraceback::PyFrame& f) const {
      return std::hash<PyCodeObject*>()(f.code) ^ std::hash<int>()(f.lasti);
    }
};

struct PyFrameEq {
  std::size_t operator() (const CapturedTraceback::PyFrame& lhs, const CapturedTraceback::PyFrame& rhs) const {
    return lhs.code == rhs.code && lhs.lasti == rhs.lasti;
  }
};

SymbolizedTracebacks symbolize(const
    std::vector<CapturedTraceback*>& to_symbolize) {

  SymbolizedTracebacks r;


  py::str filename_s = "filename";
  py::str name_s = "name";
  py::str line_s = "line";

  std::unordered_map<void*, size_t> ip_to_frame_offset;
  std::unordered_map<CapturedTraceback::PyFrame, size_t, PyFrameHash, PyFrameEq> py_to_frame_offset;
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
  if (all_cpp_ips.size() > 0) {
    r.all_frames = unwind::symbolize(all_cpp_ips);
  }

  // batch symbolization requests so we dedup frame objects
  // however, we might have to request from different python interpreters
  // make sure we flush requests before switching interpreters;
  CapturedTraceback::Python* cur_python = nullptr;
  std::vector<CapturedTraceback::PyFrame> cur_py_frames;
  size_t py_frames_size_ = 0;

  for (const auto& e : to_symbolize) {
    if (e->python_) {
      if (cur_python != e->python_ && cur_py_frames.size() > 0) {
        cur_python->appendSymbolized(cur_py_frames, r);
        cur_py_frames.clear();
      }
      cur_python = e->python_;
      for (const auto & f : e->frames_) {
        if (!py_to_frame_offset.count(f)) {
          py_to_frame_offset[f] = py_frames_size_++;
          cur_py_frames.push_back(f);
        }
      }
    }
  }
  if (cur_py_frames.size() > 0) {
    cur_python->appendSymbolized(cur_py_frames, r);
    cur_py_frames.clear();
  }
  std::vector<std::vector<uint64_t>> python_frame_fragments = std::move(r.tracebacks);

  for (const auto& sc : to_symbolize) {
    r.tracebacks.emplace_back();
    auto py_it = sc->frames_.begin();
    auto py_end = sc->frames_.end();

    bool jit_appended = false;

    auto append_python = [&](const CapturedTraceback::PyFrame& f) {
      const auto & fragment = python_frame_fragments.at(py_to_frame_offset.at(f));
      r.tracebacks.back().insert(r.tracebacks.back().end(), fragment.begin(), fragment.end());
    };

    auto append_jit = [&]() {
      if (jit_appended) {
        return;
      }
      jit_appended = true;
      for (const auto& f : sc->script_frames_) {
        unwind::Frame frame;
        frame.funcname = f.filename; // sic: torchscript puts funcname in filename field
        auto flc = f.range.file_line_col();
        if (flc) {
          size_t col;
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
      } else if (uf.funcname.rfind("torch::jit::InterpreterStateImpl::run", 0) != std::string::npos) {
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
  }
  return r;
}

} // namespace torch
