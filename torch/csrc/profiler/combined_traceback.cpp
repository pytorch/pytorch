#include <torch/csrc/profiler/combined_traceback.h>
#include <torch/csrc/utils/pythoncapi_compat.h>

namespace torch {

static std::mutex to_free_frames_mutex;
static std::vector<CapturedTraceback::PyFrame> to_free_frames;

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

std::shared_ptr<CapturedTraceback> CapturedTraceback::gather(
    bool python,
    bool script,
    bool cpp) {
  auto r = std::make_shared<CapturedTraceback>();
  if (python) {
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
      r->frames_.emplace_back(
          CapturedTraceback::PyFrame{PyFrame_GetCode(f), PyFrame_GetLasti(f)});
      auto f_back = PyFrame_GetBack(f);
      Py_XDECREF(f);
      f = f_back;
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
  std::lock_guard lock(to_free_frames_mutex);
  to_free_frames.insert(to_free_frames.end(), frames_.begin(), frames_.end());
}

std::vector<py::object> symbolize(
    std::vector<CapturedTraceback*> to_symbolize) {
  py::str filename_s = "filename";
  py::str name_s = "name";
  py::str line_s = "line";

  std::unordered_map<void*, size_t> ip_to_frame_offset; // in all_cpp_frames
  std::vector<void*> all_cpp_ips;
  struct CPPFrame {
    enum Kind { PYTHON, JIT, REPORT } kind;
    py::object frame;
  };
  std::vector<CPPFrame> all_cpp_frames;

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
    auto all_frames = unwind::symbolize(all_cpp_ips);
    for (auto& f : all_frames) {
      py::dict frame;
      frame[filename_s] = f.filename;
      frame[name_s] = f.funcname;
      frame[line_s] = f.lineno;
      CPPFrame::Kind kind = CPPFrame::REPORT;
      if (f.funcname.find("PyEval_EvalFrame") != std::string::npos) {
        kind = CPPFrame::PYTHON;
      } else if (
          f.funcname.rfind("torch::jit::InterpreterStateImpl::run", 0) !=
          std::string::npos) {
        kind = CPPFrame::JIT;
      }
      all_cpp_frames.emplace_back(CPPFrame{kind, frame});
    }
  }

  std::unordered_map<CapturedTraceback*, py::list> cached_frames;

  std::vector<py::object> result;

  for (const auto& sc : to_symbolize) {
    auto it = cached_frames.find(sc);
    if (it == cached_frames.end()) {
      py::list frames;
      auto py_it = sc->frames_.begin();
      auto py_end = sc->frames_.end();

      bool jit_appended = false;

      auto torch = py::module::import("torch");
      py::object stack_frames_for_code;
      if (py::hasattr(torch, "_inductor")) {
        py::object inductor = torch.attr("_inductor");
        if (py::hasattr(inductor, "codecache")) {
          stack_frames_for_code = inductor.attr("codecache")
                                      .attr("PyCodeCache")
                                      .attr("stack_frames_for_code");
        }
      }

      auto append_python = [&](const CapturedTraceback::PyFrame& f) {
        py::dict frame;
        py::object filename =
            py::reinterpret_borrow<py::object>(f.code->co_filename);
        frame[filename_s] = filename;
        frame[name_s] = py::reinterpret_borrow<py::object>(f.code->co_name);
        auto lineno = PyCode_Addr2Line(f.code, f.lasti);
        frame[line_s] = lineno;
        frames.append(std::move(frame));

        // find all the additional frames associated with inductor generated
        // code
        if (stack_frames_for_code.ptr()) {
          py::object extra = stack_frames_for_code(filename, lineno);
          if (!extra.is_none()) {
            for (py::handle h : extra) {
              frames.append(h);
            }
          }
        }
      };

      auto append_jit = [&]() {
        if (jit_appended) {
          return;
        }
        jit_appended = true;
        for (const auto& f : sc->script_frames_) {
          py::dict frame;
          frame[name_s] = f.filename;
          auto flc = f.range.file_line_col();
          if (flc) {
            std::string filename;
            size_t line;
            size_t col;
            std::tie(filename, line, col) = *flc;
            frame[filename_s] = filename;
            frame[line_s] = line;
          } else {
            frame[filename_s] = "??";
            frame[line_s] = 0;
          }
          frames.append(std::move(frame));
        }
      };

      for (void* f : sc->cpp_frames_) {
        const CPPFrame& wf = all_cpp_frames.at(ip_to_frame_offset.at(f));
        if (wf.kind == CPPFrame::PYTHON) {
          if (py_it != py_end) {
            append_python(*py_it++);
          }
        } else if (wf.kind == CPPFrame::JIT) {
          append_jit();
        }
        frames.append(wf.frame);
      }

      // add frames if we otherwise haven't seen the C++ frame indicating where
      // it should go
      append_jit();

      for (; py_it != py_end; ++py_it) {
        append_python(*py_it);
      }
      it = cached_frames.insert({sc, frames}).first;
    }
    result.push_back(it->second);
  }
  return result;
}

} // namespace torch
