#include <torch/csrc/dynamo/cache_entry.h>
#include <torch/csrc/dynamo/cpp_shim.h>
#include <torch/csrc/dynamo/cpython_includes.h>
#include <torch/csrc/dynamo/debug_macros.h>
#include <torch/csrc/dynamo/eval_frame.h>
#include <torch/csrc/dynamo/eval_frame_cpp.h>
#include <torch/csrc/dynamo/framelocals_mapping.h>
#include <torch/csrc/utils/python_compat.h>

extern "C" {
extern PyObject* guard_complete_hook;
}

static constexpr const char* cache_lookup_profiler_str =
    "TorchDynamo Cache Lookup";

// Remember to update the type signature for DynamoCallbackFn.__call__ in
// torch/_dynamo/types.py if this function's signature changes.
static py::object dynamo_call_callback(
    py::handle callback,
    THP_EVAL_API_FRAME_OBJECT* _frame,
    FrameLocalsMapping* locals,
    CacheEntry* cache_entry,
    FrameState* frame_state) {
  THPPyInterpreterFrame* frame = THPPyInterpreterFrame_New(_frame);
  if (frame == nullptr) {
    throw std::runtime_error(
        "Dynamo failed to initialize CPython interpreter frame wrapper");
  }
  frame->locals = (PyObject*)framelocals_mapping_to_dict(locals);

  py::object cache_entry_obj = py::none();
  if (cache_entry) {
    cache_entry_obj = py::cast(cache_entry, py::return_value_policy::reference);
  }

  py::object result = callback(
      py::handle((PyObject*)frame), cache_entry_obj, py::handle(frame_state));
  Py_DECREF(frame);
  return result;
}

static py::handle _callback_from_action(
    py::handle callback,
    FrameAction action) {
  if (action == SKIP) {
    return Py_None;
  } else if (action == RUN_ONLY) {
    return Py_False;
  }
  return callback;
}

// frame and callback are borrowed references.
// Returns new reference.
PyObject* dynamo__custom_eval_frame(
    PyThreadState* tstate,
    THP_EVAL_API_FRAME_OBJECT* frame,
    int throw_flag,
    PyObject* callback_py) {
#if IS_PYTHON_3_11_PLUS
  DEBUG_TRACE(
      "begin %s %s %i %i",
      get_frame_name(frame),
      PyUnicode_AsUTF8(F_CODE(frame)->co_filename),
      F_CODE(frame)->co_firstlineno,
      _PyInterpreterFrame_LASTI(frame));
#else
  DEBUG_TRACE(
      "begin %s %s %i %i %i",
      get_frame_name(frame),
      PyUnicode_AsUTF8(F_CODE(frame)->co_filename),
      frame->f_lineno,
      frame->f_lasti,
      frame->f_iblock);
#endif

  if (throw_flag) {
    // When unwinding generators, eval frame is called with throw_flag ==
    // true.  Frame evaluation is supposed to continue unwinding by propagating
    // the exception.  Dynamo doesn't really know how to do this, nor does it
    // really want to do this, because there's unlikely any code to capture
    // (you're going to immediately quit out of the frame, perhaps running
    // some unwinding logic along the way).  So we just run the default
    // handler in this case.
    //
    // NB: A previous version of this patch returned NULL.  This is wrong,
    // because returning NULL is *different* from unwinding an exception.
    // In particular, you will not execute things like context manager
    // __exit__ if you just return NULL.
    //
    // NB: It's /conceivable/ that you might want to actually still call the
    // Dynamo callback when throw_flag == TRUE, to give Dynamo a chance to
    // do any stack unwinding code.  But this is not really useful because
    // (1) Dynamo doesn't actually know how to do stack unwinding, so it would
    // immediately skip the frame, and (2) even if it did, this would only
    // be profitable if there was tensor code in the unwinding code.  Seems
    // unlikely.
    DEBUG_TRACE("throw %s", get_frame_name(frame));
    return dynamo_eval_frame_default(tstate, frame, throw_flag);
  }

  py::handle callback(callback_py);

  // callback to run on recursively invoked frames
  py::handle recursive_callback = callback; // borrowed
  PyCodeObject* cached_code = nullptr; // borrowed
  const char* trace_annotation = "";
  PyObject* eval_result = nullptr; // strong reference

  // exit functions
  auto eval_default = [&]() {
    eval_frame_callback_set(recursive_callback.ptr());
    eval_result = dynamo_eval_frame_default(tstate, frame, throw_flag);
    if (!callback.is(recursive_callback)) {
      // NB: Only set the callback if it's different than the recursive
      // callback! Setting the callback is dangerous in the case that `frame`
      // also sets the eval frame callback. This happens in some functions in
      // eval_frame.py. These functions should be skipped with DEFAULT recursive
      // action, so we won't accidentally overwrite the callback.
      eval_frame_callback_set(callback.ptr());
    }
  };

  // NOTE: In 3.12+, the frame evaluation function (callee) is responsible for
  // clearing/popping the frame, meaning that unless we default evaluate the
  // original frame, we are responsible for clearing it - via
  // clear_old_frame_if_python_312_plus.
  auto eval_custom = [&]() {
    eval_frame_callback_set(recursive_callback.ptr());
    DEBUG_NULL_CHECK(cached_code);
    eval_result = dynamo_eval_custom_code(
        tstate, frame, cached_code, trace_annotation, throw_flag);
    if (!callback.is(recursive_callback)) {
      eval_frame_callback_set(callback.ptr());
    }
    clear_old_frame_if_python_312_plus(tstate, frame);
  };

  auto fail = [&]() { clear_old_frame_if_python_312_plus(tstate, frame); };

#if IS_PYTHON_3_12_PLUS
  if (tstate->tracing > 0) {
    eval_default();
    return eval_result;
  }
#endif

  ExtraState* extra = get_extra_state(F_CODE(frame));

  if (callback.is(py::bool_(false)) && extra == nullptr) {
    DEBUG_TRACE("skip (run only with empty cache) %s", get_frame_name(frame));
    eval_default();
    return eval_result;
  }

  // create cache
  if (extra == nullptr) {
    extra = init_and_set_extra_state(F_CODE(frame));
  }

  // Get recursive action
  FrameExecStrategy strategy = extra_state_get_exec_strategy(extra);
  recursive_callback =
      _callback_from_action(recursive_callback, strategy.recursive_action);

  // Skip this frame
  if (strategy.cur_action == SKIP) {
    DEBUG_TRACE("skip %s", get_frame_name(frame));
    eval_default();
    return eval_result;
  }

  // default and run-only mode require guard eval
  std::unique_ptr<FrameLocalsMapping> locals =
      std::make_unique<FrameLocalsMapping>(frame);
  PyObject* backend = get_backend(callback.ptr()); // borrowed

  // We don't run the current custom_eval_frame behavior for guards.
  // So we temporarily set the callback to Py_None to drive the correct behavior
  // in the shim.
  eval_frame_callback_set(Py_None);

  DEBUG_CHECK(PyDict_CheckExact(frame->f_globals));
  DEBUG_CHECK(PyDict_CheckExact(frame->f_builtins));

  _PytorchRecordFunctionState* rf =
      _pytorch_record_function_enter(cache_lookup_profiler_str);
  PyObject* maybe_cached_code = nullptr;
  lookup(
      extra,
      locals.get(),
      backend,
      &maybe_cached_code,
      &trace_annotation,
      is_skip_guard_eval_unsafe);
  _pytorch_record_function_exit(rf);

  // A callback of Py_False indicates "run only" mode, the cache is checked,
  // but we never compile.
  bool run_only =
      strategy.cur_action == RUN_ONLY || callback.is(py::bool_(false));
  if (run_only) {
    DEBUG_TRACE("In run only mode %s", get_frame_name(frame));
  }

  if (maybe_cached_code == nullptr) {
    // guard eval failed, keep propagating
    fail();
    return eval_result;
  }

  // NB: We only do guard collectives when there are any compiled code entries
  // at all; these reduces overtriggering and we don't need to do guard
  // collectives the very first time we've seen a frame
  // TODO: We could also check if we had just created extra for the first
  // time?  Not too sure the best condition for extra->cache_entry_list
  if (guard_complete_hook != nullptr && !extra->cache_entry_list.empty()) {
    py::handle guard_complete_hook_handle(guard_complete_hook);
    // False means force compilation (someone cache missed)
    py::object res = guard_complete_hook_handle(maybe_cached_code != Py_None);
    if (!py::cast<bool>(res)) {
      maybe_cached_code = Py_None; // NB: non-owning
    }
  }

  if (maybe_cached_code != Py_None) {
    cached_code = (PyCodeObject*)maybe_cached_code;
    // used cached version
    DEBUG_TRACE("cache hit %s", get_frame_name(frame));
    eval_custom();
    return eval_result;
  }

  // cache miss
  DEBUG_TRACE("cache miss %s", get_frame_name(frame));
  if (is_skip_guard_eval_unsafe) {
    PyErr_SetString(
        PyExc_RuntimeError,
        "Recompilation triggered with skip_guard_eval_unsafe stance. "
        "This usually means that you have not warmed up your model "
        "with enough inputs such that you can guarantee no more recompilations.");
    fail();
    return eval_result;
  }

  if (run_only) {
    eval_default();
    return eval_result;
  }

  // call callback
  CacheEntry* cache_entry = extract_cache_entry(extra);
  FrameState* frame_state = extract_frame_state(extra);
  py::object callback_result;
  FrameExecStrategy new_strategy;
  bool apply_to_code = false;
  PyObject* guarded_code = nullptr;
  try {
    callback_result = dynamo_call_callback(
        callback, frame, locals.get(), cache_entry, frame_state);
    new_strategy =
        callback_result.attr("frame_exec_strategy").cast<FrameExecStrategy>();
    apply_to_code = callback_result.attr("apply_to_code").cast<bool>();
    guarded_code = callback_result.attr("guarded_code").ptr();
  } catch (py::error_already_set& e) {
    // internal exception, returning here will leak the exception into user
    // code this is useful for debugging -- but we dont want it to happen
    // outside of testing NB: we intentionally DO NOT re-enable custom
    // behavior to prevent cascading failure from internal exceptions.  The
    // upshot is if Dynamo barfs, that's it for Dynamo, even if you catch the
    // exception inside the torch.compile block we won't try to Dynamo
    // anything else.
    fail();
    e.restore();
    return eval_result;
  }

  // recursive frame action
  if (strategy.recursive_action == DEFAULT) {
    // old recursive action overrides new recursive action
    recursive_callback = _callback_from_action(
        recursive_callback, new_strategy.recursive_action);
  }

  // possibly apply frame strategy to future frames with same code object
  if (apply_to_code) {
    if (new_strategy.cur_action != DEFAULT) {
      DEBUG_TRACE("create action: %d\n", new_strategy.cur_action);
    }
    if (new_strategy.recursive_action != DEFAULT) {
      DEBUG_TRACE(
          "create recursive action: %d\n", new_strategy.recursive_action);
    }
    extra_state_set_exec_strategy(extra, new_strategy);
  }

  if (guarded_code != Py_None) {
    DEBUG_TRACE("create cache %s", get_frame_name(frame));

    // NB: We could use extract_cache_entry to get the cache_entry, but
    // extract_cache_entry returns a borrowed reference. Modifying a borrowed
    // reference seems wrong. Therefore, we directly access the
    // extra->cache_entry. extra won't be NULL here.
    CacheEntry* new_cache_entry =
        create_cache_entry(extra, guarded_code, backend);

    // Update the existing cache_entry on the extra object. This extra object
    // is sitting on the extra scratch space, we are just changing the
    // cache_entry ptr. As a result, extra now becomes the owner of CacheEntry
    // object. This will be cleaned up when set_extra_state is called.
    // Re-enable custom behavior
    cached_code = CacheEntry_get_code(new_cache_entry),
    trace_annotation = CacheEntry_get_trace_annotation(new_cache_entry);
    eval_custom();
  } else {
    eval_default();
  }
  return eval_result;
}

PyObject* set_code_exec_strategy(PyObject* dummy, PyObject* args) {
  PyObject* code_obj = nullptr;
  PyObject* strategy_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &code_obj, &strategy_obj)) {
    return nullptr;
  }
  if (!PyCode_Check(code_obj)) {
    PyErr_SetString(PyExc_TypeError, "expected a code object");
    return nullptr;
  }

  PyCodeObject* code = (PyCodeObject*)code_obj;
  ExtraState* extra = get_extra_state(code);
  if (extra == nullptr) {
    extra = init_and_set_extra_state(code);
  }

  FrameExecStrategy strategy =
      py::handle(strategy_obj).cast<FrameExecStrategy>();

  extra_state_set_exec_strategy(extra, strategy);
  Py_RETURN_NONE;
}

void skip_code_recursive(PyCodeObject* code) {
  ExtraState* extra = get_extra_state(code);
  if (extra == nullptr) {
    extra = init_and_set_extra_state(code);
  }

  FrameExecStrategy strategy =
      FrameExecStrategy{FrameAction::SKIP, FrameAction::SKIP};
  extra_state_set_exec_strategy(extra, strategy);
}
