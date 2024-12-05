// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/csrc/utils/python_compat.h>


// Many APIs have changed/don't exist anymore/untested on 3.13
#if IS_PYTHON_3_13_PLUS

#include "dim.h"

// Re-enable this some day
PyObject* Dim_init() {
    PyErr_SetString(PyExc_RuntimeError, "First class dim doesn't work with python 3.13");
    return nullptr;
}

#else

#include "minpybind.h"
#include <frameobject.h>
#include <opcode.h>
#include <utility>
#include <new>
#include <iostream>
#include <vector>
//#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/Export.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/ATen.h>
#include <memory>
#include "arena.h"
#include "dim.h"
#include "python_variable_simple.h"

#if IS_PYTHON_3_11_PLUS

#define Py_BUILD_CORE
#include "internal/pycore_opcode.h"
#undef Py_BUILD_CORE
#endif

// C++ API functions for objects to
// * construct the object, returning a ref-counted handle
// * The actual API, with methods that take/return C-typed values

// extend minpybind.h to include
// * typed handles so that -> can get to their raw API
// * object/handle distinction for the typed handles

// class Dim: ---------------
mpy::handle torch_Tensor___mul__;
mpy::handle _Tensor;
mpy::handle _Tensor_sum;
mpy::handle NamedTuple;
mpy::dict_view pointwise;
mpy::handle torch_Tensor_expand;
binaryfunc THPVariable_getitem;
objobjargproc THPVariable_setitem;
mpy::handle no_slice;
PyTypeObject* torch_Tensor;
mpy::handle torch_Tensor_copy_;
mpy::handle torch_Tensor_split;
bool pointwise_optimize = true;
PyTypeObject* DimType = nullptr;

PyObject* Tensor_getitem(PyObject* self, PyObject* index);
int Tensor_setitem(PyObject* self, PyObject* index, PyObject* value);

namespace{
void maybeInitializeGlobals() {
    // globals that depend on the python dim library,
    // which we can't lookup until we finish initializing the _C module
    if (_Tensor.ptr()) {
        return;
    }
    auto dim = mpy::import("functorch.dim");
    _Tensor = dim.attr("_Tensor");
    pointwise = dim.attr("pointwise");
    _Tensor_sum = _Tensor.attr("sum");
    DimType = (PyTypeObject*) mpy::import("functorch.dim").attr("Dim").ptr();
}

void replaceMappingIfMatches(mpy::handle tp) {
    auto T = (PyTypeObject*) tp.ptr();
    bool recurse = false;
    if (T->tp_as_mapping->mp_subscript == THPVariable_getitem) {
        T->tp_as_mapping->mp_subscript = Tensor_getitem;
        recurse = true;
    }
    if (T->tp_as_mapping->mp_ass_subscript == THPVariable_setitem) {
        T->tp_as_mapping->mp_ass_subscript = Tensor_setitem;
        recurse = true;
    }
    if (recurse) {
        auto result = tp.attr("__subclasses__").call();
        mpy::list_view lv(result);
        for (auto i : lv.enumerate()) {
            replaceMappingIfMatches(lv[i]);
        }
    }
}

void initializeGlobals(Arena & A) {
    auto torch = mpy::import("torch");
    torch_Tensor = (PyTypeObject*) torch.attr("Tensor").ptr();
    torch_Tensor___mul__ = torch.attr("Tensor").attr("__mul__");

    torch_Tensor_expand = torch.attr("_C").attr("TensorBase").attr("expand");
    torch_Tensor_split = torch.attr("_C").attr("TensorBase").attr("split");
    torch_Tensor_copy_ = torch.attr("Tensor").attr("copy_");
    auto py_TensorBase = torch.attr("_C").attr("TensorBase");
    auto TensorBase = (PyTypeObject*) py_TensorBase.ptr();
    THPVariable_getitem = TensorBase->tp_as_mapping->mp_subscript;
    THPVariable_setitem = TensorBase->tp_as_mapping->mp_ass_subscript;
    NamedTuple = mpy::import("typing").attr("NamedTuple");
    no_slice = PySlice_New(NULL, NULL, NULL);

}

mpy::handle DimensionBindError_;
mpy::handle DimensionBindError() {
    if(!DimensionBindError_.ptr()) {
        DimensionBindError_ = mpy::import("functorch.dim").attr("DimensionBindError");
    }
    return DimensionBindError_;
}

static int64_t n_dims_created = 65;

struct Dim : public mpy::base<Dim> {
    int64_t level_; // for stable comparisons in prototype
    mpy::object name_;
    Dim()
    : level_(n_dims_created++) {}
    void init(mpy::object name, int64_t s = -1) {
        name_ = std::move(name);
        size_ = s;
    }

    static bool check_exact(mpy::handle v) {
        return Py_TYPE(v.ptr()) == DimType;
    }

    int64_t size() const {
        if (size_ == -1) {
            mpy::raise_error(PyExc_ValueError, "dimension %S is unbound", name_.ptr());
        }
        return size_;
    }
    void set_size(int64_t v) {
        if (size_ == -1) {
            size_ = v;
        } else if(size_ != v) {
            mpy::raise_error(DimensionBindError(), "Dim '%R' previously bound to a dimension of size %lld cannot bind to a dimension of size %lld", this, this->size_, v);
        }
    }
    bool is_bound() const {
        return size_ != -1;
    }
    static mpy::obj<Dim> create(mpy::object name, int64_t s = -1) {
        if (!DimType) {
            maybeInitializeGlobals();
        }
        auto r = Dim::alloc(DimType);
        r->init(std::move(name), s);
        return r;
    }
    static PyTypeObject Type;
    const at::Tensor& range() {
        if (!range_.defined()) {
            range_ = at::arange(size());
        }
        return range_;
    }
    const at::Tensor& batchtensor() {
        if (!batchtensor_.defined()) {
            batchtensor_ = at::functorch::addBatchDim(range(), 0, level_);
        }
        return batchtensor_;
    }
private:
    int64_t size_{-1};
    at::Tensor range_;
    at::Tensor batchtensor_;
};


struct DimEntry {
    // union of either a negative number indicating which dimension this is from the rhs,
    // or a pointer to a first-class dimension.
    // pointers do not have their highest bit set, so checking the number is negative tells us
    // that it is not a dim.
    bool is_positional() const {
        return data_ < 0;
    }
    bool is_none() const {
        return data_ == 0;
    }
    int64_t position() const {
        return data_;
    }
    mpy::hdl<Dim> dim() const {
        Dim* result;
        std::memcpy(&result, &data_, sizeof(Dim*));
        return mpy::hdl<Dim>(result);
    }

    DimEntry()
    : data_(0) {}

    DimEntry(int64_t pos)
    : data_(pos) {
        AT_ASSERT(pos < 0);
    }
    DimEntry(mpy::hdl<Dim> d) {
       std::memcpy(&data_, &d, sizeof(int64_t));
    }
    bool operator==(const DimEntry& rhs) const {
        return data_ == rhs.data_;
    }
private:
    int64_t data_;
};

// Dim wrapper methods
DimEntry _wrap_dim(mpy::handle d, size_t N, bool keepdim) {
    if (Dim::check(d)) {
        if (keepdim) {
            mpy::raise_error(PyExc_ValueError, "cannot preserve first-class dimensions with keepdim=True");
        }
        return Dim::unchecked_wrap(d);
    } else if (mpy::is_int(d)) {
        auto i = mpy::to_int(d);
        while (i >= 0) {
            i -= N;
        }
        return i;
    } else {
        return DimEntry();
    }
}


int Dim_init(mpy::hdl<Dim> self, PyObject *args, PyObject *kwds) {
    PY_BEGIN
    static constexpr const char* kwlist[] = {"name", "size", nullptr};
    mpy::handle name;
    mpy::handle size = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", const_cast<char **>(kwlist), &name, &size)) {
        return -1;
    }
    self->init(mpy::object::borrow(name), (size.ptr() && !mpy::is_none(size)) ? mpy::to_int(size) : -1);
    return 0;
    PY_END(-1)
}

PyObject* Dim_repr(Dim* self) {
    PY_BEGIN
    mpy::object name = (self->name_.ptr()) ? self->name_ : mpy::unicode_from_string("<uninitialized dim>");
    return name.release();
    PY_END(nullptr)
}


PyObject* Dim_getsize(Dim* self, void*) {
    PY_BEGIN
    return mpy::from_int(self->size()).release();
    PY_END(nullptr)
}

int Dim_setsize(Dim* self, PyObject* size, void*) {
    PY_BEGIN
    self->set_size(mpy::to_int(size));
    return 0;
    PY_END(-1)
}

PyObject* Dim_getis_bound(Dim* self, void*) {
    return PyBool_FromLong(self->is_bound());
}

PyObject* Dim_getlevel(Dim* self, void*) {
    return PyLong_FromLong(self->level_);
}

PyObject* Dim_get_levels(Dim* self, void*) {
    mpy::tuple t(1);
    t.set(0, mpy::object::borrow(self->ptr()));
    return t.release();
}

PyObject* Dim_get_has_device(Dim* self, void*) {
    Py_RETURN_FALSE;
}

PyObject* Dim_get_tensor(Dim* self, void*) {
    return THPVariable_Wrap(self->range());
}

PyObject* Dim_get_batchtensor(Dim* self, void*) {
    return THPVariable_Wrap(self->batchtensor());
}


PyGetSetDef Dim_getsetters[] = {
    {"size", (getter) Dim_getsize, (setter) Dim_setsize,
     "Dimension size", NULL},
    {"is_bound", (getter) Dim_getis_bound, NULL, "is_bound", NULL},
    {"_level", (getter) Dim_getlevel, NULL, "_level", NULL},
    {"_levels", (getter) Dim_get_levels, NULL, "_levels", NULL},
    {"_has_device", (getter) Dim_get_has_device, NULL, "_has_device", NULL},
    {"_tensor", (getter) Dim_get_tensor, NULL, "_tensor", NULL},
    {"_batchtensor", (getter) Dim_get_batchtensor, NULL, "_batchtensor", NULL},
    {"ndim", (getter) [](PyObject* self, void*) -> PyObject* { return mpy::from_int(1).release(); }, NULL, "ndim", NULL},
    {NULL}  /* Sentinel */
};
}
PyTypeObject Dim::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_C.Dim",               /* tp_name */
    sizeof(Dim),               /* tp_basicsize */
    0,                              /* tp_itemsize */
    Dim::dealloc_stub,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    (reprfunc)Dim_repr,           /* tp_repr */
    0,                 /* tp_as_number */
    0,                              /* tp_as_sequence */
    0,                              /* tp_as_mapping */
    0,      /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,  /* tp_flags */
    "Dim Object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,  /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    0,                              /* tp_methods */
    0,                              /* tp_members */
    Dim_getsetters,                 /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc)(void*)static_cast<int(*)(mpy::hdl<Dim>,PyObject*,PyObject*)>(Dim_init),      /* tp_init */
    0,                              /* tp_alloc */
    Dim::new_stub,                      /* tp_new */
};

// class DimList ------------

struct DimList : public mpy::base<DimList> {
    mpy::object name_;
    std::vector<mpy::obj<Dim>> dims_;
    static PyTypeObject Type;
    void init(mpy::object name) {
        name_ = std::move(name);
    }
    void set_dims(std::vector<mpy::obj<Dim>> dims) {
        bound_ = true;
        dims_ = std::move(dims);
    }
    bool is_bound() {
        return bound_;
    }
    void bind_len(int64_t size) {
        if (bound_) {
            int64_t b_size = dims_.size();
            if (b_size != size) {
                mpy::raise_error(DimensionBindError(), "Dimlist has size %lld but it is being bound to size %d", b_size, size);
            }
        } else {
            bound_ = true;
            dims_.resize(size);
            for (Py_ssize_t i = 0; i < size; ++i) {
                dims_[i] = Dim::create(mpy::unicode_from_format("%S%i", name_.ptr(), (int)i));
            }
        }
    }
    int64_t size() const {
        if (!bound_) {
            mpy::raise_error(DimensionBindError(), "DimList not bound");
        }
        return dims_.size();
    }
    void set_bound(bool b) {
        bound_ = b;
    }
private:
    bool bound_ = false;
};


static int DimList_init(DimList *self, PyObject *args, PyObject *kwds);

static PyObject* DimList_repr(DimList* self) {
    PY_BEGIN
    if (self->is_bound()) {
        size_t size = self->dims_.size();
        mpy::tuple t(size);
        for(size_t i = 0; i < size; ++i) {
            t.set(i, self->dims_[i]);
        }
        return mpy::repr(t).release();
    } else if(!mpy::is_none(self->name_)) {
        return mpy::unicode_from_format("*%S", self->name_.ptr()).release();
    } else {
        return mpy::unicode_from_string("<unbound_dimlist>").release();
    }
    PY_END(nullptr)
}

static PyObject* DimList_bind(DimList *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    mpy::handle sizes;
    static const char * const _keywords[] = {"sizes", nullptr};
#if IS_PYTHON_3_12_PLUS
    static _PyArg_Parser parser = {
      .format = "O",
      .keywords = _keywords,
      .kwtuple = 0};
#else
static _PyArg_Parser parser = {"O", _keywords, 0};
#endif
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, &sizes)) {
        return nullptr;
    }
    if (!mpy::is_sequence(sizes)) {
        mpy::raise_error(PyExc_ValueError, "expected a sequence");
    }
    mpy::sequence_view seq = sizes;
    auto size = seq.size();
    self->bind_len(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
        self->dims_[i]->set_size(mpy::to_int(seq[i]));
    }
    Py_RETURN_NONE;
    PY_END(nullptr)
}

static PyObject* DimList_bind_len(DimList *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    int size;
    static const char * const _keywords[] = {"N", nullptr};
#if IS_PYTHON_3_12_PLUS
    static _PyArg_Parser parser = {
            .format = "i",
            .keywords =  _keywords,
            .kwtuple = 0};
#else
    static _PyArg_Parser parser = {"i", _keywords, 0};
#endif
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, &size)) {
        return nullptr;
    }
    self->bind_len(size);
    Py_RETURN_NONE;
    PY_END(nullptr)
}

static PyMethodDef DimList_methods[] = {
    {"bind", (PyCFunction)(void*) DimList_bind, METH_FASTCALL | METH_KEYWORDS},
    {"bind_len", (PyCFunction)(void*) DimList_bind_len, METH_FASTCALL | METH_KEYWORDS},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


static Py_ssize_t DimList_len(DimList* self) {
    PY_BEGIN
    return self->size();
    PY_END(-1)
}

static PyObject * DimList_item(DimList* self, Py_ssize_t idx) {
    PY_BEGIN
    if (!self->is_bound()) {
        mpy::raise_error(DimensionBindError(), "DimList not bound");
    }
    if (idx < 0 || (size_t) idx >= self->dims_.size()) {
        mpy::raise_error(PyExc_IndexError, "index out of bounds");
    }
    mpy::object r = self->dims_[idx];
    return r.release();
    PY_END(nullptr)
}

PySequenceMethods DimList_seq {
    (lenfunc) DimList_len, //lenfunc sq_length;
    0, //binaryfunc sq_concat;
    0, //ssizeargfunc sq_repeat;
    (ssizeargfunc) DimList_item, //ssizeargfunc sq_item;
    0, //void *was_sq_slice;
    0, //ssizeobjargproc sq_ass_item;
    0, //void *was_sq_ass_slice;
    0, //objobjproc sq_contains;

    0, //binaryfunc sq_inplace_concat;
    0, //ssizeargfunc sq_inplace_repeat;
};

static PyObject* DimList_getis_bound(DimList* self, void*) {
    return PyBool_FromLong(self->is_bound());
}

static PyGetSetDef DimList_getsetters[] = {
    {"is_bound", (getter) DimList_getis_bound, NULL, "is_bound", NULL},
    {NULL}  /* Sentinel */
};


static PyObject* DimList_subscript(DimList* self, mpy::handle idx) {
    PY_BEGIN
    if (mpy::is_int(idx)) {
        return DimList_item(self, mpy::to_int(idx));
    } else if (mpy::is_slice(idx)) {
        if (!self->is_bound()) {
            mpy::raise_error(DimensionBindError(), "DimList not bound");
        }
        mpy::slice_view s(idx, self->dims_.size());
        mpy::tuple r(s.slicelength);
        for (Py_ssize_t i = s.start, j = 0; i < s.stop; i += s.step) {
            r.set(j++,  self->dims_[i]);
        }
        return r.release();
    } else {
        mpy::raise_error(PyExc_ValueError, "expected an int or a slice");
        return nullptr;
    }
    PY_END(nullptr)
}

PyMappingMethods DimList_mapping = {
    0, //lenfunc mp_length;
    (binaryfunc)(void*) DimList_subscript, //binaryfunc mp_subscript;
    0, //objobjargproc mp_ass_subscript;
};



PyTypeObject DimList::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_C.DimList",               /* tp_name */
    sizeof(DimList),               /* tp_basicsize */
    0,                              /* tp_itemsize */
    DimList::dealloc_stub,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    (reprfunc)DimList_repr,           /* tp_repr */
    0,                 /* tp_as_number */
    &DimList_seq,                 /* tp_as_sequence */
    &DimList_mapping,             /* tp_as_mapping */
    0,      /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    0,                              /* tp_flags */
    "DimList Object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,                              /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    DimList_methods,                /* tp_methods */
    0,                              /* tp_members */
    DimList_getsetters,             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    (initproc) DimList_init,            /* tp_init */
    0,                              /* tp_alloc */
    DimList::new_stub,                      /* tp_new */
};

static int DimList_init(DimList *self, PyObject *args, PyObject *kwds) {
    PY_BEGIN
    static constexpr const char* kwlist[] = {"len_or_dims", "name", nullptr};
    mpy::handle len_or_dims = nullptr;
    PyObject* name = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", const_cast<char**>(kwlist), &len_or_dims, &name)) {
        return -1;
    }
    self->init(mpy::object::borrow(name ? name : Py_None));
    if (len_or_dims.ptr()) {
        if(mpy::is_int(len_or_dims)) {
            self->bind_len(mpy::to_int(len_or_dims));
        } else if (mpy::is_sequence(len_or_dims)) {
            mpy::sequence_view s(len_or_dims);
            std::vector<mpy::obj<Dim>> dims;
            size_t size = s.size();
            dims.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                auto r = s[i];
                if (mpy::is_int(r)) {
                    dims.emplace_back(Dim::create(mpy::unicode_from_format("%S%i", self->name_.ptr(), (int)i),  mpy::to_int(r)));
                } else {
                    dims.emplace_back(Dim::wrap(r));
                }
            }
            self->set_dims(std::move(dims));
        } else {
            PyErr_Format(PyExc_ValueError, "expected a length or a sequence of dimensions");
            return -1;
        }
        return 0;
    }
    return 0;
    PY_END(-1);
}

// Tensor -----------------------------

PyTypeObject* TensorType = nullptr; // the python wrapper type.
mpy::object run_torch_function(Arena &A, mpy::handle orig, mpy::vector_args args, bool is_pointwise);

namespace{

at::Tensor _add_batch_dims(Arena& A, at::Tensor t, Slice<DimEntry> levels_) {
    auto levels = Slice<DimEntry>();
    levels.extend(A, levels_);
    while (true) {
        int64_t min_real_index = -1;
        int64_t min_index = -1;
        int64_t min_value = INT_MAX;
        int64_t i = 0;
        int64_t r = 0;
        for (auto l : levels) {
            if (!l.is_none()) {
                if (!l.is_positional() && l.dim()->level_ < min_value) {
                    min_value = l.dim()->level_;
                    min_index = i;
                    min_real_index = r;
                }
                ++i;
            }
            ++r;
        }
        if (min_index == -1) {
            return t;
        }
        auto t2 = at::functorch::addBatchDim(std::move(t), min_index, min_value);
        t = std::move(t2);
        levels[min_real_index] = DimEntry();
    }
}



struct DelayedOperator {
    DelayedOperator(mpy::object o, mpy::vector_args a)
    : orig(std::move(o)), args(a) {
        auto all = a.size();
        // this will outlive the call so
        // take ownership of temporaries
        // in vector args
        auto buf = new mpy::handle[all];
        memcpy(buf, args.args, sizeof(mpy::handle)*all);
        args.args = buf;
        for (auto i : args.enumerate_all()) {
            Py_INCREF(args.args[i].ptr());
        }
        Py_XINCREF(args.kwnames.ptr());
    }
    ~DelayedOperator() {
        for (auto i : args.enumerate_all()) {
            Py_DECREF(args[i].ptr());
        }
        if (args.has_keywords()) {
            Py_XDECREF(args.kwnames.ptr());
        }
        delete [] args.args;
    }
    mpy::object orig;
    mpy::vector_args args;
};

void free_levels_dims(Slice<DimEntry> levels) {
    for(auto e : levels) {
        if (!e.is_positional()) {
            mpy::object::steal(e.dim());
        }
    }
}
}

struct Tensor : public mpy::base<Tensor> {
private:
    at::Tensor tensor_;
    at::Tensor batchtensor_;
    OwnedSlice<DimEntry> levels_;
    bool has_device_;
    std::unique_ptr<DelayedOperator> delayed_;
public:

    at::Tensor& tensor(Arena& A) {
        if (C10_UNLIKELY(!tensor_.defined())) {
            AT_ASSERT(delayed_);
            auto t = Tensor::wrap(run_torch_function(A, delayed_->orig, delayed_->args, true));
            tensor_ = t->tensor(A);
            delayed_.reset();
            // don't force creation of batch tensor if it wasn't alreay provided.
            batchtensor_ = t->batchtensor_;
            AT_ASSERT(levels() == t->levels());
        }
        return tensor_;
    }
    at::Tensor& batchtensor(Arena& A) {
        if (C10_UNLIKELY(!batchtensor_.defined())) {
            batchtensor_ = _add_batch_dims(A, tensor(A), levels_.slice());
        }
        return batchtensor_;
    }
    Slice<DimEntry> levels() {
        return levels_.slice();
    }
    bool has_device() {
        return has_device_;
    }
    DelayedOperator* delayed() {
        return delayed_.get();
    }
    static PyTypeObject Type;

    static bool check_exact(mpy::handle v) {
       return Py_TYPE(v.ptr()) == TensorType;
    }


    static mpy::obj<Tensor> create() {
        if (!TensorType) {
            TensorType = (PyTypeObject*) mpy::import("functorch.dim").attr("Tensor").release();
        }
        return Tensor::alloc(TensorType);
    }
    void capture_levels(Slice<DimEntry> levels) {
        // grab ownership of the dims inside levels
        for (auto l : levels) {
            if (!l.is_positional()) {
                mpy::object::borrow(l.dim()).release();
            }
        }
        levels_.set(levels, free_levels_dims);
    }
    static mpy::object from_positional(Arena & A, at::Tensor tensor, Slice<DimEntry> levels, bool has_device);
    static mpy::obj<Tensor> create_delayed(mpy::object op, mpy::vector_args args, Slice<DimEntry> levels, bool has_device);
    friend struct EnableAllLayers;
};

namespace{
// version in header does a unnecessary refcount +/-
at::functorch::BatchedTensorImpl* maybeGetBatchedImpl(const at::Tensor& tensor) {
    if (at::functorch::isBatchedTensor(tensor)) {
        return static_cast<at::functorch::BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
    }
    return nullptr;
}

TensorRef unchecked_tensor_from(mpy::handle p) {
    auto v = (THPVariable*) p.ptr();
    return TensorRef(*v->cdata);
}

static int64_t ndim_of_levels(Slice<DimEntry> levels) {
    int64_t r = 0;
    for (auto l : levels) {
        if (l.is_positional()) {
            ++r;
        }
    }
    return r;
}

struct TensorInfo {
    TensorRef tensor;
    Slice<DimEntry> levels;
    bool has_device;
    TensorRef batchedtensor;
    int64_t ndim() const {
        return ndim_of_levels(levels);
    }
    operator bool() const {
        return tensor;
    }

    static TensorInfo create(Arena& A, mpy::handle h, bool ensure_batched=true, bool ensure_present=true) {
        if (Tensor::check_exact(h)) {
            auto t = Tensor::unchecked_wrap(h);
            return TensorInfo {t->tensor(A), t->levels(), t->has_device(), ensure_batched ? t->batchtensor(A) : TensorRef()};
        } else if (Dim::check_exact(h)) {
            auto d = Dim::unchecked_wrap(h);
            return TensorInfo {d->range(), Slice<DimEntry>(A, DimEntry(d)), false, ensure_batched ? d->batchtensor() : TensorRef()};
        } else if (THPVariable_Check(h.ptr())) {
            TensorRef t = unchecked_tensor_from(h);
            Slice<DimEntry> levels;
            for (auto i : irange(-t->dim(), 0)) {
                levels.append(A, i);
            }
            return TensorInfo {t, levels, true, t};
        } else {
            if (ensure_present) {
                mpy::raise_error(PyExc_ValueError, "expected a tensor object");
            }
            return TensorInfo {};
        }
    }


};

static PyObject* py_Tensor_from_positional(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    #define ARGS(_) _(mpy::handle, tensor) _(mpy::handle, py_levels) _(int, has_device)
    MPY_PARSE_ARGS_KWNAMES("OOp", ARGS)
    #undef ARGS

    if (!THPVariable_Check(tensor.ptr())) {
        mpy::raise_error(PyExc_ValueError, "_tensor is not a Tensor?");
    }

    Slice<DimEntry> levels;
    mpy::sequence_view sq(py_levels);
    for (auto i : sq.enumerate()) {
        mpy::object v = sq[i];
        if (mpy::is_int(v)) {
            auto vi = mpy::to_int(v);
            levels.append(A, vi);
        } else {
            auto dim = Dim::wrap(std::move(v));
            mpy::hdl<Dim> hdim = dim;
            levels.append(A, hdim);
        }
    }
    return Tensor::from_positional(A, THPVariable_Unpack(tensor.ptr()), levels, has_device != 0).release();
    PY_END(nullptr)
}
}

mpy::object Tensor::from_positional(Arena & A, at::Tensor tensor, Slice<DimEntry> levels, bool has_device) {
    size_t seen_dims = 0;
    int last = 0;
    //auto sz = tensor.sizes();
    for (auto i : levels.enumerate()) {
        auto l = levels[i];
        if (l.is_positional()) {
            AT_ASSERT(last == 0 || last + 1 == l.position());
            last = l.position();
        } else {
            mpy::object::borrow(l.dim()).release();
            //AT_ASSERT(sz[i] == l.dim()->size());
            ++seen_dims;
        }
    }
    AT_ASSERT(last == 0 || last == -1);
    if (!seen_dims) {
        return mpy::object::steal(THPVariable_Wrap(tensor));
    }

    mpy::obj<Tensor> self = Tensor::create();
    self->tensor_ = std::move(tensor);
    AT_ASSERT(self->tensor_.dim() == levels.size());
    self->levels_.set(levels, free_levels_dims);
    self->has_device_ = has_device;
    mpy::object r = std::move(self);
    return r;
}


mpy::obj<Tensor> Tensor::create_delayed(mpy::object op, mpy::vector_args args, Slice<DimEntry> levels, bool has_device) {
    mpy::obj<Tensor> self = Tensor::create();
    self->capture_levels(levels);
    self->has_device_ = has_device;
    self->delayed_ = std::make_unique<DelayedOperator>(std::move(op), args);
    return self;
}

namespace{
mpy::list slice_to_list(Slice<mpy::handle> h) {
    mpy::list lst(h.size());
    for (auto i : h.enumerate()) {
        lst.set(i, mpy::object::borrow(h[i]));
    }
    return lst;
}

mpy::tuple slice_to_tuple(Slice<mpy::handle> h) {
    mpy::tuple lst(h.size());
    for (auto i : h.enumerate()) {
        lst.set(i, mpy::object::borrow(h[i]));
    }
    return lst;
}

enum UType {
    U_ELEM,
    U_TUPLE_LIKE,
    U_DICT,
};

struct Unflatten {
    mpy::object operator()(Slice<mpy::handle>& elements) {
        mpy::object r;
        switch (type) {
            case U_ELEM: {
                r = mpy::object::borrow(elements[0]);
                elements = elements.slice(1);
            } break;
            case U_TUPLE_LIKE: {
                mpy::tuple tup(children.size());
                for (auto i : children.enumerate()) {
                    tup.set(i, children[i](elements));
                }
                r = obj.call(tup);
            } break;
            case U_DICT: {
                r = mpy::object::checked_steal(PyDict_New());
                mpy::dict_view rv(r);
                mpy::dict_view d(obj);
                Py_ssize_t pos = 0;
                mpy::handle k, v;
                for (int i = 0; d.next(&pos, &k, &v); ++i) {
                    rv.set(k, children[i](elements));
                }
            } break;
        }
        return r;
    }
    UType type;
    mpy::handle obj;
    Slice<Unflatten> children;
};

Unflatten tree_flatten(Arena& A, mpy::handle agg, Slice<mpy::handle>& flat_elements) {
    Slice<Unflatten> c;
    UType utype;
    mpy::handle obj;
    if (mpy::list_view::check(agg)) {
        obj = agg.type();
        utype = U_TUPLE_LIKE;
        mpy::list_view l(agg);
        for (auto i : l.enumerate()) {
            c.append(A, tree_flatten(A, l[i], flat_elements));
        }
    } else if (mpy::tuple_view::check(agg)) {
        obj = agg.type();
        utype = U_TUPLE_LIKE;
        // includes named tuples
        mpy::tuple_view l(agg);
        for (auto i : l.enumerate()) {
            c.append(A, tree_flatten(A, l[i], flat_elements));
        }
    } else if (mpy::dict_view::check(agg)) {
        utype = U_DICT;
        mpy::dict_view d(agg);
        obj = agg;
        Py_ssize_t pos = 0;
        mpy::handle k, v;
        while (d.next(&pos, &k, &v)) {
            c.append(A, tree_flatten(A, v, flat_elements));
        }
    } else {
        utype = U_ELEM;
        flat_elements.append(A, agg);
    }
    return Unflatten {utype, obj, c};
}

struct UnflattenVectorArgs {
    mpy::vector_args operator()(Arena& A, Slice<mpy::handle>& elements) {
        if (!had_nested) {
            auto args = elements.begin();
            elements = Slice<mpy::handle>();
            return mpy::vector_args(args, nargs, kwnames);
        }
        Slice<mpy::handle> args;
        for (auto u : children) {
            args.append(A, A.autorelease(u(elements)));
        }
        return mpy::vector_args(args.begin(), nargs, kwnames);
    }
    Slice<Unflatten> children;
    Py_ssize_t nargs;
    mpy::handle kwnames;
    bool had_nested;
};

UnflattenVectorArgs tree_flatten(Arena& A, mpy::vector_args args, Slice<mpy::handle>& flat_elements) {
    UnflattenVectorArgs r;
    r.kwnames = args.kwnames;
    r.nargs = args.nargs;
    r.had_nested = false;
    auto N = args.size();
    for(auto i : irange(N)) {
        auto typ = Py_TYPE(args[i].ptr());
        // fast checks that this thing isn't something that is nested.
        bool is_element = !typ->tp_as_sequence ||  typ == torch_Tensor || typ == TensorType || typ == DimType;
        if (!is_element) {
            flat_elements.extend(A, args.args, args.args + i);
            for (auto j : irange(i)) {
                (void)j;
                r.children.append(A, Unflatten {U_ELEM});
            }
            for (auto j : irange(i, N)) {
                r.children.append(A, tree_flatten(A, args[j], flat_elements));
                if (r.children.back().type != U_ELEM) {
                    r.had_nested = true;
                }
            }
            return r;
        }
    }
    flat_elements.extend(A, args.args, args.args + N);
    return r;
}


struct UnflattenArena {
    Arena A;
    Unflatten unflatten;
};

PyObject* py_unflatten(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    #define ARGS(_) _(mpy::handle, ns)
    MPY_PARSE_ARGS_KWNAMES("O", ARGS)
    #undef ARGS
    mpy::sequence_view sv(ns);
    // because we do not have a autorelase pool yet...
    Arena A;
    Slice<mpy::handle> slice;
    mpy::handle Tuple = (PyObject*) &PyTuple_Type;
    auto inputs = Tuple.call(ns);
    mpy::tuple_view tv(inputs);
    for (auto i : tv.enumerate()) {
        slice.append(A, tv[i]);
    }
    auto AA = (UnflattenArena*) PyCapsule_GetPointer(self, "arena");
    auto r = AA->unflatten(slice).release();
    AT_ASSERT(r != nullptr);
    return r;
    PY_END(nullptr)
}

PyMethodDef py_unflatten_def = {"unflatten", (PyCFunction)(void*) py_unflatten, METH_FASTCALL | METH_KEYWORDS};

void free_unflatten_arena(PyObject * pc) {
    delete (UnflattenArena*) PyCapsule_GetPointer(pc, "arena");
}

PyObject* py_tree_flatten(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    #define ARGS(_) _(mpy::handle, tree)
    MPY_PARSE_ARGS_KWNAMES("O", ARGS)
    #undef ARGS
    auto A = new UnflattenArena;
    Slice<mpy::handle> elements;
    A->unflatten = tree_flatten(A->A, tree, elements);
    auto cap = mpy::object::checked_steal(PyCapsule_New(A, "arena", free_unflatten_arena));
    auto unflatten = mpy::object::checked_steal(PyCFunction_New(&py_unflatten_def, cap.release()));
    mpy::tuple r(2);
    r.set(0, slice_to_list(elements));
    r.set(1, std::move(unflatten));
    return r.release();
    PY_END(nullptr)
}



mpy::object tree_map(Arena& A, const std::function<mpy::handle(mpy::handle)>& fn, mpy::handle agg) {
    Slice<mpy::handle> elements;
    auto unflatten = tree_flatten(A, agg, elements);
    for (auto i : elements.enumerate()) {
        elements[i] = fn(elements[i]);
    }
    return unflatten(elements);
}

// prereq: isinstance(h, _Tensor)
int64_t _Tensor_ndim(mpy::handle h) {
    if (Tensor::check(h)) {
        int64_t r = 0;
        for (auto l : Tensor::unchecked_wrap(h)->levels()) {
            if (l.is_positional()) {
                ++r;
            }
        }
        return r;
    }
    // Dim or DelayedMulTensor
    return 0;
}

mpy::handle handle_from_tensor(Arena& A, TensorRef t) {
    // fast case: tensor is live in python
    std::optional<PyObject*> mb_obj =
        t->unsafeGetTensorImpl()->pyobj_slot()->check_pyobj(getPyInterpreter(), /*ignore_hermetic_tls=*/false);
    if (mb_obj.has_value() && !t->unsafeGetTensorImpl()->pyobj_slot()->owns_pyobj()) {
        return *mb_obj;
    }
    return A.autorelease(mpy::object::checked_steal(THPVariable_Wrap(*t)));
}
}
struct EnableAllLayers {
    EnableAllLayers(Arena& A, Slice<DimEntry> levels) {
        std::vector<std::pair<int64_t, int64_t>> layers;
        layers.reserve(levels.size());
        for (auto l : levels) {
            if (!l.is_positional()) {
                auto d = l.dim();
                levels_to_dim_.append(A, d);
            }
        }
        std::sort(levels_to_dim_.begin(), levels_to_dim_.end(), [](mpy::hdl<Dim> lhs, mpy::hdl<Dim> rhs) { return lhs->level_ < rhs->level_;});

        for (auto i : levels_to_dim_.enumerate()) {
            auto batch_size = levels_to_dim_[i]->size();
            auto level = at::functorch::initAndPushDynamicLayer(at::functorch::TransformType::Vmap, batch_size, at::functorch::RandomnessType::Different);
            if (i == 0) {
                levels_start_ = level;
            }
        }
    }

    ~EnableAllLayers() {
        auto to_remove = levels_start_ + levels_to_dim_.size() - 1;
        for (auto i : levels_to_dim_.enumerate()) {
            AT_ASSERT(at::functorch::popDynamicLayerAndDeleteMetadata().layerId() == to_remove - i);
        }
    }

    mpy::obj<Tensor> from_batched(Arena& A, at::Tensor batchedtensor, bool has_device) {
        Slice<DimEntry> levels;
        for (auto i : irange(-batchedtensor.dim(), 0)) {
            levels.append(A, i);
        }
        TensorRef tensor;
        at::functorch::BatchedTensorImpl * impl = maybeGetBatchedImpl(batchedtensor);
        while(true) {
            auto level = impl->level();
            AT_ASSERT(level >= levels_start_ && level < levels_start_ + levels_to_dim_.size());
            mpy::hdl<Dim> dim = levels_to_dim_[level - levels_start_].ptr();
            levels.insert(A, impl->bdim(), dim);
            at::functorch::BatchedTensorImpl * nimpl = maybeGetBatchedImpl(impl->value());
            if (!nimpl) {
                tensor = impl->value();
                break;
            }
            impl = nimpl;
        }

        mpy::obj<Tensor> self = Tensor::create();
        // grab ownership of the tensors
        self->tensor_ = *tensor;
        self->batchtensor_ = std::move(batchedtensor);
        self->has_device_ = has_device;
        self->capture_levels(levels);
        return self;
    }
    void inplace_update_layers(TensorRef batchtensor, Slice<DimEntry> levels) {
        // XXX - requires a patch to functorch to att set_level
        auto impl = maybeGetBatchedImpl(*batchtensor);
        for (auto i : levels_to_dim_.reversed_enumerate()) {
            if (!impl) {
                break;
            }
            if (levels.contains(levels_to_dim_[i])) {
                impl->_unsafe_set_level(levels_start_ + i);
                impl = maybeGetBatchedImpl(impl->value());

            }
        }
    }
private:
    int64_t levels_start_{};
    Slice<mpy::hdl<Dim>> levels_to_dim_;
};

namespace{
TensorRef _match_levels(Arena& A, TensorRef v, Slice<DimEntry> from_levels, Slice<DimEntry> to_levels, bool drop_levels=false) {
    if (from_levels == to_levels) {
        return v;
    }
    // drop_levels -> if a dim appears in from_levels but not to_levels, it is assumed it has stride 0.
    at::IntArrayRef sz = v->sizes();
    at::IntArrayRef sd = v->strides();
    AT_ASSERT(drop_levels || from_levels.size() <= to_levels.size());
    Slice<int64_t> nsz;
    Slice<int64_t> nsd;
    for (auto l : to_levels) {
        auto oidx = from_levels.index(l);
        if (!oidx) {
            nsz.append(A, l.is_positional() ? 1 : l.dim()->size());
            nsd.append(A, 0);
        } else {
            auto idx = *oidx;
            nsz.append(A, sz[idx]);
            nsd.append(A, sd[idx]);
        }
    }
    return A.autorelease(v->as_strided(at::IntArrayRef(nsz.begin(), nsz.end()), at::IntArrayRef(nsd.begin(), nsd.end()), v->storage_offset()));
}
}
mpy::object run_torch_function(Arena &A, mpy::handle orig, mpy::vector_args args, bool is_pointwise) {
    if (!pointwise_optimize) {
        is_pointwise = false;
    }
    // std::cout << "__torch_function__ " << ((is_pointwise) ? "pointwise" : "functorch") << " " << orig << "\n";

    Slice<mpy::hdl<Dim>> all_dims;
    Slice<mpy::handle> flat_args;
    auto unflatten_args = tree_flatten(A, args, flat_args);
    TensorRef device_holding_tensor;

    Slice<TensorInfo> infos;
    Slice<DimEntry> result_levels;
    for (auto f : flat_args) {
        infos.append(A, TensorInfo::create(A, f, !is_pointwise, false));
        if (infos.back()) {
            TensorInfo& info = infos.back();
            AT_ASSERT(is_pointwise || info.batchedtensor);
            if (!device_holding_tensor && info.has_device) {
                device_holding_tensor = infos.back().tensor;
            }
            for (auto l : info.levels) {
                if (!result_levels.contains(l)) {
                    result_levels.append(A, l);
                }
            }
        }
    }

    if (is_pointwise) {
        for (auto i : flat_args.enumerate()) {
            if (infos[i]) {
                TensorRef tensor = infos[i].tensor;
                if (device_holding_tensor && !infos[i].has_device) {
                    tensor = A.autorelease(tensor->to(device_holding_tensor->device()));
                }
                auto ml = _match_levels(A, tensor, infos[i].levels, result_levels);
                flat_args[i] = handle_from_tensor(A, std::move(ml));
            }
        }

        Slice<mpy::handle> flat_it = flat_args;
        mpy::vector_args uargs = unflatten_args(A, flat_it);

        mpy::object result = orig.call_vector(uargs);

        // fast wrap for normal case where operator just returns a tensor.
        if (THPVariable_Check(result.ptr())) {
            return Tensor::from_positional(A, THPVariable_Unpack(result.ptr()), result_levels, device_holding_tensor);
        }
        auto wrap = [&](mpy::handle h) {
            if (THPVariable_Check(h.ptr())){
                return A.autorelease(Tensor::from_positional(A, THPVariable_Unpack(h.ptr()), result_levels, device_holding_tensor));
            }
            return h;
        };
        return tree_map(A, wrap, result);
    } else {
        // std::cout << orig << " calling functorch...\n";
        // std::cout << "rl: " << result_levels << "\n";
        EnableAllLayers guard(A, result_levels);
        for (auto i : flat_args.enumerate()) {
            if (infos[i]) {
                TensorRef batched = infos[i].batchedtensor;
                if (device_holding_tensor && !infos[i].has_device) {
                    batched = A.autorelease(batched->to(device_holding_tensor->device()));
                }
                guard.inplace_update_layers(batched, infos[i].levels);
                flat_args[i] = handle_from_tensor(A, batched);
            }
        }
        Slice<mpy::handle> flat_it = flat_args;
        mpy::vector_args uargs = unflatten_args(A, flat_it);
        AT_ASSERT(flat_it.size() == 0);
        mpy::object result = orig.call_vector(uargs);
        auto wrap = [&](mpy::handle h) {
            if (THPVariable_Check(h.ptr())) {
                return A.autorelease(guard.from_batched(A, THPVariable_Unpack(h.ptr()), device_holding_tensor));
            }
            return h;
        };
        if (THPVariable_Check(result.ptr())) {
            return guard.from_batched(A, THPVariable_Unpack(result.ptr()), device_holding_tensor);
        }
        return tree_map(A, wrap, result);
    }
}

namespace{

mpy::object __torch_function__(Arena &A, mpy::handle orig, mpy::vector_args args, bool is_pointwise) {
    if (orig == torch_Tensor___mul__) {
        AT_ASSERT(args.nargs == 2 && !args.has_keywords());
        auto lhs = args[0];
        auto rhs = args[1];
        if (mpy::isinstance(lhs, _Tensor) && mpy::isinstance(rhs, _Tensor) && _Tensor_ndim(lhs) == 0 && _Tensor_ndim(rhs) == 0) {
            bool has_device = false;
            Slice<DimEntry> levels;
            for (auto i : args.enumerate_positional()) {
                auto t = TensorInfo::create(A, args[i], false);
                // something like a mask * rhs, which matrix multiplies don't correctly promote
                if (!t.tensor->is_floating_point()) {
                    return run_torch_function(A, orig, args, is_pointwise);
                }
                has_device = has_device || t.has_device;
                for (auto l : t.levels) {
                    if (!levels.contains(l)) {
                        levels.append(A, l);
                    }
                }
            }
            // std::cout << "__torch_function__ " << "delay" << " " << orig << "\n";
            return Tensor::create_delayed(mpy::object::borrow(orig), args, levels, has_device);
        }
    }
    return run_torch_function(A, orig, args, is_pointwise);
}

mpy::vector_args as_vector_args(Arena& A, mpy::handle args, mpy::handle kwargs) {
    auto pos_args = (mpy::handle*) &PyTuple_GET_ITEM(args.ptr(), 0);
    auto pos_n = PyTuple_GET_SIZE(args.ptr());
    if (!kwargs.ptr()) {
        return mpy::vector_args(pos_args, pos_n, nullptr);
    }
    Slice<mpy::handle> all_args;
    Slice<mpy::handle> kwnames;
    all_args.extend(A, pos_args, pos_args + pos_n);
    mpy::dict_view dv(kwargs);
    Py_ssize_t pos = 0;
    mpy::handle key, value;
    while (dv.next(&pos, &key, &value)) {
        all_args.append(A, value);
        kwnames.append(A, key);
    }
    return mpy::vector_args(all_args.begin(), pos_n, A.autorelease(slice_to_tuple(kwnames)));
}

PyObject* py___torch_function__(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    maybeInitializeGlobals();
    AT_ASSERT(nargs == 4 || nargs == 5);
    auto va = as_vector_args(A, args[3], nargs == 5 ? args[4] : nullptr);
    bool is_pointwise = pointwise.contains(args[1]);
    return __torch_function__(A, args[1], std::move(va), is_pointwise).release();
    PY_END(nullptr)
}

mpy::object levels_to_tuple(Slice<DimEntry> slice) {
    mpy::tuple t(slice.size());
    for (auto i : slice.enumerate()) {
        t.set(i, slice[i].is_positional() ?  mpy::from_int(slice[i].position()) : mpy::object::borrow(slice[i].dim()));
    }
    mpy::object r = std::move(t);
    return r;
}

PyObject* Tensor_ndim(Tensor* self, void*) {
    Py_ssize_t i = 0;
    for (auto l : self->levels()) {
        if (l.is_positional()) {
            ++i;
        }
    }
    return mpy::from_int(i).release();
}

PyGetSetDef Tensor_getsetters[] = {
   {"_has_device", (getter) [](PyObject* self, void*) -> PyObject* { return mpy::from_bool(((Tensor*)self)->has_device()).release(); }, NULL},
   {"_tensor", (getter) [](PyObject* self, void*) -> PyObject* {
       Arena A;
       return THPVariable_Wrap(((Tensor*)self)->tensor(A)); }, NULL},
   {"_batchtensor", (getter) [](PyObject* self, void*) -> PyObject* {
       Arena A;
       return THPVariable_Wrap(((Tensor*)self)->batchtensor(A)); }, NULL},
   {"_levels", (getter) [](PyObject* self, void*) -> PyObject* {
       PY_BEGIN
       return levels_to_tuple(((Tensor*)self)->levels()).release();
       PY_END(nullptr)
   }},
    {"ndim", (getter) Tensor_ndim, NULL, "ndim", NULL},
    {NULL}  /* Sentinel */
};

PyMethodDef Tensor_methods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};
}


PyTypeObject Tensor::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_C.Tensor",               /* tp_name */
    sizeof(Tensor),               /* tp_basicsize */
    0,                              /* tp_itemsize */
    Tensor::dealloc_stub,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    0,           /* tp_repr */
    0,                 /* tp_as_number */
    0,                 /* tp_as_sequence */
    0,             /* tp_as_mapping */
    0,      /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE , /* tp_flags */
    "Tensor Object",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,  /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    Tensor_methods,                /* tp_methods */
    0,                              /* tp_members */
    Tensor_getsetters,             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    0,            /* tp_init */
    0,                              /* tp_alloc */
    Tensor::new_stub,                      /* tp_new */
};


// dim() --------------------
#if !IS_PYTHON_3_12_PLUS
static bool relevant_op(_Py_CODEUNIT c) {
#else
static bool relevant_op(uint8_t c) {
#endif
    switch(c) {
        case STORE_NAME:
        case STORE_GLOBAL:
        case STORE_FAST:
        case STORE_DEREF:
            return true;
        default:
            return false;
    }
}

static mpy::object create_dim(mpy::object name, mpy::handle size) {
    auto d = Dim::create(std::move(name));
    if (!mpy::is_none(size)) {
        d->set_size(mpy::to_int(size));
    }
    return std::move(d);
}

static mpy::object create_dimlist(mpy::object name, mpy::handle size) {
    auto d = DimList::create(std::move(name));
    if (!mpy::is_none(size)) {
        if (mpy::is_int(size)) {
            d->bind_len(mpy::to_int(size));
        } else {
            mpy::sequence_view s(size);
            d->bind_len(s.size());
            for (auto i : irange(d->size())) {
                d->dims_[i]->set_size(mpy::to_int(s[i]));
            }
        }
    }
    return std::move(d);
}



// Python wrappers that make new reflection primitives available for older runtimes
#if !(IS_PYTHON_3_11_PLUS)
#define _PyCode_CODE(CO) ((_Py_CODEUNIT*)PyBytes_AS_STRING((CO)->co_code))
#endif

namespace{
struct PyInstDecoder {
    PyInstDecoder(PyCodeObject* code_object, int lasti)
    : code_object_(code_object), code_(_PyCode_CODE(code_object)), offset_(lasti / sizeof(_Py_CODEUNIT))  {}
    // On Windows, _PyOpcode_Caches and _PyOpcode_Deopt are private symbols
    // See https://github.com/pytorch/pytorch/issues/93854
    void next() {
    #if IS_PYTHON_3_11_PLUS
        offset_ += _PyOpcode_Caches[opcode()];
    #endif
        offset_ += 1;
    }
    int opcode() {
        auto r = _Py_OPCODE(code_[offset_]);
    #if IS_PYTHON_3_11_PLUS
        r = _PyOpcode_Deopt[r];
    #endif
        return r;
    }
    int oparg() {
        return _Py_OPARG(code_[offset_]);
    }

    mpy::object name() {
        mpy::object names;
        switch(opcode()) {
            case STORE_NAME:
            case STORE_GLOBAL:
                names = mpy::object::borrow(code_object_->co_names);
                break;
            case STORE_FAST:
                names = mpy::object::steal(PyCode_GetVarnames(code_object_));
                break;
            case STORE_DEREF:
                names = mpy::object::steal(PyCode_GetCellvars(code_object_));
                break;
            default:
                return mpy::object();
        }
        return mpy::object::steal(PySequence_GetItem(names.ptr(), oparg()));
    }
private:
    PyCodeObject* code_object_;
    _Py_CODEUNIT* code_;
    int offset_;
};

template<mpy::object (*create_object)(mpy::object, mpy::handle)>
static PyObject* _dims(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    Py_ssize_t specified_ndims = -1;
    Py_ssize_t found_ndims = 0;
    Py_ssize_t sizes = -1;
    mpy::handle n = Py_None;
    mpy::handle py_sizes = Py_None;

    if (nargs || kwnames) {
        mpy::vector_args va(args, nargs, kwnames);
        va.parse("dims", {"n", "sizes"}, {&n, &py_sizes}, 0);
        if (!mpy::is_none(py_sizes)) {
            sizes = mpy::sequence_view(py_sizes).size();
            specified_ndims = sizes;
        }
        if (!mpy::is_none(n)) {
            specified_ndims = mpy::to_int(n);
        }
    }

    PyThreadState* state = PyThreadState_GET();
    auto f = mpy::obj<PyFrameObject>::steal(PyThreadState_GetFrame(state));
    auto c = mpy::obj<PyCodeObject>::steal(PyFrame_GetCode(f.ptr()));
    auto lasti = PyFrame_GetLasti(f.ptr());
    auto decoder = PyInstDecoder(c.ptr(), lasti);
    #if IS_PYTHON_3_11_PLUS && !(IS_PYTHON_3_12_PLUS)
    // When py3.11 adapts bytecode lasti points to the precall
    // rather than the call instruction after it
    if (decoder.opcode() == PRECALL) {
        decoder.next();
    }
    // note that this opcode was removed in 3.12
    #endif
    decoder.next();

    if (relevant_op(decoder.opcode())) {
        found_ndims = 1;
    } else if (decoder.opcode() == UNPACK_SEQUENCE) {
        found_ndims = decoder.oparg();
        decoder.next();
    }

    if (specified_ndims == -1) {
        if (found_ndims == 0) {
            mpy::raise_error(PyExc_SyntaxError, "dims() must be assigned to a sequence of variable names or have argument n specified");
        }
        specified_ndims = found_ndims;
    }
    if (found_ndims != specified_ndims) {
        found_ndims = 0; // avoid taking the wrong names for dimensions
    }

    auto genobject = [&](int i) -> mpy::object {
        mpy::object name;
        if (i < found_ndims) {
            name = decoder.name();
        }
        if (!name.ptr()) {
            name = mpy::unicode_from_format("d%d", i);
            found_ndims = 0; // once we fail at finding a name, we can find any more
        } else {
            decoder.next();
        }
        return create_object(std::move(name), sizes != -1 ? mpy::sequence_view(py_sizes)[i] : mpy::handle(Py_None));
    };
    if (sizes != -1 && sizes != specified_ndims) {
        mpy::raise_error(PyExc_ValueError, "expected %d sizes but found %d", int(specified_ndims), int(sizes));
    }
    if (specified_ndims == 1) {
        return genobject(0).release();
    }
    mpy::tuple result(specified_ndims);
    for (int i = 0; i < specified_ndims; ++i) {
        result.set(i, genobject(i));
    }
    return result.release();
    PY_END(nullptr)
}

struct DotPart {
    Slice<DimEntry> dims;
    size_t total_size = 1;
    void append(Arena& A, mpy::hdl<Dim> d) {
        total_size *= d->size();
        dims.append(A, d);
    }
};

template<typename T>
static at::ArrayRef<T> as_array_ref(Slice<T> t) {
    return at::ArrayRef<T>(t.begin(), t.end());
}

static TensorRef dot_prepare(Arena& A, std::initializer_list<DotPart> parts, const TensorInfo& t) {
    Slice<DimEntry> new_levels;
    bool needs_reshape = false;
    for (auto p : parts) {
        if (p.dims.size() != 1) {
            needs_reshape = true;
        }
        new_levels.extend(A, p.dims);
    }
    auto r = _match_levels(A, t.tensor, t.levels, new_levels, true);
    if (!needs_reshape) {
        return r;
    }
    Slice<int64_t> view;
    for (auto p : parts) {
        view.append(A, p.total_size);
    }
    return A.autorelease(r->reshape(at::IntArrayRef(view.begin(), view.end())));
}

static mpy::object dot_finish(Arena& A, std::initializer_list<DotPart> parts, at::Tensor r) {
    Slice<DimEntry> result_levels;
    bool needs_reshape = false;
    for (auto p : parts) {
        if (p.dims.size() != 1) {
            needs_reshape = true;
        }
        result_levels.extend(A, p.dims);
    }
    if (needs_reshape) {
        Slice<int64_t> new_size;
        for (auto l : result_levels) {
            new_size.append(A, l.dim()->size());
        }
        r = r.reshape(at::IntArrayRef(new_size.begin(), new_size.end()));
    }
    return Tensor::from_positional(A, std::move(r), result_levels, true);
}



static mpy::object dot(Arena& A, TensorInfo lhs, TensorInfo rhs, Slice<DimEntry> sum) {
    auto lhs_strides = lhs.tensor->strides();
    auto rhs_strides = rhs.tensor->strides();

    DotPart lro_dims;
    DotPart lo_dims;
    DotPart ro_dims;
    DotPart lr_dims;

    auto insert_dim = [&] (mpy::hdl<Dim> d, std::optional<int> lhs_idx, std::optional<int> rhs_idx) {
        bool reduced = sum.contains(d);
        int64_t lhs_stride = lhs_idx ? lhs_strides[*lhs_idx] : 0;
        int64_t rhs_stride = rhs_idx ? rhs_strides[*rhs_idx] : 0;
        if (reduced) {
            // lr
            lr_dims.append(A, d);
        } else {
            if ((lhs_stride == 0) == (rhs_stride == 0)) {
                // lro
                lro_dims.append(A, d);
            } else if (lhs_stride != 0) {
                // lo
                lo_dims.append(A, d);
            } else {
                AT_ASSERT(rhs_stride != 0);
                ro_dims.append(A, d);
            }
        }
    };


    auto rhs_seen = A.allocate<bool>(rhs.levels.size());
    std::fill(rhs_seen, rhs_seen + rhs.levels.size(), false);

    for (auto i : lhs.levels.enumerate()) {
        auto d = lhs.levels[i];
        auto rhs_idx = rhs.levels.index(d);
        if (rhs_idx) {
            rhs_seen[*rhs_idx] = true;
        }
        insert_dim(d.dim(), i, rhs_idx);
    }

    for (auto i : rhs.levels.enumerate()) {
        if (rhs_seen[i]) {
            continue;
        }
        auto d = rhs.levels[i];
        insert_dim(d.dim(), std::nullopt, i);
    }

    if (lr_dims.dims.size() != sum.size()) {
        for (auto & d : sum) {
            if (!lhs.levels.contains(d) && !rhs.levels.contains(d)) {
                mpy::raise_error(DimensionBindError(), "summing over non-existant dimension %S", d.dim().ptr());
            }
        }
    }

    // std::cout << lhs.levels << " " << rhs.levels << " " << sum << "\n";
    // std::cout << lro_dims.dims << " " << lo_dims.dims << " " << ro_dims.dims << " " << lr_dims.dims << "\n";

    // no batch, just call mm
    if (lro_dims.dims.size() != 0) {
        auto lhs_ = dot_prepare(A, {lro_dims, lo_dims, lr_dims}, lhs);
        auto rhs_ = dot_prepare(A, {lro_dims, lr_dims, ro_dims}, rhs);
        return dot_finish(A, {lro_dims, lo_dims, ro_dims}, at::bmm(*lhs_, *rhs_));
    } else {
        auto lhs_ = dot_prepare(A, {lo_dims, lr_dims}, lhs);
        auto rhs_ = dot_prepare(A, {lr_dims, ro_dims}, rhs);
        return dot_finish(A, {lo_dims, ro_dims}, at::mm(*lhs_, *rhs_));
    }

}

static PyObject* test_c(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN

    Arena A;
    Slice<int> s(A, 3, 4, 5);
    AT_ASSERT(s.size() == 3 && s.capacity() == 8);
    AT_ASSERT(s[0] == 3 && s[1] == 4 && s[2] == 5);
    s.append(A, 6);
    AT_ASSERT(s[3] == 6);
    for(int i : irange(10)) {
        s.append(A, i);
    }
    AT_ASSERT(s[0] == 3 && s.back() == 9 && s.size() == 14 && s.capacity() == 16);

    Slice<int> s2(A, -1, -2, -3);
    AT_ASSERT(s2[1] == -2 && s[0] == 3);

    auto ss = s.slice(1,2);
    AT_ASSERT(ss.size() == 1);
    AT_ASSERT(ss[0] == 4);
    AT_ASSERT(ss.capacity() == 1);
    ss.append(A, -4);
    AT_ASSERT(ss.size() == 2 && ss[1] == -4);
    ss[0] = 3;
    AT_ASSERT(s[1] == 4);

    s.insert(A, s.slice(1, 4), ss);
    AT_ASSERT(s[1] == 3  && s[2] == -4 && s[3] == 0);

    auto sz = s.size();
    s.insert(A, s.slice(1, 1), 4);
    AT_ASSERT(s[1] == 4 && sz + 1 == s.size());


    Slice<int> d(A, 0, 1, 2, 3, 4);

    Slice<int> b(A, 0, 1, 2, 3, 4);
    b.insert(A, b.slice(1,1), d);
    AT_ASSERT(b.size() == 10);
    AT_ASSERT(b[1] == 0);
    AT_ASSERT(b[5] == 4);
    AT_ASSERT(b.back() == 4);

    Py_RETURN_NONE;

    PY_END(nullptr);
}


static PyObject* order(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    if (kwnames) {
        mpy::raise_error(PyExc_TypeError, "unexpected keyword arguments %S", kwnames);
    }
    AT_ASSERT(nargs-- > 0);
    Slice<DimEntry> orig_levels;
    Slice<DimEntry> levels;
    TensorRef data;
    mpy::handle self = args++[0];
    bool has_device;
    if (Tensor::check_exact(self)) {
        auto t = Tensor::unchecked_wrap(self);
        orig_levels = t->levels();
        data = t->tensor(A);
        has_device = t->has_device();
    } else {
       auto d = Dim::unchecked_wrap(self);
        orig_levels.append(A, d);
        data = d->range();
        has_device = false;
    }

    Slice<DimEntry> flat_positional_dims;
    Slice<std::pair<int, int>> to_flatten;
    levels.extend(A, orig_levels);

    int orig_ndim = ndim_of_levels(levels);
    auto append = [&](DimEntry d) {
        auto midx = levels.index(d);
        if (!midx) {
            if (d.is_positional()) {
                mpy::raise_error(PyExc_ValueError, "tensor has %d positional dimensions, but %d specified, or it was specified twice", int(orig_ndim), int(d.position() + orig_ndim));
            } else {
                mpy::raise_error(PyExc_ValueError, "tensor of dimensions %R does not contain dim %R or it was specified twice", levels_to_tuple(orig_levels).ptr(), d.dim().ptr());
            }
        }
        levels[*midx] = DimEntry();
        flat_positional_dims.append(A, d);
    };

    int n_new_positional = 0;
    for (auto i :irange(nargs)) {
        mpy::handle arg  = args[i];
        DimEntry entry = _wrap_dim(arg, orig_ndim, false);
        if (!entry.is_none()) {
            append(entry);
            ++n_new_positional;
        } else if (DimList::check(arg)) {
            auto dl = DimList::unchecked_wrap(arg);
            for (mpy::obj<Dim> & d : dl->dims_) {
                append(mpy::hdl<Dim>(d));
                ++n_new_positional;
            }
        } else {
            ++n_new_positional;
            if (!mpy::is_sequence(arg)) {
                mpy::raise_error(PyExc_ValueError, "expected a Dim, List[Dim], or Sequence[Dim]");
            }
            mpy::sequence_view sq(arg);
            auto N = sq.size();
            to_flatten.append(A, std::make_pair(flat_positional_dims.size(), N));
            for (auto j : irange(N)) {
                DimEntry e = _wrap_dim(A.autorelease(sq[j]), orig_ndim, false);
                if (e.is_none()) {
                    mpy::raise_error(PyExc_ValueError, "expected a Dim, or int");
                }
                append(e);
            }
        }
    }

    int insert_point = -1;
    Slice<DimEntry> new_levels;
    for (auto l : levels) {
        if (l.is_none()) {
            continue;
        }
        if (l.is_positional()) {
            if (insert_point == -1) {
                insert_point = new_levels.size();
                new_levels.extend(A, flat_positional_dims);
            }
        }
        new_levels.append(A, l);
    }
    if (insert_point == -1) {
        insert_point = new_levels.size();
        new_levels.extend(A, flat_positional_dims);
    }

    at::Tensor ndata = *_match_levels(A, data, orig_levels, new_levels);

    if (to_flatten.size()) {
        Slice<int64_t> view;
        auto sz = ndata.sizes();
        // before the new positional dims
        for (auto i : irange(0, insert_point)) {
            view.append(A, sz[i]);
        }
        int i = 0;
        for (auto to_flat : to_flatten) {
            for (;i < to_flat.first; ++i) {
                view.append(A, sz[insert_point + i]);
            }
            int64_t new_size = 1;
            int last = i + to_flat.second;
            for (; i < last; ++i) {
                new_size *= sz[insert_point + i];
            }
            view.append(A, new_size);
        }
        for (; i < flat_positional_dims.size(); ++i) {
            view.append(A, sz[insert_point + i]);
        }
        // after the new positional dims
        for (auto i : irange(insert_point + flat_positional_dims.size(), levels.size())) {
            view.append(A, sz[i]);
        }
        // we shorted the number of dimension, so remove them from new levels
        // we will renumber them later
        auto n_to_remove = flat_positional_dims.size() - n_new_positional;
        new_levels.insert(A, new_levels.slice(insert_point, insert_point + n_to_remove), Slice<DimEntry>());
        ndata = std::move(ndata).reshape(at::IntArrayRef(view.begin(), view.end()));
    }

    // renumber the positional dimension
    int seen = 0;
    for (auto i : new_levels.reversed_enumerate()) {
        if (new_levels[i].is_positional() || (i >= insert_point && i < insert_point + n_new_positional)) {
            new_levels[i] = --seen;
        }
    }
    return Tensor::from_positional(A, std::move(ndata), new_levels, has_device).release();

    PY_END(nullptr)
}

static PyObject* expand(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    AT_ASSERT(nargs-- > 0);
    auto info = TensorInfo::create(A, args++[0], false);
    for (auto i : irange(nargs)) {
        if (!Dim::check(args[i])) {
            maybeInitializeGlobals();
            mpy::vector_args vargs(args - 1, nargs + 1, kwnames);
            if (THPVariable_Check(args[-1])) {
                return torch_Tensor_expand.call_vector(vargs).release();
            } else {
                return __torch_function__(A, torch_Tensor_expand, vargs, false).release();
            }
        }
    }
    const at::Tensor& data = *info.tensor;
    auto levels = info.levels;
    Slice<DimEntry> new_levels;
    Slice<int64_t> sz;
    Slice<int64_t> sd;
    for (auto i : irange(nargs)) {
        auto d = Dim::unchecked_wrap(args[i]);
        if (levels.contains(d) || new_levels.contains(d)) {
            mpy::raise_error(DimensionBindError(), "expanding dimension %R already exists in tensor with dims", d.ptr());
        }
        new_levels.append(A, d);
        sz.append(A, d->size());
        sd.append(A, 0);
    }
    new_levels.extend(A, levels);
    at::IntArrayRef osz = data.sizes();
    at::IntArrayRef osd = data.strides();
    sz.extend(A, osz.begin(), osz.end());
    sd.extend(A, osd.begin(), osd.end());
    at::Tensor ndata = data.as_strided(at::IntArrayRef(sz.begin(), sz.end()), at::IntArrayRef(sd.begin(), sd.end()), data.storage_offset());
    return Tensor::from_positional(A, std::move(ndata), new_levels, info.has_device).release();
    PY_END(nullptr)
}


static void _bind_dims_to_size(Arena & A, int64_t sz, int64_t sd,
                        Slice<mpy::hdl<Dim>> dims, Slice<int64_t>& nsz, Slice<int64_t>& nsd) {
    int64_t rhs_prod = 1;
    for (auto i : dims.enumerate()) {
        if (!dims[i]->is_bound()) {
            for (auto j : irange(i + 1, dims.size())) {
                if (!dims[j]->is_bound()) {
                    mpy::raise_error(DimensionBindError(), "cannot infer the sizes of two dimensions at once %R and %R", dims[i].ptr(), dims[j].ptr());
                }
                rhs_prod *= dims[j]->size();
            }
            if (sz % rhs_prod != 0) {
                mpy::tuple tup(dims.size());
                for (auto j : dims.enumerate()) {
                    tup.set(j, dims[j]->is_bound() ? mpy::from_int(dims[j]->size()) : mpy::unicode_from_string("?"));
                }
                mpy::raise_error(DimensionBindError(), "inferred dimension does not evenly fit into larger dimension: %d vs %R", (int) sz, tup.ptr());
            }
            int64_t inferred_size = sz / rhs_prod;
            dims[i]->set_size(inferred_size);
            rhs_prod = sz;
            break;
        }
        rhs_prod *= dims[i]->size();
    }
    if (rhs_prod != sz) {
        mpy::tuple tup(dims.size());
        for (auto j : dims.enumerate()) {
            tup.set(j, mpy::object::borrow(dims[j]));
        }
        mpy::raise_error(DimensionBindError(), "Dimension sizes to do not match (%d != %d) when matching dimension pack %R", (int) sz, (int) rhs_prod, tup.ptr());
    }
    auto new_strides = A.allocate<int64_t>(dims.size());
    auto prev_stride = sd;
    for (auto i : dims.reversed_enumerate()) {
        new_strides[i] = prev_stride;
        prev_stride = dims[i]->size()*prev_stride;
    }
    for (auto i : dims.enumerate()) {
        nsd.append(A, new_strides[i]);
        nsz.append(A, dims[i]->size());
    }
}

static bool has_dims(mpy::handle d) {
    return Dim::check_exact(d) || Tensor::check_exact(d);
}

struct IndexingInfo {
    bool can_call_original; // if true, then it is safe to just call getitem or setitem, these objects do not need special handling
    bool advanced_indexing; // requires actual lookup
    TensorRef self;
    Slice<mpy::handle> flat_inputs;
    Slice<DimEntry> result_levels;
    bool has_device;
};
}

IndexingInfo getsetitem_flat(Arena& A, TensorInfo self_info, Slice<mpy::handle> input, Slice<DimEntry> keys, Slice<mpy::handle> values, bool has_dimpacks_or_none);
namespace{
Slice<mpy::handle> as_slice(mpy::tuple_view tv) {
    PyObject** begin = &PyTuple_GET_ITEM(tv.ptr(),0);
    return Slice<mpy::handle>((mpy::handle*)begin, (mpy::handle*) (begin + tv.size()));
}

Slice<mpy::handle> as_slice(mpy::list_view tv) {
    PyObject** begin = &PyList_GET_ITEM(tv.ptr(),0);
    return Slice<mpy::handle>((mpy::handle*)begin, (mpy::handle*) (begin + tv.size()));
}


bool maybe_dimpack(Slice<mpy::handle>& elements, mpy::handle s, bool check_first=true) {
    // can we avoid rechecking?
    if (mpy::list_view::check(s)) {
        mpy::list_view tv(s);
        if (!check_first || (tv.size() && Dim::check_exact(tv[0]))) {
            elements = as_slice(tv);
            return true;
        }
    }
    // can we avoid rechecking?
    if (mpy::tuple_view::check(s)) {
        mpy::tuple_view tv(s);
        if (!check_first || (tv.size() && Dim::check_exact(tv[0]))) {
            elements = as_slice(tv);
            return true;
        }
    }
    return false;
};

bool is_dimpack(mpy::handle s) {
    Slice<mpy::handle> e;
    return maybe_dimpack(e, s);
}

mpy::object invoke_getitem(Arena& A, const IndexingInfo& iinfo) {
    at::Tensor rtensor;
    if (iinfo.advanced_indexing) {
        auto self_hdl = handle_from_tensor(A, iinfo.self);
        auto tup = slice_to_tuple(iinfo.flat_inputs);
        // std::cout << "calling original getindex " << self_hdl << " " << tup << "\n";
        auto pytensor = mpy::object::checked_steal(THPVariable_getitem(self_hdl.ptr(), tup.ptr()));
        rtensor = THPVariable_Unpack(pytensor.ptr());
    } else {
        // std::cout << "skipping original getindex\n";
        rtensor = *iinfo.self;
    }
    // std::cout << "returning (from_positional)\n";
    return Tensor::from_positional(A, std::move(rtensor), iinfo.result_levels, iinfo.has_device);
}

mpy::object index(Arena& A, mpy::handle self, mpy::handle dims, mpy::handle indices) {
    maybeInitializeGlobals();
    Slice<mpy::handle> dims_list;
    Slice<mpy::handle> indices_list;
    // we allow for matching single dims to multiple dims,
    // so we first have to normalize everything into the case where there is a list on lhs and the rhs
    bool lhs_list = mpy::tuple_view::check(dims) || mpy::list_view::check(dims);
    bool rhs_list = mpy::tuple_view::check(indices) || mpy::list_view::check(indices);
    if (lhs_list && rhs_list) {
        mpy::sequence_view dv(dims);
        mpy::sequence_view ind(indices);
        Py_ssize_t N = dv.size();
        if (N != ind.size()) {
            mpy::raise_error(PyExc_TypeError, "dims (%d) and indices (%d) must have the same length", int(N), int(ind.size()));
        }
        for (auto i : irange(N)) {
            dims_list.append(A, A.autorelease(dv[i]));
            indices_list.append(A, A.autorelease(ind[i]));
        }
    } else {
        dims_list.append(A, dims);
        indices_list.append(A, indices);
    }

    // dims being indexed can be grouped together into a single index space, and we have to
    // flatten them int a single dimension before we can index them...
    auto self_info = TensorInfo::create(A, self, false);
    auto ndim = self_info.ndim();
    Slice<DimEntry> new_levels;
    Slice<DimEntry> to_flatten;
    Slice<DimEntry> dims_list_flat;
    auto parse_dim_entry = [&](mpy::handle s) -> DimEntry {
        auto d = _wrap_dim(s, ndim, false);
        if (d.is_none()) {
            mpy::raise_error(PyExc_TypeError, "expected a dimension specifyer but found %R", s.ptr());
        }
        return d;
    };
    auto dim_not_present = [&](DimEntry d) {
        if (d.is_positional()) {
            mpy::raise_error(PyExc_TypeError, "dimension %d not in tensor of %d dimensions", d.position() + ndim , ndim);
        } else {
            mpy::raise_error(PyExc_TypeError, "dimension %R not in tensor", d.dim()->ptr());
        }
    };

    for (auto i : dims_list.enumerate()) {
        Slice<mpy::handle> m;
        if (maybe_dimpack(m, dims_list[i], /*check_first=*/false)) {
            if (m.size() == 0) {
                // plausible semantics work for this to have 0 elements (e.g. the index will always be 0)
                dims_list_flat.append(A, DimEntry()); // value is just dropped
            }
            auto first = parse_dim_entry(m[0]);
            dims_list_flat.append(A, first);
            if (m.size() == 1) {
                continue;
            }
            if (to_flatten.size() == 0) {
                new_levels.extend(A, self_info.levels);
            }
            Slice<DimEntry> rest;
            for (auto i : irange(1, m.size())) {
                auto d = parse_dim_entry(m[i]);
                if (!new_levels.remove(A, d)) {
                    dim_not_present(d);
                }
                rest.append(A, d);
            }

            auto first_idx = new_levels.index(first);
            if (!first_idx) {
                dim_not_present(first);
            }
            new_levels.insert(A, new_levels.slice(*first_idx + 1, *first_idx + 1), rest);
            to_flatten.extend(A, rest);
        } else {
            dims_list_flat.append(A, parse_dim_entry(dims_list[i]));
        }
    }
    if (to_flatten.size() > 0) {
        TensorRef rearranged = _match_levels(A, self_info.tensor, self_info.levels, new_levels);
        at::IntArrayRef sizes = rearranged->sizes();
        Slice<int64_t> new_sizes;
        Slice<DimEntry> reshape_levels;
        for (auto i : new_levels.enumerate()) {
            if (to_flatten.contains(new_levels[i])) {
                new_sizes.back() *= sizes[i];
            } else {
                new_sizes.append(A, sizes[i]);
                reshape_levels.append(A, new_levels[i]);
            }
        }
        self_info.tensor = A.autorelease(rearranged->reshape(at::IntArrayRef(new_sizes.begin(), new_sizes.end())));

        self_info.levels = reshape_levels; // note: we are using the first level in a flattened group to represent the group for the rest of the op
                                           // we need to be careful not to rely the dimensions size because it doesnt match the size of the whole group
    }
    bool has_dimpacks = false;
    for (auto idx : indices_list) {
        if (mpy::tuple_view::check(idx) || mpy::list_view::check(idx)) {
            has_dimpacks = true;
            break;
        }
    }
    IndexingInfo info = getsetitem_flat(A, self_info, Slice<mpy::handle>(), dims_list_flat, indices_list, has_dimpacks);
    return invoke_getitem(A, info);
}

// true -- the indices were flattend out of a tuple, list or sequence...

Slice<mpy::handle> slice_from_sequence(Arena& A, mpy::handle value) {
    if (mpy::tuple_view::check(value)) {
        return as_slice(mpy::tuple_view(value));
    } else if (mpy::list_view::check(value)) {
        return as_slice(mpy::list_view(value));
    } else {
        mpy::sequence_view sv(value);
        Slice<mpy::handle> r;
        for (auto i : sv.enumerate()) {
            r.append(A, A.autorelease(sv[i]));
        }
        return r;
    }
}

bool extractIndices(Arena& A, mpy::handle index, Slice<mpy::handle>& indices) {
    if (mpy::tuple_view::check(index)) {
        indices.extend(A, as_slice(mpy::tuple_view(index)));
        return true;
    } else if (THPVariable_Check(index.ptr())) {
        indices.append(A, index);
        return false;
    } else if (!mpy::is_sequence(index)) {
        indices.append(A, index);
        return false;
    }
    // a copy of treatSequenceAsTuple modified to add Dim and our wrapped tensors..
    mpy::sequence_view sv(index);
    if (sv.size() >= 32) {
        indices.extend(A, slice_from_sequence(A, index));
        return true;
    }
    for (auto i : sv.enumerate()) {
        mpy::handle item;
        try {
            item = sv[i];
        } catch (mpy::exception_set & e) {
            PyErr_Clear();
            indices.append(A, index);
            return false;
        }
        if (THPVariable_Check(item.ptr()) || mpy::is_sequence(item) || PySlice_Check(item.ptr()) || item.ptr() == Py_Ellipsis || mpy::is_none(item) || has_dims(item)) {
            indices.extend(A, slice_from_sequence(A, index));
            return true;
        }
    }
    indices.append(A, index);
    return false;
}

IndexingInfo getsetitem(Arena & A, mpy::handle self, mpy::handle index, bool tensors_have_dims) {
    bool can_call_original_getitem = !tensors_have_dims;

    Slice<mpy::handle> input;
    if (has_dims(index)) {
        input.append(A, index);
    } else {
        bool is_sequence = extractIndices(A, index, input);
        // nothing about first class dims here, fallback to getitem
        if (can_call_original_getitem && !is_sequence) {
            return { true };
        }
    }

    int64_t dims_indexed = 0;
    int64_t expanding_object = -1;
    DimList* unbound_dim_list = nullptr;
    auto check_expanding = [&](int64_t i) {
        if (expanding_object != -1) {
            mpy::raise_error(DimensionBindError(), "at most one ... or unbound dimension list can exist in indexing list but found 2 at offsets %d and %d", (int) expanding_object, (int) i);
        }
        expanding_object = i;
    };
    Slice<int64_t> dimlists;

    // calculate how many dimensioned have been indexed in order to compute the size of ...
    // or expand a potentially unbound dimension list.

    bool has_dimpacks_or_none = false;
    for (auto i : input.enumerate()) {
        mpy::handle s = input[i];
        if (Dim::check_exact(s) || Tensor::check_exact(s)) {
            can_call_original_getitem = false;
            ++dims_indexed;
        } else if (s.ptr() == Py_Ellipsis) {
            check_expanding(i);
        } else if (DimList::check(s)) {
            can_call_original_getitem = false;
            auto dl = DimList::unchecked_wrap(s);
            if (!dl->is_bound()) {
                check_expanding(i);
                unbound_dim_list = dl.ptr();
            } else {
                dims_indexed += dl->dims_.size();
            }
            dimlists.append(A, i);
        } else if (mpy::is_none(s)) {
            has_dimpacks_or_none = true;
        } else if (is_dimpack(s)) {
            can_call_original_getitem = false;
            has_dimpacks_or_none = true;
            ++dims_indexed;
        } else {
            ++dims_indexed;
        }
    }

    // at this point if we haven't seen any Dim objects, we also can fallback to the original getitem.
    if (can_call_original_getitem) {
        return {true};
    }

    // std::cout << "__getitem__ " << self << " " << index << "\n";

    TensorInfo self_info = TensorInfo::create(A, self, false, true);
    auto ndim = self_info.ndim();
    if (dims_indexed > ndim) {
        mpy::raise_error(PyExc_ValueError, "at least %d indices were supplied but the tensor only has %d dimensions", (int) dims_indexed, (int) ndim);
    }
    // expand any unbound dimension list, or expand ... into individual : slices.
    auto expanding_dims = ndim - dims_indexed;
    if (expanding_object != -1) {
        if (unbound_dim_list) {
            unbound_dim_list->bind_len(expanding_dims);
        } else {
            // ...
            Slice<mpy::handle> no_slices;
            for (auto i : irange(expanding_dims)) {
                (void) i;
                no_slices.append(A, no_slice);
            }
            input.insert(A, input.slice(expanding_object, expanding_object + 1), no_slices);
        }
    }

    // flatten out any dimensions stored in dimlist elements directly into the inputs
    // std::cout << dimlists << " <- dim lists!\n";
    for (int64_t i = dimlists.size() - 1; i >=0; --i) {
        auto idx = dimlists[i];
        // we added more elements to input because of ...
        // so we need to also adjust the index to get back to where the
        // dimlist existed
        if (!unbound_dim_list && expanding_object != -1 && idx > expanding_object) {
            idx += expanding_dims;
        }
        auto dl = DimList::unchecked_wrap(input[idx]);
        // XXX would be better if we used an OwnedSlice in DimList
        Slice<mpy::handle> more_dims((mpy::handle*) &*dl->dims_.begin(), (mpy::handle*) &*dl->dims_.end());
        input.insert(A, input.slice(idx, idx + 1), more_dims);
    }

    return getsetitem_flat(A, self_info, input, Slice<DimEntry>(), Slice<mpy::handle>(), has_dimpacks_or_none);
}
}
IndexingInfo getsetitem_flat(Arena& A, TensorInfo self_info, Slice<mpy::handle> input, Slice<DimEntry> keys, Slice<mpy::handle> values, bool has_dimpacks_or_none) {
    // At this point:
    // ..., DimList have been eliminated
    // Dim, Tensor, Tuple[Dim,...], int, slice still remain


    // we have to count how many times we see a dimension.
    // A[i,j] is a simple binding operation, but A[i, i+j] or A[i, i] requires advanced indexing.
    Slice<mpy::hdl<Dim>> seen_dims;
    Slice<int64_t> seen_dims_nuses;
    auto add_dim = [&](mpy::hdl<Dim> entry) {
        auto midx = seen_dims.index(entry);
        if (!midx) {
            seen_dims.append(A, entry);
            seen_dims_nuses.append(A, 1);
        } else {
            ++seen_dims_nuses[*midx];
        }
    };

    Slice<mpy::handle> input_it = input;

    Slice<mpy::handle> flat_inputs;
    // flat inputs will start with an empty mpy::handle if the
    // actual value is in the tensor-like object in the tensor info
    Slice<TensorInfo> tensor_inputs;

    auto append_flat_handle = [&](mpy::handle h) {
        flat_inputs.append(A, h);
        tensor_inputs.append(A, TensorInfo());
    };
    TensorRef device_holding_tensor;
    auto append_tensor_input = [&](TensorInfo ti) {
        flat_inputs.append(A, mpy::handle());
        tensor_inputs.append(A, ti);
        if (ti.has_device && !device_holding_tensor) {
            device_holding_tensor = ti.tensor;
        }
    };

    Slice<int64_t> nsz;
    Slice<int64_t> nsd;
    at::IntArrayRef sz = self_info.tensor->sizes();
    at::IntArrayRef sd = self_info.tensor->strides();

    auto append_size = [&](int i) {
        if (has_dimpacks_or_none) {
            nsz.append(A, sz[i]);
            nsd.append(A, sd[i]);
        }
    };
    // std::cout << "self levels: " << self_info.levels << "\n";

    auto parse_nones = [&]() {
        while (input_it.size() && mpy::is_none(input_it[0])) {
            append_flat_handle(no_slice);
            nsz.append(A, 1);
            nsd.append(A, 0);
            input_it = input_it.slice(1);
        }
    };


    auto append_item = [&](int i, mpy::handle arg) {
        if (Dim::check_exact(arg)) {
            auto d = Dim::unchecked_wrap(arg);
            d->set_size(sz[i]);
            add_dim(d);
            append_size(i);
            append_flat_handle(arg);
            return;
        }
        auto info = TensorInfo::create(A, arg, false, false);
        if (info) {
            append_size(i);
            append_tensor_input(info);
            for (auto il : info.levels) {
                if (!il.is_positional()) {
                    add_dim(il.dim());
                }
            }
            return;
        }

        if (has_dimpacks_or_none) {
            Slice<mpy::handle> mp;
            if (maybe_dimpack(mp, arg)) {
                // dim pack
                Slice<mpy::hdl<Dim>> dim_pack;
                for (auto d : mp) {
                    dim_pack.append(A, Dim::wrap(d));
                    add_dim(dim_pack.back());
                    append_flat_handle(dim_pack.back());
                }
                _bind_dims_to_size(A, sz[i], sd[i], dim_pack, nsz, nsd);
                return;
            }
        }

        append_size(i);
        append_flat_handle(arg);
    };

    // pair up the indexing expressions with dimension of self it indexes
    // self may have first-class dims, which do not participate the indexing.
    for (auto i : self_info.levels.enumerate()) {
        auto l = self_info.levels[i];
        auto idx = keys.index(l);
        if (idx) {
            append_item(i, values[*idx]);
        } else if (l.is_positional()) {
            // grab and index from the positional list
            parse_nones();
            if (!input_it.size()) {
                // we might have fewer indices than tensor dimensions,
                // which implicitly indexes the remaining dimensions with :
                append_flat_handle(no_slice);
                append_size(i);
            } else {
                mpy::handle arg = input_it[0];
                input_it = input_it.slice(1);
                append_item(i, arg);
            }
        } else {
            add_dim(l.dim());
            append_flat_handle(l.dim());
            append_size(i);
        }
    }
    // any training Nones may have no existing dimension associated with them in self.
    parse_nones();

    // we have to restride the tensor to collapse dimension packs and introduce our none dimensions.
    if (has_dimpacks_or_none) {
        self_info.tensor = A.autorelease(self_info.tensor->as_strided(at::IntArrayRef(nsz.begin(), nsz.end()),at::IntArrayRef(nsd.begin(), nsd.end()), self_info.tensor->storage_offset()));
    }


    // figure out what the shape of the indexing tensors will be
    // and what the shape of the resulting tensor will be
    Slice<DimEntry> result_levels;
    Slice<DimEntry> index_levels;
    int64_t tensor_insert_point = -1;
    bool requires_getindex = false;
    auto mark_tensor_index = [&] {
        if (tensor_insert_point == -1) {
            tensor_insert_point = result_levels.size();
        } else if (tensor_insert_point != result_levels.size()) {
            tensor_insert_point = 0;
        }
    };
    for (auto i : flat_inputs.enumerate()) {
        auto inp = flat_inputs[i];
         if(tensor_inputs[i]) {
             requires_getindex = true;
             mark_tensor_index();
             for (auto l : tensor_inputs[i].levels) {
                 // std::cout << "Consider to add " << l << "\n";
                 if (!index_levels.contains(l)) {
                     index_levels.append(A, l);
                 }
             }
        } else if (Dim::check_exact(inp)) {
            auto d = Dim::unchecked_wrap(inp);
            // dimesions used once are just binding operations
            if (1 == seen_dims_nuses[*seen_dims.index(d)]) {
                flat_inputs[i] = no_slice;
                result_levels.append(A, d);
            } else {
                requires_getindex = true;
                flat_inputs[i] = mpy::handle();
                tensor_inputs[i] = TensorInfo {d->range(), Slice<DimEntry>(A, DimEntry(d)), false, TensorRef()};
                if (!index_levels.contains(d)) {
                     index_levels.append(A, d);
                }
                mark_tensor_index();
            }
         } else {
            if (inp.ptr() != no_slice.ptr()) {
                requires_getindex = true;
            }
            if (!mpy::is_int(inp)) {
                // note: actual positional indexes are accurately computed later
                result_levels.append(A, -1);
            }
         }
    }

    // indexing dimensions appear in the tensor at the _first use of a tensor_ in the indexing. So insert
    // the indexing leveles into the result klevels at this spot
    if (tensor_insert_point != -1) {
        result_levels.insert(A, result_levels.slice(tensor_insert_point, tensor_insert_point), index_levels);
    }

    // std::cout << "flat inputs: " << flat_inputs << "\n";
    // std::cout << "result_levels: " << result_levels << "\n";
    // std::cout << "index_levels: " << index_levels << "\n";

    // get all the tensors to be the right shape for indexing
    if (requires_getindex) {
        for (auto i : flat_inputs.enumerate()) {
            if (tensor_inputs[i]) {
                AT_ASSERT(!flat_inputs[i].ptr());
                // std::cout << "tensor " << i << " " << tensor_inputs[i].levels << "\n";
                TensorRef t = tensor_inputs[i].tensor;
                if (!tensor_inputs[i].has_device && device_holding_tensor) {
                    t = A.autorelease(t->to(device_holding_tensor->device()));
                }
                flat_inputs[i] = handle_from_tensor(A, _match_levels(A, t, tensor_inputs[i].levels, index_levels));
            }
        }
    }

    // previously we didn't know how many positional dimensions there would be so we couldn't number them right
    // so fill it in now.
    auto seen_positionals = 0;
    for (auto i : result_levels.reversed_enumerate()) {
        if (result_levels[i].is_positional()) {
            result_levels[i] = -(++seen_positionals);
        }
    }

    return IndexingInfo {false, requires_getindex, self_info.tensor, flat_inputs, result_levels, self_info.has_device};
}
namespace{
mpy::object __getitem__(Arena & A, mpy::handle self, mpy::handle index) {
    maybeInitializeGlobals();
    auto iinfo = getsetitem(A, self, index, has_dims(self));
    if (iinfo.can_call_original) {
        return mpy::object::checked_steal(THPVariable_getitem(self.ptr(), index.ptr()));
    }

    return invoke_getitem(A, iinfo);
}



void __setitem__(Arena & A, mpy::handle self, mpy::handle index, mpy::handle rhs) {
    maybeInitializeGlobals();
    auto iinfo = getsetitem(A, self, index, has_dims(self) || has_dims(rhs));
    if (iinfo.can_call_original) {
        if (-1 == THPVariable_setitem(self.ptr(), index.ptr(), rhs.ptr())) {
            throw mpy::exception_set();
        }
        return;
    }

    auto rhs_info = TensorInfo::create(A, rhs, false, false);
    if (rhs_info) { // otherwise rhs can be a scalar...
        for (auto l : rhs_info.levels) {
            if (!iinfo.result_levels.contains(l)) {
                if (l.is_positional()) {
                    mpy::raise_error(DimensionBindError(), "rhs contains too many dimensions (%d) compared to indexed value (%d)", ndim_of_levels(iinfo.result_levels), rhs_info.ndim());
                } else {
                    auto tup = levels_to_tuple(iinfo.result_levels);
                    mpy::raise_error(DimensionBindError(), "rhs of setitem contains dimension %R which is not in the dimension on the left (%R)", l.dim().ptr(), tup.ptr());
                }
            }
        }
        auto rhs_matched = _match_levels(A, rhs_info.tensor, rhs_info.levels, iinfo.result_levels);
        rhs = handle_from_tensor(A, rhs_matched);
    }
    self = handle_from_tensor(A, iinfo.self);

    if (iinfo.advanced_indexing) {
        auto tup = slice_to_tuple(iinfo.flat_inputs);
        if (-1 == THPVariable_setitem(self.ptr(), tup.ptr(), rhs.ptr())) {
            throw mpy::exception_set();
        }
    } else {
        torch_Tensor_copy_.call(self, rhs);
    }
}
}

PyObject* Tensor_getitem(PyObject* self, PyObject* index) {
    Arena A;
    PY_BEGIN
    return __getitem__(A, self, index).release();
    PY_END(nullptr);
}

int Tensor_setitem(PyObject* self, PyObject* index, PyObject* value) {
    Arena A;
    PY_BEGIN
    __setitem__(A, self, index, value);
    return 0;
    PY_END(-1);
}

namespace{
PyObject* py___getitem__(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    AT_ASSERT(nargs == 2);
    return __getitem__(A, args[0], args[1]).release();
    PY_END(nullptr)
}

PyObject* py___setitem__(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    AT_ASSERT(nargs == 3);
    __setitem__(A, args[0], args[1], args[2]);
    Py_RETURN_NONE;
    PY_END(nullptr)
}


PyObject* py_index(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    mpy::vector_args va(args, nargs, kwnames);
    mpy::handle self, dims, indices;
    va.parse("index", {"self", "dims", "indices"}, {&self, &dims, &indices}, 3);
    return index(A, self, dims, indices).release();
    PY_END(nullptr)
}


PyObject* py_stack(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    mpy::vector_args va(args, nargs, kwnames);
    mpy::handle tensors, new_dim, dim;
    va.parse("stack", {"tensors", "new_dim", "dim"}, {&tensors, &new_dim, &dim}, 2);

    Slice<DimEntry> result_levels;
    Slice<TensorInfo> infos;
    mpy::sequence_view sv(tensors);
    auto new_dim_d = Dim::wrap(new_dim);
    for (auto i : sv.enumerate()) {
        infos.append(A, TensorInfo::create(A, A.autorelease(sv[i]), false));
        for (auto l : infos.back().levels) {
            if (!result_levels.contains(l)) {
                result_levels.append(A, l);
            }
        }
    }
    new_dim_d->set_size(infos.size());
    std::vector<at::Tensor> inputs;
    inputs.reserve(infos.size());
    for (auto in : infos) {
        inputs.emplace_back(*_match_levels(A, in.tensor, in.levels, result_levels));
    }
    auto ndim = ndim_of_levels(result_levels);
    int64_t rawdim = 0;
    if (dim.ptr()) {
        auto d = _wrap_dim(dim, ndim, false);
        auto idx = result_levels.index(d);
        if (!idx) {
            mpy::raise_error(PyExc_TypeError, "Dimension %R does not exist in inputs", dim.ptr());
        }
        rawdim = *idx;
    }
    auto result = at::stack(inputs, rawdim);
    result_levels.insert(A, rawdim, new_dim_d);
    return Tensor::from_positional(A, std::move(result), result_levels, true).release();
    PY_END(nullptr)
}

PyObject* py_split(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    maybeInitializeGlobals();
    mpy::vector_args va(args, nargs, kwnames);
    mpy::handle self, split_size_or_sections, dim;
    va.parse("split", {"self", "split_size_or_sections", "dim"}, {&self, &split_size_or_sections, &dim}, 2);
    bool dim_is_object = dim.ptr() && Dim::check_exact(dim);
    Slice<mpy::handle> sizes;

    bool all_dims = true;
    bool all_ints = true;

    if (!mpy::is_int(split_size_or_sections)) {
        mpy::sequence_view sv(split_size_or_sections);
        for (auto i : sv.enumerate()) {
            sizes.append(A, A.autorelease(sv[i]));
            if (Dim::check_exact(sizes.back())) {
                all_ints = false;
            } else {
                all_dims = false;
            }
        }
    }
    if (all_ints) {
        if (dim_is_object) {
            mpy::raise_error(PyExc_TypeError, "when dim is specified as a Dim object, split sizes must also be dimensions.");
        }
        // call original split (if self has dimensions this will use torch function to do the split)
        return torch_Tensor_split.call_vector(mpy::vector_args(args, nargs, kwnames)).release();
    }
    if (!all_dims) {
        mpy::raise_error(PyExc_TypeError, "split list must be ints or dims but got a mix");
    }

    auto self_info = TensorInfo::create(A, self, false);
    auto ndim = self_info.ndim();
    if (!dim_is_object&& ndim == 0) {
        mpy::raise_error(PyExc_TypeError, "split expects at least a 1-dimension tensor");
    }
    DimEntry dim_l = dim.ptr() ? _wrap_dim(dim, ndim, false) : -ndim;

    auto idx = self_info.levels.index(dim_l);
    if (!idx) {
        if (!dim.ptr()) {
            dim = A.autorelease(mpy::from_int(0));
        }
        mpy::raise_error(PyExc_TypeError, "tensor does not comtain dimension %R", dim.ptr());
    }
    Slice<int64_t> indices;

    int64_t total_size = 0;
    Slice<int64_t> unbound;
    for (auto i : sizes.enumerate()) {
        auto d = Dim::unchecked_wrap(sizes[i]);
        if (d->is_bound()) {
            indices.append(A, d->size());
            total_size += indices.back();
        } else {
            indices.append(A, 0);
            unbound.append(A, i);
        }
    }
    auto tensor_size = self_info.tensor->sizes()[*idx];

    if (unbound.size()) {
        if (total_size > tensor_size) {
           mpy::raise_error(PyExc_TypeError, "sizes of target dimensions add up to more (%d) than source dim (%d)", int(total_size), int(tensor_size));
        }
        auto remaining_size = tensor_size - total_size;
        auto chunk_size = (remaining_size + unbound.size() - 1) / unbound.size();
        for (auto u : unbound) {
            auto sz = std::min(chunk_size, remaining_size);
            Dim::unchecked_wrap(sizes[u])->set_size(sz);
            indices[u] = sz;
            remaining_size -= sz;
        }
    } else if (tensor_size != total_size) {
        mpy::raise_error(PyExc_TypeError, "sum of sizes of target dimensions (%d) do not match the than source dim (%d)", int(total_size), int(tensor_size));
    }

    auto result_tensors = self_info.tensor->split_with_sizes(at::IntArrayRef(indices.begin(), indices.end()), *idx);
    mpy::tuple result(result_tensors.size());
    Slice<DimEntry> new_levels;
    new_levels.extend(A, self_info.levels);
    for (auto i : sizes.enumerate()) {
        new_levels[*idx] = Dim::unchecked_wrap(sizes[i]);
        result.set(i, Tensor::from_positional(A, std::move(result_tensors[i]), new_levels, true));
    }

    return result.release();

    PY_END(nullptr)
}

Slice<DimEntry> _wrap_dims(Arena& A, mpy::handle d, size_t N, bool keepdim) {
    auto de = _wrap_dim(d, N, keepdim);
    Slice<DimEntry> r;
    if (!de.is_none()) {
        r.append(A, de);
    } else {
        mpy::sequence_view sq(d);
        for (auto i : sq.enumerate()) {
            r.append(A, _wrap_dim(A.autorelease(sq[i]), N, keepdim));
        }
    }
    return r;
}

struct WrappedOperator : public mpy::base<WrappedOperator> {
    mpy::object orig;
    PyMethodDef method_def;
    mpy::object name, doc;

    bool is_pointwise = false;
    int64_t dim_offset = 0;
    int64_t keepdim_offset = 1;
    std::string dim_name;
    bool single_dim = false;
    bool reduce = true;

    static PyTypeObject Type;

    void init(mpy::object orig_, PyCFunction wrapper_implementation, std::string dim_name_="") {
        orig = std::move(orig_);
        method_def.ml_meth = wrapper_implementation;
        name = orig.attr("__name__");
        doc = orig.attr("__doc__");
        dim_name = std::move(dim_name_);
        if (!mpy::is_none(doc) && !dim_name.empty()) {
            doc = mpy::unicode_from_format("%S\nArgument '%s' can be either an integer or a torchdim.Dim object.\n", doc.ptr(), dim_name.c_str());
        }
        method_def.ml_name = mpy::is_none(name) ? "" : PyUnicode_AsUTF8(name.ptr());
        method_def.ml_doc = mpy::is_none(doc) ? "" : PyUnicode_AsUTF8(doc.ptr());
        method_def.ml_flags = METH_FASTCALL | METH_KEYWORDS;
    }

    mpy::object function() {
        return mpy::object::checked_steal(PyCFunction_New(&method_def, ptr()));
    }

};
}

PyTypeObject WrappedOperator::Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "_C.WrappedOperator",               /* tp_name */
    sizeof(WrappedOperator),               /* tp_basicsize */
    0,                              /* tp_itemsize */
    WrappedOperator::dealloc_stub,      /* tp_dealloc */
    0,                              /* tp_vectorcall_offset */
    0,                              /* tp_getattr */
    0,                              /* tp_setattr */
    0,                              /* tp_as_async */
    0,           /* tp_repr */
    0,                 /* tp_as_number */
    0,                 /* tp_as_sequence */
    0,             /* tp_as_mapping */
    0,      /* tp_hash */
    0,                              /* tp_call */
    0,                              /* tp_str */
    0,                              /* tp_getattro */
    0,                              /* tp_setattro */
    0,                              /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT, /* tp_flags */
    "Wrapped Object Holder",                   /* tp_doc */
    0,                              /* tp_traverse */
    0,                              /* tp_clear */
    0,  /* tp_richcompare */
    0,                              /* tp_weaklistoffset */
    0,                              /* tp_iter */
    0,                              /* tp_iternext */
    0,                /* tp_methods */
    0,                              /* tp_members */
    0,             /* tp_getset */
    0,                              /* tp_base */
    0,                              /* tp_dict */
    0,                              /* tp_descr_get */
    0,                              /* tp_descr_set */
    0,                              /* tp_dictoffset */
    0,            /* tp_init */
    0,                              /* tp_alloc */
    WrappedOperator::new_stub,                      /* tp_new */
};

namespace{
PyObject* patched_dim_method(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    auto self = WrappedOperator::unchecked_wrap(self_);
    PY_BEGIN

    mpy::vector_args va(args, nargs, kwnames);

    auto _getarg = [&](const char* name, int64_t offset_) -> mpy::handle {
        auto offset = offset_ + 1; // do not include self
        auto idx = va.index(name, offset);
        return idx == -1 ? mpy::handle() : va[idx];
    };
    Slice<mpy::handle> patched_args;
    patched_args.extend(A, va.begin(), va.end());
    auto _patcharg = [&](const char* name, int64_t offset_, mpy::handle value) {
        auto offset = offset_ + 1; // do not include self
        auto idx = va.index(name, offset);
        if (idx == -1) {
            mpy::raise_error(PyExc_ValueError, "Missing argument %s", name);
        }
        patched_args[idx] = value;
    };

    auto dim = _getarg(self->dim_name.c_str(), self->dim_offset);
    if (!dim.ptr()) {
        auto info = TensorInfo::create(A, args[0], true);
        EnableAllLayers l(A, info.levels);
        l.inplace_update_layers(info.batchedtensor, info.levels);
        patched_args[0] = handle_from_tensor(A, info.batchedtensor);
        auto r = self->orig.call_vector(patched_args.begin(), nargs, kwnames);
        return l.from_batched(A, THPVariable_Unpack(r.ptr()), info.has_device).release();
    }

    auto info = TensorInfo::create(A, args[0]);
    auto keepdim = false;
    if (self->reduce) {
        auto py_keepdim = _getarg("keepdim", self->keepdim_offset);
        if (py_keepdim.ptr()) {
            keepdim = mpy::to_bool(py_keepdim);
        }
    }

    auto ndim = info.ndim();
    auto dims = _wrap_dims(A, dim, ndim, keepdim);
    Slice<int64_t> dim_indices;
    auto seen = A.allocate<bool>(info.levels.size());
    std::fill(seen, seen + info.levels.size(), false);

    for (auto d : dims) {
        auto midx = info.levels.index(d);
        if (!midx) {
            auto tup = levels_to_tuple(info.levels);
            mpy::raise_error(PyExc_ValueError, "Tensor with dimensions %R does not contain one of %R\n", tup.ptr(), dim.ptr());
        }
        seen[*midx] = true;
        dim_indices.append(A, *midx);
    }
    Slice<DimEntry> new_levels;
    if (self->reduce && !keepdim) {
        for (auto i : info.levels.enumerate()) {
            if (!seen[i]) {
                new_levels.append(A, info.levels[i]);
            }
        }
    } else {
        new_levels = info.levels;
    }
    mpy::object py_indices;
    if (dim_indices.size() == 1) {
        py_indices = mpy::from_int(dim_indices[0]);
    } else {
        mpy::tuple tup(dim_indices.size());
        for (auto i : dim_indices.enumerate()) {
            tup.set(i, mpy::from_int(dim_indices[i]));
        }
        py_indices = std::move(tup);
    }
    _patcharg(self->dim_name.c_str(), self->dim_offset, py_indices);
    patched_args[0] = handle_from_tensor(A, info.tensor);
    auto r = self->orig.call_vector(patched_args.begin(), nargs, kwnames);
    auto wrap = [&](mpy::handle h) {
        if (THPVariable_Check(h.ptr())) {
            return A.autorelease(Tensor::from_positional(A, THPVariable_Unpack(h.ptr()), new_levels, info.has_device));
        }
        return h;
    };
    return tree_map(A, wrap, r).release();
    PY_END(nullptr)
}

PyObject* _wrap(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN

    #define ARGS(_) _(mpy::handle, orig) _(mpy::handle, dim_offset) _(mpy::handle, keepdim_offset) \
                    _(mpy::handle, dim_name) _(mpy::handle, single_dim) _(mpy::handle, reduce)
    MPY_PARSE_ARGS_KWNAMES("O|OOOOO", ARGS)

    std::string dim_name_str;
    if (dim_name.ptr()) {
        dim_name_str = PyUnicode_AsUTF8(dim_name.ptr());
    } else {
        dim_name_str = "dim";
    }
    auto info = WrappedOperator::create(mpy::object::borrow(orig), (PyCFunction)(void*) patched_dim_method, std::move(dim_name_str));
    if (dim_offset.ptr()) {
        info->dim_offset = mpy::to_int(dim_offset);
    }
    if (keepdim_offset.ptr()) {
        info->keepdim_offset = mpy::to_int(keepdim_offset);
    }

    if (single_dim.ptr()) {
        info->single_dim = mpy::to_bool(single_dim);
    }
    if (reduce.ptr()) {
        info->reduce = mpy::to_bool(reduce);
    }
    return info->function().release();
    #undef ARGS

    PY_END(nullptr)
}

PyObject* call_torch_function(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    Arena A;
    maybeInitializeGlobals();
    auto info = WrappedOperator::unchecked_wrap(self);
    return __torch_function__(A, info->orig, mpy::vector_args(args, nargs, kwnames), info->is_pointwise).release();
    PY_END(nullptr)
}

PyObject* _wrap_method(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    AT_ASSERT(nargs == 2);
    // XXX - ignore python function wrapped, we will call torch function directly
    mpy::handle orig = args[0];
    if (!pointwise.ptr()) {
        auto dim = mpy::import("functorch.dim");
        pointwise = dim.attr("pointwise");
    }
    auto info = WrappedOperator::create(mpy::object::borrow(orig), (PyCFunction)(void*) call_torch_function);
    info->is_pointwise = pointwise.contains(orig);
    return PyInstanceMethod_New(info->function().release());
    PY_END(nullptr);
}


PyObject* Tensor_sum(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    maybeInitializeGlobals();
    mpy::vector_args va(args, nargs, kwnames);
    auto self_ = Tensor::unchecked_wrap(args[0]);
    auto d = self_->delayed();
    if (!d) {
        return _Tensor_sum.call_vector(va).release();
    }
    mpy::handle self, dim, keepdim, dtype;
    va.parse("sum", {"self", "dim", "keepdim", "dtype"}, {&self, &dim, &keepdim, &dtype}, 1, 1);

    if (dtype.ptr() || (keepdim.ptr() && mpy::to_bool(keepdim))) {
        // std::cout << "SKIPPING fusion because dtype or keepdim=True specified\n";
        return _Tensor_sum.call_vector(va).release();
    }
    auto levels = self_->levels();

    auto N = ndim_of_levels(levels);
    auto reduced_dims = _wrap_dims(A, dim, N, false);

    return dot(A, TensorInfo::create(A, d->args[0], false), TensorInfo::create(A, d->args[1], false), reduced_dims).release();
    PY_END(nullptr)
}

PyObject* _parse_test(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    maybeInitializeGlobals();

    int required = mpy::to_int(args[0]);
    int kwonly = mpy::to_int(args[1]);

    mpy::vector_args va(args + 2, nargs - 2, kwnames);


    mpy::handle a, b, c, d;
    va.parse("_parse_test", {"a", "b", "c", "d"}, {&a, &b, &c, &d}, required, kwonly);
    mpy::tuple r(4);
    r.set(0, mpy::object::borrow(a.ptr() ? a : Py_None));
    r.set(1, mpy::object::borrow(b.ptr() ? b : Py_None));
    r.set(2, mpy::object::borrow(c.ptr() ? c : Py_None));
    r.set(3, mpy::object::borrow(d.ptr() ? d : Py_None));
    return r.release();

    PY_END(nullptr)
}

PyObject* _set_pointwise_optimize(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    mpy::handle value;
    mpy::vector_args va(args, nargs, kwnames);
    va.parse("_set_pointwise_optimization", {"value"}, {&value}, 1);
    pointwise_optimize = mpy::to_bool(value);
    Py_RETURN_NONE;
    PY_END(nullptr)
}

PyObject* _patch_tensor_class(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN

    auto torch = mpy::import("torch");
    auto py_TensorBase = torch.attr("_C").attr("TensorBase");
    replaceMappingIfMatches(py_TensorBase);

    Py_RETURN_NONE;
    PY_END(nullptr)
}


const char* dims_doc = R"""(
dims(n=None, sizes=None) -> torchdim.Dim or Tuple[torchdim.Dim, ...]

Creates and returns one or more Dim objects.

Arg:
    n (int, optional): The number of dimensions to create. Can be omitted if sizes is specified.
    sizes (List[Optional[int]], optional): A list the same size as the number of dimensions to be
      created, specifying each dimensions size, or None to leave the size unset.

Example::
    >>> batch, channel, width, height = dims(4)
    >>> batch, channel, width, height = dims(sizes=[None, 3, 224, 224])
)""";

PyMethodDef methods[] = {
    {"dims", (PyCFunction)(void*) _dims<create_dim>, METH_FASTCALL | METH_KEYWORDS, dims_doc},
    {"dimlists", (PyCFunction)(void*) _dims<create_dimlist>, METH_FASTCALL | METH_KEYWORDS},
    {"_test_c", (PyCFunction)(void*) test_c, METH_FASTCALL | METH_KEYWORDS},
    {"_wrap_method", (PyCFunction)(void*) _wrap_method, METH_FASTCALL | METH_KEYWORDS},
    {"Tensor_from_positional", (PyCFunction)(void*) py_Tensor_from_positional, METH_FASTCALL | METH_KEYWORDS},
    {"__torch_function__", (PyCFunction)(void*) py___torch_function__, METH_FASTCALL | METH_KEYWORDS},
    {"tree_flatten", (PyCFunction)(void*) py_tree_flatten, METH_FASTCALL | METH_KEYWORDS},
    {"order", (PyCFunction)(void*) order, METH_FASTCALL | METH_KEYWORDS},
    {"index", (PyCFunction)(void*) py_index, METH_FASTCALL | METH_KEYWORDS},
    {"stack", (PyCFunction)(void*) py_stack, METH_FASTCALL | METH_KEYWORDS},
    {"split", (PyCFunction)(void*) py_split, METH_FASTCALL | METH_KEYWORDS},
    {"expand", (PyCFunction)(void*) expand, METH_FASTCALL | METH_KEYWORDS},
    {"__getitem__", (PyCFunction)(void*) py___getitem__, METH_FASTCALL | METH_KEYWORDS},
    {"__setitem__", (PyCFunction)(void*) py___setitem__, METH_FASTCALL | METH_KEYWORDS},
    {"_wrap", (PyCFunction)(void*) _wrap, METH_FASTCALL | METH_KEYWORDS},
    {"Tensor_sum", (PyCFunction)(void*) Tensor_sum, METH_FASTCALL | METH_KEYWORDS},
    {"_parse_test", (PyCFunction)(void*) _parse_test, METH_FASTCALL | METH_KEYWORDS},
    {"_set_pointwise_optimize", (PyCFunction)(void*) _set_pointwise_optimize, METH_FASTCALL | METH_KEYWORDS},
    {"_patch_tensor_class", (PyCFunction)(void*) _patch_tensor_class, METH_FASTCALL | METH_KEYWORDS},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_C",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    methods
};
}

PyObject* Dim_init() {
    Arena A;
    try {
        mpy::object mod = mpy::object::checked_steal(PyModule_Create(&module_def));
        Dim::ready(mod, "Dim");
        DimList::ready(mod, "DimList");
        Tensor::ready(mod, "Tensor");
        WrappedOperator::ready(mod, "_WrappedOperator");
        Py_INCREF(&PyInstanceMethod_Type);
        PyModule_AddObject(mod.ptr(), "_instancemethod", (PyObject *)&PyInstanceMethod_Type);

        initializeGlobals(A);
        return mod.release();
    } catch(mpy::exception_set& err) {
        return nullptr;
    }
}

#endif
