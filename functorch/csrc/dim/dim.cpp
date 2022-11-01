// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "minpybind.h"
#include <frameobject.h>
#include <opcode.h>
#include <utility>
#include <new>
#include <iostream>
#include <vector>
//#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/python_compat.h>
#include <torch/csrc/Export.h>
#include <ATen/functorch/BatchedTensorImpl.h>
#include <ATen/functorch/DynamicLayer.h>
#include <ATen/ATen.h>
#include <memory>
#include "arena.h"
#include "python_variable_simple.h"


// C++ API functions for objects to
// * construct the object, returning a ref-counted handle
// * The actual API, with methods that take/return C-typed values

// extend minpybind.h to include
// * typed handles so that -> can get to their raw API
// * object/handle distinction for the typed handles

// class Dim: ---------------
py::handle torch_Tensor___mul__;
py::handle _Tensor;
py::handle _Tensor_sum;
py::handle NamedTuple;
py::dict_view pointwise;
py::handle torch_Tensor_expand;
binaryfunc THPVariable_getitem;
objobjargproc THPVariable_setitem;
py::handle no_slice;
PyTypeObject* torch_Tensor;
py::handle torch_Tensor_copy_;
py::handle torch_Tensor_split;
bool pointwise_optimize = true;
PyTypeObject* DimType = nullptr;

static void maybeInitializeGlobals() {
    // globals that depend on the python dim library,
    // which we can't lookup until we finish initializing the _C module
    if (_Tensor.ptr()) {
        return;
    }
    auto dim = py::import("functorch.dim");
    _Tensor = dim.attr("_Tensor");
    pointwise = dim.attr("pointwise");
    _Tensor_sum = _Tensor.attr("sum");
    DimType = (PyTypeObject*) py::import("functorch.dim").attr("Dim").ptr();
}

PyObject* Tensor_getitem(PyObject* self, PyObject* index);
int Tensor_setitem(PyObject* self, PyObject* index, PyObject* value);

void replaceMappingIfMatches(py::handle tp) {
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
        py::list_view lv(result);
        for (auto i : lv.enumerate()) {
            replaceMappingIfMatches(lv[i]);
        }
    }
}

static void initializeGlobals(Arena & A) {
    auto torch = py::import("torch");
    torch_Tensor = (PyTypeObject*) torch.attr("Tensor").ptr();
    torch_Tensor___mul__ = torch.attr("Tensor").attr("__mul__");

    torch_Tensor_expand = torch.attr("_C").attr("_TensorBase").attr("expand");
    torch_Tensor_split = torch.attr("_C").attr("_TensorBase").attr("split");
    torch_Tensor_copy_ = torch.attr("Tensor").attr("copy_");
    auto py_TensorBase = torch.attr("_C").attr("_TensorBase");
    auto TensorBase = (PyTypeObject*) py_TensorBase.ptr();
    THPVariable_getitem = TensorBase->tp_as_mapping->mp_subscript;
    THPVariable_setitem = TensorBase->tp_as_mapping->mp_ass_subscript;
    NamedTuple = py::import("typing").attr("NamedTuple");
    no_slice = PySlice_New(NULL, NULL, NULL);

}

py::handle DimensionBindError_;
static py::handle DimensionBindError() {
    if(!DimensionBindError_.ptr()) {
        DimensionBindError_ = py::import("functorch.dim").attr("DimensionBindError");
    }
    return DimensionBindError_;
}

static int64_t n_dims_created = 65;

struct Dim : public py::base<Dim> {
    int64_t level_; // for stable comparisons in prototype
    py::object name_;
    Dim()
    : level_(n_dims_created++) {}
    void init(py::object name, int64_t s = -1) {
        name_ = std::move(name);
        size_ = s;
    }

    static bool check_exact(py::handle v) {
        return Py_TYPE(v.ptr()) == DimType;
    }

    int64_t size() const {
        if (size_ == -1) {
            py::raise_error(PyExc_ValueError, "dimension %S is unbound", name_.ptr());
        }
        return size_;
    }
    void set_size(int64_t v) {
        if (size_ == -1) {
            size_ = v;
        } else if(size_ != v) {
            py::raise_error(DimensionBindError(), "Dim '%R' previously bound to a dimension of size %lld cannot bind to a dimension of size %lld", this, this->size_, v);
        }
    }
    bool is_bound() const {
        return size_ != -1;
    }
    static py::obj<Dim> create(py::object name, int64_t s = -1) {
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
    int64_t size_;
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
    py::hdl<Dim> dim() const {
        Dim* result;
        std::memcpy(&result, &data_, sizeof(Dim*));
        return py::hdl<Dim>(result);
    }

    DimEntry()
    : data_(0) {}

    DimEntry(int64_t pos)
    : data_(pos) {
        AT_ASSERT(pos < 0);
    }
    DimEntry(py::hdl<Dim> d) {
       std::memcpy(&data_, &d, sizeof(int64_t));
    }
    bool operator==(const DimEntry& rhs) const {
        return data_ == rhs.data_;
    }
private:
    int64_t data_;
};

std::ostream& operator<<(std::ostream& ss, DimEntry entry) {
    if (entry.is_none()) {
        ss << "None";
    } else if (entry.is_positional()) {
        ss << entry.position();
    } else {
        ss << entry.dim();
    }
    return ss;
}

// Dim wrapper methods

static int Dim_init(py::hdl<Dim> self, PyObject *args, PyObject *kwds) {
    PY_BEGIN
    static char* kwlist[] = {"name", "size", nullptr};
    py::handle name;
    py::handle size = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|O", kwlist, &name, &size)) {
        return -1;
    }
    self->init(py::object::borrow(name), (size.ptr() && !py::is_none(size)) ? py::to_int(size) : -1);
    return 0;
    PY_END(-1)
}

static PyObject* Dim_repr(Dim* self) {
    PY_BEGIN
    py::object name = (self->name_.ptr()) ? self->name_ : py::unicode_from_string("<uninitialized dim>");
    return name.release();
    PY_END(nullptr)
}


static PyObject* Dim_getsize(Dim* self, void*) {
    PY_BEGIN
    return py::from_int(self->size()).release();
    PY_END(nullptr)
}

int Dim_setsize(Dim* self, PyObject* size, void*) {
    PY_BEGIN
    self->set_size(py::to_int(size));
    return 0;
    PY_END(-1)
}

static PyObject* Dim_getis_bound(Dim* self, void*) {
    return PyBool_FromLong(self->is_bound());
}

static PyObject* Dim_getlevel(Dim* self, void*) {
    return PyLong_FromLong(self->level_);
}

static PyObject* Dim_get_levels(Dim* self, void*) {
    py::tuple t(1);
    t.set(0, py::object::borrow(self->ptr()));
    return t.release();
}

static PyObject* Dim_get_has_device(Dim* self, void*) {
    Py_RETURN_FALSE;
}

static PyObject* Dim_get_tensor(Dim* self, void*) {
    return THPVariable_Wrap(self->range());
}

static PyObject* Dim_get_batchtensor(Dim* self, void*) {
    return THPVariable_Wrap(self->batchtensor());
}


static PyGetSetDef Dim_getsetters[] = {
    {"size", (getter) Dim_getsize, (setter) Dim_setsize,
     "Dimension size", NULL},
    {"is_bound", (getter) Dim_getis_bound, NULL, "is_bound", NULL},
    {"_level", (getter) Dim_getlevel, NULL, "_level", NULL},
    {"_levels", (getter) Dim_get_levels, NULL, "_levels", NULL},
    {"_has_device", (getter) Dim_get_has_device, NULL, "_has_device", NULL},
    {"_tensor", (getter) Dim_get_tensor, NULL, "_tensor", NULL},
    {"_batchtensor", (getter) Dim_get_batchtensor, NULL, "_batchtensor", NULL},
    {"ndim", (getter) [](PyObject* self, void*) -> PyObject* { return py::from_int(1).release(); }, NULL, "ndim", NULL},
    {NULL}  /* Sentinel */
};

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
    (initproc)(void*) Dim_init,     /* tp_init */
    0,                              /* tp_alloc */
    Dim::new_stub,                      /* tp_new */
};

// class DimList ------------

struct DimList : public py::base<DimList> {
    py::object name_;
    std::vector<py::obj<Dim>> dims_;
    static PyTypeObject Type;
    void init(py::object name) {
        name_ = std::move(name);
    }
    void set_dims(std::vector<py::obj<Dim>> dims) {
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
                py::raise_error(DimensionBindError(), "Dimlist has size %lld but it is being bound to size %d", b_size, size);
            }
        } else {
            bound_ = true;
            dims_.resize(size);
            for (Py_ssize_t i = 0; i < size; ++i) {
                dims_[i] = Dim::create(py::unicode_from_format("%S%i", name_.ptr(), (int)i));
            }
        }
    }
    int64_t size() const {
        if (!bound_) {
            py::raise_error(DimensionBindError(), "DimList not bound");
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
        py::tuple t(size);
        for(size_t i = 0; i < size; ++i) {
            t.set(i, self->dims_[i]);
        }
        return py::repr(t).release();
    } else if(!py::is_none(self->name_)) {
        return py::unicode_from_format("*%S", self->name_.ptr()).release();
    } else {
        return py::unicode_from_string("<unbound_dimlist>").release();
    }
    PY_END(nullptr)
}

static PyObject* DimList_bind(DimList *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    py::handle sizes;
    static const char * const _keywords[] = {"sizes", nullptr};
    static _PyArg_Parser parser = {"O", _keywords, 0};
    if (!_PyArg_ParseStackAndKeywords(args, nargs, kwnames, &parser, &sizes)) {
        return nullptr;
    }
    if (!py::is_sequence(sizes)) {
        py::raise_error(PyExc_ValueError, "expected a sequence");
    }
    py::sequence_view seq = sizes;
    auto size = seq.size();
    self->bind_len(size);
    for (Py_ssize_t i = 0; i < size; ++i) {
        self->dims_[i]->set_size(py::to_int(seq[i]));
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
    static _PyArg_Parser parser = {"i", _keywords, 0};
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

PyObject * DimList_item(DimList* self, Py_ssize_t idx) {
    PY_BEGIN
    if (!self->is_bound()) {
        py::raise_error(DimensionBindError(), "DimList not bound");
    }
    if (idx < 0 || (size_t) idx >= self->dims_.size()) {
        py::raise_error(PyExc_IndexError, "index out of bounds");
    }
    py::object r = self->dims_[idx];
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


static PyObject* DimList_subscript(DimList* self, py::handle idx) {
    PY_BEGIN
    if (py::is_int(idx)) {
        return DimList_item(self, py::to_int(idx));
    } else if (py::is_slice(idx)) {
        if (!self->is_bound()) {
            py::raise_error(DimensionBindError(), "DimList not bound");
        }
        py::slice_view s(idx, self->dims_.size());
        py::tuple r(s.slicelength);
        for (Py_ssize_t i = s.start, j = 0; i < s.stop; i += s.step) {
            r.set(j++,  self->dims_[i]);
        }
        return r.release();
    } else {
        py::raise_error(PyExc_ValueError, "expected an int or a slice");
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
    static char* kwlist[] = {"len_or_dims", "name", nullptr};
    py::handle len_or_dims = nullptr;
    PyObject* name = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|OO", kwlist, &len_or_dims, &name)) {
        return -1;
    }
    self->init(py::object::borrow(name ? name : Py_None));
    if (len_or_dims.ptr()) {
        if(py::is_int(len_or_dims)) {
            self->bind_len(py::to_int(len_or_dims));
        } else if (py::is_sequence(len_or_dims)) {
            py::sequence_view s(len_or_dims);
            std::vector<py::obj<Dim>> dims;
            size_t size = s.size();
            dims.reserve(size);
            for (size_t i = 0; i < size; ++i) {
                auto r = s[i];
                if (py::is_int(r)) {
                    dims.emplace_back(Dim::create(py::unicode_from_format("%S%i", self->name_.ptr(), (int)i),  py::to_int(r)));
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
at::Tensor _add_batch_dims(Arena& A, at::Tensor t, Slice<DimEntry> levels_);
static py::object run_torch_function(Arena &A, py::handle orig, py::vector_args args, bool is_pointwise);
void free_levels_dims(Slice<DimEntry> levels);

struct Tensor;

struct DelayedOperator {
    DelayedOperator(py::object o, py::vector_args a)
    : orig(std::move(o)), args(a) {
        auto all = a.size();
        // this will outlive the call so
        // take ownership of temporaries
        // in vector args
        auto buf = new py::handle[all];
        memcpy(buf, args.args, sizeof(py::handle)*all);
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
    py::object orig;
    py::vector_args args;
};

struct Tensor : public py::base<Tensor> {
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

    static bool check_exact(py::handle v) {
       return Py_TYPE(v.ptr()) == TensorType;
    }


    static py::obj<Tensor> create() {
        if (!TensorType) {
            TensorType = (PyTypeObject*) py::import("functorch.dim").attr("Tensor").ptr();
        }
        return Tensor::alloc(TensorType);
    }
    void capture_levels(Slice<DimEntry> levels) {
        // grab ownership of the dims inside levels
        for (auto l : levels) {
            if (!l.is_positional()) {
                py::object::borrow(l.dim()).release();
            }
        }
        levels_.set(levels, free_levels_dims);
    }
    static py::object from_positional(Arena & A, at::Tensor tensor, Slice<DimEntry> levels, bool has_device);
    static py::obj<Tensor> create_delayed(py::object op, py::vector_args args, Slice<DimEntry> levels, bool has_device);
    friend struct EnableAllLayers;
};

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

void free_levels_dims(Slice<DimEntry> levels) {
    for(auto e : levels) {
        if (!e.is_positional()) {
            py::object::steal(e.dim());
        }
    }
}

// version in header does a unnecessary refcount +/-
inline at::functorch::BatchedTensorImpl* maybeGetBatchedImpl(const at::Tensor& tensor) {
    if (at::functorch::isBatchedTensor(tensor)) {
        return static_cast<at::functorch::BatchedTensorImpl*>(tensor.unsafeGetTensorImpl());
    }
    return nullptr;
}

inline TensorRef unchecked_tensor_from(py::handle p) {
    auto v = (THPVariable*) p.ptr();
    return TensorRef(*v->cdata);
}

int64_t ndim_of_levels(Slice<DimEntry> levels) {
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

    static TensorInfo create(Arena& A, py::handle h, bool ensure_batched=true, bool ensure_present=true) {
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
                py::raise_error(PyExc_ValueError, "expected a tensor object");
            }
            return TensorInfo {};
        }
    }


};

py::object Tensor::from_positional(Arena & A, at::Tensor tensor, Slice<DimEntry> levels, bool has_device) {
    size_t seen_dims = 0;
    int last = 0;
    //auto sz = tensor.sizes();
    for (auto i : levels.enumerate()) {
        auto l = levels[i];
        if (l.is_positional()) {
            AT_ASSERT(last == 0 || last + 1 == l.position());
            last = l.position();
        } else {
            py::object::borrow(l.dim()).release();
            //AT_ASSERT(sz[i] == l.dim()->size());
            ++seen_dims;
        }
    }
    AT_ASSERT(last == 0 || last == -1);
    if (!seen_dims) {
        return py::object::steal(THPVariable_Wrap(std::move(tensor)));
    }

    py::obj<Tensor> self = Tensor::create();
    self->tensor_ = std::move(tensor);
    AT_ASSERT(self->tensor_.dim() == levels.size());
    self->levels_.set(levels, free_levels_dims);
    self->has_device_ = has_device;
    py::object r = std::move(self);
    return r;
}


static PyObject* py_Tensor_from_positional(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    #define ARGS(_) _(py::handle, tensor) _(py::handle, py_levels) _(int, has_device)
    MPY_PARSE_ARGS_KWNAMES("OOp", ARGS)
    #undef ARGS

    if (!THPVariable_Check(tensor.ptr())) {
        py::raise_error(PyExc_ValueError, "_tensor is not a Tensor?");
    }

    Slice<DimEntry> levels;
    py::sequence_view sq(py_levels);
    for (auto i : sq.enumerate()) {
        py::object v = sq[i];
        if (py::is_int(v)) {
            auto vi = py::to_int(v);
            levels.append(A, vi);
        } else {
            auto dim = Dim::wrap(std::move(v));
            py::hdl<Dim> hdim = dim;
            levels.append(A, hdim);
        }
    }
    return Tensor::from_positional(A, THPVariable_Unpack(tensor.ptr()), levels, has_device != 0).release();
    PY_END(nullptr)
}

py::obj<Tensor> Tensor::create_delayed(py::object op, py::vector_args args, Slice<DimEntry> levels, bool has_device) {
    py::obj<Tensor> self = Tensor::create();
    self->capture_levels(levels);
    self->has_device_ = has_device;
    self->delayed_ = std::make_unique<DelayedOperator>(op, args);
    return self;
}

py::list slice_to_list(Slice<py::handle> h) {
    py::list lst(h.size());
    for (auto i : h.enumerate()) {
        lst.set(i, py::object::borrow(h[i]));
    }
    return lst;
}

py::tuple slice_to_tuple(Slice<py::handle> h) {
    py::tuple lst(h.size());
    for (auto i : h.enumerate()) {
        lst.set(i, py::object::borrow(h[i]));
    }
    return lst;
}

enum UType {
    U_ELEM,
    U_TUPLE_LIKE,
    U_DICT,
};

struct Unflatten {
    py::object operator()(Slice<py::handle>& elements) {
        py::object r;
        switch (type) {
            case U_ELEM: {
                r = py::object::borrow(elements[0]);
                elements = elements.slice(1);
            } break;
            case U_TUPLE_LIKE: {
                py::tuple tup(children.size());
                for (auto i : children.enumerate()) {
                    tup.set(i, children[i](elements));
                }
                r = obj.call(tup);
            } break;
            case U_DICT: {
                r = py::object::checked_steal(PyDict_New());
                py::dict_view rv(r);
                py::dict_view d(obj);
                Py_ssize_t pos = 0;
                py::handle k, v;
                for (int i = 0; d.next(&pos, &k, &v); ++i) {
                    rv.set(k, children[i](elements));
                }
            } break;
        }
        return r;
    }
    UType type;
    py::handle obj;
    Slice<Unflatten> children;
};

Unflatten tree_flatten(Arena& A, py::handle agg, Slice<py::handle>& flat_elements) {
    Slice<Unflatten> c;
    UType utype;
    py::handle obj;
    if (py::list_view::check(agg)) {
        obj = agg.type();
        utype = U_TUPLE_LIKE;
        py::list_view l(agg);
        for (auto i : l.enumerate()) {
            c.append(A, tree_flatten(A, l[i], flat_elements));
        }
    } else if (py::tuple_view::check(agg)) {
        obj = agg.type();
        utype = U_TUPLE_LIKE;
        // includes named tuples
        py::tuple_view l(agg);
        for (auto i : l.enumerate()) {
            c.append(A, tree_flatten(A, l[i], flat_elements));
        }
    } else if (py::dict_view::check(agg)) {
        utype = U_DICT;
        py::dict_view d(agg);
        obj = agg;
        Py_ssize_t pos = 0;
        py::handle k, v;
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
    py::vector_args operator()(Arena& A, Slice<py::handle>& elements) {
        if (!had_nested) {
            auto args = elements.begin();
            elements = Slice<py::handle>();
            return py::vector_args(args, nargs, kwnames);
        }
        Slice<py::handle> args;
        for (auto u : children) {
            args.append(A, A.autorelease(u(elements)));
        }
        return py::vector_args(args.begin(), nargs, kwnames);
    }
    Slice<Unflatten> children;
    Py_ssize_t nargs;
    py::handle kwnames;
    bool had_nested;
};

UnflattenVectorArgs tree_flatten(Arena& A, py::vector_args args, Slice<py::handle>& flat_elements) {
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

static PyObject* py_unflatten(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    #define ARGS(_) _(py::handle, ns)
    MPY_PARSE_ARGS_KWNAMES("O", ARGS)
    #undef ARGS
    py::sequence_view sv(ns);
    // because we do not have a autorelase pool yet...
    Arena A;
    Slice<py::handle> slice;
    py::handle Tuple = (PyObject*) &PyTuple_Type;
    auto inputs = Tuple.call(ns);
    py::tuple_view tv(inputs);
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

static PyObject* py_tree_flatten(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    #define ARGS(_) _(py::handle, tree)
    MPY_PARSE_ARGS_KWNAMES("O", ARGS)
    #undef ARGS
    auto A = new UnflattenArena;
    Slice<py::handle> elements;
    A->unflatten = tree_flatten(A->A, tree, elements);
    auto cap = py::object::checked_steal(PyCapsule_New(A, "arena", free_unflatten_arena));
    auto unflatten = py::object::checked_steal(PyCFunction_New(&py_unflatten_def, cap.release()));
    py::tuple r(2);
    r.set(0, slice_to_list(elements));
    r.set(1, std::move(unflatten));
    return r.release();
    PY_END(nullptr)
}



py::object tree_map(Arena& A, std::function<py::handle(py::handle)> fn, py::handle agg) {
    Slice<py::handle> elements;
    auto unflatten = tree_flatten(A, agg, elements);
    for (auto i : elements.enumerate()) {
        elements[i] = fn(elements[i]);
    }
    return unflatten(elements);
}

// prereq: isinstance(h, _Tensor)
inline int64_t _Tensor_ndim(py::handle h) {
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

inline py::handle handle_from_tensor(Arena& A, TensorRef t) {
    // fast case: tensor is live in python
    c10::optional<PyObject*> mb_obj =
        t->unsafeGetTensorImpl()->check_pyobj(getPyInterpreter());
    if (mb_obj.has_value() && !t->unsafeGetTensorImpl()->owns_pyobj()) {
        return *mb_obj;
    }
    return A.autorelease(py::object::checked_steal(THPVariable_Wrap(*t)));
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
        std::sort(levels_to_dim_.begin(), levels_to_dim_.end(), [](py::hdl<Dim> lhs, py::hdl<Dim> rhs) { return lhs->level_ < rhs->level_;});

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

    py::obj<Tensor> from_batched(Arena& A, at::Tensor batchedtensor, bool has_device) {
        Slice<DimEntry> levels;
        for (auto i : irange(-batchedtensor.dim(), 0)) {
            levels.append(A, i);
        }
        TensorRef tensor;
        at::functorch::BatchedTensorImpl * impl = maybeGetBatchedImpl(batchedtensor);
        while(true) {
            auto level = impl->level();
            AT_ASSERT(level >= levels_start_ && level < levels_start_ + levels_to_dim_.size());
            py::hdl<Dim> dim = levels_to_dim_[level - levels_start_].ptr();
            levels.insert(A, impl->bdim(), dim);
            at::functorch::BatchedTensorImpl * nimpl = maybeGetBatchedImpl(impl->value());
            if (!nimpl) {
                tensor = impl->value();
                break;
            }
            impl = nimpl;
        }

        py::obj<Tensor> self = Tensor::create();
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
    int64_t levels_start_;
    Slice<py::hdl<Dim>> levels_to_dim_;
};

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

static py::object run_torch_function(Arena &A, py::handle orig, py::vector_args args, bool is_pointwise) {
    if (!pointwise_optimize) {
        is_pointwise = false;
    }
    // std::cout << "__torch_function__ " << ((is_pointwise) ? "pointwise" : "functorch") << " " << orig << "\n";

    Slice<py::hdl<Dim>> all_dims;
    Slice<py::handle> flat_args;
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

        Slice<py::handle> flat_it = flat_args;
        py::vector_args uargs = unflatten_args(A, flat_it);

        py::object result = orig.call_vector(uargs);

        // fast wrap for normal case where operator just returns a tensor.
        if (THPVariable_Check(result.ptr())) {
            return Tensor::from_positional(A, THPVariable_Unpack(result.ptr()), result_levels, device_holding_tensor);
        }
        auto wrap = [&](py::handle h) {
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
        Slice<py::handle> flat_it = flat_args;
        py::vector_args uargs = unflatten_args(A, flat_it);
        AT_ASSERT(flat_it.size() == 0);
        py::object result = orig.call_vector(uargs);
        auto wrap = [&](py::handle h) {
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


static py::object __torch_function__(Arena &A, py::handle orig, py::vector_args args, bool is_pointwise) {
    if (orig == torch_Tensor___mul__) {
        AT_ASSERT(args.nargs == 2 && !args.has_keywords());
        auto lhs = args[0];
        auto rhs = args[1];
        if (py::isinstance(lhs, _Tensor) && py::isinstance(rhs, _Tensor) && _Tensor_ndim(lhs) == 0 && _Tensor_ndim(rhs) == 0) {
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
            return Tensor::create_delayed(py::object::borrow(orig), args, levels, has_device);
        }
    }
    return run_torch_function(A, orig, args, is_pointwise);
}

py::vector_args as_vector_args(Arena& A, py::handle args, py::handle kwargs) {
    auto pos_args = (py::handle*) &PyTuple_GET_ITEM(args.ptr(), 0);
    auto pos_n = PyTuple_GET_SIZE(args.ptr());
    if (!kwargs.ptr()) {
        return py::vector_args(pos_args, pos_n, nullptr);
    }
    Slice<py::handle> all_args;
    Slice<py::handle> kwnames;
    all_args.extend(A, pos_args, pos_args + pos_n);
    py::dict_view dv(kwargs);
    Py_ssize_t pos = 0;
    py::handle key, value;
    while (dv.next(&pos, &key, &value)) {
        all_args.append(A, value);
        kwnames.append(A, key);
    }
    return py::vector_args(all_args.begin(), pos_n, A.autorelease(slice_to_tuple(kwnames)));
}

static PyObject* py___torch_function__(PyObject *self,
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

py::object levels_to_tuple(Slice<DimEntry> slice) {
    py::tuple t(slice.size());
    for (auto i : slice.enumerate()) {
        t.set(i, slice[i].is_positional() ?  py::from_int(slice[i].position()) : py::object::borrow(slice[i].dim()));
    }
    py::object r = std::move(t);
    return r;
}

PyObject* Tensor_ndim(Tensor* self, void*) {
    Py_ssize_t i = 0;
    for (auto l : self->levels()) {
        if (l.is_positional()) {
            ++i;
        }
    }
    return py::from_int(i).release();
}

static PyGetSetDef Tensor_getsetters[] = {
   {"_has_device", (getter) [](PyObject* self, void*) -> PyObject* { return py::from_bool(((Tensor*)self)->has_device()).release(); }, NULL},
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

static PyMethodDef Tensor_methods[] = {
    {NULL, NULL, 0, NULL}        /* Sentinel */
};


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

bool relevant_op(_Py_CODEUNIT c) {
    switch(_Py_OPCODE(c)) {
        case STORE_NAME:
        case STORE_GLOBAL:
        case STORE_FAST:
        case STORE_DEREF:
            return true;
        default:
            return false;
    }
}

py::object getname(PyCodeObject* code, _Py_CODEUNIT c) {
    PyObject* names = NULL;
    switch(_Py_OPCODE(c)) {
        case STORE_NAME:
        case STORE_GLOBAL:
          names = code->co_names;
          break;
        case STORE_FAST:
#if PY_VERSION_HEX < 0x030b0000
          names = code->co_varnames;
#else
          names = PyCode_GetVarnames(code);
#endif
          break;
        case STORE_DEREF:
#if PY_VERSION_HEX < 0x030b0000
          names = code->co_cellvars;
#else
          names = PyCode_GetCellvars(code);
#endif
          break;
        default:
            return py::object();
    }
    return py::object::steal(PySequence_GetItem(names, _Py_OPARG(c)));
}

py::object create_dim(py::object name, py::handle size) {
    auto d = Dim::create(std::move(name));
    if (!py::is_none(size)) {
        d->set_size(py::to_int(size));
    }
    return std::move(d);
}

py::object create_dimlist(py::object name, py::handle size) {
    auto d = DimList::create(std::move(name));
    if (!py::is_none(size)) {
        if (py::is_int(size)) {
            d->bind_len(py::to_int(size));
        } else {
            py::sequence_view s(size);
            d->bind_len(s.size());
            for (auto i : irange(d->size())) {
                d->dims_[i]->set_size(py::to_int(s[i]));
            }
        }
    }
    return std::move(d);
}



// Python wrappers that make new reflection primitives available for older runtimes
#if PY_VERSION_HEX < 0x030b0000
#define _PyCode_CODE(CO) ((_Py_CODEUNIT*)PyBytes_AS_STRING((CO)->co_code))
#endif

template<py::object (*create_object)(py::object, py::handle)>
static PyObject* _dims(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    Py_ssize_t specified_ndims = -1;
    Py_ssize_t found_ndims = 0;
    Py_ssize_t sizes = -1;
    py::handle n = Py_None;
    py::handle py_sizes = Py_None;

    if (nargs || kwnames) {
        py::vector_args va(args, nargs, kwnames);
        va.parse("dims", {"n", "sizes"}, {&n, &py_sizes}, 0);
        if (!py::is_none(py_sizes)) {
            sizes = py::sequence_view(py_sizes).size();
            specified_ndims = sizes;
        }
        if (!py::is_none(n)) {
            specified_ndims = py::to_int(n);
        }
    }

    PyThreadState* state = PyThreadState_GET();
    auto f = py::obj<PyFrameObject>::steal(PyThreadState_GetFrame(state));
    auto c = py::obj<PyCodeObject>::steal(PyFrame_GetCode(f.ptr()));
    auto code = _PyCode_CODE(c.ptr());
#if PY_VERSION_HEX >= 0x030a00f0
    int first = PyFrame_GetLasti(f.ptr()) + 1;
#else
    int first = PyFrame_GetLasti(f.ptr()) /  2 + 1;
#endif
    auto unpack = code[first];
    int names_start = first;
    if (relevant_op(unpack)) {
        found_ndims = 1;
    } else if (_Py_OPCODE(unpack) == UNPACK_SEQUENCE) {
        found_ndims = _Py_OPARG(unpack);
        names_start++;
    }

    if (specified_ndims == -1) {
        if (found_ndims == 0) {
            py::raise_error(PyExc_SyntaxError, "dims() must be assigned to a sequence of variable names or have argument n specified");
        }
        specified_ndims = found_ndims;
    }
    if (found_ndims != specified_ndims) {
        found_ndims = 0; // avoid taking the wrong names for dimensions
    }

    auto genobject = [&](int i) -> py::object {
        py::object name;
        if (i < found_ndims) {
            name = getname(c.ptr(), code[names_start + i]);
        }
        if (!name.ptr()) {
            name = py::unicode_from_format("d%d", i);
            found_ndims = 0; // once we fail at finding a name, we can find any more
        }
        return create_object(std::move(name), sizes != -1 ? py::sequence_view(py_sizes)[i] : py::handle(Py_None));
    };
    if (sizes != -1 && sizes != specified_ndims) {
        py::raise_error(PyExc_ValueError, "expected %d sizes but found %d", int(specified_ndims), int(sizes));
    }
    if (specified_ndims == 1) {
        return genobject(0).release();
    }
    py::tuple result(specified_ndims);
    for (int i = 0; i < specified_ndims; ++i) {
        result.set(i, genobject(i));
    }
    return result.release();
    PY_END(nullptr)
}

int64_t dim_index(const std::vector<py::obj<Dim>>& dims, py::hdl<Dim> dim) {
    for (int64_t i = 0, N  = dims.size(); i < N; ++i) {
        if (dims[i].ptr() == dim.ptr()) {
            return i;
        }
    }
    return -1;
}


struct DotPart {
    Slice<DimEntry> dims;
    size_t total_size = 1;
    void append(Arena& A, py::hdl<Dim> d) {
        total_size *= d->size();
        dims.append(A, d);
    }
};

template<typename T>
static at::ArrayRef<T> as_array_ref(Slice<T> t) {
    return at::ArrayRef<T>(t.begin(), t.end());
}

TensorRef dot_prepare(Arena& A, std::initializer_list<DotPart> parts, const TensorInfo& t) {
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

py::object dot_finish(Arena& A, std::initializer_list<DotPart> parts, at::Tensor r) {
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



py::object dot(Arena& A, TensorInfo lhs, TensorInfo rhs, Slice<DimEntry> sum) {
    auto lhs_strides = lhs.tensor->strides();
    auto rhs_strides = rhs.tensor->strides();

    DotPart lro_dims;
    DotPart lo_dims;
    DotPart ro_dims;
    DotPart lr_dims;

    auto insert_dim = [&] (py::hdl<Dim> d, at::optional<int> lhs_idx, at::optional<int> rhs_idx) {
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
        insert_dim(d.dim(), at::nullopt, i);
    }

    if (lr_dims.dims.size() != sum.size()) {
        for (auto & d : sum) {
            if (!lhs.levels.contains(d) && !rhs.levels.contains(d)) {
                py::raise_error(DimensionBindError(), "summing over non-existant dimension %S", d.dim().ptr());
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

static DimEntry _wrap_dim(py::handle d, size_t N, bool keepdim);

static PyObject* order(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    if (kwnames) {
        py::raise_error(PyExc_TypeError, "unexpected keyword arguments %S", kwnames);
    }
    AT_ASSERT(nargs-- > 0);
    Slice<DimEntry> orig_levels;
    Slice<DimEntry> levels;
    TensorRef data;
    py::handle self = args++[0];
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
                py::raise_error(PyExc_ValueError, "tensor has %d positional dimensions, but %d specified, or it was specified twice", int(orig_ndim), int(d.position() + orig_ndim));
            } else {
                py::raise_error(PyExc_ValueError, "tensor of dimensions %R does not contain dim %R or it was specified twice", levels_to_tuple(orig_levels).ptr(), d.dim().ptr());
            }
        }
        levels[*midx] = DimEntry();
        flat_positional_dims.append(A, d);
    };

    int n_new_positional = 0;
    for (auto i :irange(nargs)) {
        py::handle arg  = args[i];
        DimEntry entry = _wrap_dim(arg, orig_ndim, false);
        if (!entry.is_none()) {
            append(entry);
            ++n_new_positional;
        } else if (DimList::check(arg)) {
            auto dl = DimList::unchecked_wrap(arg);
            for (py::obj<Dim> & d : dl->dims_) {
                append(py::hdl<Dim>(d));
                ++n_new_positional;
            }
        } else {
            ++n_new_positional;
            if (!py::is_sequence(arg)) {
                py::raise_error(PyExc_ValueError, "expected a Dim, List[Dim], or Sequence[Dim]");
            }
            py::sequence_view sq(arg);
            auto N = sq.size();
            to_flatten.append(A, std::make_pair(flat_positional_dims.size(), N));
            for (auto j : irange(N)) {
                DimEntry e = _wrap_dim(A.autorelease(sq[j]), orig_ndim, false);
                if (e.is_none()) {
                    py::raise_error(PyExc_ValueError, "expected a Dim, or int");
                }
                append(e);
            }
        }
    }

    int ndim = 0;
    int insert_point = -1;
    Slice<DimEntry> new_levels;
    for (auto l : levels) {
        if (l.is_none()) {
            continue;
        }
        if (l.is_positional()) {
            ndim++;
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
            py::vector_args vargs(args - 1, nargs + 1, kwnames);
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
            py::raise_error(DimensionBindError(), "expanding dimension %R already exists in tensor with dims", d.ptr());
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


void _bind_dims_to_size(Arena & A, int64_t sz, int64_t sd,
                        Slice<py::hdl<Dim>> dims, Slice<int64_t>& nsz, Slice<int64_t>& nsd) {
    int64_t rhs_prod = 1;
    for (auto i : dims.enumerate()) {
        if (!dims[i]->is_bound()) {
            for (auto j : irange(i + 1, dims.size())) {
                if (!dims[j]->is_bound()) {
                    py::raise_error(DimensionBindError(), "cannot infer the sizes of two dimensions at once %R and %R", dims[i].ptr(), dims[j].ptr());
                }
                rhs_prod *= dims[j]->size();
            }
            if (sz % rhs_prod != 0) {
                py::tuple tup(dims.size());
                for (auto j : dims.enumerate()) {
                    tup.set(j, dims[j]->is_bound() ? py::from_int(dims[j]->size()) : py::unicode_from_string("?"));
                }
                py::raise_error(DimensionBindError(), "inferred dimension does not evenly fit into larger dimension: %d vs %R", (int) sz, tup.ptr());
            }
            int64_t inferred_size = sz / rhs_prod;
            dims[i]->set_size(inferred_size);
            rhs_prod = sz;
            break;
        }
        rhs_prod *= dims[i]->size();
    }
    if (rhs_prod != sz) {
        py::tuple tup(dims.size());
        for (auto j : dims.enumerate()) {
            tup.set(j, py::object::borrow(dims[j]));
        }
        py::raise_error(DimensionBindError(), "Dimension sizes to do not match (%d != %d) when matching dimension pack %R", (int) sz, (int) rhs_prod, tup.ptr());
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

inline bool has_dims(py::handle d) {
    return Dim::check_exact(d) || Tensor::check_exact(d);
}

struct IndexingInfo {
    bool can_call_original; // if true, then it is safe to just call getitem or setitem, these objects do not need special handling
    bool advanced_indexing; // requires actual lookup
    TensorRef self;
    Slice<py::handle> flat_inputs;
    Slice<DimEntry> result_levels;
    bool has_device;
};

static Slice<py::handle> as_slice(py::tuple_view tv) {
    PyObject** begin = &PyTuple_GET_ITEM(tv.ptr(),0);
    return Slice<py::handle>((py::handle*)begin, (py::handle*) (begin + tv.size()));
}

static Slice<py::handle> as_slice(py::list_view tv) {
    PyObject** begin = &PyList_GET_ITEM(tv.ptr(),0);
    return Slice<py::handle>((py::handle*)begin, (py::handle*) (begin + tv.size()));
}


bool maybe_dimpack(Slice<py::handle>& elements, py::handle s, bool check_first=true) {
    // can we avoid rechecking?
    if (py::list_view::check(s)) {
        py::list_view tv(s);
        if (!check_first || (tv.size() && Dim::check_exact(tv[0]))) {
            elements = as_slice(tv);
            return true;
        }
    }
    // can we avoid rechecking?
    if (py::tuple_view::check(s)) {
        py::tuple_view tv(s);
        if (!check_first || (tv.size() && Dim::check_exact(tv[0]))) {
            elements = as_slice(tv);
            return true;
        }
    }
    return false;
};

bool is_dimpack(py::handle s) {
    Slice<py::handle> e;
    return maybe_dimpack(e, s);
}

IndexingInfo getsetitem_flat(Arena& A, TensorInfo self_info, Slice<py::handle> input, Slice<DimEntry> keys, Slice<py::handle> values, bool has_dimpacks_or_none);
static py::object invoke_getitem(Arena& A, const IndexingInfo& iinfo);

static py::object index(Arena& A, py::handle self, py::handle dims, py::handle indices) {
    maybeInitializeGlobals();
    Slice<py::handle> dims_list;
    Slice<py::handle> indices_list;
    // we allow for matching single dims to multiple dims,
    // so we first have to normalize everything into the case where there is a list on lhs and the rhs
    bool lhs_list = py::tuple_view::check(dims) || py::list_view::check(dims);
    bool rhs_list = py::tuple_view::check(indices) || py::list_view::check(indices);
    if (lhs_list && rhs_list) {
        py::sequence_view dv(dims);
        py::sequence_view ind(indices);
        Py_ssize_t N = dv.size();
        if (N != ind.size()) {
            py::raise_error(PyExc_TypeError, "dims (%d) and indices (%d) must have the same length", int(N), int(ind.size()));
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
    auto parse_dim_entry = [&](py::handle s) -> DimEntry {
        auto d = _wrap_dim(s, ndim, false);
        if (d.is_none()) {
            py::raise_error(PyExc_TypeError, "expected a dimension specifyer but found %R", s.ptr());
        }
        return d;
    };
    auto dim_not_present = [&](DimEntry d) {
        if (d.is_positional()) {
            py::raise_error(PyExc_TypeError, "dimension %d not in tensor of %d dimensions", d.position() + ndim , ndim);
        } else {
            py::raise_error(PyExc_TypeError, "dimension %R not in tensor", d.dim()->ptr());
        }
    };

    for (auto i : dims_list.enumerate()) {
        Slice<py::handle> m;
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
        if (py::tuple_view::check(idx) || py::list_view::check(idx)) {
            has_dimpacks = true;
            break;
        }
    }
    IndexingInfo info = getsetitem_flat(A, self_info, Slice<py::handle>(), dims_list_flat, indices_list, has_dimpacks);
    return invoke_getitem(A, info);
}

// true -- the indices were flattend out of a tuple, list or sequence...

Slice<py::handle> slice_from_sequence(Arena& A, py::handle value) {
    if (py::tuple_view::check(value)) {
        return as_slice(py::tuple_view(value));
    } else if (py::list_view::check(value)) {
        return as_slice(py::list_view(value));
    } else {
        py::sequence_view sv(value);
        Slice<py::handle> r;
        for (auto i : sv.enumerate()) {
            r.append(A, A.autorelease(sv[i]));
        }
        return r;
    }
}

bool extractIndices(Arena& A, py::handle index, Slice<py::handle>& indices) {
    if (py::tuple_view::check(index)) {
        indices.extend(A, as_slice(py::tuple_view(index)));
        return true;
    } else if (THPVariable_Check(index.ptr())) {
        indices.append(A, index);
        return false;
    } else if (!py::is_sequence(index)) {
        indices.append(A, index);
        return false;
    }
    // a copy of treatSequenceAsTuple modified to add Dim and our wrapped tensors..
    py::sequence_view sv(index);
    if (sv.size() >= 32) {
        indices.extend(A, slice_from_sequence(A, index));
        return true;
    }
    for (auto i : sv.enumerate()) {
        py::handle item;
        try {
            item = sv[i];
        } catch (py::exception_set & e) {
            PyErr_Clear();
            indices.append(A, index);
            return false;
        }
        if (THPVariable_Check(item.ptr()) || py::is_sequence(item) || PySlice_Check(item.ptr()) || item.ptr() == Py_Ellipsis || py::is_none(item) || has_dims(item)) {
            indices.extend(A, slice_from_sequence(A, index));
            return true;
        }
    }
    indices.append(A, index);
    return false;
}

static IndexingInfo getsetitem(Arena & A, py::handle self, py::handle index, bool tensors_have_dims) {
    bool can_call_original_getitem = !tensors_have_dims;

    Slice<py::handle> input;
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
            py::raise_error(DimensionBindError(), "at most one ... or unbound dimension list can exist in indexing list but found 2 at offsets %d and %d", (int) expanding_object, (int) i);
        }
        expanding_object = i;
    };
    Slice<int64_t> dimlists;

    // calculate how many dimensioned have been indexed in order to compute the size of ...
    // or expand a potentially unbound dimension list.

    bool has_dimpacks_or_none = false;
    for (auto i : input.enumerate()) {
        py::handle s = input[i];
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
        } else if (py::is_none(s)) {
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
        py::raise_error(PyExc_ValueError, "at least %d indices were supplied but the tensor only has %d dimensions", (int) dims_indexed, (int) ndim);
    }
    // expand any unbound dimension list, or expand ... into individual : slices.
    auto expanding_dims = ndim - dims_indexed;
    if (expanding_object != -1) {
        if (unbound_dim_list) {
            unbound_dim_list->bind_len(expanding_dims);
        } else {
            // ...
            Slice<py::handle> no_slices;
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
        Slice<py::handle> more_dims((py::handle*) &*dl->dims_.begin(), (py::handle*) &*dl->dims_.end());
        input.insert(A, input.slice(idx, idx + 1), more_dims);
    }

    return getsetitem_flat(A, self_info, input, Slice<DimEntry>(), Slice<py::handle>(), has_dimpacks_or_none);
}

IndexingInfo getsetitem_flat(Arena& A, TensorInfo self_info, Slice<py::handle> input, Slice<DimEntry> keys, Slice<py::handle> values, bool has_dimpacks_or_none) {
    // At this point:
    // ..., DimList have been eliminated
    // Dim, Tensor, Tuple[Dim,...], int, slice still remain


    // we have to count how many times we see a dimension.
    // A[i,j] is a simple binding operation, but A[i, i+j] or A[i, i] requires advanced indexing.
    Slice<py::hdl<Dim>> seen_dims;
    Slice<int64_t> seen_dims_nuses;
    auto add_dim = [&](py::hdl<Dim> entry) {
        auto midx = seen_dims.index(entry);
        if (!midx) {
            seen_dims.append(A, entry);
            seen_dims_nuses.append(A, 1);
        } else {
            ++seen_dims_nuses[*midx];
        }
    };

    Slice<py::handle> input_it = input;

    Slice<py::handle> flat_inputs;
    // flat inputs will start with an empty py::handle if the
    // actual value is in the tensor-like object in the tensor info
    Slice<TensorInfo> tensor_inputs;

    auto append_flat_handle = [&](py::handle h) {
        flat_inputs.append(A, h);
        tensor_inputs.append(A, TensorInfo());
    };
    TensorRef device_holding_tensor;
    auto append_tensor_input = [&](TensorInfo ti) {
        flat_inputs.append(A, py::handle());
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
        while (input_it.size() && py::is_none(input_it[0])) {
            append_flat_handle(no_slice);
            nsz.append(A, 1);
            nsd.append(A, 0);
            input_it = input_it.slice(1);
        }
    };


    auto append_item = [&](int i, py::handle arg) {
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
            Slice<py::handle> mp;
            if (maybe_dimpack(mp, arg)) {
                // dim pack
                Slice<py::hdl<Dim>> dim_pack;
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
                py::handle arg = input_it[0];
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
                flat_inputs[i] = py::handle();
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
            if (!py::is_int(inp)) {
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

static py::object invoke_getitem(Arena& A, const IndexingInfo& iinfo) {
    at::Tensor rtensor;
    if (iinfo.advanced_indexing) {
        auto self_hdl = handle_from_tensor(A, iinfo.self);
        auto tup = slice_to_tuple(iinfo.flat_inputs);
        // std::cout << "calling original getindex " << self_hdl << " " << tup << "\n";
        auto pytensor = py::object::checked_steal(THPVariable_getitem(self_hdl.ptr(), tup.ptr()));
        rtensor = THPVariable_Unpack(pytensor.ptr());
    } else {
        // std::cout << "skipping original getindex\n";
        rtensor = *iinfo.self;
    }
    // std::cout << "returning (from_positional)\n";
    return Tensor::from_positional(A, std::move(rtensor), iinfo.result_levels, iinfo.has_device);
}

static py::object __getitem__(Arena & A, py::handle self, py::handle index) {
    maybeInitializeGlobals();
    auto iinfo = getsetitem(A, self, index, has_dims(self));
    if (iinfo.can_call_original) {
        return py::object::checked_steal(THPVariable_getitem(self.ptr(), index.ptr()));
    }

    return invoke_getitem(A, iinfo);
}


PyObject* Tensor_getitem(PyObject* self, PyObject* index) {
    Arena A;
    PY_BEGIN
    return __getitem__(A, self, index).release();
    PY_END(nullptr);
}

static void __setitem__(Arena & A, py::handle self, py::handle index, py::handle rhs) {
    maybeInitializeGlobals();
    auto iinfo = getsetitem(A, self, index, has_dims(self) || has_dims(rhs));
    if (iinfo.can_call_original) {
        if (-1 == THPVariable_setitem(self.ptr(), index.ptr(), rhs.ptr())) {
            throw py::exception_set();
        }
        return;
    }

    auto rhs_info = TensorInfo::create(A, rhs, false, false);
    if (rhs_info) { // otherwise rhs can be a scalar...
        for (auto l : rhs_info.levels) {
            if (!iinfo.result_levels.contains(l)) {
                if (l.is_positional()) {
                    py::raise_error(DimensionBindError(), "rhs contains too many dimensions (%d) compared to indexed value (%d)", ndim_of_levels(iinfo.result_levels), rhs_info.ndim());
                } else {
                    auto tup = levels_to_tuple(iinfo.result_levels);
                    py::raise_error(DimensionBindError(), "rhs of setitem contains dimension %R which is not in the dimension on the left (%R)", l.dim().ptr(), tup.ptr());
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
            throw py::exception_set();
        }
    } else {
        torch_Tensor_copy_.call(self, rhs);
    }
}


int Tensor_setitem(PyObject* self, PyObject* index, PyObject* value) {
    Arena A;
    PY_BEGIN
    __setitem__(A, self, index, value);
    return 0;
    PY_END(-1);
}

static PyObject* py___getitem__(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    AT_ASSERT(nargs == 2);
    return __getitem__(A, args[0], args[1]).release();
    PY_END(nullptr)
}

static PyObject* py___setitem__(PyObject *_,
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


static PyObject* py_index(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    py::vector_args va(args, nargs, kwnames);
    py::handle self, dims, indices;
    va.parse("index", {"self", "dims", "indices"}, {&self, &dims, &indices}, 3);
    return index(A, self, dims, indices).release();
    PY_END(nullptr)
}


static PyObject* py_stack(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    py::vector_args va(args, nargs, kwnames);
    py::handle tensors, new_dim, dim;
    va.parse("stack", {"tensors", "new_dim", "dim"}, {&tensors, &new_dim, &dim}, 2);

    Slice<DimEntry> result_levels;
    Slice<TensorInfo> infos;
    py::sequence_view sv(tensors);
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
            py::raise_error(PyExc_TypeError, "Dimension %R does not exist in inputs", dim);
        }
        rawdim = *idx;
    }
    auto result = at::stack(inputs, rawdim);
    result_levels.insert(A, rawdim, new_dim_d);
    return Tensor::from_positional(A, std::move(result), result_levels, true).release();
    PY_END(nullptr)
}

static PyObject* py_split(PyObject *_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    maybeInitializeGlobals();
    py::vector_args va(args, nargs, kwnames);
    py::handle self, split_size_or_sections, dim;
    va.parse("split", {"self", "split_size_or_sections", "dim"}, {&self, &split_size_or_sections, &dim}, 2);
    bool dim_is_object = dim.ptr() && Dim::check_exact(dim);
    Slice<py::handle> sizes;

    bool all_dims = true;
    bool all_ints = true;

    if (!py::is_int(split_size_or_sections)) {
        py::sequence_view sv(split_size_or_sections);
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
            py::raise_error(PyExc_TypeError, "when dim is specified as a Dim object, split sizes must also be dimensions.");
        }
        // call original split (if self has dimensions this will use torch function to do the split)
        return torch_Tensor_split.call_vector(py::vector_args(args, nargs, kwnames)).release();
    }
    if (!all_dims) {
        py::raise_error(PyExc_TypeError, "split list must be ints or dims but got a mix");
    }

    auto self_info = TensorInfo::create(A, self, false);
    auto ndim = self_info.ndim();
    if (!dim_is_object&& ndim == 0) {
        py::raise_error(PyExc_TypeError, "split expects at least a 1-dimension tensor");
    }
    DimEntry dim_l = dim.ptr() ? _wrap_dim(dim, ndim, false) : -ndim;

    auto idx = self_info.levels.index(dim_l);
    if (!idx) {
        if (!dim.ptr()) {
            dim = A.autorelease(py::from_int(0));
        }
        py::raise_error(PyExc_TypeError, "tensor does not comtain dimension %R", dim.ptr());
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
           py::raise_error(PyExc_TypeError, "sizes of target dimensions add up to more (%d) than source dim (%d)", int(total_size), int(tensor_size));
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
        py::raise_error(PyExc_TypeError, "sum of sizes of target dimensions (%d) do not match the than source dim (%d)", int(total_size), int(tensor_size));
    }

    auto result_tensors = self_info.tensor->split_with_sizes(at::IntArrayRef(indices.begin(), indices.end()), *idx);
    py::tuple result(result_tensors.size());
    Slice<DimEntry> new_levels;
    new_levels.extend(A, self_info.levels);
    for (auto i : sizes.enumerate()) {
        new_levels[*idx] = Dim::unchecked_wrap(sizes[i]);
        result.set(i, Tensor::from_positional(A, std::move(result_tensors[i]), new_levels, true));
    }

    return result.release();

    PY_END(nullptr)
}


static DimEntry _wrap_dim(py::handle d, size_t N, bool keepdim) {
    if (Dim::check(d)) {
        if (keepdim) {
            py::raise_error(PyExc_ValueError, "cannot preserve first-class dimensions with keepdim=True");
        }
        return Dim::unchecked_wrap(d);
    } else if (py::is_int(d)) {
        auto i = py::to_int(d);
        while (i >= 0) {
            i -= N;
        }
        return i;
    } else {
        return DimEntry();
    }
}

static Slice<DimEntry> _wrap_dims(Arena& A, py::handle d, size_t N, bool keepdim) {
    auto de = _wrap_dim(d, N, keepdim);
    Slice<DimEntry> r;
    if (!de.is_none()) {
        r.append(A, de);
    } else {
        py::sequence_view sq(d);
        for (auto i : sq.enumerate()) {
            r.append(A, _wrap_dim(A.autorelease(sq[i]), N, keepdim));
        }
    }
    return r;
}

struct WrappedOperator : public py::base<WrappedOperator> {
    py::object orig;
    PyMethodDef method_def;
    py::object name, doc;

    bool is_pointwise = false;
    int64_t dim_offset = 0;
    int64_t keepdim_offset = 1;
    std::string dim_name;
    bool single_dim = false;
    bool reduce = true;

    static PyTypeObject Type;

    void init(py::object orig_, PyCFunction wrapper_implementation, std::string dim_name_="") {
        orig = std::move(orig_);
        method_def.ml_meth = wrapper_implementation;
        name = orig.attr("__name__");
        doc = orig.attr("__doc__");
        dim_name = std::move(dim_name_);
        if (!py::is_none(doc) && dim_name.size() > 0) {
            doc = py::unicode_from_format("%S\nArgument '%s' can be either an integer or a torchdim.Dim object.\n", doc.ptr(), dim_name.c_str());
        }
        method_def.ml_name = py::is_none(name) ? "" : PyUnicode_AsUTF8(name.ptr());
        method_def.ml_doc = py::is_none(doc) ? "" : PyUnicode_AsUTF8(doc.ptr());
        method_def.ml_flags = METH_FASTCALL | METH_KEYWORDS;
    }

    py::object function() {
        return py::object::checked_steal(PyCFunction_New(&method_def, ptr()));
    }

};

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

static PyObject* patched_dim_method(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    auto self = WrappedOperator::unchecked_wrap(self_);
    PY_BEGIN

    py::vector_args va(args, nargs, kwnames);

    auto _getarg = [&](const char* name, int64_t offset_) -> py::handle {
        auto offset = offset_ + 1; // do not include self
        auto idx = va.index(name, offset);
        return idx == -1 ? py::handle() : va[idx];
    };
    Slice<py::handle> patched_args;
    patched_args.extend(A, va.begin(), va.end());
    auto _patcharg = [&](const char* name, int64_t offset_, py::handle value) {
        auto offset = offset_ + 1; // do not include self
        auto idx = va.index(name, offset);
        if (idx == -1) {
            py::raise_error(PyExc_ValueError, "Missing argument %s", name);
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
            keepdim = py::to_bool(py_keepdim);
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
            py::raise_error(PyExc_ValueError, "Tensor with dimensions %R does not contain one of %R\n", tup.ptr(), dim.ptr());
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
    py::object py_indices;
    if (dim_indices.size() == 1) {
        py_indices = py::from_int(dim_indices[0]);
    } else {
        py::tuple tup(dim_indices.size());
        for (auto i : dim_indices.enumerate()) {
            tup.set(i, py::from_int(dim_indices[i]));
        }
        py_indices = std::move(tup);
    }
    _patcharg(self->dim_name.c_str(), self->dim_offset, py_indices);
    patched_args[0] = handle_from_tensor(A, info.tensor);
    auto r = self->orig.call_vector(patched_args.begin(), nargs, kwnames);
    auto wrap = [&](py::handle h) {
        if (THPVariable_Check(h.ptr())) {
            return A.autorelease(Tensor::from_positional(A, THPVariable_Unpack(h.ptr()), new_levels, info.has_device));
        }
        return h;
    };
    return tree_map(A, wrap, r).release();
    PY_END(nullptr)
}

static PyObject* _wrap(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN

    #define ARGS(_) _(py::handle, orig) _(py::handle, dim_offset) _(py::handle, keepdim_offset) \
                    _(py::handle, dim_name) _(py::handle, single_dim) _(py::handle, reduce)
    MPY_PARSE_ARGS_KWNAMES("O|OOOOO", ARGS)

    std::string dim_name_str;
    if (dim_name.ptr()) {
        dim_name_str = PyUnicode_AsUTF8(dim_name.ptr());
    } else {
        dim_name_str = "dim";
    }
    auto info = WrappedOperator::create(py::object::borrow(orig), (PyCFunction)(void*) patched_dim_method, std::move(dim_name_str));
    if (dim_offset.ptr()) {
        info->dim_offset = py::to_int(dim_offset);
    }
    if (keepdim_offset.ptr()) {
        info->keepdim_offset = py::to_int(keepdim_offset);
    }

    if (single_dim.ptr()) {
        info->single_dim = py::to_bool(single_dim);
    }
    if (reduce.ptr()) {
        info->reduce = py::to_bool(reduce);
    }
    return info->function().release();
    #undef ARGS

    PY_END(nullptr)
}

static PyObject* call_torch_function(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    Arena A;
    maybeInitializeGlobals();
    auto info = WrappedOperator::unchecked_wrap(self);
    return __torch_function__(A, info->orig, py::vector_args(args, nargs, kwnames), info->is_pointwise).release();
    PY_END(nullptr)
}

static PyObject* _wrap_method(PyObject *self,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    AT_ASSERT(nargs == 2);
    // XXX - ignore python function wrapped, we will call torch function directly
    py::handle orig = args[0];
    if (!pointwise.ptr()) {
        auto dim = py::import("functorch.dim");
        pointwise = dim.attr("pointwise");
    }
    auto info = WrappedOperator::create(py::object::borrow(orig), (PyCFunction)(void*) call_torch_function);
    info->is_pointwise = pointwise.contains(orig);
    return PyInstanceMethod_New(info->function().release());
    PY_END(nullptr);
}


static PyObject* Tensor_sum(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    Arena A;
    PY_BEGIN
    maybeInitializeGlobals();
    py::vector_args va(args, nargs, kwnames);
    auto self_ = Tensor::unchecked_wrap(args[0]);
    auto d = self_->delayed();
    if (!d) {
        return _Tensor_sum.call_vector(va).release();
    }
    py::handle self, dim, keepdim, dtype;
    va.parse("sum", {"self", "dim", "keepdim", "dtype"}, {&self, &dim, &keepdim, &dtype}, 1, 1);

    if (dtype.ptr() || (keepdim.ptr() && py::to_bool(keepdim))) {
        // std::cout << "SKIPPING fusion because dtype or keepdim=True specified\n";
        return _Tensor_sum.call_vector(va).release();
    }
    auto levels = self_->levels();

    auto N = ndim_of_levels(levels);
    auto reduced_dims = _wrap_dims(A, dim, N, false);

    return dot(A, TensorInfo::create(A, d->args[0], false), TensorInfo::create(A, d->args[1], false), reduced_dims).release();
    PY_END(nullptr)
}

static PyObject* _parse_test(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    maybeInitializeGlobals();

    int required = py::to_int(args[0]);
    int kwonly = py::to_int(args[1]);

    py::vector_args va(args + 2, nargs - 2, kwnames);


    py::handle a, b, c, d;
    va.parse("_parse_test", {"a", "b", "c", "d"}, {&a, &b, &c, &d}, required, kwonly);
    py::tuple r(4);
    r.set(0, py::object::borrow(a.ptr() ? a : Py_None));
    r.set(1, py::object::borrow(b.ptr() ? b : Py_None));
    r.set(2, py::object::borrow(c.ptr() ? c : Py_None));
    r.set(3, py::object::borrow(d.ptr() ? d : Py_None));
    return r.release();

    PY_END(nullptr)
}

static PyObject* _set_pointwise_optimize(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN
    py::handle value;
    py::vector_args va(args, nargs, kwnames);
    va.parse("_set_pointwise_optimization", {"value"}, {&value}, 1);
    pointwise_optimize = py::to_bool(value);
    Py_RETURN_NONE;
    PY_END(nullptr)
}

static PyObject* _patch_tensor_class(PyObject * self_,
                      PyObject *const *args,
                      Py_ssize_t nargs,
                      PyObject *kwnames) {
    PY_BEGIN

    auto torch = py::import("torch");
    auto py_TensorBase = torch.attr("_C").attr("_TensorBase");
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

static PyMethodDef methods[] = {
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

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "_C",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    methods
};

PyObject* Dim_init(void) {
    Arena A;
    try {
        py::object mod = py::object::checked_steal(PyModule_Create(&module_def));
        Dim::ready(mod, "Dim");
        DimList::ready(mod, "DimList");
        Tensor::ready(mod, "Tensor");
        WrappedOperator::ready(mod, "_WrappedOperator");
        Py_INCREF(&PyInstanceMethod_Type);
        PyModule_AddObject(mod.ptr(), "_instancemethod", (PyObject *)&PyInstanceMethod_Type);

        initializeGlobals(A);
        return mod.release();
    } catch(py::exception_set& err) {
        return nullptr;
    }
}
