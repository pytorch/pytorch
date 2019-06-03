/*
    pybind11/eigen.h: Transparent conversion for dense and sparse Eigen matrices

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "numpy.h"

#if defined(__INTEL_COMPILER)
#  pragma warning(disable: 1682) // implicit conversion of a 64-bit integral type to a smaller integral type (potential portability problem)
#elif defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#  ifdef __clang__
//   Eigen generates a bunch of implicit-copy-constructor-is-deprecated warnings with -Wdeprecated
//   under Clang, so disable that warning here:
#    pragma GCC diagnostic ignored "-Wdeprecated"
#  endif
#  if __GNUC__ >= 7
#    pragma GCC diagnostic ignored "-Wint-in-bool-context"
#  endif
#endif

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#  pragma warning(disable: 4996) // warning C4996: std::unary_negate is deprecated in C++17
#endif

#include <Eigen/Core>
#include <Eigen/SparseCore>

// Eigen prior to 3.2.7 doesn't have proper move constructors--but worse, some classes get implicit
// move constructors that break things.  We could detect this an explicitly copy, but an extra copy
// of matrices seems highly undesirable.
static_assert(EIGEN_VERSION_AT_LEAST(3,2,7), "Eigen support in pybind11 requires Eigen >= 3.2.7");

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

// Provide a convenience alias for easier pass-by-ref usage with fully dynamic strides:
using EigenDStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
template <typename MatrixType> using EigenDRef = Eigen::Ref<MatrixType, 0, EigenDStride>;
template <typename MatrixType> using EigenDMap = Eigen::Map<MatrixType, 0, EigenDStride>;

NAMESPACE_BEGIN(detail)

#if EIGEN_VERSION_AT_LEAST(3,3,0)
using EigenIndex = Eigen::Index;
#else
using EigenIndex = EIGEN_DEFAULT_DENSE_INDEX_TYPE;
#endif

// Matches Eigen::Map, Eigen::Ref, blocks, etc:
template <typename T> using is_eigen_dense_map = all_of<is_template_base_of<Eigen::DenseBase, T>, std::is_base_of<Eigen::MapBase<T, Eigen::ReadOnlyAccessors>, T>>;
template <typename T> using is_eigen_mutable_map = std::is_base_of<Eigen::MapBase<T, Eigen::WriteAccessors>, T>;
template <typename T> using is_eigen_dense_plain = all_of<negation<is_eigen_dense_map<T>>, is_template_base_of<Eigen::PlainObjectBase, T>>;
template <typename T> using is_eigen_sparse = is_template_base_of<Eigen::SparseMatrixBase, T>;
// Test for objects inheriting from EigenBase<Derived> that aren't captured by the above.  This
// basically covers anything that can be assigned to a dense matrix but that don't have a typical
// matrix data layout that can be copied from their .data().  For example, DiagonalMatrix and
// SelfAdjointView fall into this category.
template <typename T> using is_eigen_other = all_of<
    is_template_base_of<Eigen::EigenBase, T>,
    negation<any_of<is_eigen_dense_map<T>, is_eigen_dense_plain<T>, is_eigen_sparse<T>>>
>;

// Captures numpy/eigen conformability status (returned by EigenProps::conformable()):
template <bool EigenRowMajor> struct EigenConformable {
    bool conformable = false;
    EigenIndex rows = 0, cols = 0;
    EigenDStride stride{0, 0};      // Only valid if negativestrides is false!
    bool negativestrides = false;   // If true, do not use stride!

    EigenConformable(bool fits = false) : conformable{fits} {}
    // Matrix type:
    EigenConformable(EigenIndex r, EigenIndex c,
            EigenIndex rstride, EigenIndex cstride) :
        conformable{true}, rows{r}, cols{c} {
        // TODO: when Eigen bug #747 is fixed, remove the tests for non-negativity. http://eigen.tuxfamily.org/bz/show_bug.cgi?id=747
        if (rstride < 0 || cstride < 0) {
            negativestrides = true;
        } else {
            stride = {EigenRowMajor ? rstride : cstride /* outer stride */,
                      EigenRowMajor ? cstride : rstride /* inner stride */ };
        }
    }
    // Vector type:
    EigenConformable(EigenIndex r, EigenIndex c, EigenIndex stride)
        : EigenConformable(r, c, r == 1 ? c*stride : stride, c == 1 ? r : r*stride) {}

    template <typename props> bool stride_compatible() const {
        // To have compatible strides, we need (on both dimensions) one of fully dynamic strides,
        // matching strides, or a dimension size of 1 (in which case the stride value is irrelevant)
        return
            !negativestrides &&
            (props::inner_stride == Eigen::Dynamic || props::inner_stride == stride.inner() ||
                (EigenRowMajor ? cols : rows) == 1) &&
            (props::outer_stride == Eigen::Dynamic || props::outer_stride == stride.outer() ||
                (EigenRowMajor ? rows : cols) == 1);
    }
    operator bool() const { return conformable; }
};

template <typename Type> struct eigen_extract_stride { using type = Type; };
template <typename PlainObjectType, int MapOptions, typename StrideType>
struct eigen_extract_stride<Eigen::Map<PlainObjectType, MapOptions, StrideType>> { using type = StrideType; };
template <typename PlainObjectType, int Options, typename StrideType>
struct eigen_extract_stride<Eigen::Ref<PlainObjectType, Options, StrideType>> { using type = StrideType; };

// Helper struct for extracting information from an Eigen type
template <typename Type_> struct EigenProps {
    using Type = Type_;
    using Scalar = typename Type::Scalar;
    using StrideType = typename eigen_extract_stride<Type>::type;
    static constexpr EigenIndex
        rows = Type::RowsAtCompileTime,
        cols = Type::ColsAtCompileTime,
        size = Type::SizeAtCompileTime;
    static constexpr bool
        row_major = Type::IsRowMajor,
        vector = Type::IsVectorAtCompileTime, // At least one dimension has fixed size 1
        fixed_rows = rows != Eigen::Dynamic,
        fixed_cols = cols != Eigen::Dynamic,
        fixed = size != Eigen::Dynamic, // Fully-fixed size
        dynamic = !fixed_rows && !fixed_cols; // Fully-dynamic size

    template <EigenIndex i, EigenIndex ifzero> using if_zero = std::integral_constant<EigenIndex, i == 0 ? ifzero : i>;
    static constexpr EigenIndex inner_stride = if_zero<StrideType::InnerStrideAtCompileTime, 1>::value,
                                outer_stride = if_zero<StrideType::OuterStrideAtCompileTime,
                                                       vector ? size : row_major ? cols : rows>::value;
    static constexpr bool dynamic_stride = inner_stride == Eigen::Dynamic && outer_stride == Eigen::Dynamic;
    static constexpr bool requires_row_major = !dynamic_stride && !vector && (row_major ? inner_stride : outer_stride) == 1;
    static constexpr bool requires_col_major = !dynamic_stride && !vector && (row_major ? outer_stride : inner_stride) == 1;

    // Takes an input array and determines whether we can make it fit into the Eigen type.  If
    // the array is a vector, we attempt to fit it into either an Eigen 1xN or Nx1 vector
    // (preferring the latter if it will fit in either, i.e. for a fully dynamic matrix type).
    static EigenConformable<row_major> conformable(const array &a) {
        const auto dims = a.ndim();
        if (dims < 1 || dims > 2)
            return false;

        if (dims == 2) { // Matrix type: require exact match (or dynamic)

            EigenIndex
                np_rows = a.shape(0),
                np_cols = a.shape(1),
                np_rstride = a.strides(0) / static_cast<ssize_t>(sizeof(Scalar)),
                np_cstride = a.strides(1) / static_cast<ssize_t>(sizeof(Scalar));
            if ((fixed_rows && np_rows != rows) || (fixed_cols && np_cols != cols))
                return false;

            return {np_rows, np_cols, np_rstride, np_cstride};
        }

        // Otherwise we're storing an n-vector.  Only one of the strides will be used, but whichever
        // is used, we want the (single) numpy stride value.
        const EigenIndex n = a.shape(0),
              stride = a.strides(0) / static_cast<ssize_t>(sizeof(Scalar));

        if (vector) { // Eigen type is a compile-time vector
            if (fixed && size != n)
                return false; // Vector size mismatch
            return {rows == 1 ? 1 : n, cols == 1 ? 1 : n, stride};
        }
        else if (fixed) {
            // The type has a fixed size, but is not a vector: abort
            return false;
        }
        else if (fixed_cols) {
            // Since this isn't a vector, cols must be != 1.  We allow this only if it exactly
            // equals the number of elements (rows is Dynamic, and so 1 row is allowed).
            if (cols != n) return false;
            return {1, n, stride};
        }
        else {
            // Otherwise it's either fully dynamic, or column dynamic; both become a column vector
            if (fixed_rows && rows != n) return false;
            return {n, 1, stride};
        }
    }

    static constexpr bool show_writeable = is_eigen_dense_map<Type>::value && is_eigen_mutable_map<Type>::value;
    static constexpr bool show_order = is_eigen_dense_map<Type>::value;
    static constexpr bool show_c_contiguous = show_order && requires_row_major;
    static constexpr bool show_f_contiguous = !show_c_contiguous && show_order && requires_col_major;

    static constexpr auto descriptor =
        _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name +
        _("[")  + _<fixed_rows>(_<(size_t) rows>(), _("m")) +
        _(", ") + _<fixed_cols>(_<(size_t) cols>(), _("n")) +
        _("]") +
        // For a reference type (e.g. Ref<MatrixXd>) we have other constraints that might need to be
        // satisfied: writeable=True (for a mutable reference), and, depending on the map's stride
        // options, possibly f_contiguous or c_contiguous.  We include them in the descriptor output
        // to provide some hint as to why a TypeError is occurring (otherwise it can be confusing to
        // see that a function accepts a 'numpy.ndarray[float64[3,2]]' and an error message that you
        // *gave* a numpy.ndarray of the right type and dimensions.
        _<show_writeable>(", flags.writeable", "") +
        _<show_c_contiguous>(", flags.c_contiguous", "") +
        _<show_f_contiguous>(", flags.f_contiguous", "") +
        _("]");
};

// Casts an Eigen type to numpy array.  If given a base, the numpy array references the src data,
// otherwise it'll make a copy.  writeable lets you turn off the writeable flag for the array.
template <typename props> handle eigen_array_cast(typename props::Type const &src, handle base = handle(), bool writeable = true) {
    constexpr ssize_t elem_size = sizeof(typename props::Scalar);
    array a;
    if (props::vector)
        a = array({ src.size() }, { elem_size * src.innerStride() }, src.data(), base);
    else
        a = array({ src.rows(), src.cols() }, { elem_size * src.rowStride(), elem_size * src.colStride() },
                  src.data(), base);

    if (!writeable)
        array_proxy(a.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;

    return a.release();
}

// Takes an lvalue ref to some Eigen type and a (python) base object, creating a numpy array that
// reference the Eigen object's data with `base` as the python-registered base class (if omitted,
// the base will be set to None, and lifetime management is up to the caller).  The numpy array is
// non-writeable if the given type is const.
template <typename props, typename Type>
handle eigen_ref_array(Type &src, handle parent = none()) {
    // none here is to get past array's should-we-copy detection, which currently always
    // copies when there is no base.  Setting the base to None should be harmless.
    return eigen_array_cast<props>(src, parent, !std::is_const<Type>::value);
}

// Takes a pointer to some dense, plain Eigen type, builds a capsule around it, then returns a numpy
// array that references the encapsulated data with a python-side reference to the capsule to tie
// its destruction to that of any dependent python objects.  Const-ness is determined by whether or
// not the Type of the pointer given is const.
template <typename props, typename Type, typename = enable_if_t<is_eigen_dense_plain<Type>::value>>
handle eigen_encapsulate(Type *src) {
    capsule base(src, [](void *o) { delete static_cast<Type *>(o); });
    return eigen_ref_array<props>(*src, base);
}

// Type caster for regular, dense matrix types (e.g. MatrixXd), but not maps/refs/etc. of dense
// types.
template<typename Type>
struct type_caster<Type, enable_if_t<is_eigen_dense_plain<Type>::value>> {
    using Scalar = typename Type::Scalar;
    using props = EigenProps<Type>;

    bool load(handle src, bool convert) {
        // If we're in no-convert mode, only load if given an array of the correct type
        if (!convert && !isinstance<array_t<Scalar>>(src))
            return false;

        // Coerce into an array, but don't do type conversion yet; the copy below handles it.
        auto buf = array::ensure(src);

        if (!buf)
            return false;

        auto dims = buf.ndim();
        if (dims < 1 || dims > 2)
            return false;

        auto fits = props::conformable(buf);
        if (!fits)
            return false;

        // Allocate the new type, then build a numpy reference into it
        value = Type(fits.rows, fits.cols);
        auto ref = reinterpret_steal<array>(eigen_ref_array<props>(value));
        if (dims == 1) ref = ref.squeeze();
        else if (ref.ndim() == 1) buf = buf.squeeze();

        int result = detail::npy_api::get().PyArray_CopyInto_(ref.ptr(), buf.ptr());

        if (result < 0) { // Copy failed!
            PyErr_Clear();
            return false;
        }

        return true;
    }

private:

    // Cast implementation
    template <typename CType>
    static handle cast_impl(CType *src, return_value_policy policy, handle parent) {
        switch (policy) {
            case return_value_policy::take_ownership:
            case return_value_policy::automatic:
                return eigen_encapsulate<props>(src);
            case return_value_policy::move:
                return eigen_encapsulate<props>(new CType(std::move(*src)));
            case return_value_policy::copy:
                return eigen_array_cast<props>(*src);
            case return_value_policy::reference:
            case return_value_policy::automatic_reference:
                return eigen_ref_array<props>(*src);
            case return_value_policy::reference_internal:
                return eigen_ref_array<props>(*src, parent);
            default:
                throw cast_error("unhandled return_value_policy: should not happen!");
        };
    }

public:

    // Normal returned non-reference, non-const value:
    static handle cast(Type &&src, return_value_policy /* policy */, handle parent) {
        return cast_impl(&src, return_value_policy::move, parent);
    }
    // If you return a non-reference const, we mark the numpy array readonly:
    static handle cast(const Type &&src, return_value_policy /* policy */, handle parent) {
        return cast_impl(&src, return_value_policy::move, parent);
    }
    // lvalue reference return; default (automatic) becomes copy
    static handle cast(Type &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast_impl(&src, policy, parent);
    }
    // const lvalue reference return; default (automatic) becomes copy
    static handle cast(const Type &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast(&src, policy, parent);
    }
    // non-const pointer return
    static handle cast(Type *src, return_value_policy policy, handle parent) {
        return cast_impl(src, policy, parent);
    }
    // const pointer return
    static handle cast(const Type *src, return_value_policy policy, handle parent) {
        return cast_impl(src, policy, parent);
    }

    static constexpr auto name = props::descriptor;

    operator Type*() { return &value; }
    operator Type&() { return value; }
    operator Type&&() && { return std::move(value); }
    template <typename T> using cast_op_type = movable_cast_op_type<T>;

private:
    Type value;
};

// Base class for casting reference/map/block/etc. objects back to python.
template <typename MapType> struct eigen_map_caster {
private:
    using props = EigenProps<MapType>;

public:

    // Directly referencing a ref/map's data is a bit dangerous (whatever the map/ref points to has
    // to stay around), but we'll allow it under the assumption that you know what you're doing (and
    // have an appropriate keep_alive in place).  We return a numpy array pointing directly at the
    // ref's data (The numpy array ends up read-only if the ref was to a const matrix type.) Note
    // that this means you need to ensure you don't destroy the object in some other way (e.g. with
    // an appropriate keep_alive, or with a reference to a statically allocated matrix).
    static handle cast(const MapType &src, return_value_policy policy, handle parent) {
        switch (policy) {
            case return_value_policy::copy:
                return eigen_array_cast<props>(src);
            case return_value_policy::reference_internal:
                return eigen_array_cast<props>(src, parent, is_eigen_mutable_map<MapType>::value);
            case return_value_policy::reference:
            case return_value_policy::automatic:
            case return_value_policy::automatic_reference:
                return eigen_array_cast<props>(src, none(), is_eigen_mutable_map<MapType>::value);
            default:
                // move, take_ownership don't make any sense for a ref/map:
                pybind11_fail("Invalid return_value_policy for Eigen Map/Ref/Block type");
        }
    }

    static constexpr auto name = props::descriptor;

    // Explicitly delete these: support python -> C++ conversion on these (i.e. these can be return
    // types but not bound arguments).  We still provide them (with an explicitly delete) so that
    // you end up here if you try anyway.
    bool load(handle, bool) = delete;
    operator MapType() = delete;
    template <typename> using cast_op_type = MapType;
};

// We can return any map-like object (but can only load Refs, specialized next):
template <typename Type> struct type_caster<Type, enable_if_t<is_eigen_dense_map<Type>::value>>
    : eigen_map_caster<Type> {};

// Loader for Ref<...> arguments.  See the documentation for info on how to make this work without
// copying (it requires some extra effort in many cases).
template <typename PlainObjectType, typename StrideType>
struct type_caster<
    Eigen::Ref<PlainObjectType, 0, StrideType>,
    enable_if_t<is_eigen_dense_map<Eigen::Ref<PlainObjectType, 0, StrideType>>::value>
> : public eigen_map_caster<Eigen::Ref<PlainObjectType, 0, StrideType>> {
private:
    using Type = Eigen::Ref<PlainObjectType, 0, StrideType>;
    using props = EigenProps<Type>;
    using Scalar = typename props::Scalar;
    using MapType = Eigen::Map<PlainObjectType, 0, StrideType>;
    using Array = array_t<Scalar, array::forcecast |
                ((props::row_major ? props::inner_stride : props::outer_stride) == 1 ? array::c_style :
                 (props::row_major ? props::outer_stride : props::inner_stride) == 1 ? array::f_style : 0)>;
    static constexpr bool need_writeable = is_eigen_mutable_map<Type>::value;
    // Delay construction (these have no default constructor)
    std::unique_ptr<MapType> map;
    std::unique_ptr<Type> ref;
    // Our array.  When possible, this is just a numpy array pointing to the source data, but
    // sometimes we can't avoid copying (e.g. input is not a numpy array at all, has an incompatible
    // layout, or is an array of a type that needs to be converted).  Using a numpy temporary
    // (rather than an Eigen temporary) saves an extra copy when we need both type conversion and
    // storage order conversion.  (Note that we refuse to use this temporary copy when loading an
    // argument for a Ref<M> with M non-const, i.e. a read-write reference).
    Array copy_or_ref;
public:
    bool load(handle src, bool convert) {
        // First check whether what we have is already an array of the right type.  If not, we can't
        // avoid a copy (because the copy is also going to do type conversion).
        bool need_copy = !isinstance<Array>(src);

        EigenConformable<props::row_major> fits;
        if (!need_copy) {
            // We don't need a converting copy, but we also need to check whether the strides are
            // compatible with the Ref's stride requirements
            Array aref = reinterpret_borrow<Array>(src);

            if (aref && (!need_writeable || aref.writeable())) {
                fits = props::conformable(aref);
                if (!fits) return false; // Incompatible dimensions
                if (!fits.template stride_compatible<props>())
                    need_copy = true;
                else
                    copy_or_ref = std::move(aref);
            }
            else {
                need_copy = true;
            }
        }

        if (need_copy) {
            // We need to copy: If we need a mutable reference, or we're not supposed to convert
            // (either because we're in the no-convert overload pass, or because we're explicitly
            // instructed not to copy (via `py::arg().noconvert()`) we have to fail loading.
            if (!convert || need_writeable) return false;

            Array copy = Array::ensure(src);
            if (!copy) return false;
            fits = props::conformable(copy);
            if (!fits || !fits.template stride_compatible<props>())
                return false;
            copy_or_ref = std::move(copy);
            loader_life_support::add_patient(copy_or_ref);
        }

        ref.reset();
        map.reset(new MapType(data(copy_or_ref), fits.rows, fits.cols, make_stride(fits.stride.outer(), fits.stride.inner())));
        ref.reset(new Type(*map));

        return true;
    }

    operator Type*() { return ref.get(); }
    operator Type&() { return *ref; }
    template <typename _T> using cast_op_type = pybind11::detail::cast_op_type<_T>;

private:
    template <typename T = Type, enable_if_t<is_eigen_mutable_map<T>::value, int> = 0>
    Scalar *data(Array &a) { return a.mutable_data(); }

    template <typename T = Type, enable_if_t<!is_eigen_mutable_map<T>::value, int> = 0>
    const Scalar *data(Array &a) { return a.data(); }

    // Attempt to figure out a constructor of `Stride` that will work.
    // If both strides are fixed, use a default constructor:
    template <typename S> using stride_ctor_default = bool_constant<
        S::InnerStrideAtCompileTime != Eigen::Dynamic && S::OuterStrideAtCompileTime != Eigen::Dynamic &&
        std::is_default_constructible<S>::value>;
    // Otherwise, if there is a two-index constructor, assume it is (outer,inner) like
    // Eigen::Stride, and use it:
    template <typename S> using stride_ctor_dual = bool_constant<
        !stride_ctor_default<S>::value && std::is_constructible<S, EigenIndex, EigenIndex>::value>;
    // Otherwise, if there is a one-index constructor, and just one of the strides is dynamic, use
    // it (passing whichever stride is dynamic).
    template <typename S> using stride_ctor_outer = bool_constant<
        !any_of<stride_ctor_default<S>, stride_ctor_dual<S>>::value &&
        S::OuterStrideAtCompileTime == Eigen::Dynamic && S::InnerStrideAtCompileTime != Eigen::Dynamic &&
        std::is_constructible<S, EigenIndex>::value>;
    template <typename S> using stride_ctor_inner = bool_constant<
        !any_of<stride_ctor_default<S>, stride_ctor_dual<S>>::value &&
        S::InnerStrideAtCompileTime == Eigen::Dynamic && S::OuterStrideAtCompileTime != Eigen::Dynamic &&
        std::is_constructible<S, EigenIndex>::value>;

    template <typename S = StrideType, enable_if_t<stride_ctor_default<S>::value, int> = 0>
    static S make_stride(EigenIndex, EigenIndex) { return S(); }
    template <typename S = StrideType, enable_if_t<stride_ctor_dual<S>::value, int> = 0>
    static S make_stride(EigenIndex outer, EigenIndex inner) { return S(outer, inner); }
    template <typename S = StrideType, enable_if_t<stride_ctor_outer<S>::value, int> = 0>
    static S make_stride(EigenIndex outer, EigenIndex) { return S(outer); }
    template <typename S = StrideType, enable_if_t<stride_ctor_inner<S>::value, int> = 0>
    static S make_stride(EigenIndex, EigenIndex inner) { return S(inner); }

};

// type_caster for special matrix types (e.g. DiagonalMatrix), which are EigenBase, but not
// EigenDense (i.e. they don't have a data(), at least not with the usual matrix layout).
// load() is not supported, but we can cast them into the python domain by first copying to a
// regular Eigen::Matrix, then casting that.
template <typename Type>
struct type_caster<Type, enable_if_t<is_eigen_other<Type>::value>> {
protected:
    using Matrix = Eigen::Matrix<typename Type::Scalar, Type::RowsAtCompileTime, Type::ColsAtCompileTime>;
    using props = EigenProps<Matrix>;
public:
    static handle cast(const Type &src, return_value_policy /* policy */, handle /* parent */) {
        handle h = eigen_encapsulate<props>(new Matrix(src));
        return h;
    }
    static handle cast(const Type *src, return_value_policy policy, handle parent) { return cast(*src, policy, parent); }

    static constexpr auto name = props::descriptor;

    // Explicitly delete these: support python -> C++ conversion on these (i.e. these can be return
    // types but not bound arguments).  We still provide them (with an explicitly delete) so that
    // you end up here if you try anyway.
    bool load(handle, bool) = delete;
    operator Type() = delete;
    template <typename> using cast_op_type = Type;
};

template<typename Type>
struct type_caster<Type, enable_if_t<is_eigen_sparse<Type>::value>> {
    typedef typename Type::Scalar Scalar;
    typedef remove_reference_t<decltype(*std::declval<Type>().outerIndexPtr())> StorageIndex;
    typedef typename Type::Index Index;
    static constexpr bool rowMajor = Type::IsRowMajor;

    bool load(handle src, bool) {
        if (!src)
            return false;

        auto obj = reinterpret_borrow<object>(src);
        object sparse_module = module::import("scipy.sparse");
        object matrix_type = sparse_module.attr(
            rowMajor ? "csr_matrix" : "csc_matrix");

        if (!obj.get_type().is(matrix_type)) {
            try {
                obj = matrix_type(obj);
            } catch (const error_already_set &) {
                return false;
            }
        }

        auto values = array_t<Scalar>((object) obj.attr("data"));
        auto innerIndices = array_t<StorageIndex>((object) obj.attr("indices"));
        auto outerIndices = array_t<StorageIndex>((object) obj.attr("indptr"));
        auto shape = pybind11::tuple((pybind11::object) obj.attr("shape"));
        auto nnz = obj.attr("nnz").cast<Index>();

        if (!values || !innerIndices || !outerIndices)
            return false;

        value = Eigen::MappedSparseMatrix<Scalar, Type::Flags, StorageIndex>(
            shape[0].cast<Index>(), shape[1].cast<Index>(), nnz,
            outerIndices.mutable_data(), innerIndices.mutable_data(), values.mutable_data());

        return true;
    }

    static handle cast(const Type &src, return_value_policy /* policy */, handle /* parent */) {
        const_cast<Type&>(src).makeCompressed();

        object matrix_type = module::import("scipy.sparse").attr(
            rowMajor ? "csr_matrix" : "csc_matrix");

        array data(src.nonZeros(), src.valuePtr());
        array outerIndices((rowMajor ? src.rows() : src.cols()) + 1, src.outerIndexPtr());
        array innerIndices(src.nonZeros(), src.innerIndexPtr());

        return matrix_type(
            std::make_tuple(data, innerIndices, outerIndices),
            std::make_pair(src.rows(), src.cols())
        ).release();
    }

    PYBIND11_TYPE_CASTER(Type, _<(Type::IsRowMajor) != 0>("scipy.sparse.csr_matrix[", "scipy.sparse.csc_matrix[")
            + npy_format_descriptor<Scalar>::name + _("]"));
};

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif
