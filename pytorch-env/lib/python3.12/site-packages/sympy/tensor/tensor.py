"""
This module defines tensors with abstract index notation.

The abstract index notation has been first formalized by Penrose.

Tensor indices are formal objects, with a tensor type; there is no
notion of index range, it is only possible to assign the dimension,
used to trace the Kronecker delta; the dimension can be a Symbol.

The Einstein summation convention is used.
The covariant indices are indicated with a minus sign in front of the index.

For instance the tensor ``t = p(a)*A(b,c)*q(-c)`` has the index ``c``
contracted.

A tensor expression ``t`` can be called; called with its
indices in sorted order it is equal to itself:
in the above example ``t(a, b) == t``;
one can call ``t`` with different indices; ``t(c, d) == p(c)*A(d,a)*q(-a)``.

The contracted indices are dummy indices, internally they have no name,
the indices being represented by a graph-like structure.

Tensors are put in canonical form using ``canon_bp``, which uses
the Butler-Portugal algorithm for canonicalization using the monoterm
symmetries of the tensors.

If there is a (anti)symmetric metric, the indices can be raised and
lowered when the tensor is put in canonical form.
"""

from __future__ import annotations
from typing import Any
from functools import reduce
from math import prod

from abc import abstractmethod, ABC
from collections import defaultdict
import operator
import itertools
from sympy.core.numbers import (Integer, Rational)
from sympy.combinatorics import Permutation
from sympy.combinatorics.tensor_can import get_symmetric_group_sgs, \
    bsgs_direct_product, canonicalize, riemann_bsgs
from sympy.core import Basic, Expr, sympify, Add, Mul, S
from sympy.core.cache import clear_cache
from sympy.core.containers import Tuple, Dict
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol, symbols
from sympy.core.sympify import CantSympify, _sympify
from sympy.core.operations import AssocOp
from sympy.external.gmpy import SYMPY_INTS
from sympy.matrices import eye
from sympy.utilities.exceptions import (sympy_deprecation_warning,
                                        SymPyDeprecationWarning,
                                        ignore_warnings)
from sympy.utilities.decorator import memoize_property, deprecated
from sympy.utilities.iterables import sift


def deprecate_data():
    sympy_deprecation_warning(
        """
        The data attribute of TensorIndexType is deprecated. Use The
        replace_with_arrays() method instead.
        """,
        deprecated_since_version="1.4",
        active_deprecations_target="deprecated-tensorindextype-attrs",
        stacklevel=4,
    )

def deprecate_fun_eval():
    sympy_deprecation_warning(
        """
        The Tensor.fun_eval() method is deprecated. Use
        Tensor.substitute_indices() instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-tensor-fun-eval",
        stacklevel=4,
    )


def deprecate_call():
    sympy_deprecation_warning(
        """
        Calling a tensor like Tensor(*indices) is deprecated. Use
        Tensor.substitute_indices() instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-tensor-fun-eval",
        stacklevel=4,
    )


class _IndexStructure(CantSympify):
    """
    This class handles the indices (free and dummy ones). It contains the
    algorithms to manage the dummy indices replacements and contractions of
    free indices under multiplications of tensor expressions, as well as stuff
    related to canonicalization sorting, getting the permutation of the
    expression and so on. It also includes tools to get the ``TensorIndex``
    objects corresponding to the given index structure.
    """

    def __init__(self, free, dum, index_types, indices, canon_bp=False):
        self.free = free
        self.dum = dum
        self.index_types = index_types
        self.indices = indices
        self._ext_rank = len(self.free) + 2*len(self.dum)
        self.dum.sort(key=lambda x: x[0])

    @staticmethod
    def from_indices(*indices):
        """
        Create a new ``_IndexStructure`` object from a list of ``indices``.

        Explanation
        ===========

        ``indices``     ``TensorIndex`` objects, the indices. Contractions are
                        detected upon construction.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, _IndexStructure
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2, m3 = tensor_indices('m0,m1,m2,m3', Lorentz)
        >>> _IndexStructure.from_indices(m0, m1, -m1, m3)
        _IndexStructure([(m0, 0), (m3, 3)], [(1, 2)], [Lorentz, Lorentz, Lorentz, Lorentz])
        """

        free, dum = _IndexStructure._free_dum_from_indices(*indices)
        index_types = [i.tensor_index_type for i in indices]
        indices = _IndexStructure._replace_dummy_names(indices, free, dum)
        return _IndexStructure(free, dum, index_types, indices)

    @staticmethod
    def from_components_free_dum(components, free, dum):
        index_types = []
        for component in components:
            index_types.extend(component.index_types)
        indices = _IndexStructure.generate_indices_from_free_dum_index_types(free, dum, index_types)
        return _IndexStructure(free, dum, index_types, indices)

    @staticmethod
    def _free_dum_from_indices(*indices):
        """
        Convert ``indices`` into ``free``, ``dum`` for single component tensor.

        Explanation
        ===========

        ``free``     list of tuples ``(index, pos, 0)``,
                     where ``pos`` is the position of index in
                     the list of indices formed by the component tensors

        ``dum``      list of tuples ``(pos_contr, pos_cov, 0, 0)``

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, \
            _IndexStructure
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2, m3 = tensor_indices('m0,m1,m2,m3', Lorentz)
        >>> _IndexStructure._free_dum_from_indices(m0, m1, -m1, m3)
        ([(m0, 0), (m3, 3)], [(1, 2)])
        """
        n = len(indices)
        if n == 1:
            return [(indices[0], 0)], []

        # find the positions of the free indices and of the dummy indices
        free = [True]*len(indices)
        index_dict = {}
        dum = []
        for i, index in enumerate(indices):
            name = index.name
            typ = index.tensor_index_type
            contr = index.is_up
            if (name, typ) in index_dict:
                # found a pair of dummy indices
                is_contr, pos = index_dict[(name, typ)]
                # check consistency and update free
                if is_contr:
                    if contr:
                        raise ValueError('two equal contravariant indices in slots %d and %d' %(pos, i))
                    else:
                        free[pos] = False
                        free[i] = False
                else:
                    if contr:
                        free[pos] = False
                        free[i] = False
                    else:
                        raise ValueError('two equal covariant indices in slots %d and %d' %(pos, i))
                if contr:
                    dum.append((i, pos))
                else:
                    dum.append((pos, i))
            else:
                index_dict[(name, typ)] = index.is_up, i

        free = [(index, i) for i, index in enumerate(indices) if free[i]]
        free.sort()
        return free, dum

    def get_indices(self):
        """
        Get a list of indices, creating new tensor indices to complete dummy indices.
        """
        return self.indices[:]

    @staticmethod
    def generate_indices_from_free_dum_index_types(free, dum, index_types):
        indices = [None]*(len(free)+2*len(dum))
        for idx, pos in free:
            indices[pos] = idx

        generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
        for pos1, pos2 in dum:
            typ1 = index_types[pos1]
            indname = generate_dummy_name(typ1)
            indices[pos1] = TensorIndex(indname, typ1, True)
            indices[pos2] = TensorIndex(indname, typ1, False)

        return _IndexStructure._replace_dummy_names(indices, free, dum)

    @staticmethod
    def _get_generator_for_dummy_indices(free):
        cdt = defaultdict(int)
        # if the free indices have names with dummy_name, start with an
        # index higher than those for the dummy indices
        # to avoid name collisions
        for indx, ipos in free:
            if indx.name.split('_')[0] == indx.tensor_index_type.dummy_name:
                cdt[indx.tensor_index_type] = max(cdt[indx.tensor_index_type], int(indx.name.split('_')[1]) + 1)

        def dummy_name_gen(tensor_index_type):
            nd = str(cdt[tensor_index_type])
            cdt[tensor_index_type] += 1
            return tensor_index_type.dummy_name + '_' + nd

        return dummy_name_gen

    @staticmethod
    def _replace_dummy_names(indices, free, dum):
        dum.sort(key=lambda x: x[0])
        new_indices = list(indices)
        assert len(indices) == len(free) + 2*len(dum)
        generate_dummy_name = _IndexStructure._get_generator_for_dummy_indices(free)
        for ipos1, ipos2 in dum:
            typ1 = new_indices[ipos1].tensor_index_type
            indname = generate_dummy_name(typ1)
            new_indices[ipos1] = TensorIndex(indname, typ1, True)
            new_indices[ipos2] = TensorIndex(indname, typ1, False)
        return new_indices

    def get_free_indices(self) -> list[TensorIndex]:
        """
        Get a list of free indices.
        """
        # get sorted indices according to their position:
        free = sorted(self.free, key=lambda x: x[1])
        return [i[0] for i in free]

    def __str__(self):
        return "_IndexStructure({}, {}, {})".format(self.free, self.dum, self.index_types)

    def __repr__(self):
        return self.__str__()

    def _get_sorted_free_indices_for_canon(self):
        sorted_free = self.free[:]
        sorted_free.sort(key=lambda x: x[0])
        return sorted_free

    def _get_sorted_dum_indices_for_canon(self):
        return sorted(self.dum, key=lambda x: x[0])

    def _get_lexicographically_sorted_index_types(self):
        permutation = self.indices_canon_args()[0]
        index_types = [None]*self._ext_rank
        for i, it in enumerate(self.index_types):
            index_types[permutation(i)] = it
        return index_types

    def _get_lexicographically_sorted_indices(self):
        permutation = self.indices_canon_args()[0]
        indices = [None]*self._ext_rank
        for i, it in enumerate(self.indices):
            indices[permutation(i)] = it
        return indices

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns a ``_IndexStructure`` instance corresponding to the permutation ``g``.

        Explanation
        ===========

        ``g``  permutation corresponding to the tensor in the representation
        used in canonicalization

        ``is_canon_bp``   if True, then ``g`` is the permutation
        corresponding to the canonical form of the tensor
        """
        sorted_free = [i[0] for i in self._get_sorted_free_indices_for_canon()]
        lex_index_types = self._get_lexicographically_sorted_index_types()
        lex_indices = self._get_lexicographically_sorted_indices()
        nfree = len(sorted_free)
        rank = self._ext_rank
        dum = [[None]*2 for i in range((rank - nfree)//2)]
        free = []

        index_types = [None]*rank
        indices = [None]*rank
        for i in range(rank):
            gi = g[i]
            index_types[i] = lex_index_types[gi]
            indices[i] = lex_indices[gi]
            if gi < nfree:
                ind = sorted_free[gi]
                assert index_types[i] == sorted_free[gi].tensor_index_type
                free.append((ind, i))
            else:
                j = gi - nfree
                idum, cov = divmod(j, 2)
                if cov:
                    dum[idum][1] = i
                else:
                    dum[idum][0] = i
        dum = [tuple(x) for x in dum]

        return _IndexStructure(free, dum, index_types, indices)

    def indices_canon_args(self):
        """
        Returns ``(g, dummies, msym, v)``, the entries of ``canonicalize``

        See ``canonicalize`` in ``tensor_can.py`` in combinatorics module.
        """
        # to be called after sorted_components
        from sympy.combinatorics.permutations import _af_new
        n = self._ext_rank
        g = [None]*n + [n, n+1]

        # Converts the symmetry of the metric into msym from .canonicalize()
        # method in the combinatorics module
        def metric_symmetry_to_msym(metric):
            if metric is None:
                return None
            sym = metric.symmetry
            if sym == TensorSymmetry.fully_symmetric(2):
                return 0
            if sym == TensorSymmetry.fully_symmetric(-2):
                return 1
            return None

        # ordered indices: first the free indices, ordered by types
        # then the dummy indices, ordered by types and contravariant before
        # covariant
        # g[position in tensor] = position in ordered indices
        for i, (indx, ipos) in enumerate(self._get_sorted_free_indices_for_canon()):
            g[ipos] = i
        pos = len(self.free)
        j = len(self.free)
        dummies = []
        prev = None
        a = []
        msym = []
        for ipos1, ipos2 in self._get_sorted_dum_indices_for_canon():
            g[ipos1] = j
            g[ipos2] = j + 1
            j += 2
            typ = self.index_types[ipos1]
            if typ != prev:
                if a:
                    dummies.append(a)
                a = [pos, pos + 1]
                prev = typ
                msym.append(metric_symmetry_to_msym(typ.metric))
            else:
                a.extend([pos, pos + 1])
            pos += 2
        if a:
            dummies.append(a)

        return _af_new(g), dummies, msym


def components_canon_args(components):
    numtyp = []
    prev = None
    for t in components:
        if t == prev:
            numtyp[-1][1] += 1
        else:
            prev = t
            numtyp.append([prev, 1])
    v = []
    for h, n in numtyp:
        if h.comm in (0, 1):
            comm = h.comm
        else:
            comm = TensorManager.get_comm(h.comm, h.comm)
        v.append((h.symmetry.base, h.symmetry.generators, n, comm))
    return v


class _TensorDataLazyEvaluator(CantSympify):
    """
    EXPERIMENTAL: do not rely on this class, it may change without deprecation
    warnings in future versions of SymPy.

    Explanation
    ===========

    This object contains the logic to associate components data to a tensor
    expression. Components data are set via the ``.data`` property of tensor
    expressions, is stored inside this class as a mapping between the tensor
    expression and the ``ndarray``.

    Computations are executed lazily: whereas the tensor expressions can have
    contractions, tensor products, and additions, components data are not
    computed until they are accessed by reading the ``.data`` property
    associated to the tensor expression.
    """
    _substitutions_dict: dict[Any, Any] = {}
    _substitutions_dict_tensmul: dict[Any, Any] = {}

    def __getitem__(self, key):
        dat = self._get(key)
        if dat is None:
            return None

        from .array import NDimArray
        if not isinstance(dat, NDimArray):
            return dat

        if dat.rank() == 0:
            return dat[()]
        elif dat.rank() == 1 and len(dat) == 1:
            return dat[0]
        return dat

    def _get(self, key):
        """
        Retrieve ``data`` associated with ``key``.

        Explanation
        ===========

        This algorithm looks into ``self._substitutions_dict`` for all
        ``TensorHead`` in the ``TensExpr`` (or just ``TensorHead`` if key is a
        TensorHead instance). It reconstructs the components data that the
        tensor expression should have by performing on components data the
        operations that correspond to the abstract tensor operations applied.

        Metric tensor is handled in a different manner: it is pre-computed in
        ``self._substitutions_dict_tensmul``.
        """
        if key in self._substitutions_dict:
            return self._substitutions_dict[key]

        if isinstance(key, TensorHead):
            return None

        if isinstance(key, Tensor):
            # special case to handle metrics. Metric tensors cannot be
            # constructed through contraction by the metric, their
            # components show if they are a matrix or its inverse.
            signature = tuple([i.is_up for i in key.get_indices()])
            srch = (key.component,) + signature
            if srch in self._substitutions_dict_tensmul:
                return self._substitutions_dict_tensmul[srch]
            array_list = [self.data_from_tensor(key)]
            return self.data_contract_dum(array_list, key.dum, key.ext_rank)

        if isinstance(key, TensMul):
            tensmul_args = key.args
            if len(tensmul_args) == 1 and len(tensmul_args[0].components) == 1:
                # special case to handle metrics. Metric tensors cannot be
                # constructed through contraction by the metric, their
                # components show if they are a matrix or its inverse.
                signature = tuple([i.is_up for i in tensmul_args[0].get_indices()])
                srch = (tensmul_args[0].components[0],) + signature
                if srch in self._substitutions_dict_tensmul:
                    return self._substitutions_dict_tensmul[srch]
            #data_list = [self.data_from_tensor(i) for i in tensmul_args if isinstance(i, TensExpr)]
            data_list = [self.data_from_tensor(i) if isinstance(i, Tensor) else i.data for i in tensmul_args if isinstance(i, TensExpr)]
            coeff = prod([i for i in tensmul_args if not isinstance(i, TensExpr)])
            if all(i is None for i in data_list):
                return None
            if any(i is None for i in data_list):
                raise ValueError("Mixing tensors with associated components "\
                                 "data with tensors without components data")
            data_result = self.data_contract_dum(data_list, key.dum, key.ext_rank)
            return coeff*data_result

        if isinstance(key, TensAdd):
            data_list = []
            free_args_list = []
            for arg in key.args:
                if isinstance(arg, TensExpr):
                    data_list.append(arg.data)
                    free_args_list.append([x[0] for x in arg.free])
                else:
                    data_list.append(arg)
                    free_args_list.append([])
            if all(i is None for i in data_list):
                return None
            if any(i is None for i in data_list):
                raise ValueError("Mixing tensors with associated components "\
                                 "data with tensors without components data")

            sum_list = []
            from .array import permutedims
            for data, free_args in zip(data_list, free_args_list):
                if len(free_args) < 2:
                    sum_list.append(data)
                else:
                    free_args_pos = {y: x for x, y in enumerate(free_args)}
                    axes = [free_args_pos[arg] for arg in key.free_args]
                    sum_list.append(permutedims(data, axes))
            return reduce(lambda x, y: x+y, sum_list)

        return None

    @staticmethod
    def data_contract_dum(ndarray_list, dum, ext_rank):
        from .array import tensorproduct, tensorcontraction, MutableDenseNDimArray
        arrays = list(map(MutableDenseNDimArray, ndarray_list))
        prodarr = tensorproduct(*arrays)
        return tensorcontraction(prodarr, *dum)

    def data_tensorhead_from_tensmul(self, data, tensmul, tensorhead):
        """
        This method is used when assigning components data to a ``TensMul``
        object, it converts components data to a fully contravariant ndarray,
        which is then stored according to the ``TensorHead`` key.
        """
        if data is None:
            return None

        return self._correct_signature_from_indices(
            data,
            tensmul.get_indices(),
            tensmul.free,
            tensmul.dum,
            True)

    def data_from_tensor(self, tensor):
        """
        This method corrects the components data to the right signature
        (covariant/contravariant) using the metric associated with each
        ``TensorIndexType``.
        """
        tensorhead = tensor.component

        if tensorhead.data is None:
            return None

        return self._correct_signature_from_indices(
            tensorhead.data,
            tensor.get_indices(),
            tensor.free,
            tensor.dum)

    def _assign_data_to_tensor_expr(self, key, data):
        if isinstance(key, TensAdd):
            raise ValueError('cannot assign data to TensAdd')
        # here it is assumed that `key` is a `TensMul` instance.
        if len(key.components) != 1:
            raise ValueError('cannot assign data to TensMul with multiple components')
        tensorhead = key.components[0]
        newdata = self.data_tensorhead_from_tensmul(data, key, tensorhead)
        return tensorhead, newdata

    def _check_permutations_on_data(self, tens, data):
        from .array import permutedims
        from .array.arrayop import Flatten

        if isinstance(tens, TensorHead):
            rank = tens.rank
            generators = tens.symmetry.generators
        elif isinstance(tens, Tensor):
            rank = tens.rank
            generators = tens.components[0].symmetry.generators
        elif isinstance(tens, TensorIndexType):
            rank = tens.metric.rank
            generators = tens.metric.symmetry.generators

        # Every generator is a permutation, check that by permuting the array
        # by that permutation, the array will be the same, except for a
        # possible sign change if the permutation admits it.
        for gener in generators:
            sign_change = +1 if (gener(rank) == rank) else -1
            data_swapped = data
            last_data = data
            permute_axes = list(map(gener, range(rank)))
            # the order of a permutation is the number of times to get the
            # identity by applying that permutation.
            for i in range(gener.order()-1):
                data_swapped = permutedims(data_swapped, permute_axes)
                # if any value in the difference array is non-zero, raise an error:
                if any(Flatten(last_data - sign_change*data_swapped)):
                    raise ValueError("Component data symmetry structure error")
                last_data = data_swapped

    def __setitem__(self, key, value):
        """
        Set the components data of a tensor object/expression.

        Explanation
        ===========

        Components data are transformed to the all-contravariant form and stored
        with the corresponding ``TensorHead`` object. If a ``TensorHead`` object
        cannot be uniquely identified, it will raise an error.
        """
        data = _TensorDataLazyEvaluator.parse_data(value)
        self._check_permutations_on_data(key, data)

        # TensorHead and TensorIndexType can be assigned data directly, while
        # TensMul must first convert data to a fully contravariant form, and
        # assign it to its corresponding TensorHead single component.
        if not isinstance(key, (TensorHead, TensorIndexType)):
            key, data = self._assign_data_to_tensor_expr(key, data)

        if isinstance(key, TensorHead):
            for dim, indextype in zip(data.shape, key.index_types):
                if indextype.data is None:
                    raise ValueError("index type {} has no components data"\
                    " associated (needed to raise/lower index)".format(indextype))
                if not indextype.dim.is_number:
                    continue
                if dim != indextype.dim:
                    raise ValueError("wrong dimension of ndarray")
        self._substitutions_dict[key] = data

    def __delitem__(self, key):
        del self._substitutions_dict[key]

    def __contains__(self, key):
        return key in self._substitutions_dict

    def add_metric_data(self, metric, data):
        """
        Assign data to the ``metric`` tensor. The metric tensor behaves in an
        anomalous way when raising and lowering indices.

        Explanation
        ===========

        A fully covariant metric is the inverse transpose of the fully
        contravariant metric (it is meant matrix inverse). If the metric is
        symmetric, the transpose is not necessary and mixed
        covariant/contravariant metrics are Kronecker deltas.
        """
        # hard assignment, data should not be added to `TensorHead` for metric:
        # the problem with `TensorHead` is that the metric is anomalous, i.e.
        # raising and lowering the index means considering the metric or its
        # inverse, this is not the case for other tensors.
        self._substitutions_dict_tensmul[metric, True, True] = data
        inverse_transpose = self.inverse_transpose_matrix(data)
        # in symmetric spaces, the transpose is the same as the original matrix,
        # the full covariant metric tensor is the inverse transpose, so this
        # code will be able to handle non-symmetric metrics.
        self._substitutions_dict_tensmul[metric, False, False] = inverse_transpose
        # now mixed cases, these are identical to the unit matrix if the metric
        # is symmetric.
        m = data.tomatrix()
        invt = inverse_transpose.tomatrix()
        self._substitutions_dict_tensmul[metric, True, False] = m * invt
        self._substitutions_dict_tensmul[metric, False, True] = invt * m

    @staticmethod
    def _flip_index_by_metric(data, metric, pos):
        from .array import tensorproduct, tensorcontraction

        mdim = metric.rank()
        ddim = data.rank()

        if pos == 0:
            data = tensorcontraction(
                tensorproduct(
                    metric,
                    data
                ),
                (1, mdim+pos)
            )
        else:
            data = tensorcontraction(
                tensorproduct(
                    data,
                    metric
                ),
                (pos, ddim)
            )
        return data

    @staticmethod
    def inverse_matrix(ndarray):
        m = ndarray.tomatrix().inv()
        return _TensorDataLazyEvaluator.parse_data(m)

    @staticmethod
    def inverse_transpose_matrix(ndarray):
        m = ndarray.tomatrix().inv().T
        return _TensorDataLazyEvaluator.parse_data(m)

    @staticmethod
    def _correct_signature_from_indices(data, indices, free, dum, inverse=False):
        """
        Utility function to correct the values inside the components data
        ndarray according to whether indices are covariant or contravariant.

        It uses the metric matrix to lower values of covariant indices.
        """
        # change the ndarray values according covariantness/contravariantness of the indices
        # use the metric
        for i, indx in enumerate(indices):
            if not indx.is_up and not inverse:
                data = _TensorDataLazyEvaluator._flip_index_by_metric(data, indx.tensor_index_type.data, i)
            elif not indx.is_up and inverse:
                data = _TensorDataLazyEvaluator._flip_index_by_metric(
                    data,
                    _TensorDataLazyEvaluator.inverse_matrix(indx.tensor_index_type.data),
                    i
                )
        return data

    @staticmethod
    def _sort_data_axes(old, new):
        from .array import permutedims

        new_data = old.data.copy()

        old_free = [i[0] for i in old.free]
        new_free = [i[0] for i in new.free]

        for i in range(len(new_free)):
            for j in range(i, len(old_free)):
                if old_free[j] == new_free[i]:
                    old_free[i], old_free[j] = old_free[j], old_free[i]
                    new_data = permutedims(new_data, (i, j))
                    break
        return new_data

    @staticmethod
    def add_rearrange_tensmul_parts(new_tensmul, old_tensmul):
        def sorted_compo():
            return _TensorDataLazyEvaluator._sort_data_axes(old_tensmul, new_tensmul)

        _TensorDataLazyEvaluator._substitutions_dict[new_tensmul] = sorted_compo()

    @staticmethod
    def parse_data(data):
        """
        Transform ``data`` to array. The parameter ``data`` may
        contain data in various formats, e.g. nested lists, SymPy ``Matrix``,
        and so on.

        Examples
        ========

        >>> from sympy.tensor.tensor import _TensorDataLazyEvaluator
        >>> _TensorDataLazyEvaluator.parse_data([1, 3, -6, 12])
        [1, 3, -6, 12]

        >>> _TensorDataLazyEvaluator.parse_data([[1, 2], [4, 7]])
        [[1, 2], [4, 7]]
        """
        from .array import MutableDenseNDimArray

        if not isinstance(data, MutableDenseNDimArray):
            if len(data) == 2 and hasattr(data[0], '__call__'):
                data = MutableDenseNDimArray(data[0], data[1])
            else:
                data = MutableDenseNDimArray(data)
        return data

_tensor_data_substitution_dict = _TensorDataLazyEvaluator()


class _TensorManager:
    """
    Class to manage tensor properties.

    Notes
    =====

    Tensors belong to tensor commutation groups; each group has a label
    ``comm``; there are predefined labels:

    ``0``   tensors commuting with any other tensor

    ``1``   tensors anticommuting among themselves

    ``2``   tensors not commuting, apart with those with ``comm=0``

    Other groups can be defined using ``set_comm``; tensors in those
    groups commute with those with ``comm=0``; by default they
    do not commute with any other group.
    """
    def __init__(self):
        self._comm_init()

    def _comm_init(self):
        self._comm = [{} for i in range(3)]
        for i in range(3):
            self._comm[0][i] = 0
            self._comm[i][0] = 0
        self._comm[1][1] = 1
        self._comm[2][1] = None
        self._comm[1][2] = None
        self._comm_symbols2i = {0:0, 1:1, 2:2}
        self._comm_i2symbol = {0:0, 1:1, 2:2}

    @property
    def comm(self):
        return self._comm

    def comm_symbols2i(self, i):
        """
        Get the commutation group number corresponding to ``i``.

        ``i`` can be a symbol or a number or a string.

        If ``i`` is not already defined its commutation group number
        is set.
        """
        if i not in self._comm_symbols2i:
            n = len(self._comm)
            self._comm.append({})
            self._comm[n][0] = 0
            self._comm[0][n] = 0
            self._comm_symbols2i[i] = n
            self._comm_i2symbol[n] = i
            return n
        return self._comm_symbols2i[i]

    def comm_i2symbol(self, i):
        """
        Returns the symbol corresponding to the commutation group number.
        """
        return self._comm_i2symbol[i]

    def set_comm(self, i, j, c):
        """
        Set the commutation parameter ``c`` for commutation groups ``i, j``.

        Parameters
        ==========

        i, j : symbols representing commutation groups

        c  :  group commutation number

        Notes
        =====

        ``i, j`` can be symbols, strings or numbers,
        apart from ``0, 1`` and ``2`` which are reserved respectively
        for commuting, anticommuting tensors and tensors not commuting
        with any other group apart with the commuting tensors.
        For the remaining cases, use this method to set the commutation rules;
        by default ``c=None``.

        The group commutation number ``c`` is assigned in correspondence
        to the group commutation symbols; it can be

        0        commuting

        1        anticommuting

        None     no commutation property

        Examples
        ========

        ``G`` and ``GH`` do not commute with themselves and commute with
        each other; A is commuting.

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, TensorManager, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz')
        >>> i0,i1,i2,i3,i4 = tensor_indices('i0:5', Lorentz)
        >>> A = TensorHead('A', [Lorentz])
        >>> G = TensorHead('G', [Lorentz], TensorSymmetry.no_symmetry(1), 'Gcomm')
        >>> GH = TensorHead('GH', [Lorentz], TensorSymmetry.no_symmetry(1), 'GHcomm')
        >>> TensorManager.set_comm('Gcomm', 'GHcomm', 0)
        >>> (GH(i1)*G(i0)).canon_bp()
        G(i0)*GH(i1)
        >>> (G(i1)*G(i0)).canon_bp()
        G(i1)*G(i0)
        >>> (G(i1)*A(i0)).canon_bp()
        A(i0)*G(i1)
        """
        if c not in (0, 1, None):
            raise ValueError('`c` can assume only the values 0, 1 or None')

        i = sympify(i)
        j = sympify(j)

        if i not in self._comm_symbols2i:
            n = len(self._comm)
            self._comm.append({})
            self._comm[n][0] = 0
            self._comm[0][n] = 0
            self._comm_symbols2i[i] = n
            self._comm_i2symbol[n] = i
        if j not in self._comm_symbols2i:
            n = len(self._comm)
            self._comm.append({})
            self._comm[0][n] = 0
            self._comm[n][0] = 0
            self._comm_symbols2i[j] = n
            self._comm_i2symbol[n] = j
        ni = self._comm_symbols2i[i]
        nj = self._comm_symbols2i[j]
        self._comm[ni][nj] = c
        self._comm[nj][ni] = c

        """
        Cached sympy functions (e.g. expand) may have cached the results of
        expressions involving tensors, but those results may not be valid after
        changing the commutation properties. To stay on the safe side, we clear
        the cache of all functions.
        """
        clear_cache()

    def set_comms(self, *args):
        """
        Set the commutation group numbers ``c`` for symbols ``i, j``.

        Parameters
        ==========

        args : sequence of ``(i, j, c)``
        """
        for i, j, c in args:
            self.set_comm(i, j, c)

    def get_comm(self, i, j):
        """
        Return the commutation parameter for commutation group numbers ``i, j``

        see ``_TensorManager.set_comm``
        """
        return self._comm[i].get(j, 0 if i == 0 or j == 0 else None)

    def clear(self):
        """
        Clear the TensorManager.
        """
        self._comm_init()


TensorManager = _TensorManager()


class TensorIndexType(Basic):
    """
    A TensorIndexType is characterized by its name and its metric.

    Parameters
    ==========

    name : name of the tensor type
    dummy_name : name of the head of dummy indices
    dim : dimension, it can be a symbol or an integer or ``None``
    eps_dim : dimension of the epsilon tensor
    metric_symmetry : integer that denotes metric symmetry or ``None`` for no metric
    metric_name : string with the name of the metric tensor

    Attributes
    ==========

    ``metric`` : the metric tensor
    ``delta`` : ``Kronecker delta``
    ``epsilon`` : the ``Levi-Civita epsilon`` tensor
    ``data`` : (deprecated) a property to add ``ndarray`` values, to work in a specified basis.

    Notes
    =====

    The possible values of the ``metric_symmetry`` parameter are:

        ``1``   :   metric tensor is fully symmetric
        ``0``   :   metric tensor possesses no index symmetry
        ``-1``  :   metric tensor is fully antisymmetric
        ``None``:   there is no metric tensor (metric equals to ``None``)

    The metric is assumed to be symmetric by default. It can also be set
    to a custom tensor by the ``.set_metric()`` method.

    If there is a metric the metric is used to raise and lower indices.

    In the case of non-symmetric metric, the following raising and
    lowering conventions will be adopted:

    ``psi(a) = g(a, b)*psi(-b); chi(-a) = chi(b)*g(-b, -a)``

    From these it is easy to find:

    ``g(-a, b) = delta(-a, b)``

    where ``delta(-a, b) = delta(b, -a)`` is the ``Kronecker delta``
    (see ``TensorIndex`` for the conventions on indices).
    For antisymmetric metrics there is also the following equality:

    ``g(a, -b) = -delta(a, -b)``

    If there is no metric it is not possible to raise or lower indices;
    e.g. the index of the defining representation of ``SU(N)``
    is 'covariant' and the conjugate representation is
    'contravariant'; for ``N > 2`` they are linearly independent.

    ``eps_dim`` is by default equal to ``dim``, if the latter is an integer;
    else it can be assigned (for use in naive dimensional regularization);
    if ``eps_dim`` is not an integer ``epsilon`` is ``None``.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> Lorentz.metric
    metric(Lorentz,Lorentz)
    """

    def __new__(cls, name, dummy_name=None, dim=None, eps_dim=None,
                metric_symmetry=1, metric_name='metric', **kwargs):
        if 'dummy_fmt' in kwargs:
            dummy_fmt = kwargs['dummy_fmt']
            sympy_deprecation_warning(
                f"""
                The dummy_fmt keyword to TensorIndexType is deprecated. Use
                dummy_name={dummy_fmt} instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-tensorindextype-dummy-fmt",
            )
            dummy_name = dummy_fmt

        if isinstance(name, str):
            name = Symbol(name)

        if dummy_name is None:
            dummy_name = str(name)[0]
        if isinstance(dummy_name, str):
            dummy_name = Symbol(dummy_name)

        if dim is None:
            dim = Symbol("dim_" + dummy_name.name)
        else:
            dim = sympify(dim)

        if eps_dim is None:
            eps_dim = dim
        else:
            eps_dim = sympify(eps_dim)

        metric_symmetry = sympify(metric_symmetry)

        if isinstance(metric_name, str):
            metric_name = Symbol(metric_name)

        if 'metric' in kwargs:
            SymPyDeprecationWarning(
                """
                The 'metric' keyword argument to TensorIndexType is
                deprecated. Use the 'metric_symmetry' keyword argument or the
                TensorIndexType.set_metric() method instead.
                """,
                deprecated_since_version="1.5",
                active_deprecations_target="deprecated-tensorindextype-metric",
            )
            metric = kwargs.get('metric')
            if metric is not None:
                if metric in (True, False, 0, 1):
                    metric_name = 'metric'
                    #metric_antisym = metric
                else:
                    metric_name = metric.name
                    #metric_antisym = metric.antisym

                if metric:
                    metric_symmetry = -1
                else:
                    metric_symmetry = 1

        obj = Basic.__new__(cls, name, dummy_name, dim, eps_dim,
                            metric_symmetry, metric_name)

        obj._autogenerated = []
        return obj

    @property
    def name(self):
        return self.args[0].name

    @property
    def dummy_name(self):
        return self.args[1].name

    @property
    def dim(self):
        return self.args[2]

    @property
    def eps_dim(self):
        return self.args[3]

    @memoize_property
    def metric(self):
        metric_symmetry = self.args[4]
        metric_name = self.args[5]
        if metric_symmetry is None:
            return None

        if metric_symmetry == 0:
            symmetry = TensorSymmetry.no_symmetry(2)
        elif metric_symmetry == 1:
            symmetry = TensorSymmetry.fully_symmetric(2)
        elif metric_symmetry == -1:
            symmetry = TensorSymmetry.fully_symmetric(-2)

        return TensorHead(metric_name, [self]*2, symmetry)

    @memoize_property
    def delta(self):
        return TensorHead('KD', [self]*2, TensorSymmetry.fully_symmetric(2))

    @memoize_property
    def epsilon(self):
        if not isinstance(self.eps_dim, (SYMPY_INTS, Integer)):
            return None
        symmetry = TensorSymmetry.fully_symmetric(-self.eps_dim)
        return TensorHead('Eps', [self]*self.eps_dim, symmetry)

    def set_metric(self, tensor):
        self._metric = tensor

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name

    __repr__ = __str__

    # Everything below this line is deprecated

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self]

    @data.setter
    def data(self, data):
        deprecate_data()
        # This assignment is a bit controversial, should metric components be assigned
        # to the metric only or also to the TensorIndexType object? The advantage here
        # is the ability to assign a 1D array and transform it to a 2D diagonal array.
        from .array import MutableDenseNDimArray

        data = _TensorDataLazyEvaluator.parse_data(data)
        if data.rank() > 2:
            raise ValueError("data have to be of rank 1 (diagonal metric) or 2.")
        if data.rank() == 1:
            if self.dim.is_number:
                nda_dim = data.shape[0]
                if nda_dim != self.dim:
                    raise ValueError("Dimension mismatch")

            dim = data.shape[0]
            newndarray = MutableDenseNDimArray.zeros(dim, dim)
            for i, val in enumerate(data):
                newndarray[i, i] = val
            data = newndarray
        dim1, dim2 = data.shape
        if dim1 != dim2:
            raise ValueError("Non-square matrix tensor.")
        if self.dim.is_number:
            if self.dim != dim1:
                raise ValueError("Dimension mismatch")
        _tensor_data_substitution_dict[self] = data
        _tensor_data_substitution_dict.add_metric_data(self.metric, data)
        with ignore_warnings(SymPyDeprecationWarning):
            delta = self.get_kronecker_delta()
        i1 = TensorIndex('i1', self)
        i2 = TensorIndex('i2', self)
        with ignore_warnings(SymPyDeprecationWarning):
            delta(i1, -i2).data = _TensorDataLazyEvaluator.parse_data(eye(dim1))

    @data.deleter
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self]
            if self.metric in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self.metric]

    @deprecated(
        """
        The TensorIndexType.get_kronecker_delta() method is deprecated. Use
        the TensorIndexType.delta attribute instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-tensorindextype-methods",
    )
    def get_kronecker_delta(self):
        sym2 = TensorSymmetry(get_symmetric_group_sgs(2))
        delta = TensorHead('KD', [self]*2, sym2)
        return delta

    @deprecated(
        """
        The TensorIndexType.get_epsilon() method is deprecated. Use
        the TensorIndexType.epsilon attribute instead.
        """,
        deprecated_since_version="1.5",
        active_deprecations_target="deprecated-tensorindextype-methods",
    )
    def get_epsilon(self):
        if not isinstance(self._eps_dim, (SYMPY_INTS, Integer)):
            return None
        sym = TensorSymmetry(get_symmetric_group_sgs(self._eps_dim, 1))
        epsilon = TensorHead('Eps', [self]*self._eps_dim, sym)
        return epsilon

    def _components_data_full_destroy(self):
        """
        EXPERIMENTAL: do not rely on this API method.

        This destroys components data associated to the ``TensorIndexType``, if
        any, specifically:

        * metric tensor data
        * Kronecker tensor data
        """
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]

        def delete_tensmul_data(key):
            if key in _tensor_data_substitution_dict._substitutions_dict_tensmul:
                del _tensor_data_substitution_dict._substitutions_dict_tensmul[key]

        # delete metric data:
        delete_tensmul_data((self.metric, True, True))
        delete_tensmul_data((self.metric, True, False))
        delete_tensmul_data((self.metric, False, True))
        delete_tensmul_data((self.metric, False, False))

        # delete delta tensor data:
        delta = self.get_kronecker_delta()
        if delta in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[delta]


class TensorIndex(Basic):
    """
    Represents a tensor index

    Parameters
    ==========

    name : name of the index, or ``True`` if you want it to be automatically assigned
    tensor_index_type : ``TensorIndexType`` of the index
    is_up :  flag for contravariant index (is_up=True by default)

    Attributes
    ==========

    ``name``
    ``tensor_index_type``
    ``is_up``

    Notes
    =====

    Tensor indices are contracted with the Einstein summation convention.

    An index can be in contravariant or in covariant form; in the latter
    case it is represented prepending a ``-`` to the index name. Adding
    ``-`` to a covariant (is_up=False) index makes it contravariant.

    Dummy indices have a name with head given by
    ``tensor_inde_type.dummy_name`` with underscore and a number.

    Similar to ``symbols`` multiple contravariant indices can be created
    at once using ``tensor_indices(s, typ)``, where ``s`` is a string
    of names.


    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorIndex, TensorHead, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> mu = TensorIndex('mu', Lorentz, is_up=False)
    >>> nu, rho = tensor_indices('nu, rho', Lorentz)
    >>> A = TensorHead('A', [Lorentz, Lorentz])
    >>> A(mu, nu)
    A(-mu, nu)
    >>> A(-mu, -rho)
    A(mu, -rho)
    >>> A(mu, -mu)
    A(-L_0, L_0)
    """
    def __new__(cls, name, tensor_index_type, is_up=True):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        elif name is True:
            name = "_i{}".format(len(tensor_index_type._autogenerated))
            name_symbol = Symbol(name)
            tensor_index_type._autogenerated.append(name_symbol)
        else:
            raise ValueError("invalid name")

        is_up = sympify(is_up)
        return Basic.__new__(cls, name_symbol, tensor_index_type, is_up)

    @property
    def name(self):
        return self.args[0].name

    @property
    def tensor_index_type(self):
        return self.args[1]

    @property
    def is_up(self):
        return self.args[2]

    def _print(self):
        s = self.name
        if not self.is_up:
            s = '-%s' % s
        return s

    def __lt__(self, other):
        return ((self.tensor_index_type, self.name) <
                (other.tensor_index_type, other.name))

    def __neg__(self):
        t1 = TensorIndex(self.name, self.tensor_index_type,
                (not self.is_up))
        return t1


def tensor_indices(s, typ):
    """
    Returns list of tensor indices given their names and their types.

    Parameters
    ==========

    s : string of comma separated names of indices

    typ : ``TensorIndexType`` of the indices

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
    """
    if isinstance(s, str):
        a = [x.name for x in symbols(s, seq=True)]
    else:
        raise ValueError('expecting a string')

    tilist = [TensorIndex(i, typ) for i in a]
    if len(tilist) == 1:
        return tilist[0]
    return tilist


class TensorSymmetry(Basic):
    """
    Monoterm symmetry of a tensor (i.e. any symmetric or anti-symmetric
    index permutation). For the relevant terminology see ``tensor_can.py``
    section of the combinatorics module.

    Parameters
    ==========

    bsgs : tuple ``(base, sgs)`` BSGS of the symmetry of the tensor

    Attributes
    ==========

    ``base`` : base of the BSGS
    ``generators`` : generators of the BSGS
    ``rank`` : rank of the tensor

    Notes
    =====

    A tensor can have an arbitrary monoterm symmetry provided by its BSGS.
    Multiterm symmetries, like the cyclic symmetry of the Riemann tensor
    (i.e., Bianchi identity), are not covered. See combinatorics module for
    information on how to generate BSGS for a general index permutation group.
    Simple symmetries can be generated using built-in methods.

    See Also
    ========

    sympy.combinatorics.tensor_can.get_symmetric_group_sgs

    Examples
    ========

    Define a symmetric tensor of rank 2

    >>> from sympy.tensor.tensor import TensorIndexType, TensorSymmetry, get_symmetric_group_sgs, TensorHead
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> sym = TensorSymmetry(get_symmetric_group_sgs(2))
    >>> T = TensorHead('T', [Lorentz]*2, sym)

    Note, that the same can also be done using built-in TensorSymmetry methods

    >>> sym2 = TensorSymmetry.fully_symmetric(2)
    >>> sym == sym2
    True
    """
    def __new__(cls, *args, **kw_args):
        if len(args) == 1:
            base, generators = args[0]
        elif len(args) == 2:
            base, generators = args
        else:
            raise TypeError("bsgs required, either two separate parameters or one tuple")

        if not isinstance(base, Tuple):
            base = Tuple(*base)
        if not isinstance(generators, Tuple):
            generators = Tuple(*generators)

        return Basic.__new__(cls, base, generators, **kw_args)

    @property
    def base(self):
        return self.args[0]

    @property
    def generators(self):
        return self.args[1]

    @property
    def rank(self):
        return self.generators[0].size - 2

    @classmethod
    def fully_symmetric(cls, rank):
        """
        Returns a fully symmetric (antisymmetric if ``rank``<0)
        TensorSymmetry object for ``abs(rank)`` indices.
        """
        if rank > 0:
            bsgs = get_symmetric_group_sgs(rank, False)
        elif rank < 0:
            bsgs = get_symmetric_group_sgs(-rank, True)
        elif rank == 0:
            bsgs = ([], [Permutation(1)])
        return TensorSymmetry(bsgs)

    @classmethod
    def direct_product(cls, *args):
        """
        Returns a TensorSymmetry object that is being a direct product of
        fully (anti-)symmetric index permutation groups.

        Notes
        =====

        Some examples for different values of ``(*args)``:
        ``(1)``         vector, equivalent to ``TensorSymmetry.fully_symmetric(1)``
        ``(2)``         tensor with 2 symmetric indices, equivalent to ``.fully_symmetric(2)``
        ``(-2)``        tensor with 2 antisymmetric indices, equivalent to ``.fully_symmetric(-2)``
        ``(2, -2)``     tensor with the first 2 indices commuting and the last 2 anticommuting
        ``(1, 1, 1)``   tensor with 3 indices without any symmetry
        """
        base, sgs = [], [Permutation(1)]
        for arg in args:
            if arg > 0:
                bsgs2 = get_symmetric_group_sgs(arg, False)
            elif arg < 0:
                bsgs2 = get_symmetric_group_sgs(-arg, True)
            else:
                continue
            base, sgs = bsgs_direct_product(base, sgs, *bsgs2)

        return TensorSymmetry(base, sgs)

    @classmethod
    def riemann(cls):
        """
        Returns a monotorem symmetry of the Riemann tensor
        """
        return TensorSymmetry(riemann_bsgs)

    @classmethod
    def no_symmetry(cls, rank):
        """
        TensorSymmetry object for ``rank`` indices with no symmetry
        """
        return TensorSymmetry([], [Permutation(rank+1)])


@deprecated(
    """
    The tensorsymmetry() function is deprecated. Use the TensorSymmetry
    constructor instead.
    """,
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensorsymmetry",
)
def tensorsymmetry(*args):
    """
    Returns a ``TensorSymmetry`` object. This method is deprecated, use
    ``TensorSymmetry.direct_product()`` or ``.riemann()`` instead.

    Explanation
    ===========

    One can represent a tensor with any monoterm slot symmetry group
    using a BSGS.

    ``args`` can be a BSGS
    ``args[0]``    base
    ``args[1]``    sgs

    Usually tensors are in (direct products of) representations
    of the symmetric group;
    ``args`` can be a list of lists representing the shapes of Young tableaux

    Notes
    =====

    For instance:
    ``[[1]]``       vector
    ``[[1]*n]``     symmetric tensor of rank ``n``
    ``[[n]]``       antisymmetric tensor of rank ``n``
    ``[[2, 2]]``    monoterm slot symmetry of the Riemann tensor
    ``[[1],[1]]``   vector*vector
    ``[[2],[1],[1]`` (antisymmetric tensor)*vector*vector

    Notice that with the shape ``[2, 2]`` we associate only the monoterm
    symmetries of the Riemann tensor; this is an abuse of notation,
    since the shape ``[2, 2]`` corresponds usually to the irreducible
    representation characterized by the monoterm symmetries and by the
    cyclic symmetry.
    """
    from sympy.combinatorics import Permutation

    def tableau2bsgs(a):
        if len(a) == 1:
            # antisymmetric vector
            n = a[0]
            bsgs = get_symmetric_group_sgs(n, 1)
        else:
            if all(x == 1 for x in a):
                # symmetric vector
                n = len(a)
                bsgs = get_symmetric_group_sgs(n)
            elif a == [2, 2]:
                bsgs = riemann_bsgs
            else:
                raise NotImplementedError
        return bsgs

    if not args:
        return TensorSymmetry(Tuple(), Tuple(Permutation(1)))

    if len(args) == 2 and isinstance(args[1][0], Permutation):
        return TensorSymmetry(args)
    base, sgs = tableau2bsgs(args[0])
    for a in args[1:]:
        basex, sgsx = tableau2bsgs(a)
        base, sgs = bsgs_direct_product(base, sgs, basex, sgsx)
    return TensorSymmetry(Tuple(base, sgs))

@deprecated(
    "TensorType is deprecated. Use tensor_heads() instead.",
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensortype",
)
class TensorType(Basic):
    """
    Class of tensor types. Deprecated, use tensor_heads() instead.

    Parameters
    ==========

    index_types : list of ``TensorIndexType`` of the tensor indices
    symmetry : ``TensorSymmetry`` of the tensor

    Attributes
    ==========

    ``index_types``
    ``symmetry``
    ``types`` : list of ``TensorIndexType`` without repetitions
    """
    is_commutative = False

    def __new__(cls, index_types, symmetry, **kw_args):
        assert symmetry.rank == len(index_types)
        obj = Basic.__new__(cls, Tuple(*index_types), symmetry, **kw_args)
        return obj

    @property
    def index_types(self):
        return self.args[0]

    @property
    def symmetry(self):
        return self.args[1]

    @property
    def types(self):
        return sorted(set(self.index_types), key=lambda x: x.name)

    def __str__(self):
        return 'TensorType(%s)' % ([str(x) for x in self.index_types])

    def __call__(self, s, comm=0):
        """
        Return a TensorHead object or a list of TensorHead objects.

        Parameters
        ==========

        s : name or string of names.

        comm : Commutation group.

        see ``_TensorManager.set_comm``
        """
        if isinstance(s, str):
            names = [x.name for x in symbols(s, seq=True)]
        else:
            raise ValueError('expecting a string')
        if len(names) == 1:
            return TensorHead(names[0], self.index_types, self.symmetry, comm)
        else:
            return [TensorHead(name, self.index_types, self.symmetry, comm) for name in names]


@deprecated(
    """
    The tensorhead() function is deprecated. Use tensor_heads() instead.
    """,
    deprecated_since_version="1.5",
    active_deprecations_target="deprecated-tensorhead",
)
def tensorhead(name, typ, sym=None, comm=0):
    """
    Function generating tensorhead(s). This method is deprecated,
    use TensorHead constructor or tensor_heads() instead.

    Parameters
    ==========

    name : name or sequence of names (as in ``symbols``)

    typ :  index types

    sym :  same as ``*args`` in ``tensorsymmetry``

    comm : commutation group number
    see ``_TensorManager.set_comm``
    """
    if sym is None:
        sym = [[1] for i in range(len(typ))]
    with ignore_warnings(SymPyDeprecationWarning):
        sym = tensorsymmetry(*sym)
    return TensorHead(name, typ, sym, comm)


class TensorHead(Basic):
    """
    Tensor head of the tensor.

    Parameters
    ==========

    name : name of the tensor
    index_types : list of TensorIndexType
    symmetry : TensorSymmetry of the tensor
    comm : commutation group number

    Attributes
    ==========

    ``name``
    ``index_types``
    ``rank`` : total number of indices
    ``symmetry``
    ``comm`` : commutation group

    Notes
    =====

    Similar to ``symbols`` multiple TensorHeads can be created using
    ``tensorhead(s, typ, sym=None, comm=0)`` function, where ``s``
    is the string of names and ``sym`` is the monoterm tensor symmetry
    (see ``tensorsymmetry``).

    A ``TensorHead`` belongs to a commutation group, defined by a
    symbol on number ``comm`` (see ``_TensorManager.set_comm``);
    tensors in a commutation group have the same commutation properties;
    by default ``comm`` is ``0``, the group of the commuting tensors.

    Examples
    ========

    Define a fully antisymmetric tensor of rank 2:

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorSymmetry
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> asym2 = TensorSymmetry.fully_symmetric(-2)
    >>> A = TensorHead('A', [Lorentz, Lorentz], asym2)

    Examples with ndarray values, the components data assigned to the
    ``TensorHead`` object are assumed to be in a fully-contravariant
    representation. In case it is necessary to assign components data which
    represents the values of a non-fully covariant tensor, see the other
    examples.

    >>> from sympy.tensor.tensor import tensor_indices
    >>> from sympy import diag
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> i0, i1 = tensor_indices('i0:2', Lorentz)

    Specify a replacement dictionary to keep track of the arrays to use for
    replacements in the tensorial expression. The ``TensorIndexType`` is
    associated to the metric used for contractions (in fully covariant form):

    >>> repl = {Lorentz: diag(1, -1, -1, -1)}

    Let's see some examples of working with components with the electromagnetic
    tensor:

    >>> from sympy import symbols
    >>> Ex, Ey, Ez, Bx, By, Bz = symbols('E_x E_y E_z B_x B_y B_z')
    >>> c = symbols('c', positive=True)

    Let's define `F`, an antisymmetric tensor:

    >>> F = TensorHead('F', [Lorentz, Lorentz], asym2)

    Let's update the dictionary to contain the matrix to use in the
    replacements:

    >>> repl.update({F(-i0, -i1): [
    ... [0, Ex/c, Ey/c, Ez/c],
    ... [-Ex/c, 0, -Bz, By],
    ... [-Ey/c, Bz, 0, -Bx],
    ... [-Ez/c, -By, Bx, 0]]})

    Now it is possible to retrieve the contravariant form of the Electromagnetic
    tensor:

    >>> F(i0, i1).replace_with_arrays(repl, [i0, i1])
    [[0, -E_x/c, -E_y/c, -E_z/c], [E_x/c, 0, -B_z, B_y], [E_y/c, B_z, 0, -B_x], [E_z/c, -B_y, B_x, 0]]

    and the mixed contravariant-covariant form:

    >>> F(i0, -i1).replace_with_arrays(repl, [i0, -i1])
    [[0, E_x/c, E_y/c, E_z/c], [E_x/c, 0, B_z, -B_y], [E_y/c, -B_z, 0, B_x], [E_z/c, B_y, -B_x, 0]]

    Energy-momentum of a particle may be represented as:

    >>> from sympy import symbols
    >>> P = TensorHead('P', [Lorentz], TensorSymmetry.no_symmetry(1))
    >>> E, px, py, pz = symbols('E p_x p_y p_z', positive=True)
    >>> repl.update({P(i0): [E, px, py, pz]})

    The contravariant and covariant components are, respectively:

    >>> P(i0).replace_with_arrays(repl, [i0])
    [E, p_x, p_y, p_z]
    >>> P(-i0).replace_with_arrays(repl, [-i0])
    [E, -p_x, -p_y, -p_z]

    The contraction of a 1-index tensor by itself:

    >>> expr = P(i0)*P(-i0)
    >>> expr.replace_with_arrays(repl, [])
    E**2 - p_x**2 - p_y**2 - p_z**2
    """
    is_commutative = False

    def __new__(cls, name, index_types, symmetry=None, comm=0):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        else:
            raise ValueError("invalid name")

        if symmetry is None:
            symmetry = TensorSymmetry.no_symmetry(len(index_types))
        else:
            assert symmetry.rank == len(index_types)

        obj = Basic.__new__(cls, name_symbol, Tuple(*index_types), symmetry, sympify(comm))
        return obj

    @property
    def name(self):
        return self.args[0].name

    @property
    def index_types(self):
        return list(self.args[1])

    @property
    def symmetry(self):
        return self.args[2]

    @property
    def comm(self):
        return TensorManager.comm_symbols2i(self.args[3])

    @property
    def rank(self):
        return len(self.index_types)

    def __lt__(self, other):
        return (self.name, self.index_types) < (other.name, other.index_types)

    def commutes_with(self, other):
        """
        Returns ``0`` if ``self`` and ``other`` commute, ``1`` if they anticommute.

        Returns ``None`` if ``self`` and ``other`` neither commute nor anticommute.
        """
        r = TensorManager.get_comm(self.comm, other.comm)
        return r

    def _print(self):
        return '%s(%s)' %(self.name, ','.join([str(x) for x in self.index_types]))

    def __call__(self, *indices, **kw_args):
        """
        Returns a tensor with indices.

        Explanation
        ===========

        There is a special behavior in case of indices denoted by ``True``,
        they are considered auto-matrix indices, their slots are automatically
        filled, and confer to the tensor the behavior of a matrix or vector
        upon multiplication with another tensor containing auto-matrix indices
        of the same ``TensorIndexType``. This means indices get summed over the
        same way as in matrix multiplication. For matrix behavior, define two
        auto-matrix indices, for vector behavior define just one.

        Indices can also be strings, in which case the attribute
        ``index_types`` is used to convert them to proper ``TensorIndex``.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, TensorHead
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> a, b = tensor_indices('a,b', Lorentz)
        >>> A = TensorHead('A', [Lorentz]*2, TensorSymmetry.no_symmetry(2))
        >>> t = A(a, -b)
        >>> t
        A(a, -b)

        """

        updated_indices = []
        for idx, typ in zip(indices, self.index_types):
            if isinstance(idx, str):
                idx = idx.strip().replace(" ", "")
                if idx.startswith('-'):
                    updated_indices.append(TensorIndex(idx[1:], typ,
                                           is_up=False))
                else:
                    updated_indices.append(TensorIndex(idx, typ))
            else:
                updated_indices.append(idx)

        updated_indices += indices[len(updated_indices):]

        tensor = Tensor(self, updated_indices, **kw_args)
        return tensor.doit()

    # Everything below this line is deprecated

    def __pow__(self, other):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self.data is None:
                raise ValueError("No power on abstract tensors.")
            from .array import tensorproduct, tensorcontraction
            metrics = [_.data for _ in self.index_types]

            marray = self.data
            marraydim = marray.rank()
            for metric in metrics:
                marray = tensorproduct(marray, metric, marray)
                marray = tensorcontraction(marray, (0, marraydim), (marraydim+1, marraydim+2))

            return marray ** (other * S.Half)

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self]

    @data.setter
    def data(self, data):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            _tensor_data_substitution_dict[self] = data

    @data.deleter
    def data(self):
        deprecate_data()
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]

    def __iter__(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data.__iter__()

    def _components_data_full_destroy(self):
        """
        EXPERIMENTAL: do not rely on this API method.

        Destroy components data associated to the ``TensorHead`` object, this
        checks for attached components data, and destroys components data too.
        """
        # do not garbage collect Kronecker tensor (it should be done by
        # ``TensorIndexType`` garbage collection)
        deprecate_data()
        if self.name == "KD":
            return

        # the data attached to a tensor must be deleted only by the TensorHead
        # destructor. If the TensorHead is deleted, it means that there are no
        # more instances of that tensor anywhere.
        if self in _tensor_data_substitution_dict:
            del _tensor_data_substitution_dict[self]


def tensor_heads(s, index_types, symmetry=None, comm=0):
    """
    Returns a sequence of TensorHeads from a string `s`
    """
    if isinstance(s, str):
        names = [x.name for x in symbols(s, seq=True)]
    else:
        raise ValueError('expecting a string')

    thlist = [TensorHead(name, index_types, symmetry, comm) for name in names]
    if len(thlist) == 1:
        return thlist[0]
    return thlist


class TensExpr(Expr, ABC):
    """
    Abstract base class for tensor expressions

    Notes
    =====

    A tensor expression is an expression formed by tensors;
    currently the sums of tensors are distributed.

    A ``TensExpr`` can be a ``TensAdd`` or a ``TensMul``.

    ``TensMul`` objects are formed by products of component tensors,
    and include a coefficient, which is a SymPy expression.


    In the internal representation contracted indices are represented
    by ``(ipos1, ipos2, icomp1, icomp2)``, where ``icomp1`` is the position
    of the component tensor with contravariant index, ``ipos1`` is the
    slot which the index occupies in that component tensor.

    Contracted indices are therefore nameless in the internal representation.
    """

    _op_priority = 12.0
    is_commutative = False

    def __neg__(self):
        return self*S.NegativeOne

    def __abs__(self):
        raise NotImplementedError

    def __add__(self, other):
        return TensAdd(self, other).doit()

    def __radd__(self, other):
        return TensAdd(other, self).doit()

    def __sub__(self, other):
        return TensAdd(self, -other).doit()

    def __rsub__(self, other):
        return TensAdd(other, -self).doit()

    def __mul__(self, other):
        """
        Multiply two tensors using Einstein summation convention.

        Explanation
        ===========

        If the two tensors have an index in common, one contravariant
        and the other covariant, in their product the indices are summed

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t1 = p(m0)
        >>> t2 = q(-m0)
        >>> t1*t2
        p(L_0)*q(-L_0)
        """
        return TensMul(self, other).doit()

    def __rmul__(self, other):
        return TensMul(other, self).doit()

    def __truediv__(self, other):
        other = _sympify(other)
        if isinstance(other, TensExpr):
            raise ValueError('cannot divide by a tensor')
        return TensMul(self, S.One/other).doit()

    def __rtruediv__(self, other):
        raise ValueError('cannot divide by a tensor')

    def __pow__(self, other):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self.data is None:
                raise ValueError("No power without ndarray data.")
            from .array import tensorproduct, tensorcontraction
            free = self.free
            marray = self.data
            mdim = marray.rank()
            for metric in free:
                marray = tensorcontraction(
                    tensorproduct(
                    marray,
                    metric[0].tensor_index_type.data,
                    marray),
                    (0, mdim), (mdim+1, mdim+2)
                )
            return marray ** (other * S.Half)

    def __rpow__(self, other):
        raise NotImplementedError

    @property
    @abstractmethod
    def nocoeff(self):
        raise NotImplementedError("abstract method")

    @property
    @abstractmethod
    def coeff(self):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def get_indices(self):
        raise NotImplementedError("abstract method")

    @abstractmethod
    def get_free_indices(self) -> list[TensorIndex]:
        raise NotImplementedError("abstract method")

    @abstractmethod
    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        raise NotImplementedError("abstract method")

    def fun_eval(self, *index_tuples):
        deprecate_fun_eval()
        return self.substitute_indices(*index_tuples)

    def get_matrix(self):
        """
        DEPRECATED: do not use.

        Returns ndarray components data as a matrix, if components data are
        available and ndarray dimension does not exceed 2.
        """
        from sympy.matrices.dense import Matrix
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if 0 < self.rank <= 2:
                rows = self.data.shape[0]
                columns = self.data.shape[1] if self.rank == 2 else 1
                if self.rank == 2:
                    mat_list = [] * rows
                    for i in range(rows):
                        mat_list.append([])
                        for j in range(columns):
                            mat_list[i].append(self[i, j])
                else:
                    mat_list = [None] * rows
                    for i in range(rows):
                        mat_list[i] = self[i]
                return Matrix(mat_list)
            else:
                raise NotImplementedError(
                    "missing multidimensional reduction to matrix.")

    @staticmethod
    def _get_indices_permutation(indices1, indices2):
        return [indices1.index(i) for i in indices2]

    def expand(self, **hints):
        return _expand(self, **hints).doit()

    def _expand(self, **kwargs):
        return self

    def _get_free_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_free_indices_set())
        return indset

    def _get_dummy_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_dummy_indices_set())
        return indset

    def _get_indices_set(self):
        indset = set()
        for arg in self.args:
            if isinstance(arg, TensExpr):
                indset.update(arg._get_indices_set())
        return indset

    @property
    def _iterate_dummy_indices(self):
        dummy_set = self._get_dummy_indices_set()

        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                if expr in dummy_set:
                    yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos+(p,))

        return recursor(self, ())

    @property
    def _iterate_free_indices(self):
        free_set = self._get_free_indices_set()

        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                if expr in free_set:
                    yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos+(p,))

        return recursor(self, ())

    @property
    def _iterate_indices(self):
        def recursor(expr, pos):
            if isinstance(expr, TensorIndex):
                yield (expr, pos)
            elif isinstance(expr, (Tuple, TensExpr)):
                for p, arg in enumerate(expr.args):
                    yield from recursor(arg, pos+(p,))

        return recursor(self, ())

    @staticmethod
    def _contract_and_permute_with_metric(metric, array, pos, dim):
        # TODO: add possibility of metric after (spinors)
        from .array import tensorcontraction, tensorproduct, permutedims

        array = tensorcontraction(tensorproduct(metric, array), (1, 2+pos))
        permu = list(range(dim))
        permu[0], permu[pos] = permu[pos], permu[0]
        return permutedims(array, permu)

    @staticmethod
    def _match_indices_with_other_tensor(array, free_ind1, free_ind2, replacement_dict):
        from .array import permutedims

        index_types1 = [i.tensor_index_type for i in free_ind1]

        # Check if variance of indices needs to be fixed:
        pos2up = []
        pos2down = []
        free2remaining = free_ind2[:]
        for pos1, index1 in enumerate(free_ind1):
            if index1 in free2remaining:
                pos2 = free2remaining.index(index1)
                free2remaining[pos2] = None
                continue
            if -index1 in free2remaining:
                pos2 = free2remaining.index(-index1)
                free2remaining[pos2] = None
                free_ind2[pos2] = index1
                if index1.is_up:
                    pos2up.append(pos2)
                else:
                    pos2down.append(pos2)
            else:
                index2 = free2remaining[pos1]
                if index2 is None:
                    raise ValueError("incompatible indices: %s and %s" % (free_ind1, free_ind2))
                free2remaining[pos1] = None
                free_ind2[pos1] = index1
                if index1.is_up ^ index2.is_up:
                    if index1.is_up:
                        pos2up.append(pos1)
                    else:
                        pos2down.append(pos1)

        if len(set(free_ind1) & set(free_ind2)) < len(free_ind1):
            raise ValueError("incompatible indices: %s and %s" % (free_ind1, free_ind2))

        # Raise indices:
        for pos in pos2up:
            index_type_pos = index_types1[pos]
            if index_type_pos not in replacement_dict:
                raise ValueError("No metric provided to lower index")
            metric = replacement_dict[index_type_pos]
            metric_inverse = _TensorDataLazyEvaluator.inverse_matrix(metric)
            array = TensExpr._contract_and_permute_with_metric(metric_inverse, array, pos, len(free_ind1))
        # Lower indices:
        for pos in pos2down:
            index_type_pos = index_types1[pos]
            if index_type_pos not in replacement_dict:
                raise ValueError("No metric provided to lower index")
            metric = replacement_dict[index_type_pos]
            array = TensExpr._contract_and_permute_with_metric(metric, array, pos, len(free_ind1))

        if free_ind1:
            permutation = TensExpr._get_indices_permutation(free_ind2, free_ind1)
            array = permutedims(array, permutation)

        if hasattr(array, "rank") and array.rank() == 0:
            array = array[()]

        return free_ind2, array

    def replace_with_arrays(self, replacement_dict, indices=None):
        """
        Replace the tensorial expressions with arrays. The final array will
        correspond to the N-dimensional array with indices arranged according
        to ``indices``.

        Parameters
        ==========

        replacement_dict
            dictionary containing the replacement rules for tensors.
        indices
            the index order with respect to which the array is read. The
            original index order will be used if no value is passed.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices
        >>> from sympy.tensor.tensor import TensorHead
        >>> from sympy import symbols, diag

        >>> L = TensorIndexType("L")
        >>> i, j = tensor_indices("i j", L)
        >>> A = TensorHead("A", [L])
        >>> A(i).replace_with_arrays({A(i): [1, 2]}, [i])
        [1, 2]

        Since 'indices' is optional, we can also call replace_with_arrays by
        this way if no specific index order is needed:

        >>> A(i).replace_with_arrays({A(i): [1, 2]})
        [1, 2]

        >>> expr = A(i)*A(j)
        >>> expr.replace_with_arrays({A(i): [1, 2]})
        [[1, 2], [2, 4]]

        For contractions, specify the metric of the ``TensorIndexType``, which
        in this case is ``L``, in its covariant form:

        >>> expr = A(i)*A(-i)
        >>> expr.replace_with_arrays({A(i): [1, 2], L: diag(1, -1)})
        -3

        Symmetrization of an array:

        >>> H = TensorHead("H", [L, L])
        >>> a, b, c, d = symbols("a b c d")
        >>> expr = H(i, j)/2 + H(j, i)/2
        >>> expr.replace_with_arrays({H(i, j): [[a, b], [c, d]]})
        [[a, b/2 + c/2], [b/2 + c/2, d]]

        Anti-symmetrization of an array:

        >>> expr = H(i, j)/2 - H(j, i)/2
        >>> repl = {H(i, j): [[a, b], [c, d]]}
        >>> expr.replace_with_arrays(repl)
        [[0, b/2 - c/2], [-b/2 + c/2, 0]]

        The same expression can be read as the transpose by inverting ``i`` and
        ``j``:

        >>> expr.replace_with_arrays(repl, [j, i])
        [[0, -b/2 + c/2], [b/2 - c/2, 0]]
        """
        from .array import Array

        indices = indices or []
        remap = {k.args[0] if k.is_up else -k.args[0]: k for k in self.get_free_indices()}
        for i, index in enumerate(indices):
            if isinstance(index, (Symbol, Mul)):
                if index in remap:
                    indices[i] = remap[index]
                else:
                    indices[i] = -remap[-index]

        replacement_dict = {tensor: Array(array) for tensor, array in replacement_dict.items()}

        # Check dimensions of replaced arrays:
        for tensor, array in replacement_dict.items():
            if isinstance(tensor, TensorIndexType):
                expected_shape = [tensor.dim for i in range(2)]
            else:
                expected_shape = [index_type.dim for index_type in tensor.index_types]
            if len(expected_shape) != array.rank() or (not all(dim1 == dim2 if
                dim1.is_number else True for dim1, dim2 in zip(expected_shape,
                array.shape))):
                raise ValueError("shapes for tensor %s expected to be %s, "\
                    "replacement array shape is %s" % (tensor, expected_shape,
                    array.shape))

        ret_indices, array = self._extract_data(replacement_dict)

        last_indices, array = self._match_indices_with_other_tensor(array, indices, ret_indices, replacement_dict)
        return array

    def _check_add_Sum(self, expr, index_symbols):
        from sympy.concrete.summations import Sum
        indices = self.get_indices()
        dum = self.dum
        sum_indices = [ (index_symbols[i], 0,
            indices[i].tensor_index_type.dim-1) for i, j in dum]
        if sum_indices:
            expr = Sum(expr, *sum_indices)
        return expr

    def _expand_partial_derivative(self):
        # simply delegate the _expand_partial_derivative() to
        # its arguments to expand a possibly found PartialDerivative
        return self.func(*[
                    a._expand_partial_derivative()
                    if isinstance(a, TensExpr) else a
                    for a in self.args])


class TensAdd(TensExpr, AssocOp):
    """
    Sum of tensors.

    Parameters
    ==========

    free_args : list of the free indices

    Attributes
    ==========

    ``args`` : tuple of addends
    ``rank`` : rank of the tensor
    ``free_args`` : list of the free indices in sorted order

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_heads, tensor_indices
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> a, b = tensor_indices('a,b', Lorentz)
    >>> p, q = tensor_heads('p,q', [Lorentz])
    >>> t = p(a) + q(a); t
    p(a) + q(a)

    Examples with components data added to the tensor expression:

    >>> from sympy import symbols, diag
    >>> x, y, z, t = symbols("x y z t")
    >>> repl = {}
    >>> repl[Lorentz] = diag(1, -1, -1, -1)
    >>> repl[p(a)] = [1, 2, 3, 4]
    >>> repl[q(a)] = [x, y, z, t]

    The following are: 2**2 - 3**2 - 2**2 - 7**2 ==> -58

    >>> expr = p(a) + q(a)
    >>> expr.replace_with_arrays(repl, [a])
    [x + 1, y + 2, z + 3, t + 4]
    """

    def __new__(cls, *args, **kw_args):
        args = [_sympify(x) for x in args if x]
        args = TensAdd._tensAdd_flatten(args)
        args.sort(key=default_sort_key)
        if not args:
            return S.Zero
        if len(args) == 1:
            return args[0]

        return Basic.__new__(cls, *args, **kw_args)

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    def get_free_indices(self) -> list[TensorIndex]:
        return self.free_indices

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        newargs = [arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg in self.args]
        return self.func(*newargs)

    @memoize_property
    def rank(self):
        if isinstance(self.args[0], TensExpr):
            return self.args[0].rank
        else:
            return 0

    @memoize_property
    def free_args(self):
        if isinstance(self.args[0], TensExpr):
            return self.args[0].free_args
        else:
            return []

    @memoize_property
    def free_indices(self):
        if isinstance(self.args[0], TensExpr):
            return self.args[0].get_free_indices()
        else:
            return set()

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args

        # if any of the args are zero (after doit), drop them. Otherwise, _tensAdd_check will complain about non-matching indices, even though the TensAdd is correctly formed.
        args = [arg for arg in args if arg != S.Zero]

        if len(args) == 0:
            return S.Zero
        elif len(args) == 1:
            return args[0]

        # now check that all addends have the same indices:
        TensAdd._tensAdd_check(args)

        # Collect terms appearing more than once, differing by their coefficients:
        args = TensAdd._tensAdd_collect_terms(args)

        # collect canonicalized terms
        def sort_key(t):
            if not isinstance(t, TensExpr):
                return [], [], []
            if hasattr(t, "_index_structure") and hasattr(t, "components"):
                x = get_index_structure(t)
                return t.components, x.free, x.dum
            return [], [], []
        args.sort(key=sort_key)

        if not args:
            return S.Zero
        # it there is only a component tensor return it
        if len(args) == 1:
            return args[0]

        obj = self.func(*args)
        return obj

    @staticmethod
    def _tensAdd_flatten(args):
        # flatten TensAdd, coerce terms which are not tensors to tensors
        a = []
        for x in args:
            if isinstance(x, (Add, TensAdd)):
                a.extend(list(x.args))
            else:
                a.append(x)
        args = [x for x in a if x.coeff]
        return args

    @staticmethod
    def _tensAdd_check(args):
        # check that all addends have the same free indices

        def get_indices_set(x: Expr) -> set[TensorIndex]:
            if isinstance(x, TensExpr):
                return set(x.get_free_indices())
            return set()

        indices0 = get_indices_set(args[0])
        list_indices = [get_indices_set(arg) for arg in args[1:]]
        if not all(x == indices0 for x in list_indices):
            raise ValueError('all tensors must have the same indices')

    @staticmethod
    def _tensAdd_collect_terms(args):
        # collect TensMul terms differing at most by their coefficient
        terms_dict = defaultdict(list)
        scalars = S.Zero
        if isinstance(args[0], TensExpr):
            free_indices = set(args[0].get_free_indices())
        else:
            free_indices = set()

        for arg in args:
            if not isinstance(arg, TensExpr):
                if free_indices != set():
                    raise ValueError("wrong valence")
                scalars += arg
                continue
            if free_indices != set(arg.get_free_indices()):
                raise ValueError("wrong valence")
            # TODO: what is the part which is not a coeff?
            # needs an implementation similar to .as_coeff_Mul()
            terms_dict[arg.nocoeff].append(arg.coeff)

        new_args = [TensMul(Add(*coeff), t).doit() for t, coeff in terms_dict.items() if Add(*coeff) != 0]
        if isinstance(scalars, Add):
            new_args = list(scalars.args) + new_args
        elif scalars != 0:
            new_args = [scalars] + new_args
        return new_args

    def get_indices(self):
        indices = []
        for arg in self.args:
            indices.extend([i for i in get_indices(arg) if i not in indices])
        return indices

    def _expand(self, **hints):
        return TensAdd(*[_expand(i, **hints) for i in self.args])

    def __call__(self, *indices):
        deprecate_call()
        free_args = self.free_args
        indices = list(indices)
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        if indices == free_args:
            return self
        index_tuples = list(zip(free_args, indices))
        a = [x.func(*x.substitute_indices(*index_tuples).args) for x in self.args]
        res = TensAdd(*a).doit()
        return res

    def canon_bp(self):
        """
        Canonicalize using the Butler-Portugal algorithm for canonicalization
        under monoterm symmetries.
        """
        expr = self.expand()
        args = [canon_bp(x) for x in expr.args]
        res = TensAdd(*args).doit()
        return res

    def equals(self, other):
        other = _sympify(other)
        if isinstance(other, TensMul) and other.coeff == 0:
            return all(x.coeff == 0 for x in self.args)
        if isinstance(other, TensExpr):
            if self.rank != other.rank:
                return False
        if isinstance(other, TensAdd):
            if set(self.args) != set(other.args):
                return False
            else:
                return True
        t = self - other
        if not isinstance(t, TensExpr):
            return t == 0
        else:
            if isinstance(t, TensMul):
                return t.coeff == 0
            else:
                return all(x.coeff == 0 for x in t.args)

    def __getitem__(self, item):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]

    def contract_delta(self, delta):
        args = [x.contract_delta(delta) for x in self.args]
        t = TensAdd(*args).doit()
        return canon_bp(t)

    def contract_metric(self, g):
        """
        Raise or lower indices with the metric ``g``.

        Parameters
        ==========

        g :  metric

        contract_all : if True, eliminate all ``g`` which are contracted

        Notes
        =====

        see the ``TensorIndexType`` docstring for the contraction conventions
        """

        args = [contract_metric(x, g) for x in self.args]
        t = TensAdd(*args).doit()
        return canon_bp(t)

    def substitute_indices(self, *index_tuples):
        new_args = []
        for arg in self.args:
            if isinstance(arg, TensExpr):
                arg = arg.substitute_indices(*index_tuples)
            new_args.append(arg)
        return TensAdd(*new_args).doit()

    def _print(self):
        a = []
        args = self.args
        for x in args:
            a.append(str(x))
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _extract_data(self, replacement_dict):
        from sympy.tensor.array import Array, permutedims
        args_indices, arrays = zip(*[
            arg._extract_data(replacement_dict) if
            isinstance(arg, TensExpr) else ([], arg) for arg in self.args
        ])
        arrays = [Array(i) for i in arrays]
        ref_indices = args_indices[0]
        for i in range(1, len(args_indices)):
            indices = args_indices[i]
            array = arrays[i]
            permutation = TensMul._get_indices_permutation(indices, ref_indices)
            arrays[i] = permutedims(array, permutation)
        return ref_indices, sum(arrays, Array.zeros(*array.shape))

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self.expand()]

    @data.setter
    def data(self, data):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            _tensor_data_substitution_dict[self] = data

    @data.deleter
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self]

    def __iter__(self):
        deprecate_data()
        if not self.data:
            raise ValueError("No iteration on abstract tensors")
        return self.data.flatten().__iter__()

    def _eval_rewrite_as_Indexed(self, *args, **kwargs):
        return Add.fromiter(args)

    def _eval_partial_derivative(self, s):
        # Evaluation like Add
        list_addends = []
        for a in self.args:
            if isinstance(a, TensExpr):
                list_addends.append(a._eval_partial_derivative(s))
            # do not call diff if s is no symbol
            elif s._diff_wrt:
                list_addends.append(a._eval_derivative(s))

        return self.func(*list_addends)


class Tensor(TensExpr):
    """
    Base tensor class, i.e. this represents a tensor, the single unit to be
    put into an expression.

    Explanation
    ===========

    This object is usually created from a ``TensorHead``, by attaching indices
    to it. Indices preceded by a minus sign are considered contravariant,
    otherwise covariant.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead
    >>> Lorentz = TensorIndexType("Lorentz", dummy_name="L")
    >>> mu, nu = tensor_indices('mu nu', Lorentz)
    >>> A = TensorHead("A", [Lorentz, Lorentz])
    >>> A(mu, -nu)
    A(mu, -nu)
    >>> A(mu, -mu)
    A(L_0, -L_0)

    It is also possible to use symbols instead of inidices (appropriate indices
    are then generated automatically).

    >>> from sympy import Symbol
    >>> x = Symbol('x')
    >>> A(x, mu)
    A(x, mu)
    >>> A(x, -x)
    A(L_0, -L_0)

    """

    is_commutative = False

    _index_structure = None  # type: _IndexStructure
    args: tuple[TensorHead, Tuple]

    def __new__(cls, tensor_head, indices, *, is_canon_bp=False, **kw_args):
        indices = cls._parse_indices(tensor_head, indices)
        obj = Basic.__new__(cls, tensor_head, Tuple(*indices), **kw_args)
        obj._index_structure = _IndexStructure.from_indices(*indices)
        obj._free = obj._index_structure.free[:]
        obj._dum = obj._index_structure.dum[:]
        obj._ext_rank = obj._index_structure._ext_rank
        obj._coeff = S.One
        obj._nocoeff = obj
        obj._component = tensor_head
        obj._components = [tensor_head]
        if tensor_head.rank != len(indices):
            raise ValueError("wrong number of indices")
        obj.is_canon_bp = is_canon_bp
        obj._index_map = Tensor._build_index_map(indices, obj._index_structure)
        return obj

    @property
    def free(self):
        return self._free

    @property
    def dum(self):
        return self._dum

    @property
    def ext_rank(self):
        return self._ext_rank

    @property
    def coeff(self):
        return self._coeff

    @property
    def nocoeff(self):
        return self._nocoeff

    @property
    def component(self):
        return self._component

    @property
    def components(self):
        return self._components

    @property
    def head(self):
        return self.args[0]

    @property
    def indices(self):
        return self.args[1]

    @property
    def free_indices(self):
        return set(self._index_structure.get_free_indices())

    @property
    def index_types(self):
        return self.head.index_types

    @property
    def rank(self):
        return len(self.free_indices)

    @staticmethod
    def _build_index_map(indices, index_structure):
        index_map = {}
        for idx in indices:
            index_map[idx] = (indices.index(idx),)
        return index_map

    def doit(self, **hints):
        args, indices, free, dum = TensMul._tensMul_contract_indices([self])
        return args[0]

    @staticmethod
    def _parse_indices(tensor_head, indices):
        if not isinstance(indices, (tuple, list, Tuple)):
            raise TypeError("indices should be an array, got %s" % type(indices))
        indices = list(indices)
        for i, index in enumerate(indices):
            if isinstance(index, Symbol):
                indices[i] = TensorIndex(index, tensor_head.index_types[i], True)
            elif isinstance(index, Mul):
                c, e = index.as_coeff_Mul()
                if c == -1 and isinstance(e, Symbol):
                    indices[i] = TensorIndex(e, tensor_head.index_types[i], False)
                else:
                    raise ValueError("index not understood: %s" % index)
            elif not isinstance(index, TensorIndex):
                raise TypeError("wrong type for index: %s is %s" % (index, type(index)))
        return indices

    def _set_new_index_structure(self, im, is_canon_bp=False):
        indices = im.get_indices()
        return self._set_indices(*indices, is_canon_bp=is_canon_bp)

    def _set_indices(self, *indices, is_canon_bp=False, **kw_args):
        if len(indices) != self.ext_rank:
            raise ValueError("indices length mismatch")
        return self.func(self.args[0], indices, is_canon_bp=is_canon_bp).doit()

    def _get_free_indices_set(self):
        return {i[0] for i in self._index_structure.free}

    def _get_dummy_indices_set(self):
        dummy_pos = set(itertools.chain(*self._index_structure.dum))
        return {idx for i, idx in enumerate(self.args[1]) if i in dummy_pos}

    def _get_indices_set(self):
        return set(self.args[1].args)

    @property
    def free_in_args(self):
        return [(ind, pos, 0) for ind, pos in self.free]

    @property
    def dum_in_args(self):
        return [(p1, p2, 0, 0) for p1, p2 in self.dum]

    @property
    def free_args(self):
        return sorted([x[0] for x in self.free])

    def commutes_with(self, other):
        """
        :param other:
        :return:
            0  commute
            1  anticommute
            None  neither commute nor anticommute
        """
        if not isinstance(other, TensExpr):
            return 0
        elif isinstance(other, Tensor):
            return self.component.commutes_with(other.component)
        return NotImplementedError

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns the tensor corresponding to the permutation ``g``.

        For further details, see the method in ``TIDS`` with the same name.
        """
        return perm2tensor(self, g, is_canon_bp)

    def canon_bp(self):
        if self.is_canon_bp:
            return self
        expr = self.expand()
        g, dummies, msym = expr._index_structure.indices_canon_args()
        v = components_canon_args([expr.component])
        can = canonicalize(g, dummies, msym, *v)
        if can == 0:
            return S.Zero
        tensor = self.perm2tensor(can, True)
        return tensor

    def split(self):
        return [self]

    def _expand(self, **kwargs):
        return self

    def sorted_components(self):
        return self

    def get_indices(self) -> list[TensorIndex]:
        """
        Get a list of indices, corresponding to those of the tensor.
        """
        return list(self.args[1])

    def get_free_indices(self) -> list[TensorIndex]:
        """
        Get a list of free indices, corresponding to those of the tensor.
        """
        return self._index_structure.get_free_indices()

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        # TODO: this could be optimized by only swapping the indices
        # instead of visiting the whole expression tree:
        return self.xreplace(repl)

    def as_base_exp(self):
        return self, S.One

    def substitute_indices(self, *index_tuples):
        """
        Return a tensor with free indices substituted according to ``index_tuples``.

        ``index_types`` list of tuples ``(old_index, new_index)``.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> i, j, k, l = tensor_indices('i,j,k,l', Lorentz)
        >>> A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        >>> t = A(i, k)*B(-k, -j); t
        A(i, L_0)*B(-L_0, -j)
        >>> t.substitute_indices((i, k),(-j, l))
        A(k, L_0)*B(-L_0, l)
        """
        indices = []
        for index in self.indices:
            for ind_old, ind_new in index_tuples:
                if (index.name == ind_old.name and index.tensor_index_type ==
                                                   ind_old.tensor_index_type):
                    if index.is_up == ind_old.is_up:
                        indices.append(ind_new)
                    else:
                        indices.append(-ind_new)
                    break
            else:
                indices.append(index)
        return self.head(*indices)

    def _get_symmetrized_forms(self):
        """
        Return a list giving all possible permutations of self that are allowed by its symmetries.
        """
        comp = self.component
        gens = comp.symmetry.generators
        rank = comp.rank

        old_perms = None
        new_perms = {self}
        while new_perms != old_perms:
            old_perms = new_perms.copy()
            for tens in old_perms:
                for gen in gens:
                    inds = tens.get_indices()
                    per = [gen.apply(i) for i in range(0,rank)]
                    sign = (-1)**(gen.apply(rank) - rank)
                    ind_map = dict(zip(inds, [inds[i] for i in per]))
                    new_perms.add( sign * tens._replace_indices(ind_map) )

        return new_perms

    def matches(self, expr, repl_dict=None, old=False):
        expr = sympify(expr)

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        #simple checks
        if self == expr:
            return repl_dict
        if not isinstance(expr, Tensor):
            return None
        if self.head != expr.head:
            return None

        #Now consider all index symmetries of expr, and see if any of them allow a match.
        for new_expr in expr._get_symmetrized_forms():
            m = self._matches(new_expr, repl_dict, old=old)
            if m is not None:
                repl_dict.update(m)
                return repl_dict

        return None

    def _matches(self, expr, repl_dict=None, old=False):
        """
        This does not account for index symmetries of expr
        """
        expr = sympify(expr)

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        #simple checks
        if self == expr:
            return repl_dict
        if not isinstance(expr, Tensor):
            return None
        if self.head != expr.head:
            return None

        s_indices = self.get_indices()
        e_indices = expr.get_indices()

        if len(s_indices) != len(e_indices):
            return None

        for i in range(len(s_indices)):
            s_ind = s_indices[i]
            m = s_ind.matches(e_indices[i])
            if m is None:
                return None
            elif -s_ind in repl_dict.keys() and -repl_dict[-s_ind] != m[s_ind]:
                return None
            else:
                repl_dict.update(m)

        return repl_dict

    def __call__(self, *indices):
        deprecate_call()
        free_args = self.free_args
        indices = list(indices)
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        if indices == free_args:
            return self
        t = self.substitute_indices(*list(zip(free_args, indices)))

        # object is rebuilt in order to make sure that all contracted indices
        # get recognized as dummies, but only if there are contracted indices.
        if len({i if i.is_up else -i for i in indices}) != len(indices):
            return t.func(*t.args)
        return t

    # TODO: put this into TensExpr?
    def __iter__(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data.__iter__()

    # TODO: put this into TensExpr?
    def __getitem__(self, item):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]

    def _extract_data(self, replacement_dict):
        from .array import Array
        for k, v in replacement_dict.items():
            if isinstance(k, Tensor) and k.args[0] == self.args[0]:
                other = k
                array = v
                break
        else:
            raise ValueError("%s not found in %s" % (self, replacement_dict))

        # TODO: inefficient, this should be done at root level only:
        replacement_dict = {k: Array(v) for k, v in replacement_dict.items()}
        array = Array(array)

        dum1 = self.dum
        dum2 = other.dum

        if len(dum2) > 0:
            for pair in dum2:
                # allow `dum2` if the contained values are also in `dum1`.
                if pair not in dum1:
                    raise NotImplementedError("%s with contractions is not implemented" % other)
            # Remove elements in `dum2` from `dum1`:
            dum1 = [pair for pair in dum1 if pair not in dum2]
        if len(dum1) > 0:
            indices1 = self.get_indices()
            indices2 = other.get_indices()
            repl = {}
            for p1, p2 in dum1:
                repl[indices2[p2]] = -indices2[p1]
                for pos in (p1, p2):
                    if indices1[pos].is_up ^ indices2[pos].is_up:
                        metric = replacement_dict[indices1[pos].tensor_index_type]
                        if indices1[pos].is_up:
                            metric = _TensorDataLazyEvaluator.inverse_matrix(metric)
                        array = self._contract_and_permute_with_metric(metric, array, pos, len(indices2))
            other = other.xreplace(repl).doit()
            array = _TensorDataLazyEvaluator.data_contract_dum([array], dum1, len(indices2))

        free_ind1 = self.get_free_indices()
        free_ind2 = other.get_free_indices()

        return self._match_indices_with_other_tensor(array, free_ind1, free_ind2, replacement_dict)

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return _tensor_data_substitution_dict[self]

    @data.setter
    def data(self, data):
        deprecate_data()
        # TODO: check data compatibility with properties of tensor.
        with ignore_warnings(SymPyDeprecationWarning):
            _tensor_data_substitution_dict[self] = data

    @data.deleter
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self]
            if self.metric in _tensor_data_substitution_dict:
                del _tensor_data_substitution_dict[self.metric]

    def _print(self):
        indices = [str(ind) for ind in self.indices]
        component = self.component
        if component.rank > 0:
            return ('%s(%s)' % (component.name, ', '.join(indices)))
        else:
            return ('%s' % component.name)

    def equals(self, other):
        if other == 0:
            return self.coeff == 0
        other = _sympify(other)
        if not isinstance(other, TensExpr):
            assert not self.components
            return S.One == other

        def _get_compar_comp(self):
            t = self.canon_bp()
            r = (t.coeff, tuple(t.components), \
                    tuple(sorted(t.free)), tuple(sorted(t.dum)))
            return r

        return _get_compar_comp(self) == _get_compar_comp(other)

    def contract_metric(self, g):
        # if metric is not the same, ignore this step:
        if self.component != g:
            return self
        # in case there are free components, do not perform anything:
        if len(self.free) != 0:
            return self

        #antisym = g.index_types[0].metric_antisym
        if g.symmetry == TensorSymmetry.fully_symmetric(-2):
            antisym = 1
        elif g.symmetry == TensorSymmetry.fully_symmetric(2):
            antisym = 0
        elif g.symmetry == TensorSymmetry.no_symmetry(2):
            antisym = None
        else:
            raise NotImplementedError
        sign = S.One
        typ = g.index_types[0]

        if not antisym:
            # g(i, -i)
            sign = sign*typ.dim
        else:
            # g(i, -i)
            sign = sign*typ.dim

            dp0, dp1 = self.dum[0]
            if dp0 < dp1:
                # g(i, -i) = -D with antisymmetric metric
                sign = -sign

        return sign

    def contract_delta(self, metric):
        return self.contract_metric(metric)

    def _eval_rewrite_as_Indexed(self, tens, indices, **kwargs):
        from sympy.tensor.indexed import Indexed
        # TODO: replace .args[0] with .name:
        index_symbols = [i.args[0] for i in self.get_indices()]
        expr = Indexed(tens.args[0], *index_symbols)
        return self._check_add_Sum(expr, index_symbols)

    def _eval_partial_derivative(self, s):  # type: (Tensor) -> Expr

        if not isinstance(s, Tensor):
            return S.Zero
        else:

            # @a_i/@a_k = delta_i^k
            # @a_i/@a^k = g_ij delta^j_k
            # @a^i/@a^k = delta^i_k
            # @a^i/@a_k = g^ij delta_j^k
            # TODO: if there is no metric present, the derivative should be zero?

            if self.head != s.head:
                return S.Zero

            # if heads are the same, provide delta and/or metric products
            # for every free index pair in the appropriate tensor
            # assumed that the free indices are in proper order
            # A contravariante index in the derivative becomes covariant
            # after performing the derivative and vice versa

            kronecker_delta_list = [1]

            # not guarantee a correct index order

            for (count, (iself, iother)) in enumerate(zip(self.get_free_indices(), s.get_free_indices())):
                if iself.tensor_index_type != iother.tensor_index_type:
                    raise ValueError("index types not compatible")
                else:
                    tensor_index_type = iself.tensor_index_type
                    tensor_metric = tensor_index_type.metric
                    dummy = TensorIndex("d_" + str(count), tensor_index_type,
                                        is_up=iself.is_up)
                    if iself.is_up == iother.is_up:
                        kroneckerdelta = tensor_index_type.delta(iself, -iother)
                    else:
                        kroneckerdelta = (
                            TensMul(tensor_metric(iself, dummy),
                                    tensor_index_type.delta(-dummy, -iother))
                        )
                    kronecker_delta_list.append(kroneckerdelta)
            return TensMul.fromiter(kronecker_delta_list).doit()
            # doit necessary to rename dummy indices accordingly


class TensMul(TensExpr, AssocOp):
    """
    Product of tensors.

    Parameters
    ==========

    coeff : SymPy coefficient of the tensor
    args

    Attributes
    ==========

    ``components`` : list of ``TensorHead`` of the component tensors
    ``types`` : list of nonrepeated ``TensorIndexType``
    ``free`` : list of ``(ind, ipos, icomp)``, see Notes
    ``dum`` : list of ``(ipos1, ipos2, icomp1, icomp2)``, see Notes
    ``ext_rank`` : rank of the tensor counting the dummy indices
    ``rank`` : rank of the tensor
    ``coeff`` : SymPy coefficient of the tensor
    ``free_args`` : list of the free indices in sorted order
    ``is_canon_bp`` : ``True`` if the tensor in in canonical form

    Notes
    =====

    ``args[0]``   list of ``TensorHead`` of the component tensors.

    ``args[1]``   list of ``(ind, ipos, icomp)``
    where ``ind`` is a free index, ``ipos`` is the slot position
    of ``ind`` in the ``icomp``-th component tensor.

    ``args[2]`` list of tuples representing dummy indices.
    ``(ipos1, ipos2, icomp1, icomp2)`` indicates that the contravariant
    dummy index is the ``ipos1``-th slot position in the ``icomp1``-th
    component tensor; the corresponding covariant index is
    in the ``ipos2`` slot position in the ``icomp2``-th component tensor.

    """
    identity = S.One

    _index_structure = None  # type: _IndexStructure

    def __new__(cls, *args, **kw_args):
        is_canon_bp = kw_args.get('is_canon_bp', False)
        args = list(map(_sympify, args))

        """
        If the internal dummy indices in one arg conflict with the free indices
        of the remaining args, we need to rename those internal dummy indices.
        """
        free = [get_free_indices(arg) for arg in args]
        free = set(itertools.chain(*free)) #flatten free
        newargs = []
        for arg in args:
            dum_this = set(get_dummy_indices(arg))
            dum_other = [get_dummy_indices(a) for a in newargs]
            dum_other = set(itertools.chain(*dum_other)) #flatten dum_other
            free_this = set(get_free_indices(arg))
            if len(dum_this.intersection(free)) > 0:
                exclude = free_this.union(free, dum_other)
                newarg = TensMul._dedupe_indices(arg, exclude)
            else:
                newarg = arg
            newargs.append(newarg)

        args = newargs

        # Flatten:
        args = [i for arg in args for i in (arg.args if isinstance(arg, (TensMul, Mul)) else [arg])]

        args, indices, free, dum = TensMul._tensMul_contract_indices(args, replace_indices=False)

        # Data for indices:
        index_types = [i.tensor_index_type for i in indices]
        index_structure = _IndexStructure(free, dum, index_types, indices, canon_bp=is_canon_bp)

        obj = TensExpr.__new__(cls, *args)
        obj._indices = indices
        obj._index_types = index_types[:]
        obj._index_structure = index_structure
        obj._free = index_structure.free[:]
        obj._dum = index_structure.dum[:]
        obj._free_indices = {x[0] for x in obj.free}
        obj._rank = len(obj.free)
        obj._ext_rank = len(obj._index_structure.free) + 2*len(obj._index_structure.dum)
        obj._coeff = S.One
        obj._is_canon_bp = is_canon_bp
        return obj

    index_types = property(lambda self: self._index_types)
    free = property(lambda self: self._free)
    dum = property(lambda self: self._dum)
    free_indices = property(lambda self: self._free_indices)
    rank = property(lambda self: self._rank)
    ext_rank = property(lambda self: self._ext_rank)

    @staticmethod
    def _indices_to_free_dum(args_indices):
        free2pos1 = {}
        free2pos2 = {}
        dummy_data = []
        indices = []

        # Notation for positions (to better understand the code):
        # `pos1`: position in the `args`.
        # `pos2`: position in the indices.

        # Example:
        # A(i, j)*B(k, m, n)*C(p)
        # `pos1` of `n` is 1 because it's in `B` (second `args` of TensMul).
        # `pos2` of `n` is 4 because it's the fifth overall index.

        # Counter for the index position wrt the whole expression:
        pos2 = 0

        for pos1, arg_indices in enumerate(args_indices):

            for index in arg_indices:
                if not isinstance(index, TensorIndex):
                    raise TypeError("expected TensorIndex")
                if -index in free2pos1:
                    # Dummy index detected:
                    other_pos1 = free2pos1.pop(-index)
                    other_pos2 = free2pos2.pop(-index)
                    if index.is_up:
                        dummy_data.append((index, pos1, other_pos1, pos2, other_pos2))
                    else:
                        dummy_data.append((-index, other_pos1, pos1, other_pos2, pos2))
                    indices.append(index)
                elif index in free2pos1:
                    raise ValueError("Repeated index: %s" % index)
                else:
                    free2pos1[index] = pos1
                    free2pos2[index] = pos2
                    indices.append(index)
                pos2 += 1

        free = list(free2pos2.items())
        free_names = [i.name for i in free2pos2.keys()]

        dummy_data.sort(key=lambda x: x[3])
        return indices, free, free_names, dummy_data

    @staticmethod
    def _dummy_data_to_dum(dummy_data):
        return [(p2a, p2b) for (i, p1a, p1b, p2a, p2b) in dummy_data]

    @staticmethod
    def _tensMul_contract_indices(args, replace_indices=True):
        replacements = [{} for _ in args]

        #_index_order = all(_has_index_order(arg) for arg in args)

        args_indices = [get_indices(arg) for arg in args]
        indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)

        cdt = defaultdict(int)

        def dummy_name_gen(tensor_index_type):
            nd = str(cdt[tensor_index_type])
            cdt[tensor_index_type] += 1
            return tensor_index_type.dummy_name + '_' + nd

        if replace_indices:
            for old_index, pos1cov, pos1contra, pos2cov, pos2contra in dummy_data:
                index_type = old_index.tensor_index_type
                while True:
                    dummy_name = dummy_name_gen(index_type)
                    if dummy_name not in free_names:
                        break
                dummy = TensorIndex(dummy_name, index_type, True)
                replacements[pos1cov][old_index] = dummy
                replacements[pos1contra][-old_index] = -dummy
                indices[pos2cov] = dummy
                indices[pos2contra] = -dummy
            args = [
                arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg
                for arg, repl in zip(args, replacements)]

        dum = TensMul._dummy_data_to_dum(dummy_data)
        return args, indices, free, dum

    @staticmethod
    def _get_components_from_args(args):
        """
        Get a list of ``Tensor`` objects having the same ``TIDS`` if multiplied
        by one another.
        """
        components = []
        for arg in args:
            if not isinstance(arg, TensExpr):
                continue
            if isinstance(arg, TensAdd):
                continue
            components.extend(arg.components)
        return components

    @staticmethod
    def _rebuild_tensors_list(args, index_structure):
        indices = index_structure.get_indices()
        #tensors = [None for i in components]  # pre-allocate list
        ind_pos = 0
        for i, arg in enumerate(args):
            if not isinstance(arg, TensExpr):
                continue
            prev_pos = ind_pos
            ind_pos += arg.ext_rank
            args[i] = Tensor(arg.component, indices[prev_pos:ind_pos])

    def doit(self, **hints):
        is_canon_bp = self._is_canon_bp
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]

            """
            There may now be conflicts between dummy indices of different args
            (each arg's doit method does not have any information about which
            dummy indices are already used in the other args), so we
            deduplicate them.
            """
            rule = dict(zip(self.args, args))
            rule = self._dedupe_indices_in_rule(rule)
            args = [rule[a] for a in self.args]

        else:
            args = self.args

        args = [arg for arg in args if arg != self.identity]

        # Extract non-tensor coefficients:
        coeff = reduce(lambda a, b: a*b, [arg for arg in args if not isinstance(arg, TensExpr)], S.One)
        args = [arg for arg in args if isinstance(arg, TensExpr)]

        if len(args) == 0:
            return coeff

        if coeff != self.identity:
            args = [coeff] + args
        if coeff == 0:
            return S.Zero

        if len(args) == 1:
            return args[0]

        args, indices, free, dum = TensMul._tensMul_contract_indices(args)

        # Data for indices:
        index_types = [i.tensor_index_type for i in indices]
        index_structure = _IndexStructure(free, dum, index_types, indices, canon_bp=is_canon_bp)

        obj = self.func(*args)
        obj._index_types = index_types
        obj._index_structure = index_structure
        obj._ext_rank = len(obj._index_structure.free) + 2*len(obj._index_structure.dum)
        obj._coeff = coeff
        obj._is_canon_bp = is_canon_bp
        return obj

    # TODO: this method should be private
    # TODO: should this method be renamed _from_components_free_dum ?
    @staticmethod
    def from_data(coeff, components, free, dum, **kw_args):
        return TensMul(coeff, *TensMul._get_tensors_from_components_free_dum(components, free, dum), **kw_args).doit()

    @staticmethod
    def _get_tensors_from_components_free_dum(components, free, dum):
        """
        Get a list of ``Tensor`` objects by distributing ``free`` and ``dum`` indices on the ``components``.
        """
        index_structure = _IndexStructure.from_components_free_dum(components, free, dum)
        indices = index_structure.get_indices()
        tensors = [None for i in components]  # pre-allocate list

        # distribute indices on components to build a list of tensors:
        ind_pos = 0
        for i, component in enumerate(components):
            prev_pos = ind_pos
            ind_pos += component.rank
            tensors[i] = Tensor(component, indices[prev_pos:ind_pos])
        return tensors

    def _get_free_indices_set(self):
        return {i[0] for i in self.free}

    def _get_dummy_indices_set(self):
        dummy_pos = set(itertools.chain(*self.dum))
        return {idx for i, idx in enumerate(self._index_structure.get_indices()) if i in dummy_pos}

    def _get_position_offset_for_indices(self):
        arg_offset = [None for i in range(self.ext_rank)]
        counter = 0
        for arg in self.args:
            if not isinstance(arg, TensExpr):
                continue
            for j in range(arg.ext_rank):
                arg_offset[j + counter] = counter
            counter += arg.ext_rank
        return arg_offset

    @property
    def free_args(self):
        return sorted([x[0] for x in self.free])

    @property
    def components(self):
        return self._get_components_from_args(self.args)

    @property
    def free_in_args(self):
        arg_offset = self._get_position_offset_for_indices()
        argpos = self._get_indices_to_args_pos()
        return [(ind, pos-arg_offset[pos], argpos[pos]) for (ind, pos) in self.free]

    @property
    def coeff(self):
        # return Mul.fromiter([c for c in self.args if not isinstance(c, TensExpr)])
        return self._coeff

    @property
    def nocoeff(self):
        return self.func(*[t for t in self.args if isinstance(t, TensExpr)]).doit()

    @property
    def dum_in_args(self):
        arg_offset = self._get_position_offset_for_indices()
        argpos = self._get_indices_to_args_pos()
        return [(p1-arg_offset[p1], p2-arg_offset[p2], argpos[p1], argpos[p2]) for p1, p2 in self.dum]

    def equals(self, other):
        if other == 0:
            return self.coeff == 0
        other = _sympify(other)
        if not isinstance(other, TensExpr):
            assert not self.components
            return self.coeff == other

        return self.canon_bp() == other.canon_bp()

    def get_indices(self):
        """
        Returns the list of indices of the tensor.

        Explanation
        ===========

        The indices are listed in the order in which they appear in the
        component tensors.
        The dummy indices are given a name which does not collide with
        the names of the free indices.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m1)*g(m0,m2)
        >>> t.get_indices()
        [m1, m0, m2]
        >>> t2 = p(m1)*g(-m1, m2)
        >>> t2.get_indices()
        [L_0, -L_0, m2]
        """
        return self._indices

    def get_free_indices(self) -> list[TensorIndex]:
        """
        Returns the list of free indices of the tensor.

        Explanation
        ===========

        The indices are listed in the order in which they appear in the
        component tensors.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m1)*g(m0,m2)
        >>> t.get_free_indices()
        [m1, m0, m2]
        >>> t2 = p(m1)*g(-m1, m2)
        >>> t2.get_free_indices()
        [m2]
        """
        return self._index_structure.get_free_indices()

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        return self.func(*[arg._replace_indices(repl) if isinstance(arg, TensExpr) else arg for arg in self.args])

    def split(self):
        """
        Returns a list of tensors, whose product is ``self``.

        Explanation
        ===========

        Dummy indices contracted among different tensor components
        become free indices with the same name as the one used to
        represent the dummy indices.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> a, b, c, d = tensor_indices('a,b,c,d', Lorentz)
        >>> A, B = tensor_heads('A,B', [Lorentz]*2, TensorSymmetry.fully_symmetric(2))
        >>> t = A(a,b)*B(-b,c)
        >>> t
        A(a, L_0)*B(-L_0, c)
        >>> t.split()
        [A(a, L_0), B(-L_0, c)]
        """
        if self.args == ():
            return [self]
        splitp = []
        res = 1
        for arg in self.args:
            if isinstance(arg, Tensor):
                splitp.append(res*arg)
                res = 1
            else:
                res *= arg
        return splitp

    def _expand(self, **hints):
        # TODO: temporary solution, in the future this should be linked to
        # `Expr.expand`.
        args = [_expand(arg, **hints) for arg in self.args]
        args1 = [arg.args if isinstance(arg, (Add, TensAdd)) else (arg,) for arg in args]
        return TensAdd(*[
            TensMul(*i) for i in itertools.product(*args1)]
        )

    def __neg__(self):
        return TensMul(S.NegativeOne, self, is_canon_bp=self._is_canon_bp).doit()

    def __getitem__(self, item):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            return self.data[item]

    def _get_args_for_traditional_printer(self):
        args = list(self.args)
        if self.coeff.could_extract_minus_sign():
            # expressions like "-A(a)"
            sign = "-"
            if args[0] == S.NegativeOne:
                args = args[1:]
            else:
                args[0] = -args[0]
        else:
            sign = ""
        return sign, args

    def _sort_args_for_sorted_components(self):
        """
        Returns the ``args`` sorted according to the components commutation
        properties.

        Explanation
        ===========

        The sorting is done taking into account the commutation group
        of the component tensors.
        """
        cv = [arg for arg in self.args if isinstance(arg, TensExpr)]
        sign = 1
        n = len(cv) - 1
        for i in range(n):
            for j in range(n, i, -1):
                c = cv[j-1].commutes_with(cv[j])
                # if `c` is `None`, it does neither commute nor anticommute, skip:
                if c not in (0, 1):
                    continue
                typ1 = sorted(set(cv[j-1].component.index_types), key=lambda x: x.name)
                typ2 = sorted(set(cv[j].component.index_types), key=lambda x: x.name)
                if (typ1, cv[j-1].component.name) > (typ2, cv[j].component.name):
                    cv[j-1], cv[j] = cv[j], cv[j-1]
                    # if `c` is 1, the anticommute, so change sign:
                    if c:
                        sign = -sign

        coeff = sign * self.coeff
        if coeff != 1:
            return [coeff] + cv
        return cv

    def sorted_components(self):
        """
        Returns a tensor product with sorted components.
        """
        return TensMul(*self._sort_args_for_sorted_components()).doit()

    def perm2tensor(self, g, is_canon_bp=False):
        """
        Returns the tensor corresponding to the permutation ``g``

        For further details, see the method in ``TIDS`` with the same name.
        """
        return perm2tensor(self, g, is_canon_bp=is_canon_bp)

    def canon_bp(self):
        """
        Canonicalize using the Butler-Portugal algorithm for canonicalization
        under monoterm symmetries.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, TensorSymmetry
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> A = TensorHead('A', [Lorentz]*2, TensorSymmetry.fully_symmetric(-2))
        >>> t = A(m0,-m1)*A(m1,-m0)
        >>> t.canon_bp()
        -A(L_0, L_1)*A(-L_0, -L_1)
        >>> t = A(m0,-m1)*A(m1,-m2)*A(m2,-m0)
        >>> t.canon_bp()
        0
        """
        if self._is_canon_bp:
            return self
        expr = self.expand()
        if isinstance(expr, TensAdd):
            return expr.canon_bp()
        if not expr.components:
            return expr
        t = expr.sorted_components()
        g, dummies, msym = t._index_structure.indices_canon_args()
        v = components_canon_args(t.components)
        can = canonicalize(g, dummies, msym, *v)
        if can == 0:
            return S.Zero
        tmul = t.perm2tensor(can, True)
        return tmul

    def contract_delta(self, delta):
        t = self.contract_metric(delta)
        return t

    def _get_indices_to_args_pos(self):
        """
        Get a dict mapping the index position to TensMul's argument number.
        """
        pos_map = {}
        pos_counter = 0
        for arg_i, arg in enumerate(self.args):
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)
            for i in range(arg.ext_rank):
                pos_map[pos_counter] = arg_i
                pos_counter += 1
        return pos_map

    def contract_metric(self, g):
        """
        Raise or lower indices with the metric ``g``.

        Parameters
        ==========

        g : metric

        Notes
        =====

        See the ``TensorIndexType`` docstring for the contraction conventions.

        Examples
        ========

        >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, tensor_heads
        >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
        >>> m0, m1, m2 = tensor_indices('m0,m1,m2', Lorentz)
        >>> g = Lorentz.metric
        >>> p, q = tensor_heads('p,q', [Lorentz])
        >>> t = p(m0)*q(m1)*g(-m0, -m1)
        >>> t.canon_bp()
        metric(L_0, L_1)*p(-L_0)*q(-L_1)
        >>> t.contract_metric(g).canon_bp()
        p(L_0)*q(-L_0)
        """
        expr = self.expand()
        if self != expr:
            expr = canon_bp(expr)
            return contract_metric(expr, g)
        pos_map = self._get_indices_to_args_pos()
        args = list(self.args)

        #antisym = g.index_types[0].metric_antisym
        if g.symmetry == TensorSymmetry.fully_symmetric(-2):
            antisym = 1
        elif g.symmetry == TensorSymmetry.fully_symmetric(2):
            antisym = 0
        elif g.symmetry == TensorSymmetry.no_symmetry(2):
            antisym = None
        else:
            raise NotImplementedError

        # list of positions of the metric ``g`` inside ``args``
        gpos = [i for i, x in enumerate(self.args) if isinstance(x, Tensor) and x.component == g]
        if not gpos:
            return self

        # Sign is either 1 or -1, to correct the sign after metric contraction
        # (for spinor indices).
        sign = 1
        dum = self.dum[:]
        free = self.free[:]
        elim = set()
        for gposx in gpos:
            if gposx in elim:
                continue
            free1 = [x for x in free if pos_map[x[1]] == gposx]
            dum1 = [x for x in dum if pos_map[x[0]] == gposx or pos_map[x[1]] == gposx]
            if not dum1:
                continue
            elim.add(gposx)
            # subs with the multiplication neutral element, that is, remove it:
            args[gposx] = 1
            if len(dum1) == 2:
                if not antisym:
                    dum10, dum11 = dum1
                    if pos_map[dum10[1]] == gposx:
                        # the index with pos p0 contravariant
                        p0 = dum10[0]
                    else:
                        # the index with pos p0 is covariant
                        p0 = dum10[1]
                    if pos_map[dum11[1]] == gposx:
                        # the index with pos p1 is contravariant
                        p1 = dum11[0]
                    else:
                        # the index with pos p1 is covariant
                        p1 = dum11[1]

                    dum.append((p0, p1))
                else:
                    dum10, dum11 = dum1
                    # change the sign to bring the indices of the metric to contravariant
                    # form; change the sign if dum10 has the metric index in position 0
                    if pos_map[dum10[1]] == gposx:
                        # the index with pos p0 is contravariant
                        p0 = dum10[0]
                        if dum10[1] == 1:
                            sign = -sign
                    else:
                        # the index with pos p0 is covariant
                        p0 = dum10[1]
                        if dum10[0] == 0:
                            sign = -sign
                    if pos_map[dum11[1]] == gposx:
                        # the index with pos p1 is contravariant
                        p1 = dum11[0]
                        sign = -sign
                    else:
                        # the index with pos p1 is covariant
                        p1 = dum11[1]

                    dum.append((p0, p1))

            elif len(dum1) == 1:
                if not antisym:
                    dp0, dp1 = dum1[0]
                    if pos_map[dp0] == pos_map[dp1]:
                        # g(i, -i)
                        typ = g.index_types[0]
                        sign = sign*typ.dim

                    else:
                        # g(i0, i1)*p(-i1)
                        if pos_map[dp0] == gposx:
                            p1 = dp1
                        else:
                            p1 = dp0

                        ind, p = free1[0]
                        free.append((ind, p1))
                else:
                    dp0, dp1 = dum1[0]
                    if pos_map[dp0] == pos_map[dp1]:
                        # g(i, -i)
                        typ = g.index_types[0]
                        sign = sign*typ.dim

                        if dp0 < dp1:
                            # g(i, -i) = -D with antisymmetric metric
                            sign = -sign
                    else:
                        # g(i0, i1)*p(-i1)
                        if pos_map[dp0] == gposx:
                            p1 = dp1
                            if dp0 == 0:
                                sign = -sign
                        else:
                            p1 = dp0
                        ind, p = free1[0]
                        free.append((ind, p1))
            dum = [x for x in dum if x not in dum1]
            free = [x for x in free if x not in free1]

        # shift positions:
        shift = 0
        shifts = [0]*len(args)
        for i in range(len(args)):
            if i in elim:
                shift += 2
                continue
            shifts[i] = shift
        free = [(ind, p - shifts[pos_map[p]]) for (ind, p) in free if pos_map[p] not in elim]
        dum = [(p0 - shifts[pos_map[p0]], p1 - shifts[pos_map[p1]]) for p0, p1 in dum if pos_map[p0] not in elim and pos_map[p1] not in elim]

        res = sign*TensMul(*args).doit()
        if not isinstance(res, TensExpr):
            return res
        im = _IndexStructure.from_components_free_dum(res.components, free, dum)
        return res._set_new_index_structure(im)

    def _set_new_index_structure(self, im, is_canon_bp=False):
        indices = im.get_indices()
        return self._set_indices(*indices, is_canon_bp=is_canon_bp)

    def _set_indices(self, *indices, is_canon_bp=False, **kw_args):
        if len(indices) != self.ext_rank:
            raise ValueError("indices length mismatch")
        args = list(self.args)[:]
        pos = 0
        for i, arg in enumerate(args):
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)
            ext_rank = arg.ext_rank
            args[i] = arg._set_indices(*indices[pos:pos+ext_rank])
            pos += ext_rank
        return TensMul(*args, is_canon_bp=is_canon_bp).doit()

    @staticmethod
    def _index_replacement_for_contract_metric(args, free, dum):
        for arg in args:
            if not isinstance(arg, TensExpr):
                continue
            assert isinstance(arg, Tensor)

    def substitute_indices(self, *index_tuples):
        new_args = []
        for arg in self.args:
            if isinstance(arg, TensExpr):
                arg = arg.substitute_indices(*index_tuples)
            new_args.append(arg)
        return TensMul(*new_args).doit()

    def __call__(self, *indices):
        deprecate_call()
        free_args = self.free_args
        indices = list(indices)
        if [x.tensor_index_type for x in indices] != [x.tensor_index_type for x in free_args]:
            raise ValueError('incompatible types')
        if indices == free_args:
            return self
        t = self.substitute_indices(*list(zip(free_args, indices)))

        # object is rebuilt in order to make sure that all contracted indices
        # get recognized as dummies, but only if there are contracted indices.
        if len({i if i.is_up else -i for i in indices}) != len(indices):
            return t.func(*t.args)
        return t

    def _extract_data(self, replacement_dict):
        args_indices, arrays = zip(*[arg._extract_data(replacement_dict) for arg in self.args if isinstance(arg, TensExpr)])
        coeff = reduce(operator.mul, [a for a in self.args if not isinstance(a, TensExpr)], S.One)
        indices, free, free_names, dummy_data = TensMul._indices_to_free_dum(args_indices)
        dum = TensMul._dummy_data_to_dum(dummy_data)
        ext_rank = self.ext_rank
        free.sort(key=lambda x: x[1])
        free_indices = [i[0] for i in free]
        return free_indices, coeff*_TensorDataLazyEvaluator.data_contract_dum(arrays, dum, ext_rank)

    @property
    def data(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            dat = _tensor_data_substitution_dict[self.expand()]
        return dat

    @data.setter
    def data(self, data):
        deprecate_data()
        raise ValueError("Not possible to set component data to a tensor expression")

    @data.deleter
    def data(self):
        deprecate_data()
        raise ValueError("Not possible to delete component data to a tensor expression")

    def __iter__(self):
        deprecate_data()
        with ignore_warnings(SymPyDeprecationWarning):
            if self.data is None:
                raise ValueError("No iteration on abstract tensors")
            return self.data.__iter__()

    @staticmethod
    def _dedupe_indices(new, exclude):
        """
        exclude: set
        new: TensExpr

        If ``new`` has any dummy indices that are in ``exclude``, return a version
        of new with those indices replaced. If no replacements are needed,
        return None

        """
        exclude = set(exclude)
        dums_new = set(get_dummy_indices(new))
        free_new = set(get_free_indices(new))

        conflicts = dums_new.intersection(exclude)
        if len(conflicts) == 0:
            return None

        """
        ``exclude_for_gen`` is to be passed to ``_IndexStructure._get_generator_for_dummy_indices()``.
        Since the latter does not use the index position for anything, we just
        set it as ``None`` here.
        """
        exclude.update(dums_new)
        exclude.update(free_new)
        exclude_for_gen = [(i, None) for i in exclude]
        gen = _IndexStructure._get_generator_for_dummy_indices(exclude_for_gen)
        repl = {}
        for d in conflicts:
            if -d in repl.keys():
                continue
            newname = gen(d.tensor_index_type)
            new_d = d.func(newname, *d.args[1:])
            repl[d] = new_d
            repl[-d] = -new_d

        if len(repl) == 0:
            return None

        new_renamed = new._replace_indices(repl)
        return new_renamed

    def _dedupe_indices_in_rule(self, rule):
        """
        rule: dict

        This applies TensMul._dedupe_indices on all values of rule.

        """
        index_rules = {k:v for k,v in rule.items() if isinstance(k, TensorIndex)}
        other_rules = {k:v for k,v in rule.items() if k not in index_rules.keys()}
        exclude = set(self.get_indices())

        newrule = {}
        newrule.update(index_rules)
        exclude.update(index_rules.keys())
        exclude.update(index_rules.values())
        for old, new in other_rules.items():
            new_renamed = TensMul._dedupe_indices(new, exclude)
            if old == new or new_renamed is None:
                newrule[old] = new
            else:
                newrule[old] = new_renamed
                exclude.update(get_indices(new_renamed))
        return newrule

    def _eval_rewrite_as_Indexed(self, *args, **kwargs):
        from sympy.concrete.summations import Sum
        index_symbols = [i.args[0] for i in self.get_indices()]
        args = [arg.args[0] if isinstance(arg, Sum) else arg for arg in args]
        expr = Mul.fromiter(args)
        return self._check_add_Sum(expr, index_symbols)

    def _eval_partial_derivative(self, s):
        # Evaluation like Mul
        terms = []
        for i, arg in enumerate(self.args):
            # checking whether some tensor instance is differentiated
            # or some other thing is necessary, but ugly
            if isinstance(arg, TensExpr):
                d = arg._eval_partial_derivative(s)
            else:
                # do not call diff is s is no symbol
                if s._diff_wrt:
                    d = arg._eval_derivative(s)
                else:
                    d = S.Zero
            if d:
                terms.append(TensMul.fromiter(self.args[:i] + (d,) + self.args[i + 1:]))
        return TensAdd.fromiter(terms)


class TensorElement(TensExpr):
    """
    Tensor with evaluated components.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, TensorHead, TensorSymmetry
    >>> from sympy import symbols
    >>> L = TensorIndexType("L")
    >>> i, j, k = symbols("i j k")
    >>> A = TensorHead("A", [L, L], TensorSymmetry.fully_symmetric(2))
    >>> A(i, j).get_free_indices()
    [i, j]

    If we want to set component ``i`` to a specific value, use the
    ``TensorElement`` class:

    >>> from sympy.tensor.tensor import TensorElement
    >>> te = TensorElement(A(i, j), {i: 2})

    As index ``i`` has been accessed (``{i: 2}`` is the evaluation of its 3rd
    element), the free indices will only contain ``j``:

    >>> te.get_free_indices()
    [j]
    """

    def __new__(cls, expr, index_map):
        if not isinstance(expr, Tensor):
            # remap
            if not isinstance(expr, TensExpr):
                raise TypeError("%s is not a tensor expression" % expr)
            return expr.func(*[TensorElement(arg, index_map) for arg in expr.args])
        expr_free_indices = expr.get_free_indices()
        name_translation = {i.args[0]: i for i in expr_free_indices}
        index_map = {name_translation.get(index, index): value for index, value in index_map.items()}
        index_map = {index: value for index, value in index_map.items() if index in expr_free_indices}
        if len(index_map) == 0:
            return expr
        free_indices = [i for i in expr_free_indices if i not in index_map.keys()]
        index_map = Dict(index_map)
        obj = TensExpr.__new__(cls, expr, index_map)
        obj._free_indices = free_indices
        return obj

    @property
    def free(self):
        return [(index, i) for i, index in enumerate(self.get_free_indices())]

    @property
    def dum(self):
        # TODO: inherit dummies from expr
        return []

    @property
    def expr(self):
        return self._args[0]

    @property
    def index_map(self):
        return self._args[1]

    @property
    def coeff(self):
        return S.One

    @property
    def nocoeff(self):
        return self

    def get_free_indices(self):
        return self._free_indices

    def _replace_indices(self, repl: dict[TensorIndex, TensorIndex]) -> TensExpr:
        # TODO: can be improved:
        return self.xreplace(repl)

    def get_indices(self):
        return self.get_free_indices()

    def _extract_data(self, replacement_dict):
        ret_indices, array = self.expr._extract_data(replacement_dict)
        index_map = self.index_map
        slice_tuple = tuple(index_map.get(i, slice(None)) for i in ret_indices)
        ret_indices = [i for i in ret_indices if i not in index_map]
        array = array.__getitem__(slice_tuple)
        return ret_indices, array


class WildTensorHead(TensorHead):
    """
    A wild object that is used to create ``WildTensor`` instances

    Explanation
    ===========

    Examples
    ========
    >>> from sympy.tensor.tensor import TensorHead, TensorIndex, WildTensorHead, TensorIndexType
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)

    A WildTensorHead can be created without specifying a ``TensorIndexType``

    >>> W = WildTensorHead("W")

    Calling it with a ``TensorIndex`` creates a ``WildTensor`` instance.

    >>> type(W(p))
    <class 'sympy.tensor.tensor.WildTensor'>

    The ``TensorIndexType`` is automatically detected from the index that is passed

    >>> W(p).component
    W(R3)

    Calling it with no indices returns an object that can match tensors with any number of indices.

    >>> K = TensorHead('K', [R3])
    >>> Q = TensorHead('Q', [R3, R3])
    >>> W().matches(K(p))
    {W: K(p)}
    >>> W().matches(Q(p,q))
    {W: Q(p, q)}

    If you want to ignore the order of indices while matching, pass ``unordered_indices=True``.

    >>> U = WildTensorHead("U", unordered_indices=True)
    >>> W(p,q).matches(Q(q,p))
    >>> U(p,q).matches(Q(q,p))
    {U(R3,R3): _WildTensExpr(Q(q, p))}

    Parameters
    ==========
    name : name of the tensor
    unordered_indices : whether the order of the indices matters for matching
        (default: False)

    See also
    ========
    ``WildTensor``
    ``TensorHead``

    """
    def __new__(cls, name, index_types=None, symmetry=None, comm=0,  unordered_indices=False):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        else:
            raise ValueError("invalid name")

        if index_types is None:
            index_types = []

        if symmetry is None:
            symmetry = TensorSymmetry.no_symmetry(len(index_types))
        else:
            assert symmetry.rank == len(index_types)

        if symmetry != TensorSymmetry.no_symmetry(len(index_types)):
            raise NotImplementedError("Wild matching based on symmetry is not implemented.")

        obj = Basic.__new__(cls, name_symbol, Tuple(*index_types), sympify(symmetry), sympify(comm), sympify(unordered_indices))

        return obj

    @property
    def unordered_indices(self):
        return self.args[4]

    def __call__(self, *indices, **kwargs):
        tensor = WildTensor(self, indices, **kwargs)
        return tensor.doit()


class WildTensor(Tensor):
    """
    A wild object which matches ``Tensor`` instances

    Explanation
    ===========
    This is instantiated by attaching indices to a ``WildTensorHead`` instance.

    Examples
    ========
    >>> from sympy.tensor.tensor import TensorHead, TensorIndex, WildTensorHead, TensorIndexType
    >>> W = WildTensorHead("W")
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)
    >>> K = TensorHead('K', [R3])
    >>> Q = TensorHead('Q', [R3, R3])

    Matching also takes the indices into account
    >>> W(p).matches(K(p))
    {W(R3): _WildTensExpr(K(p))}
    >>> W(p).matches(K(q))
    >>> W(p).matches(K(-p))

    If you want to match objects with any number of indices, just use a ``WildTensor`` with no indices.
    >>> W().matches(K(p))
    {W: K(p)}
    >>> W().matches(Q(p,q))
    {W: Q(p, q)}

    See Also
    ========
    ``WildTensorHead``
    ``Tensor``

    """
    def __new__(cls, tensor_head, indices, **kw_args):
        is_canon_bp = kw_args.pop("is_canon_bp", False)

        if tensor_head.func == TensorHead:
            """
            If someone tried to call WildTensor by supplying a TensorHead (not a WildTensorHead), return a normal tensor instead. This is helpful when using subs on an expression to replace occurrences of a WildTensorHead with a TensorHead.
            """
            return Tensor(tensor_head, indices, is_canon_bp=is_canon_bp, **kw_args)
        elif tensor_head.func == _WildTensExpr:
            return tensor_head(*indices)

        indices = cls._parse_indices(tensor_head, indices)
        index_types = [ind.tensor_index_type for ind in indices]
        tensor_head = tensor_head.func(
            tensor_head.name,
            index_types,
            symmetry=None,
            comm=tensor_head.comm,
            unordered_indices=tensor_head.unordered_indices,
            )

        obj = Basic.__new__(cls, tensor_head, Tuple(*indices))
        obj.name = tensor_head.name
        obj._index_structure = _IndexStructure.from_indices(*indices)
        obj._free = obj._index_structure.free[:]
        obj._dum = obj._index_structure.dum[:]
        obj._ext_rank = obj._index_structure._ext_rank
        obj._coeff = S.One
        obj._nocoeff = obj
        obj._component = tensor_head
        obj._components = [tensor_head]
        if tensor_head.rank != len(indices):
            raise ValueError("wrong number of indices")
        obj.is_canon_bp = is_canon_bp
        obj._index_map = obj._build_index_map(indices, obj._index_structure)

        return obj


    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, TensExpr) and expr != S(1):
            return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        if len(self.indices) > 0:
            if not hasattr(expr, "get_free_indices"):
                return None
            expr_indices = expr.get_free_indices()
            if len(expr_indices) != len(self.indices):
                return None
            if self._component.unordered_indices:
                m = self._match_indices_ignoring_order(expr)
                if m is None:
                    return None
                else:
                    repl_dict.update(m)
            else:
                for i in range(len(expr_indices)):
                    m = self.indices[i].matches(expr_indices[i])
                    if m is None:
                        return None
                    else:
                        repl_dict.update(m)

            repl_dict[self.component] = _WildTensExpr(expr)
        else:
            #If no indices were passed to the WildTensor, it may match tensors with any number of indices.
            repl_dict[self] = expr

        return repl_dict

    def _match_indices_ignoring_order(self, expr, repl_dict=None, old=False):
        """
        Helper method for matches. Checks if the indices of self and expr
        match disregarding index ordering.
        """
        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        def siftkey(ind):
            if isinstance(ind, WildTensorIndex):
                if ind.ignore_updown:
                    return "wild, updown"
                else:
                    return "wild"
            else:
                return "nonwild"

        indices_sifted = sift(self.indices, siftkey)

        matched_indices = []
        expr_indices_remaining = expr.get_indices()
        for ind in indices_sifted["nonwild"]:
            matched_this_ind = False
            for e_ind in expr_indices_remaining:
                if e_ind in matched_indices:
                    continue
                m = ind.matches(e_ind)
                if m is not None:
                    matched_this_ind = True
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            if not matched_this_ind:
                return None

        expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]
        for ind in indices_sifted["wild"]:
            matched_this_ind = False
            for e_ind in expr_indices_remaining:
                m = ind.matches(e_ind)
                if m is not None:
                    if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                        return None
                    matched_this_ind = True
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            if not matched_this_ind:
                return None

        expr_indices_remaining = [i for i in expr_indices_remaining if i not in matched_indices]
        for ind in indices_sifted["wild, updown"]:
            matched_this_ind = False
            for e_ind in expr_indices_remaining:
                m = ind.matches(e_ind)
                if m is not None:
                    if -ind in repl_dict.keys() and -repl_dict[-ind] != m[ind]:
                        return None
                    matched_this_ind = True
                    repl_dict.update(m)
                    matched_indices.append(e_ind)
                    break
            if not matched_this_ind:
                return None

        if len(matched_indices) < len(self.indices):
            return None
        else:
            return repl_dict

class WildTensorIndex(TensorIndex):
    """
    A wild object that matches TensorIndex instances.

    Examples
    ========
    >>> from sympy.tensor.tensor import TensorIndex, TensorIndexType, WildTensorIndex
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex("p", R3)

    By default, covariant indices only match with covariant indices (and
    similarly for contravariant)

    >>> q = WildTensorIndex("q", R3)
    >>> (q).matches(p)
    {q: p}
    >>> (q).matches(-p)

    If you want matching to ignore whether the index is co/contra-variant, set
    ignore_updown=True

    >>> r = WildTensorIndex("r", R3, ignore_updown=True)
    >>> (r).matches(-p)
    {r: -p}
    >>> (r).matches(p)
    {r: p}

    Parameters
    ==========
    name : name of the index (string), or ``True`` if you want it to be
        automatically assigned
    tensor_index_type : ``TensorIndexType`` of the index
    is_up :  flag for contravariant index (is_up=True by default)
    ignore_updown : bool, Whether this should match both co- and contra-variant
        indices (default:False)
    """
    def __new__(cls, name, tensor_index_type, is_up=True, ignore_updown=False):
        if isinstance(name, str):
            name_symbol = Symbol(name)
        elif isinstance(name, Symbol):
            name_symbol = name
        elif name is True:
            name = "_i{}".format(len(tensor_index_type._autogenerated))
            name_symbol = Symbol(name)
            tensor_index_type._autogenerated.append(name_symbol)
        else:
            raise ValueError("invalid name")

        is_up = sympify(is_up)
        ignore_updown = sympify(ignore_updown)
        return Basic.__new__(cls, name_symbol, tensor_index_type, is_up, ignore_updown)

    @property
    def ignore_updown(self):
        return self.args[3]

    def __neg__(self):
        t1 = WildTensorIndex(self.name, self.tensor_index_type,
                (not self.is_up), self.ignore_updown)
        return t1

    def matches(self, expr, repl_dict=None, old=False):
        if not isinstance(expr, TensorIndex):
            return None
        if self.tensor_index_type != expr.tensor_index_type:
            return None
        if not self.ignore_updown:
            if self.is_up != expr.is_up:
                return None

        if repl_dict is None:
            repl_dict = {}
        else:
            repl_dict = repl_dict.copy()

        repl_dict[self] = expr
        return repl_dict


class _WildTensExpr(Basic):
    """
    INTERNAL USE ONLY

    This is an object that helps with replacement of WildTensors in expressions.
    When this object is set as the tensor_head of a WildTensor, it replaces the
    WildTensor by a TensExpr (passed when initializing this object).

    Examples
    ========
    >>> from sympy.tensor.tensor import WildTensorHead, TensorIndex, TensorHead, TensorIndexType
    >>> W = WildTensorHead("W")
    >>> R3 = TensorIndexType('R3', dim=3)
    >>> p = TensorIndex('p', R3)
    >>> q = TensorIndex('q', R3)
    >>> K = TensorHead('K', [R3])
    >>> print( ( K(p) ).replace( W(p), W(q)*W(-q)*W(p) ) )
    K(R_0)*K(-R_0)*K(p)

    """
    def __init__(self, expr):
        if not isinstance(expr, TensExpr):
            raise TypeError("_WildTensExpr expects a TensExpr as argument")
        self.expr = expr

    def __call__(self, *indices):
        return self.expr._replace_indices(dict(zip(self.expr.get_free_indices(), indices)))

    def __neg__(self):
        return self.func(self.expr*S.NegativeOne)

    def __abs__(self):
        raise NotImplementedError

    def __add__(self, other):
        if other.func != self.func:
            raise TypeError(f"Cannot add {self.func} to {other.func}")
        return self.func(self.expr+other.expr)

    def __radd__(self, other):
        if other.func != self.func:
            raise TypeError(f"Cannot add {self.func} to {other.func}")
        return self.func(other.expr+self.expr)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        raise NotImplementedError

    def __rmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rtruediv__(self, other):
        raise NotImplementedError

    def __pow__(self, other):
        raise NotImplementedError

    def __rpow__(self, other):
        raise NotImplementedError


def canon_bp(p):
    """
    Butler-Portugal canonicalization. See ``tensor_can.py`` from the
    combinatorics module for the details.
    """
    if isinstance(p, TensExpr):
        return p.canon_bp()
    return p


def tensor_mul(*a):
    """
    product of tensors
    """
    if not a:
        return TensMul.from_data(S.One, [], [], [])
    t = a[0]
    for tx in a[1:]:
        t = t*tx
    return t


def riemann_cyclic_replace(t_r):
    """
    replace Riemann tensor with an equivalent expression

    ``R(m,n,p,q) -> 2/3*R(m,n,p,q) - 1/3*R(m,q,n,p) + 1/3*R(m,p,n,q)``

    """
    free = sorted(t_r.free, key=lambda x: x[1])
    m, n, p, q = [x[0] for x in free]
    t0 = t_r*Rational(2, 3)
    t1 = -t_r.substitute_indices((m,m),(n,q),(p,n),(q,p))*Rational(1, 3)
    t2 = t_r.substitute_indices((m,m),(n,p),(p,n),(q,q))*Rational(1, 3)
    t3 = t0 + t1 + t2
    return t3

def riemann_cyclic(t2):
    """
    Replace each Riemann tensor with an equivalent expression
    satisfying the cyclic identity.

    This trick is discussed in the reference guide to Cadabra.

    Examples
    ========

    >>> from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorHead, riemann_cyclic, TensorSymmetry
    >>> Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    >>> i, j, k, l = tensor_indices('i,j,k,l', Lorentz)
    >>> R = TensorHead('R', [Lorentz]*4, TensorSymmetry.riemann())
    >>> t = R(i,j,k,l)*(R(-i,-j,-k,-l) - 2*R(-i,-k,-j,-l))
    >>> riemann_cyclic(t)
    0
    """
    t2 = t2.expand()
    if isinstance(t2, (TensMul, Tensor)):
        args = [t2]
    else:
        args = t2.args
    a1 = [x.split() for x in args]
    a2 = [[riemann_cyclic_replace(tx) for tx in y] for y in a1]
    a3 = [tensor_mul(*v) for v in a2]
    t3 = TensAdd(*a3).doit()
    if not t3:
        return t3
    else:
        return canon_bp(t3)


def get_lines(ex, index_type):
    """
    Returns ``(lines, traces, rest)`` for an index type,
    where ``lines`` is the list of list of positions of a matrix line,
    ``traces`` is the list of list of traced matrix lines,
    ``rest`` is the rest of the elements of the tensor.
    """
    def _join_lines(a):
        i = 0
        while i < len(a):
            x = a[i]
            xend = x[-1]
            xstart = x[0]
            hit = True
            while hit:
                hit = False
                for j in range(i + 1, len(a)):
                    if j >= len(a):
                        break
                    if a[j][0] == xend:
                        hit = True
                        x.extend(a[j][1:])
                        xend = x[-1]
                        a.pop(j)
                        continue
                    if a[j][0] == xstart:
                        hit = True
                        a[i] = reversed(a[j][1:]) + x
                        x = a[i]
                        xstart = a[i][0]
                        a.pop(j)
                        continue
                    if a[j][-1] == xend:
                        hit = True
                        x.extend(reversed(a[j][:-1]))
                        xend = x[-1]
                        a.pop(j)
                        continue
                    if a[j][-1] == xstart:
                        hit = True
                        a[i] = a[j][:-1] + x
                        x = a[i]
                        xstart = x[0]
                        a.pop(j)
                        continue
            i += 1
        return a

    arguments = ex.args
    dt = {}
    for c in ex.args:
        if not isinstance(c, TensExpr):
            continue
        if c in dt:
            continue
        index_types = c.index_types
        a = []
        for i in range(len(index_types)):
            if index_types[i] is index_type:
                a.append(i)
        if len(a) > 2:
            raise ValueError('at most two indices of type %s allowed' % index_type)
        if len(a) == 2:
            dt[c] = a
    #dum = ex.dum
    lines = []
    traces = []
    traces1 = []
    #indices_to_args_pos = ex._get_indices_to_args_pos()
    # TODO: add a dum_to_components_map ?
    for p0, p1, c0, c1 in ex.dum_in_args:
        if arguments[c0] not in dt:
            continue
        if c0 == c1:
            traces.append([c0])
            continue
        ta0 = dt[arguments[c0]]
        ta1 = dt[arguments[c1]]
        if p0 not in ta0:
            continue
        if ta0.index(p0) == ta1.index(p1):
            # case gamma(i,s0,-s1) in c0, gamma(j,-s0,s2) in c1;
            # to deal with this case one could add to the position
            # a flag for transposition;
            # one could write [(c0, False), (c1, True)]
            raise NotImplementedError
        # if p0 == ta0[1] then G in pos c0 is mult on the right by G in c1
        # if p0 == ta0[0] then G in pos c1 is mult on the right by G in c0
        ta0 = dt[arguments[c0]]
        b0, b1 = (c0, c1) if p0 == ta0[1]  else (c1, c0)
        lines1 = lines[:]
        for line in lines:
            if line[-1] == b0:
                if line[0] == b1:
                    n = line.index(min(line))
                    traces1.append(line)
                    traces.append(line[n:] + line[:n])
                else:
                    line.append(b1)
                break
            elif line[0] == b1:
                line.insert(0, b0)
                break
        else:
            lines1.append([b0, b1])

        lines = [x for x in lines1 if x not in traces1]
        lines = _join_lines(lines)
    rest = []
    for line in lines:
        for y in line:
            rest.append(y)
    for line in traces:
        for y in line:
            rest.append(y)
    rest = [x for x in range(len(arguments)) if x not in rest]

    return lines, traces, rest


def get_free_indices(t):
    if not isinstance(t, TensExpr):
        return ()
    return t.get_free_indices()


def get_indices(t):
    if not isinstance(t, TensExpr):
        return ()
    return t.get_indices()

def get_dummy_indices(t):
    if not isinstance(t, TensExpr):
        return ()
    inds = t.get_indices()
    free = t.get_free_indices()
    return [i for i in inds if i not in free]

def get_index_structure(t):
    if isinstance(t, TensExpr):
        return t._index_structure
    return _IndexStructure([], [], [], [])


def get_coeff(t):
    if isinstance(t, Tensor):
        return S.One
    if isinstance(t, TensMul):
        return t.coeff
    if isinstance(t, TensExpr):
        raise ValueError("no coefficient associated to this tensor expression")
    return t

def contract_metric(t, g):
    if isinstance(t, TensExpr):
        return t.contract_metric(g)
    return t


def perm2tensor(t, g, is_canon_bp=False):
    """
    Returns the tensor corresponding to the permutation ``g``

    For further details, see the method in ``TIDS`` with the same name.
    """
    if not isinstance(t, TensExpr):
        return t
    elif isinstance(t, (Tensor, TensMul)):
        nim = get_index_structure(t).perm2tensor(g, is_canon_bp=is_canon_bp)
        res = t._set_new_index_structure(nim, is_canon_bp=is_canon_bp)
        if g[-1] != len(g) - 1:
            return -res

        return res
    raise NotImplementedError()


def substitute_indices(t, *index_tuples):
    if not isinstance(t, TensExpr):
        return t
    return t.substitute_indices(*index_tuples)


def _expand(expr, **kwargs):
    if isinstance(expr, TensExpr):
        return expr._expand(**kwargs)
    else:
        return expr.expand(**kwargs)


def get_postprocessor(cls):
    def _postprocessor(expr):
        tens_class = {Mul: TensMul, Add: TensAdd}[cls]
        if any(isinstance(a, TensExpr) for a in expr.args):
            return tens_class(*expr.args)
        else:
            return expr

    return _postprocessor

Basic._constructor_postprocessor_mapping[TensExpr] = {
    "Mul": [get_postprocessor(Mul)],
}
