def _get_all_methods():
    return dir(torch.Tensor)


def _check_behavior(fn):
    try:
        return fn()
    except TypeError:
        pass
    except RuntimeError:
        pass
    except AttributeError:
        pass
    except ValueError:
        pass
    except IndexError:
        pass
    except AssertionError:
        pass
    return False


def get_unary_pointwise_methods():

    class Mock(object):
        pass

    result = []
    a = torch.Tensor([2, 3])
    for method in _get_all_methods():
        def _check():
            # Exercise call behavior of a method
            # we consider pointwise unary
            if method[-1] == '_':
                b = a.clone()
                getattr(b, method)()
            else:
                fn = getattr(a, method)
                b = fn()
            if a.size() != b.size():
                return False
            # setattr(Mock, method, fn)
            return True
        if _check_behavior(_check):
            result.append(method)
    logging.warning(result)
    logging.warning(len(result))
    return result


def get_binary_pointwise_methods():

    class Mock(object):
        pass

    result = []
    a = torch.Tensor([2, 3])
    c = torch.Tensor([4, 5])
    for method in _get_all_methods():
        def _check():
            # Exercise call behavior of a method
            # we consider pointwise unary
            if method[-1] == '_':
                b = a.clone()
                getattr(b, method)(c)
            else:
                fn = getattr(a, method)
                b = fn(c)
            if a.size() != b.size():
                return False
            # setattr(Mock, method, fn)
            return True
        if _check_behavior(_check):
            result.append(method)
    logging.warning(result)
    logging.warning(len(result))
    return result


# TODO: Some of these overlap with other methods
# (notably binary pointwise) due to overloading
def get_pointwise_scalar_methods():

    class Mock(object):
        pass

    result = []
    a = torch.Tensor([2, 3])
    for method in _get_all_methods():
        def _check():
            # Exercise call behavior of a method
            # we consider pointwise unary
            scal = 4.3
            if method[-1] == '_':
                b = a.clone()
                getattr(b, method)(scal)
            else:
                fn = getattr(a, method)
                b = fn(scal)
            if a.size() != b.size():
                return False
            # setattr(Mock, method, fn)
            return True
        if _check_behavior(_check):
            result.append(method)
    logging.warning("H1")
    logging.warning(result)
    logging.warning(len(result))
    logging.warning("H2")
    return result


# XXX: Temporarily moved here to clean up batched.py


def monkey_pointwise_unary(fn):
    # TODO: Differentiate inplace from out of place
    def make_pointwise_unary_fn_impl(fn):
        def fn_impl(self):
            ret = getattr(self.tensor, fn)()
            if ret is not None:
                return BatchedTensor(ret, self.mask.clone())
        return fn_impl

    setattr(BatchedTensor, fn, make_pointwise_unary_fn_impl(fn))


def monkey_pointwise_binary(fn):
    # TODO: Differentiate inplace from out of place
    def make_pointwise_binary_fn_impl(fn):
        def fn_impl(self, other):
            assert torch.equal(self.mask, other.mask), \
                    "Can only operate on tensors of same shape."
            ret = getattr(self.tensor, fn)(other.tensor)
            if ret is not None:
                return BatchedTensor(ret, self.mask.clone())
        return fn_impl

    setattr(BatchedTensor, fn, make_pointwise_binary_fn_impl(fn))


def monkey_not_implemented(fn):

    def make_pointwise_not_implemented_fn_impl(fn):
        def fn_impl(self, other):
            raise NotImplementedError()
        return fn_impl

    setattr(BatchedTensor, fn, make_pointwise_not_implemented_fn_impl(fn))


def _monkey_module():

    implemented = [
        # Already implemented or prefer default / notimpl
        '__subclasshook__',
        '__class__',
        '__dir__',
        '__init__',
        '__len__',
        '__reduce__',
        '__repr__',
        '__sizeof__',
        '__str__',
        '__getitem__',
        'copy',
        'index',

        # Tensor ops
        'transpose',
        'transpose_',
        'size',
        'reshape',

        # FIXME
        '__eq__',

        # From skipped
        '__dict__',
        '__new__',
        '__getattribute__',
        '__module__',
        '__weakref__',
        '__delattr__',
        '__setattr__',
        '__format__',
    ]

    skipped = [
        '__and__',
        '__array_priority__',
        '__array_wrap__',
        '__bool__',
        '__cuda_array_interface__',
        '__deepcopy__',
        '__delitem__',
        '__doc__',
        '__float__',
        '__iand__',
        '__index__',
        '__int__',
        '__invert__',
        '__ior__',
        '__ipow__',
        '__ixor__',
        '__long__',
        '__nonzero__',
        '__or__',
        '__setstate__',
        '__xor__',
        '_backward_hooks',
        '_base',
        '_cdata',
        '_coalesced_',
        '_dimI',
        '_dimV',
        '_grad',
        '_grad_fn',
        '_indices',
        '_make_subclass',
        '_nnz',
        '_values',
        '_version',
    ]

    for fn in implemented:
        assert getattr(BatchedTensor, fn)
        assert getattr(BatchedTensor, fn)

    for fn in op_types.get_unary_pointwise_methods():
        if fn in implemented:
            continue
        if getattr(BatchedTensor, fn, None) is not None:
            logging.warning("unary fn " + fn + " already implemented")
        monkey_pointwise_unary(fn)

    for fn in op_types.get_binary_pointwise_methods():
        if fn in implemented:
            continue
        if getattr(BatchedTensor, fn, None) is not None:
            logging.warning("binary fn " + fn + " already implemented")
        monkey_pointwise_binary(fn)

    for fn in skipped:
        if getattr(BatchedTensor, fn, None) is not None:
            logging.warning("fn " + fn + " already implemented")
        if fn not in implemented:
            monkey_not_implemented(fn)

    # For now set functions we successfully characterized to skipped
    skipped = list(set(op_types.get_pointwise_scalar_methods()) | set(skipped))

    covered = op_types.get_unary_pointwise_methods()
    covered += op_types.get_binary_pointwise_methods()
    covered += implemented
    covered += skipped
    all_methods = op_types._get_all_methods()
    not_covered = [method for method in all_methods if method not in covered]
    return not_covered
