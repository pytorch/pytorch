import torch


# This is a very odd way to define a test case that uses CALL_INTRINSIC_1 4 and
# CALL_INTRINSIC_2 7. The problem is those intrinsics only exists on Python
# 3.12+ and the Python code to produce them is not backwards compatible with
# previous versions.
class TestPy312Intrinsics:
    @classmethod
    def _default_update(cls):
        def f[T](a:'This is a new annotation'):
            """This is a test"""
            pass
        f.attr = 'This is also a test'
        f.__wrapped__ = "This is a bald faced lie"
        def wrapper(b:'This is the prior annotation'):
            pass
        return wrapper, f

    def test_default_update(self):
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            wrapper, f = TestPy312Intrinsics._default_update()
            return x + 1
        x = torch.randn(2)
        fn(x)
