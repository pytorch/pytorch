# set_linter: detect (most) uses of set and badly fix them

## The problem

Since 3.7, Python has had a `dict` class with a deterministic iteration order
corresponding to the order of insertion of its items.

But `set()` has no such guarantee, and this has lead to hard-to-find issues
in at least `torch._inductor`.

## The solution

`set` should be replaced whenever possible by the class
`torch.utils._ordereed_set.OrderedSet` which has the same API as `set` but
uses `dict`'s deterministic iteration order.

`set_linter.py` can detect many uses of `set` in Python code and report them.
With the `--fix` option, it can

A linter needs to
