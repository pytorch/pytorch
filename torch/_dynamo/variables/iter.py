MAX_CYCLE = 3000

from typing import List, Optional

from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable


class IteratorVariable(VariableTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def next_variables(self, tx):
        assert self.mutable_local
        self.init_item(tx)
        ret_val = self.item.add_options(self)
        self.next_item(tx)
        next_iter = type(self)(
            *self.items(),
            mutable_local=MutableLocal(),
            recursively_contains=self.recursively_contains,
            **VariableTracker.propagate([self]),
        )
        tx.replace_all(self, next_iter)
        return ret_val, next_iter

    def next_item(self, tx):
        pass

    def init_item(self, tx):
        pass

    def items(self):
        return []


class RepeatIteratorVariable(IteratorVariable):
    def __init__(self, item: VariableTracker, **kwargs):
        super().__init__(**kwargs)
        self.item = item

    def items(self):
        return [self.item]


class CountIteratorVariable(IteratorVariable):
    def __init__(self, item: int = 0, step: int = 1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(item, VariableTracker):
            item = ConstantVariable.create(item)
        if not isinstance(step, VariableTracker):
            step = ConstantVariable.create(step)
        self.item = item
        self.step = step

    def items(self):
        return [self.item, self.step]

    def next_item(self, tx):
        self.item.call_method(tx, "__add__", [self.step], {})


class CycleIteratorVariable(IteratorVariable):
    def __init__(
        self,
        iterator: IteratorVariable,
        saved: List[VariableTracker] = None,
        saved_index: int = 0,
        saved_len: int = 0,
        item: Optional[VariableTracker] = None,
        **kwargs,
    ):
        if saved is None:
            saved = []
        super().__init__(**kwargs)
        self.iterator = iterator
        self.saved = saved
        self.saved_index = saved_index
        self.saved_len = len(saved)
        self.item = item

    def items(self):
        return [self.iterator, self.saved, self.saved_len, self.saved_index, self.item]

    def init_item(self, tx):
        if self.item is None:
            self.next_item(tx)

    def next_item(self, tx):
        if self.iterator is not None:
            try:
                val, next_iter = self.iterator.next_variables(tx)
                tx.replace_all(self.iterator, next_iter)
                self.saved.append(val)
                if len(self.saved) > MAX_CYCLE:
                    unimplemented(
                        "input iterator to itertools.cycle has too many items"
                    )
                self.item = val
            except StopIteration:
                self.iterator = None
                self.saved_len = len(self.saved)
                self.saved_index = 0
                self.next_item(tx)
        elif self.saved_len > 0:
            ret = self.saved[self.saved_index]
            self.saved_index += 1
            self.saved_index %= self.saved_len
            self.item = ret
        else:
            raise StopIteration
