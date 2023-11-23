MAX_CYCLE = 3000

from typing import List, Optional

from ..exc import unimplemented

from .base import VariableTracker
from .constant import ConstantVariable


class IteratorVariable(VariableTracker):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def next_variables(self, tx):
        unimplemented("abstract method, must implement")


class RepeatIteratorVariable(IteratorVariable):
    def __init__(self, item: VariableTracker, **kwargs):
        super().__init__(**kwargs)
        self.item = item

    # Repeat needs no mutation, clone self
    def next_variables(self, tx):
        return self.item.clone(), self


class CountIteratorVariable(IteratorVariable):
    def __init__(self, item: int = 0, step: int = 1, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(item, VariableTracker):
            item = ConstantVariable.create(item)
        if not isinstance(step, VariableTracker):
            step = ConstantVariable.create(step)
        self.item = item
        self.step = step

    def next_variables(self, tx):
        assert self.mutable_local
        next_item = self.item.call_method(tx, "__add__", [self.step], {})
        next_iter = self.clone(item=next_item)
        tx.replace_all(self, next_iter)
        return self.item, next_iter


class CycleIteratorVariable(IteratorVariable):
    def __init__(
        self,
        iterator: IteratorVariable,
        saved: List[VariableTracker] = None,
        saved_index: int = 0,
        item: Optional[VariableTracker] = None,
        **kwargs,
    ):
        if saved is None:
            saved = []
        super().__init__(**kwargs)
        self.iterator = iterator
        self.saved = saved
        self.saved_index = saved_index
        self.item = item

    def next_variables(self, tx):
        assert self.mutable_local

        if self.iterator is not None:
            try:
                new_item, next_inner_iter = self.iterator.next_variables(tx)
                tx.replace_all(self.iterator, next_inner_iter)
                if len(self.saved) > MAX_CYCLE:
                    unimplemented(
                        "input iterator to itertools.cycle has too many items"
                    )
                next_iter = self.clone(
                    iterator=next_inner_iter,
                    saved=self.saved + [new_item],
                    item=new_item,
                )

                tx.replace_all(self, next_iter)
                if self.item is None:
                    return next_iter.next_variables(tx)
                return self.item, next_iter
            except StopIteration:
                next_iter = self.clone(iterator=None)
                # this is redundant as next_iter will do the same
                # but we do it anyway for safety
                tx.replace_all(self, next_iter)
                return next_iter.next_variables(tx)
        elif len(self.saved) > 0:
            next_iter = self.clone(
                saved_index=(self.saved_index + 1) % len(self.saved),
                item=self.saved[self.saved_index],
            )
            tx.replace_all(self, next_iter)
            return self.item, next_iter
        else:
            raise StopIteration
        return self.item, next_iter
