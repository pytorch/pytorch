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
        return self.item, self


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
        tx.output.side_effects.mutation(self)
        next_item = self.item.call_method(tx, "__add__", [self.step], {})
        self.item = next_item
        return self.item, self


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
                new_item, _ = self.iterator.next_variables(tx)
                if len(self.saved) > MAX_CYCLE:
                    unimplemented(
                        "input iterator to itertools.cycle has too many items"
                    )
                tx.output.side_effects.mutation(self)
                self.saved.append(new_item)
                self.item = new_item
                if self.item is None:
                    return self.next_variables(tx)
                return self.item, self
            except StopIteration:
                self.iterator = None
                return self.next_variables(tx)
        elif len(self.saved) > 0:
            tx.output.side_effects.mutation(self)
            self.saved_index = (self.saved_index + 1) % len(self.saved)
            return self.item, self
        else:
            raise StopIteration
