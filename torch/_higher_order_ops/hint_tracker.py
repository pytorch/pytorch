import collections
import json
from typing import Any, Deque, TypeVar

S = TypeVar("S", bound="ContextHintTracker")


class ContextHintTracker:
    hint_stack: Deque[str] = collections.deque()

    def __init__(self, hint: str):
        self.hint = hint

    def __enter__(self: S) -> S:
        ContextHintTracker.hint_stack.append(self.hint)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        ContextHintTracker.hint_stack.pop()

    @staticmethod
    def get_last_hint() -> str | None:
        if len(ContextHintTracker.hint_stack):
            return ContextHintTracker.hint_stack[-1]
        else:
            return None

    @staticmethod
    def get_hints_merged() -> str | None:
        if len(ContextHintTracker.hint_stack):
            merged_hints = {}
            for hint in ContextHintTracker.hint_stack:
                hint_object = json.loads(hint)
                merged_hints.update(hint_object)

            if ContextHintTracker.is_preserve_order_requested():
                merged_hints.update(
                    {"exec_order": UserOrderTracker.get_new_order_number()}
                )
            return json.dumps(merged_hints)
        else:
            return None

    @staticmethod
    def is_preserve_order_requested() -> bool:
        if len(ContextHintTracker.hint_stack):
            for hint in ContextHintTracker.hint_stack:
                hint_object = json.loads(hint)
                if "preserve_order" in hint_object and (
                    (
                        isinstance(hint_object["preserve_order"], bool)
                        and hint_object["preserve_order"]
                    )
                    or (
                        isinstance(hint_object["preserve_order"], str)
                        and hint_object["preserve_order"].lower() == "true"
                    )
                ):
                    return True
            return False
        else:
            return False


class UserOrderTracker:
    op_count = 0

    @staticmethod
    def get_new_order_number() -> int:
        UserOrderTracker.op_count += 1
        return UserOrderTracker.op_count
