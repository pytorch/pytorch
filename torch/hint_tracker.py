import collections
import json

class ContextHintTracker:
    hint_stack = collections.deque()

    def __init__(self, hint):
        self.hint = hint

    def __enter__(self):
        ContextHintTracker.hint_stack.append(self.hint)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        ContextHintTracker.hint_stack.pop()

    def get_last_hint():
        if len(ContextHintTracker.hint_stack):
            return ContextHintTracker.hint_stack[-1]
        else:
            return None

    def get_hints_merged():
        if len(ContextHintTracker.hint_stack):
            merged_hints = {}
            for hint in ContextHintTracker.hint_stack:
                hint_object = json.loads(hint)
                merged_hints.update(hint_object)
            return json.dumps(merged_hints)
        else:
            return None