from collections import Counter

class ExecutionEngine(object):
    def __init__(self):
        pass

    def _compute_dependencies(self, function):
        dependencies = {}
        seen = {function}
        queue = [function]
        while len(queue) > 0:
            fn = queue.pop()
            for prev_fn, arg_id in fn.previous_functions:
                if prev_fn not in dependencies:
                    dependencies[prev_fn] = [Counter() for _ in prev_fn.output_ids]
                output_idx = prev_fn.output_ids[arg_id]
                dependencies[prev_fn][output_idx][fn] += 1
                if prev_fn not in seen:
                    queue.append(prev_fn)
                    seen.add(prev_fn)
        return dependencies

    def _free_backward_dependency(self, dependencies, prev_fn, fn, arg_id):
        deps = dependencies[prev_fn]
        output_idx = prev_fn.output_ids[arg_id]
        output_deps = deps[output_idx]
        output_deps[fn] -= 1
        if output_deps[fn] == 0:
            del output_deps[fn]
        return output_idx


    def _is_ready_for_backward(self, dependencies, function):
        for deps in dependencies[function]:
            if len(deps) > 0:
                return False
        return True

    def run_backward(self, variable, grad):
        ready = [(variable.creator, (grad,))]
        not_ready = {}

        dependencies = self._compute_dependencies(variable.creator)

        while len(ready) > 0:
            fn, grad = ready.pop()
            # TODO: double-buffering
            grad_input = fn._do_backward(*grad)
            for (prev_fn, arg_id), d_prev_fn in zip(fn.previous_functions, grad_input):
                if not prev_fn.requires_grad:
                    assert d_prev_fn is None
                    continue
                output_nr = self._free_backward_dependency(dependencies, prev_fn, fn, arg_id)
                is_ready = self._is_ready_for_backward(dependencies, prev_fn)
                if is_ready:
                    if prev_fn in not_ready:
                        prev_grad = not_ready[prev_fn]
                        if not prev_grad[output_nr]:
                            prev_grad[output_nr] = d_prev_fn
                        else:
                            prev_grad[output_nr].add_(d_prev_fn)
                        del not_ready[prev_fn]
                    else:
                        assert output_nr == 0
                        prev_grad = (d_prev_fn,)
                    ready.append((prev_fn, prev_grad))
                else:
                    if prev_fn in not_ready:
                        prev_grad = not_ready[prev_fn]
                    else:
                        prev_grad = [None for _ in prev_fn.output_ids]

                    if not prev_grad[output_nr]:
                        prev_grad[output_nr] = d_prev_fn
                    else:
                        prev_grad[output_nr].add_(d_prev_fn)

                    not_ready[prev_fn] = prev_grad


