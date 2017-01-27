from collections import deque, defaultdict
from torch._C import _ImperativeEngine as ImperativeEngine
from .variable import Variable


class BasicEngine(object):

    def _compute_dependencies(self, function):
        dependencies = defaultdict(int)
        seen = {function}
        queue = [function]
        while len(queue) > 0:
            fn = queue.pop()
            for prev_fn, output_nr in fn.previous_functions:
                if not prev_fn.requires_grad or isinstance(prev_fn, Variable):
                    continue
                dependencies[prev_fn] += 1
                if prev_fn not in seen:
                    queue.append(prev_fn)
                    seen.add(prev_fn)
        return dependencies

    def _free_backward_dependency(self, dependencies, prev_fn):
        dependencies[prev_fn] -= 1
        if dependencies[prev_fn] == 0:
            del dependencies[prev_fn]
            return True
        return False

    def _add_grad(self, need_copy, prev_grad, output_nr, d_prev_fn):
        copy_id = (id(prev_grad), output_nr)
        if not prev_grad[output_nr]:
            prev_grad[output_nr] = d_prev_fn
            need_copy.add(copy_id)
        else:
            grad_tensor = prev_grad[output_nr]
            if copy_id in need_copy:
                need_copy.remove(copy_id)
                grad_tensor = grad_tensor.clone()
                prev_grad[output_nr] = grad_tensor
            grad_tensor.add_(d_prev_fn)

    def run_backward(self, variable, grad, retain_variables):
        if variable.creator is None:
            variable._do_backward((grad,), retain_variables)
            return

        initial_grad = [None for _ in range(variable.creator.num_outputs)]
        initial_grad[variable.output_nr] = grad
        ready = deque([(variable.creator, initial_grad)])
        not_ready = {}
        need_copy = set()

        dependencies = self._compute_dependencies(variable.creator)

        while len(ready) > 0:
            fn, grad = ready.pop()
            grad_input = fn._do_backward(tuple(grad), retain_variables)
            for (prev_fn, output_nr), d_prev_fn in zip(fn.previous_functions, grad_input):
                if not prev_fn.requires_grad:
                    # TODO: check that d_prev_fn is None and warn otherwise
                    continue
                if isinstance(prev_fn, Variable):
                    prev_fn._do_backward((d_prev_fn,), retain_variables)
                    continue
                is_ready = self._free_backward_dependency(dependencies, prev_fn)
                if is_ready:
                    if prev_fn in not_ready:
                        prev_grad = not_ready[prev_fn]
                        self._add_grad(need_copy, prev_grad, output_nr, d_prev_fn)
                    else:
                        if prev_fn.num_outputs != 1:
                            raise RuntimeError("one of the function outputs "
                                               "wasn't used - this is an error not, but "
                                               "it's going to be fixed soon")
                        prev_grad = (d_prev_fn,)
                    ready.appendleft((prev_fn, prev_grad))
                else:
                    if prev_fn in not_ready:
                        prev_grad = not_ready[prev_fn]
                    else:
                        prev_grad = [None for _ in range(prev_fn.num_outputs)]

                    self._add_grad(need_copy, prev_grad, output_nr, d_prev_fn)
                    not_ready[prev_fn] = prev_grad
