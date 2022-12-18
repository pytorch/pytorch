# This file establishes the public comptime interface to Dynamo.
# This allows Dynamo users to execute arbitrary Python code while
# Dynamo is symbolically evaluating their original programs.
#
# The goal of the public API is to give users rope, without actually
# leaking private implementation details of Dynamo.

import dis
import traceback

from .exc import unimplemented


class ComptimeVar:
    """
    A ComptimeVar represents a Python value, at some particular point
    in time, in the Python code we are symbolically evaluating with
    torchdynamo.  This must be distinguished from a runtime value, as
    at compile-time there are some properties of the variable we
    do not know (for example, if the ComptimeVar represents a Tensor,
    we only know metadata about the tensor; we do NOT know what the
    actual data in the Tensor is.)
    """

    def __init__(self, v):
        self.__variable = v

    def as_proxy(self):
        """
        Returns an fx.Proxy (or tuple/list of fx.Proxy) representing
        this variable in the FX graph we are assembling to pass
        to the user compiler.

        This method only works for variables we actually track in
        the FX graph, aka Tensors (and ints, if you are compiling
        with dynamic shapes).  In particular, if you have a list
        or tuple of tensors, you will get a list/tuple of proxies
        (not a single proxy representing the entire list/tuple).
        """
        return self.__variable.as_proxy()

    def is_proxy(self):
        """
        Returns True if as_proxy() would succeed.
        """
        return self.__variable.is_proxy()

    def as_fake(self):
        """
        Returns a "fake" value (either a FakeTensor or a SymInt)
        representing the variable in question.  This only works
        for variables that denote Tensor or int.  You can use
        this to query metadata; e.g., v.as_fake().size(0) will
        tell you the compile-time known size of the tensor.

        WARNING: Do NOT mutate the returned tensor.
        """
        return self.__variable.as_proxy().node.meta["example_value"]

    def python_type(self):
        """
        Returns what type(v) would have returned for the variable
        at compile time.
        """
        return self.__variable.python_type()

    def as_python_constant(self):
        """
        Returns the Python value this variable would have, but only if it is
        completely known at compile-time (e.g., it is constant).

        WARNING: Do NOT mutate the returned constant.  The returned constant
        may or may not correspond to the actual value this variable may take
        on at runtime; for example, if the variable in question is a constant
        list, we may return a copy of that list.
        """
        return self.__variable.as_python_constant()

    def is_python_constant(self):
        """
        Returns True if as_python_constant would succeed.
        """
        return self.__variable.is_python_constant()

    def _i_will_not_complain_if_bc_breaks_VariableTracker(self):
        """
        Returns the internal data structure VariableTracker that Dynamo uses
        to represent variables at compile time.  There are no BC guarantees on
        this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if you rely on
        it.
        """
        return self.__variable

    def __repr__(self):
        # TODO: The default repr is pretty bad, do better
        return repr(self.__variable)

    # TODO: API for adding a custom guard


class ComptimeContext:
    """
    This context class provides access to a public API for Dynamo's internals.
    If there is something here you would find useful that is missing, please
    file a feature request at https://github.com/pytorch/pytorch/
    """

    def __init__(self, tx):
        self.__tx = tx

    def get_local(self, name: str) -> ComptimeVar:
        """
        Retrieve the compile-time known information about a local.
        """
        return ComptimeVar(self.__tx.symbolic_locals[name])

    def graph_break(self, msg="ComptimeContext.graph_break"):
        """
        Manually trigger a graph break
        """
        unimplemented(msg)

    def graph(self):
        """
        Retrieve the partially constructed FX graph that would be
        passed to the user compiler after compilation.
        """
        return self.__tx.output.graph

    def print_graph(self, *, verbose=True, file=None):
        """
        Print the partially constructed FX graph that would be passed
        to the user compiler after compilation.
        """
        print(
            self.__tx.output.graph.python_code("self", verbose=verbose).src, file=file
        )

    def print_disas(self, *, file=None):
        """
        Print the current series of opcodes being executed (not including
        parent frames), including where you are in the particular opcode
        stream.
        """
        print(
            dis.Bytecode(
                self.__tx.f_code,
                current_offset=self.__tx.instructions[
                    self.__tx.instruction_pointer
                ].offset,
            ).dis(),
            file=file,
        )

    def print_stack(self, *, file=None):
        """
        Print the current Python bytecode interpreter stack.  Note that
        this is NOT the same as the traceback; use print_bt() to print
        that.  Note that this will typically be empty, as comptime
        cannot currently be used in an expression context where there
        would be intermediates on the stack.  If you would find this
        useful, please file a bug at https://github.com/pytorch/pytorch/
        """
        # TODO: improve
        # TODO: without an expression level comptime this is not very useful
        for s in self.__tx.stack:
            print(f"- {s}", file=file)

    def print_locals(self, *, file=None):
        """
        Print all of the locals available in the current context.
        By default this view is very limited; you can get more information
        about any individual local using get_local().
        """
        # TODO: improve by improving the VariableTracker printing
        for k, v in self.__tx.symbolic_locals.items():
            print(f"{k} = {v}", file=file)

    def print_bt(self, *, file=None):
        """
        Print the user code backtrace, starting at the beginning of the
        frame Dynamo started evaluating.  Note that this MAY NOT go all
        the way to the torch.compile invocation, as we may have done
        a graph break and are compiling an intermediate frame as the
        starting point.  If you think the other behavior would be better,
        file a bug at https://github.com/pytorch/pytorch/
        """
        stack = []
        tx = self.__tx
        while tx is not None:
            stack.append(tx.frame_summary())
            tx = getattr(tx, "parent", None)
        print(
            "".join(traceback.StackSummary.from_list(reversed(stack)).format()),
            file=file,
        )

    def print_guards(self, *, file=None):
        """
        Print the currently installed guards for the Dynamo context.
        This does NOT include guards associated with variables that
        may or may not be installed in the future if those variables
        are used.
        """
        # TODO: improve print format, current guard format is extremely
        # verbose
        print(
            "\n".join(f"-{str(guard)}" for guard in sorted(self.__tx.output.guards)),
            file=file,
        )

    def _i_will_not_complain_if_bc_breaks_InstructionTranslator(self):
        """
        Returns the internal data structure InstructionTranslator that Dynamo
        uses to track state of symbolic evaluation.  There are no BC
        guarantees on this API and WE RESERVE THE RIGHT TO BREAK YOUR CODE if
        you rely on it.
        """
        return self.__tx
