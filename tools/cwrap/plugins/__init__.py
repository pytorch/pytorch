
class CWrapPlugin(object):
    """Base class from which all cwrap plugins should inherit.

    Override any of the following methods to implement the desired wrapping
    behavior.
    """

    def initialize(self, cwrap):
        """Initialize the Plugin class prior to calling any other functions.

        It is used to give the Plugin access to the cwrap object's helper
        functions and state.

        Args:
            cwrap: the cwrap object performing the wrapping.

        """
        pass

    def get_type_check(self, arg, option):
        """Used to generate code for runtime checks of object types.

        The type can be found in arg['type']. For example, it could be
        THTensor*. If this Plugin recognizes the type in arg, it should
        return a Template string containing code that checks whether a
        Python object is of this type. For example, the return type in
        this case would be:

        Template('(PyObject*)Py_TYPE($arg) == THPTensorClass')

        As a simpler example, if the type == 'bool' then we would return:

        Template('PyBool_Check($arg)')

        Note that the name of the identifier that will be subsituted must be
        $arg.

        Args:
            arg: a Python object with a 'type' field representing the type
            to generate a check string for.
            option: dictionary containing the information for this specific
            option.

        Returns:
            A Template string as described above, or None if this Plugin does
            not have a corresponding type check for the passed type.

        """
        pass

    def get_type_unpack(self, arg, option):
        """Used to generate code unpacking of Python objects into C types.

        Similar to get_type_check, but for unpacking Python objects into their
        corresponding C types. The type is once again accessible via
        arg['type']. This time we return a Template string that unpacks an
        object. For a THTensor*, we know that the corresponding PyTorch type is
        a THPTensor*, so we need to get the cdata from the object. So we would
        return:

        Template('((THPTensor*)$arg)->cdata')

        For a simpler type, such as a long, we could do:

        Template('PyLong_AsLong($arg)')

        though in practice we will use our own custom unpacking code. Once
        again, $arg must be used as the identifier.

        Args:
            arg: a Python object with a 'type' field representing the type
            to generate a unpack string for.
            option: dictionary containing the information for this specific
            option.

        Returns:
            A Template string as described above, or None if this Plugin does
            not have a corresponding type unpack for the passed type.

        """
        pass

    def get_return_wrapper(self, option):
        """Used to generate code wrapping a function's return value.

        Wrapped functions should always return a PyObject *. However,
        internally, the code will be working with C objects or primitives.
        Therefore, if a function has a return value we need to convert it back
        to a PyObject * before the function returns. Plugins can override this
        function to generate wrapper code for returning specific C types. The
        type is accessible via option['return'].

        Continuing on with our THTensor* example, we might do something like:

        Template('return THPTensor_(New)($result);')

        In general, you want to do return <statement>; In this case, we call
        into THP's library routine that takes a THTensor* (the $result
        identifier) and returns a PyObject *.

        For a bool, we could do Template('return PyBool_FromLong($result);').

        Note that in other cases, our logic might be more complicated. For
        example, if our return value is also an argument to the function call,
        we could need to increase the reference count prior to returning.

        Args:
            option: dictionary containing the information for this specific
            option.

        Returns:
            A Template string as described above, or None if this Plugin does
            not have a corresponding return wrapper for the functions return
            type or specifier.

        """
        pass

    def get_wrapper_template(self, declaration):
        """Used to create a code template to wrap the options.

        This function returns a Template string that contains the function call
        for the overall declaration, including the method definition, opening
        and closing brackets, and any additional code within the method body.
        Look through the examples to get a sense of what this might look like.
        The only requirements are that it contains unsubstituted template
        identifiers for anything the cwrap engine expects.

        Note that for any declaration only one Plugin can generate the wrapper
        template.

        Args:
            declaration: the declaration for the wrapped method.

        Returns:
            A template string representing the entire function declaration,
            with identifiers as necessary.

        """
        pass

    def get_assign_args(self, arguments):
        """Used to modify argument metadata prior to assignment.

        We have already setup argument checking, and how to unpack arguments.
        This function allows you to modify the metadata of an argument prior to
        actually performing the assignment. For example, you might want to
        check that an argument is of a specific type, but when unpacking it you
        might want to treat it as a different type. This function will allow
        you to do stuff like that --> e.g. you could set the 'type' field for a
        particular argument to be something else.

        Args:
            arguments: a list of argument metadata dictionaries.

        Returns:
            The same list of arguments, with any modifications as you see fit.

        """
        pass

    def get_arg_accessor(self, arg, option):
        """Used to generate a string for accessing the passed arg.

        One of the key components of the YAML definition for a method to be
        wrapped are the arguments to that method. Override this function to
        show how to access that specific arg in the code. For example, you
        might do something different if the argument is a keyword argument, or
        a constant, or self. The base cwrap plugin has a fallback arg accessor
        for loading elements from the args PyObject * tuple passed to the
        function.

        Its best to look at some of the existing Plugins to get a sense of what
        one might do.

        Args:
            arg: a dictionary specifying attributes of the arg to be accessed
            option: dictionary containing the information for this specific
            option.

        Returns:
            A a string (note: not a Template string!) of code that can be used
            to access the given arg. If the plugin does not know how to access
            the arg, return None.
        """
        pass

    def process_full_file(self, code):
        """Used to modify the code for the entire output file.

        The last thing any plugin can do. Code contains the results of wrapping
        all the declarations. The plugin can do things like adding header
        guards, include statements, etc.

        Args:
            code: a string source code for the wrapped declarations.

        Returns:
            The same code, modified as the plugin sees fit.

        """
        return code

    def process_single_check(self, code, arg, arg_accessor):
        """Used to postprocess a type check.

        Above we defined a function get_type_check that returns a Template
        string that allows for type checking a PyObject * for a specific type.
        In this function, the passed "code" is a combination of that type check
        along with a specific arg_accessor pasted in. For example:

        '(PyObject*)Py_TYPE(PyTuple_GET_ITEM(args, 1)) == THPTensorClass'

        This function can be overriden to support modifying this check string.
        For example, if an argument can be null, we might want to check and see
        if the type is Py_None, as well.

        Args:
            code: The string code representing a type check for a specific
            argument being accessed.
            arg: dictionary containing properties of that specific argument
            arg_accessor: the arg_accessor string for that specific argument.
            Note that this is likely also embedded in code, but if you want to
            be able to access this arg and throw away the other code, you can
            do so.

        Returns:
            A string representing the processed check/access string for this
            arg. If the plugin does not know how to modify a specific input, it
            should return the original code.

        """
        return code

    def process_all_checks(self, code, option):
        """Used to generate additional checks based on all the individual ones.

        After individually processing each argument with get_type_check,
        get_arg_accessor, process_single_check, this function allows you to
        inspect the combined checks and do any additional checking/modify that
        string as you see fit. In particular, given code is a string like:

        CHECK_TYPE(GET_ARG(0)) && CHECK_TYPE(GET_ARG(1)) && ..

        We can process it as we see fit. For example, we may want to add a
        check at the beginning that we have the specified number of arguments.

        Args:
            code: A string representing each argument check separated by an
            '&&'. code can be None if there are no arguments to be checked.
            option: dictionary containing the information for this specific
            option.

        Returns:
            The modified code string with any additional checks, or just the
            existing code if no modifications are to be made.

        """
        return code

    def process_single_unpack(self, code, arg, arg_accessor):
        """Used to postprocess a type unpack.

        Same as process_single_check above, but for type unpacking. E.g. an
        example code could be:

        PyLong_FromLong(PyTuple_GET_ITEM(args, 0))

        And this code could modify that as it sees fit. For example, if the
        result of accessing the argument is None, we would not want to call the
        unpacking code.

        Args:
            code: The string code representing a type unpack for a specific
            argument being accessed.
            arg: dictionary containing properties of that specific argument
            arg_accessor: the arg_accessor string for that specific argument.
            Note that this is likely also embedded in code, but if you want to
            be able to access this arg and throw away the other code, you can
            do so.

        Returns:
            A string representing the processed unpack/access string for this
            arg. If the plugin does not know how to modify a specific input, it
            should return the original code.

        """
        return code

    def process_all_call_arg(self, code, option):
        """Used to modify the arguments to the underlying C function call.

        Code is the string of comma-separated arguments that will be passed to
        the wrapped C function. You can use this function to modify that string
        as you see fit. For example, THP prepends the LIBRARY_STATE definition
        so that the generated code will follow the conventions it uses for
        writing one function for both TH/THC calls.

        Args:
            code: A string as described above.
            option: dictionary containing the information for this specific
            option.

        Returns:
            The same code, modified as the plugin sees fit.

        """
        return code

    def process_option_code(self, code, option):
        """Used to modify the entire code body for an option.

        Code in this case is a string containing the entire generated code for
        a specific option. Note that this body includes the checks for each
        option, i.e. if (type checks for one permutation) { ... } else if (type
        checks for another permutation) { ... } etc.

        Args:
            code: string representing the generated code for the option
            option: dictionary containing the information for this specific
            option.

        Returns:
            The same code, modified as the plugin sees fit.

        """
        return code

    def process_wrapper(self, code, declaration):
        """Used to modify the entire code body for a declaration.

        Code in this case is a string containing the entire generated code for
        a specific declaration. This code can be modified as the plugin sees
        fit. For example, we might want to wrap the function in preprocessor
        guards if it is only enabled for floats.

        Args:
            code: string representing the generated code for the declaration
            declaration: the declaration metadata.

        Returns:
            The same code, modified as the plugin sees fit.

        """
        return code

    def process_declarations(self, declarations):
        """Used to process/modify the function's declaration.

        Cwrap loads the YAML of a function to be cwrap'd into a dictionary.
        This is known as the declaration. The cwrap code sets some defaults as
        necessary, and then passes this dictionary to process_declarations.
        Overriding this code allows the plugin to modify this declaration as it
        sees fit prior to any code generation. The plugin may add, remove or
        modify the fields of the declaration dictionary. It can also save state
        to the Plugin for use in subsequent function overrides.

        Its best to look at some of the existing Plugins to get a sense of what
        one might do.

        Args:
            declarations: a list of declarations, i.e. dictionaries that define
            the function(s) being wrapped. Note that this can be plural, so the
            function must take care to modify each input declaration.

        Returns:
            Those same declarations, modified as the Plugin sees fit. Note that
            you could insert a declaration, if you wanted to take an input
            declaration and e.g. wrap it multiple times.

        """
        return declarations

    def process_option_code_template(self, template, option):
        """Used to modify the code template for the option.

        The "code template" can be thought of the actual body implementing the
        wrapped function call --> i.e. it is not the argument check,
        assignment, etc. but the actual logic of the function. The template is
        a list containing two operations: the $call, and the $return_result.
        These represent the "locations" where the function call will happen,
        and the function will return.

        This function can modify the list to insert arbitrary code around the
        $call and $return_result. For example, one might want to wrap the code
        in a try/catch, or post-process the result in some way. This allows a
        plugin to do that.

        Args:
            template: a list containing $call and $return_result, in addition
            to any arbitrary code inserted by other plugins.
            option: dictionary containing the information for this specific
            option.

        Returns:
            The same "code template", possibly modified by this plugin.

        """
        return template

    def process_pre_arg_assign(self, template, option):
        """Used to include any code before argument assignment.

        This function can be used to insert any code that will be part of the
        resulting function. The code is inserted after argument checks occur,
        but before argument assignment.

        Args:
            template: String representing the code to be inserted. If other
            plugins have included code for pre_arg_assign, it will be included
            here.
            option: dictionary containing the information for this specific
            option.

        Returns:
            template, with any additional code if needed.

        """
        return template


from .StandaloneExtension import StandaloneExtension
from .NullableArguments import NullableArguments
from .OptionalArguments import OptionalArguments
from .ArgcountChecker import ArgcountChecker
from .ArgumentReferences import ArgumentReferences
from .BeforeAfterCall import BeforeAfterCall
from .ConstantArguments import ConstantArguments
from .ReturnArguments import ReturnArguments
from .GILRelease import GILRelease
from .AutoGPU import AutoGPU
from .CuDNNPlugin import CuDNNPlugin
from .WrapDim import WrapDim
from .Broadcast import Broadcast
