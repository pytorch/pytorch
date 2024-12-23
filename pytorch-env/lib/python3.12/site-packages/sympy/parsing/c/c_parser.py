from sympy.external import import_module
import os

cin = import_module('clang.cindex', import_kwargs = {'fromlist': ['cindex']})

"""
This module contains all the necessary Classes and Function used to Parse C and
C++ code into SymPy expression
The module serves as a backend for SymPyExpression to parse C code
It is also dependent on Clang's AST and SymPy's Codegen AST.
The module only supports the features currently supported by the Clang and
codegen AST which will be updated as the development of codegen AST and this
module progresses.
You might find unexpected bugs and exceptions while using the module, feel free
to report them to the SymPy Issue Tracker

Features Supported
==================

- Variable Declarations (integers and reals)
- Assignment (using integer & floating literal and function calls)
- Function Definitions and Declaration
- Function Calls
- Compound statements, Return statements

Notes
=====

The module is dependent on an external dependency which needs to be installed
to use the features of this module.

Clang: The C and C++ compiler which is used to extract an AST from the provided
C source code.

References
==========

.. [1] https://github.com/sympy/sympy/issues
.. [2] https://clang.llvm.org/docs/
.. [3] https://clang.llvm.org/docs/IntroductionToTheClangAST.html

"""

if cin:
    from sympy.codegen.ast import (Variable, Integer, Float,
        FunctionPrototype, FunctionDefinition, FunctionCall,
        none, Return, Assignment, intc, int8, int16, int64,
        uint8, uint16, uint32, uint64, float32, float64, float80,
        aug_assign, bool_, While, CodeBlock)
    from sympy.codegen.cnodes import (PreDecrement, PostDecrement,
        PreIncrement, PostIncrement)
    from sympy.core import Add, Mod, Mul, Pow, Rel
    from sympy.logic.boolalg import And, as_Boolean, Not, Or
    from sympy.core.symbol import Symbol
    from sympy.core.sympify import sympify
    from sympy.logic.boolalg import (false, true)
    import sys
    import tempfile

    class BaseParser:
        """Base Class for the C parser"""

        def __init__(self):
            """Initializes the Base parser creating a Clang AST index"""
            self.index = cin.Index.create()

        def diagnostics(self, out):
            """Diagostics function for the Clang AST"""
            for diag in self.tu.diagnostics:
                # tu = translation unit
                print('%s %s (line %s, col %s) %s' % (
                        {
                            4: 'FATAL',
                            3: 'ERROR',
                            2: 'WARNING',
                            1: 'NOTE',
                            0: 'IGNORED',
                        }[diag.severity],
                        diag.location.file,
                        diag.location.line,
                        diag.location.column,
                        diag.spelling
                    ), file=out)

    class CCodeConverter(BaseParser):
        """The Code Convereter for Clang AST

        The converter object takes the C source code or file as input and
        converts them to SymPy Expressions.
        """

        def __init__(self):
            """Initializes the code converter"""
            super().__init__()
            self._py_nodes = []
            self._data_types = {
                "void": {
                    cin.TypeKind.VOID: none
                },
                "bool": {
                    cin.TypeKind.BOOL: bool_
                },
                "int": {
                    cin.TypeKind.SCHAR: int8,
                    cin.TypeKind.SHORT: int16,
                    cin.TypeKind.INT: intc,
                    cin.TypeKind.LONG: int64,
                    cin.TypeKind.UCHAR: uint8,
                    cin.TypeKind.USHORT: uint16,
                    cin.TypeKind.UINT: uint32,
                    cin.TypeKind.ULONG: uint64
                },
                "float": {
                    cin.TypeKind.FLOAT: float32,
                    cin.TypeKind.DOUBLE: float64,
                    cin.TypeKind.LONGDOUBLE: float80
                }
            }

        def parse(self, filename, flags):
            """Function to parse a file with C source code

            It takes the filename as an attribute and creates a Clang AST
            Translation Unit parsing the file.
            Then the transformation function is called on the translation unit,
            whose reults are collected into a list which is returned by the
            function.

            Parameters
            ==========

            filename : string
                Path to the C file to be parsed

            flags: list
                Arguments to be passed to Clang while parsing the C code

            Returns
            =======

            py_nodes: list
                A list of SymPy AST nodes

            """
            filepath = os.path.abspath(filename)
            self.tu = self.index.parse(
                filepath,
                args=flags,
                options=cin.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            for child in self.tu.cursor.get_children():
                if child.kind == cin.CursorKind.VAR_DECL or child.kind == cin.CursorKind.FUNCTION_DECL:
                    self._py_nodes.append(self.transform(child))
            return self._py_nodes

        def parse_str(self, source, flags):
            """Function to parse a string with C source code

            It takes the source code as an attribute, stores it in a temporary
            file and creates a Clang AST Translation Unit parsing the file.
            Then the transformation function is called on the translation unit,
            whose reults are collected into a list which is returned by the
            function.

            Parameters
            ==========

            source : string
                A string containing the C source code to be parsed

            flags: list
                Arguments to be passed to Clang while parsing the C code

            Returns
            =======

            py_nodes: list
                A list of SymPy AST nodes

            """
            file = tempfile.NamedTemporaryFile(mode = 'w+', suffix = '.cpp')
            file.write(source)
            file.seek(0)
            self.tu = self.index.parse(
                file.name,
                args=flags,
                options=cin.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
            )
            file.close()
            for child in self.tu.cursor.get_children():
                if child.kind == cin.CursorKind.VAR_DECL or child.kind == cin.CursorKind.FUNCTION_DECL:
                    self._py_nodes.append(self.transform(child))
            return self._py_nodes

        def transform(self, node):
            """Transformation Function for Clang AST nodes

            It determines the kind of node and calls the respective
            transformation function for that node.

            Raises
            ======

            NotImplementedError : if the transformation for the provided node
            is not implemented

            """
            handler = getattr(self, 'transform_%s' % node.kind.name.lower(), None)

            if handler is None:
                print(
                    "Ignoring node of type %s (%s)" % (
                        node.kind,
                        ' '.join(
                            t.spelling for t in node.get_tokens())
                        ),
                    file=sys.stderr
                )

            return handler(node)

        def transform_var_decl(self, node):
            """Transformation Function for Variable Declaration

            Used to create nodes for variable declarations and assignments with
            values or function call for the respective nodes in the clang AST

            Returns
            =======

            A variable node as Declaration, with the initial value if given

            Raises
            ======

            NotImplementedError : if called for data types not currently
            implemented

            Notes
            =====

            The function currently supports following data types:

            Boolean:
                bool, _Bool

            Integer:
                8-bit: signed char and unsigned char
                16-bit: short, short int, signed short,
                    signed short int, unsigned short, unsigned short int
                32-bit: int, signed int, unsigned int
                64-bit: long, long int, signed long,
                    signed long int, unsigned long, unsigned long int

            Floating point:
                Single Precision: float
                Double Precision: double
                Extended Precision: long double

            """
            if node.type.kind in self._data_types["int"]:
                type = self._data_types["int"][node.type.kind]
            elif node.type.kind in self._data_types["float"]:
                type = self._data_types["float"][node.type.kind]
            elif node.type.kind in self._data_types["bool"]:
                type = self._data_types["bool"][node.type.kind]
            else:
                raise NotImplementedError("Only bool, int "
                    "and float are supported")
            try:
                children = node.get_children()
                child = next(children)

                #ignoring namespace and type details for the variable
                while child.kind == cin.CursorKind.NAMESPACE_REF or child.kind == cin.CursorKind.TYPE_REF:
                    child = next(children)

                val = self.transform(child)

                supported_rhs = [
                    cin.CursorKind.INTEGER_LITERAL,
                    cin.CursorKind.FLOATING_LITERAL,
                    cin.CursorKind.UNEXPOSED_EXPR,
                    cin.CursorKind.BINARY_OPERATOR,
                    cin.CursorKind.PAREN_EXPR,
                    cin.CursorKind.UNARY_OPERATOR,
                    cin.CursorKind.CXX_BOOL_LITERAL_EXPR
                ]

                if child.kind in supported_rhs:
                    if isinstance(val, str):
                        value = Symbol(val)
                    elif isinstance(val, bool):
                        if node.type.kind in self._data_types["int"]:
                            value = Integer(0) if val == False else Integer(1)
                        elif node.type.kind in self._data_types["float"]:
                            value = Float(0.0) if val == False else Float(1.0)
                        elif node.type.kind in self._data_types["bool"]:
                            value = sympify(val)
                    elif isinstance(val, (Integer, int, Float, float)):
                        if node.type.kind in self._data_types["int"]:
                            value = Integer(val)
                        elif node.type.kind in self._data_types["float"]:
                            value = Float(val)
                        elif node.type.kind in self._data_types["bool"]:
                            value = sympify(bool(val))
                    else:
                        value = val

                    return Variable(
                    node.spelling
                    ).as_Declaration(
                        type = type,
                        value = value
                    )

                elif child.kind == cin.CursorKind.CALL_EXPR:
                    return Variable(
                        node.spelling
                        ).as_Declaration(
                            value = val
                        )

                else:
                    raise NotImplementedError("Given "
                        "variable declaration \"{}\" "
                        "is not possible to parse yet!"
                        .format(" ".join(
                            t.spelling for t in node.get_tokens()
                            )
                        ))

            except StopIteration:
                return Variable(
                node.spelling
                ).as_Declaration(
                    type = type
                )

        def transform_function_decl(self, node):
            """Transformation Function For Function Declaration

            Used to create nodes for function declarations and definitions for
            the respective nodes in the clang AST

            Returns
            =======

            function : Codegen AST node
                - FunctionPrototype node if function body is not present
                - FunctionDefinition node if the function body is present


            """

            if node.result_type.kind in self._data_types["int"]:
                ret_type = self._data_types["int"][node.result_type.kind]
            elif node.result_type.kind in self._data_types["float"]:
                ret_type = self._data_types["float"][node.result_type.kind]
            elif node.result_type.kind in self._data_types["bool"]:
                ret_type = self._data_types["bool"][node.result_type.kind]
            elif node.result_type.kind in self._data_types["void"]:
                ret_type = self._data_types["void"][node.result_type.kind]
            else:
                raise NotImplementedError("Only void, bool, int "
                    "and float are supported")
            body = []
            param = []

            # Subsequent nodes will be the parameters for the function.
            for child in node.get_children():
                decl = self.transform(child)
                if child.kind == cin.CursorKind.PARM_DECL:
                    param.append(decl)
                elif child.kind == cin.CursorKind.COMPOUND_STMT:
                    for val in decl:
                        body.append(val)
                else:
                    body.append(decl)

            if body == []:
                function = FunctionPrototype(
                    return_type = ret_type,
                    name = node.spelling,
                    parameters = param
                )
            else:
                function = FunctionDefinition(
                    return_type = ret_type,
                    name = node.spelling,
                    parameters = param,
                    body = body
                )
            return function

        def transform_parm_decl(self, node):
            """Transformation function for Parameter Declaration

            Used to create parameter nodes for the required functions for the
            respective nodes in the clang AST

            Returns
            =======

            param : Codegen AST Node
                Variable node with the value and type of the variable

            Raises
            ======

            ValueError if multiple children encountered in the parameter node

            """
            if node.type.kind in self._data_types["int"]:
                type = self._data_types["int"][node.type.kind]
            elif node.type.kind in self._data_types["float"]:
                type = self._data_types["float"][node.type.kind]
            elif node.type.kind in self._data_types["bool"]:
                type = self._data_types["bool"][node.type.kind]
            else:
                raise NotImplementedError("Only bool, int "
                    "and float are supported")
            try:
                children = node.get_children()
                child = next(children)

                # Any namespace nodes can be ignored
                while child.kind in [cin.CursorKind.NAMESPACE_REF,
                                     cin.CursorKind.TYPE_REF,
                                     cin.CursorKind.TEMPLATE_REF]:
                    child = next(children)

                # If there is a child, it is the default value of the parameter.
                lit = self.transform(child)
                if node.type.kind in self._data_types["int"]:
                    val = Integer(lit)
                elif node.type.kind in self._data_types["float"]:
                    val = Float(lit)
                elif node.type.kind in self._data_types["bool"]:
                    val = sympify(bool(lit))
                else:
                    raise NotImplementedError("Only bool, int "
                        "and float are supported")

                param = Variable(
                    node.spelling
                ).as_Declaration(
                    type = type,
                    value = val
                )
            except StopIteration:
                param = Variable(
                    node.spelling
                ).as_Declaration(
                    type = type
                )

            try:
                self.transform(next(children))
                raise ValueError("Can't handle multiple children on parameter")
            except StopIteration:
                pass

            return param

        def transform_integer_literal(self, node):
            """Transformation function for integer literal

            Used to get the value and type of the given integer literal.

            Returns
            =======

            val : list
                List with two arguments type and Value
                type contains the type of the integer
                value contains the value stored in the variable

            Notes
            =====

            Only Base Integer type supported for now

            """
            try:
                value = next(node.get_tokens()).spelling
            except StopIteration:
                # No tokens
                value = node.literal
            return int(value)

        def transform_floating_literal(self, node):
            """Transformation function for floating literal

            Used to get the value and type of the given floating literal.

            Returns
            =======

            val : list
                List with two arguments type and Value
                type contains the type of float
                value contains the value stored in the variable

            Notes
            =====

            Only Base Float type supported for now

            """
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                # No tokens
                value = node.literal
            return float(value)

        def transform_string_literal(self, node):
            #TODO: No string type in AST
            #type =
            #try:
            #    value = next(node.get_tokens()).spelling
            #except (StopIteration, ValueError):
                # No tokens
            #    value = node.literal
            #val = [type, value]
            #return val
            pass

        def transform_character_literal(self, node):
            """Transformation function for character literal

            Used to get the value of the given character literal.

            Returns
            =======

            val : int
                val contains the ascii value of the character literal

            Notes
            =====

            Only for cases where character is assigned to a integer value,
            since character literal is not in SymPy AST

            """
            try:
               value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                # No tokens
               value = node.literal
            return ord(str(value[1]))

        def transform_cxx_bool_literal_expr(self, node):
            """Transformation function for boolean literal

            Used to get the value of the given boolean literal.

            Returns
            =======

            value : bool
                value contains the boolean value of the variable

            """
            try:
                value = next(node.get_tokens()).spelling
            except (StopIteration, ValueError):
                value = node.literal
            return True if value == 'true' else False

        def transform_unexposed_decl(self,node):
            """Transformation function for unexposed declarations"""
            pass

        def transform_unexposed_expr(self, node):
            """Transformation function for unexposed expression

            Unexposed expressions are used to wrap float, double literals and
            expressions

            Returns
            =======

            expr : Codegen AST Node
                the result from the wrapped expression

            None : NoneType
                No childs are found for the node

            Raises
            ======

            ValueError if the expression contains multiple children

            """
            # Ignore unexposed nodes; pass whatever is the first
            # (and should be only) child unaltered.
            try:
                children = node.get_children()
                expr = self.transform(next(children))
            except StopIteration:
                return None

            try:
                next(children)
                raise ValueError("Unexposed expression has > 1 children.")
            except StopIteration:
                pass

            return expr

        def transform_decl_ref_expr(self, node):
            """Returns the name of the declaration reference"""
            return node.spelling

        def transform_call_expr(self, node):
            """Transformation function for a call expression

            Used to create function call nodes for the function calls present
            in the C code

            Returns
            =======

            FunctionCall : Codegen AST Node
                FunctionCall node with parameters if any parameters are present

            """
            param = []
            children = node.get_children()
            child = next(children)

            while child.kind == cin.CursorKind.NAMESPACE_REF:
                child = next(children)
            while child.kind == cin.CursorKind.TYPE_REF:
                child = next(children)

            first_child = self.transform(child)
            try:
                for child in children:
                    arg = self.transform(child)
                    if child.kind == cin.CursorKind.INTEGER_LITERAL:
                        param.append(Integer(arg))
                    elif child.kind == cin.CursorKind.FLOATING_LITERAL:
                        param.append(Float(arg))
                    else:
                        param.append(arg)
                return FunctionCall(first_child, param)

            except StopIteration:
                return FunctionCall(first_child)

        def transform_return_stmt(self, node):
            """Returns the Return Node for a return statement"""
            return Return(next(node.get_children()).spelling)

        def transform_compound_stmt(self, node):
            """Transformation function for compond statemets

            Returns
            =======

            expr : list
                list of Nodes for the expressions present in the statement

            None : NoneType
                if the compound statement is empty

            """
            expr = []
            children = node.get_children()

            for child in children:
                expr.append(self.transform(child))
            return expr

        def transform_decl_stmt(self, node):
            """Transformation function for declaration statements

            These statements are used to wrap different kinds of declararions
            like variable or function declaration
            The function calls the transformer function for the child of the
            given node

            Returns
            =======

            statement : Codegen AST Node
                contains the node returned by the children node for the type of
                declaration

            Raises
            ======

            ValueError if multiple children present

            """
            try:
                children = node.get_children()
                statement = self.transform(next(children))
            except StopIteration:
                pass

            try:
                self.transform(next(children))
                raise ValueError("Don't know how to handle multiple statements")
            except StopIteration:
                pass

            return statement

        def transform_paren_expr(self, node):
            """Transformation function for Parenthesized expressions

            Returns the result from its children nodes

            """
            return self.transform(next(node.get_children()))

        def transform_compound_assignment_operator(self, node):
            """Transformation function for handling shorthand operators

            Returns
            =======

            augmented_assignment_expression: Codegen AST node
                    shorthand assignment expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If the shorthand operator for bitwise operators
                (~=, ^=, &=, |=, <<=, >>=) is encountered

            """
            return self.transform_binary_operator(node)

        def transform_unary_operator(self, node):
            """Transformation function for handling unary operators

            Returns
            =======

            unary_expression: Codegen AST node
                    simplified unary expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If dereferencing operator(*), address operator(&) or
                bitwise NOT operator(~) is encountered

            """
            # supported operators list
            operators_list = ['+', '-', '++', '--', '!']
            tokens = list(node.get_tokens())

            # it can be either pre increment/decrement or any other operator from the list
            if tokens[0].spelling in operators_list:
                child = self.transform(next(node.get_children()))
                # (decl_ref) e.g.; int a = ++b; or simply ++b;
                if isinstance(child, str):
                    if tokens[0].spelling == '+':
                        return Symbol(child)
                    if tokens[0].spelling == '-':
                        return Mul(Symbol(child), -1)
                    if tokens[0].spelling == '++':
                        return PreIncrement(Symbol(child))
                    if tokens[0].spelling == '--':
                        return PreDecrement(Symbol(child))
                    if tokens[0].spelling == '!':
                        return Not(Symbol(child))
                # e.g.; int a = -1; or int b = -(1 + 2);
                else:
                    if tokens[0].spelling == '+':
                        return child
                    if tokens[0].spelling == '-':
                        return Mul(child, -1)
                    if tokens[0].spelling == '!':
                        return Not(sympify(bool(child)))

            # it can be either post increment/decrement
            # since variable name is obtained in token[0].spelling
            elif tokens[1].spelling in ['++', '--']:
                child = self.transform(next(node.get_children()))
                if tokens[1].spelling == '++':
                    return PostIncrement(Symbol(child))
                if tokens[1].spelling == '--':
                    return PostDecrement(Symbol(child))
            else:
                raise NotImplementedError("Dereferencing operator, "
                    "Address operator and bitwise NOT operator "
                    "have not been implemented yet!")

        def transform_binary_operator(self, node):
            """Transformation function for handling binary operators

            Returns
            =======

            binary_expression: Codegen AST node
                    simplified binary expression represented as Codegen AST

            Raises
            ======

            NotImplementedError
                If a bitwise operator or
                unary operator(which is a child of any binary
                operator in Clang AST) is encountered

            """
            # get all the tokens of assignment
            # and store it in the tokens list
            tokens = list(node.get_tokens())

            # supported operators list
            operators_list = ['+', '-', '*', '/', '%','=',
            '>', '>=', '<', '<=', '==', '!=', '&&', '||', '+=', '-=',
            '*=', '/=', '%=']

            # this stack will contain variable content
            # and type of variable in the rhs
            combined_variables_stack = []

            # this stack will contain operators
            # to be processed in the rhs
            operators_stack = []

            # iterate through every token
            for token in tokens:
                # token is either '(', ')' or
                # any of the supported operators from the operator list
                if token.kind == cin.TokenKind.PUNCTUATION:

                    # push '(' to the operators stack
                    if token.spelling == '(':
                        operators_stack.append('(')

                    elif token.spelling == ')':
                        # keep adding the expression to the
                        # combined variables stack unless
                        # '(' is found
                        while (operators_stack
                            and operators_stack[-1] != '('):
                            if len(combined_variables_stack) < 2:
                                raise NotImplementedError(
                                    "Unary operators as a part of "
                                    "binary operators is not "
                                    "supported yet!")
                            rhs = combined_variables_stack.pop()
                            lhs = combined_variables_stack.pop()
                            operator = operators_stack.pop()
                            combined_variables_stack.append(
                                self.perform_operation(
                                lhs, rhs, operator))

                        # pop '('
                        operators_stack.pop()

                    # token is an operator (supported)
                    elif token.spelling in operators_list:
                        while (operators_stack
                            and self.priority_of(token.spelling)
                            <= self.priority_of(
                            operators_stack[-1])):
                            if len(combined_variables_stack) < 2:
                                raise NotImplementedError(
                                    "Unary operators as a part of "
                                    "binary operators is not "
                                    "supported yet!")
                            rhs = combined_variables_stack.pop()
                            lhs = combined_variables_stack.pop()
                            operator = operators_stack.pop()
                            combined_variables_stack.append(
                                self.perform_operation(
                                lhs, rhs, operator))

                        # push current operator
                        operators_stack.append(token.spelling)

                    # token is a bitwise operator
                    elif token.spelling in ['&', '|', '^', '<<', '>>']:
                        raise NotImplementedError(
                            "Bitwise operator has not been "
                            "implemented yet!")

                    # token is a shorthand bitwise operator
                    elif token.spelling in ['&=', '|=', '^=', '<<=',
                    '>>=']:
                        raise NotImplementedError(
                            "Shorthand bitwise operator has not been "
                            "implemented yet!")
                    else:
                        raise NotImplementedError(
                            "Given token {} is not implemented yet!"
                            .format(token.spelling))

                # token is an identifier(variable)
                elif token.kind == cin.TokenKind.IDENTIFIER:
                    combined_variables_stack.append(
                        [token.spelling, 'identifier'])

                # token is a literal
                elif token.kind == cin.TokenKind.LITERAL:
                    combined_variables_stack.append(
                        [token.spelling, 'literal'])

                # token is a keyword, either true or false
                elif (token.kind == cin.TokenKind.KEYWORD
                    and token.spelling in ['true', 'false']):
                    combined_variables_stack.append(
                        [token.spelling, 'boolean'])
                else:
                    raise NotImplementedError(
                        "Given token {} is not implemented yet!"
                        .format(token.spelling))

            # process remaining operators
            while operators_stack:
                if len(combined_variables_stack) < 2:
                    raise NotImplementedError(
                        "Unary operators as a part of "
                        "binary operators is not "
                        "supported yet!")
                rhs = combined_variables_stack.pop()
                lhs = combined_variables_stack.pop()
                operator = operators_stack.pop()
                combined_variables_stack.append(
                    self.perform_operation(lhs, rhs, operator))

            return combined_variables_stack[-1][0]

        def priority_of(self, op):
            """To get the priority of given operator"""
            if op in ['=', '+=', '-=', '*=', '/=', '%=']:
                return 1
            if op in ['&&', '||']:
                return 2
            if op in ['<', '<=', '>', '>=', '==', '!=']:
                return 3
            if op in ['+', '-']:
                return 4
            if op in ['*', '/', '%']:
                return 5
            return 0

        def perform_operation(self, lhs, rhs, op):
            """Performs operation supported by the SymPy core

            Returns
            =======

            combined_variable: list
                contains variable content and type of variable

            """
            lhs_value = self.get_expr_for_operand(lhs)
            rhs_value = self.get_expr_for_operand(rhs)
            if op == '+':
                return [Add(lhs_value, rhs_value), 'expr']
            if op == '-':
                return [Add(lhs_value, -rhs_value), 'expr']
            if op == '*':
                return [Mul(lhs_value, rhs_value), 'expr']
            if op == '/':
                return [Mul(lhs_value, Pow(rhs_value, Integer(-1))), 'expr']
            if op == '%':
                return [Mod(lhs_value, rhs_value), 'expr']
            if op in ['<', '<=', '>', '>=', '==', '!=']:
                return [Rel(lhs_value, rhs_value, op), 'expr']
            if op == '&&':
                return [And(as_Boolean(lhs_value), as_Boolean(rhs_value)), 'expr']
            if op == '||':
                return [Or(as_Boolean(lhs_value), as_Boolean(rhs_value)), 'expr']
            if op == '=':
                return [Assignment(Variable(lhs_value), rhs_value), 'expr']
            if op in ['+=', '-=', '*=', '/=', '%=']:
                return [aug_assign(Variable(lhs_value), op[0], rhs_value), 'expr']

        def get_expr_for_operand(self, combined_variable):
            """Gives out SymPy Codegen AST node

            AST node returned is corresponding to
            combined variable passed.Combined variable contains
            variable content and type of variable

            """
            if combined_variable[1] == 'identifier':
                return Symbol(combined_variable[0])
            if combined_variable[1] == 'literal':
                if '.' in combined_variable[0]:
                    return Float(float(combined_variable[0]))
                else:
                    return Integer(int(combined_variable[0]))
            if combined_variable[1] == 'expr':
                return combined_variable[0]
            if combined_variable[1] == 'boolean':
                    return true if combined_variable[0] == 'true' else false

        def transform_null_stmt(self, node):
            """Handles Null Statement and returns None"""
            return none

        def transform_while_stmt(self, node):
            """Transformation function for handling while statement

            Returns
            =======

            while statement : Codegen AST Node
                contains the while statement node having condition and
                statement block

            """
            children = node.get_children()

            condition = self.transform(next(children))
            statements = self.transform(next(children))

            if isinstance(statements, list):
                statement_block = CodeBlock(*statements)
            else:
                statement_block = CodeBlock(statements)

            return While(condition, statement_block)



else:
    class CCodeConverter():  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("Module not Installed")


def parse_c(source):
    """Function for converting a C source code

    The function reads the source code present in the given file and parses it
    to give out SymPy Expressions

    Returns
    =======

    src : list
        List of Python expression strings

    """
    converter = CCodeConverter()
    if os.path.exists(source):
        src = converter.parse(source, flags = [])
    else:
        src = converter.parse_str(source, flags = [])
    return src
