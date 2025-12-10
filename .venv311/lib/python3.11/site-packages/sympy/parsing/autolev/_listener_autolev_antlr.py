import collections
import warnings

from sympy.external import import_module

autolevparser = import_module('sympy.parsing.autolev._antlr.autolevparser',
                              import_kwargs={'fromlist': ['AutolevParser']})
autolevlexer = import_module('sympy.parsing.autolev._antlr.autolevlexer',
                             import_kwargs={'fromlist': ['AutolevLexer']})
autolevlistener = import_module('sympy.parsing.autolev._antlr.autolevlistener',
                                import_kwargs={'fromlist': ['AutolevListener']})

AutolevParser = getattr(autolevparser, 'AutolevParser', None)
AutolevLexer = getattr(autolevlexer, 'AutolevLexer', None)
AutolevListener = getattr(autolevlistener, 'AutolevListener', None)


def strfunc(z):
    if z == 0:
        return ""
    elif z == 1:
        return "_d"
    else:
        return "_" + "d" * z

def declare_phy_entities(self, ctx, phy_type, i, j=None):
    if phy_type in ("frame", "newtonian"):
        declare_frames(self, ctx, i, j)
    elif phy_type == "particle":
        declare_particles(self, ctx, i, j)
    elif phy_type == "point":
        declare_points(self, ctx, i, j)
    elif phy_type == "bodies":
        declare_bodies(self, ctx, i, j)

def declare_frames(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()
    name2 = "frame_" + name1
    if self.getValue(ctx.parentCtx.varType()) == "newtonian":
        self.newtonian = name2

    self.symbol_table2.update({name1: name2})

    self.symbol_table.update({name1 + "1>": name2 + ".x"})
    self.symbol_table.update({name1 + "2>": name2 + ".y"})
    self.symbol_table.update({name1 + "3>": name2 + ".z"})

    self.type2.update({name1: "frame"})
    self.write(name2 + " = " + "_me.ReferenceFrame('" + name1 + "')\n")

def declare_points(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    name2 = "point_" + name1

    self.symbol_table2.update({name1: name2})
    self.type2.update({name1: "point"})
    self.write(name2 + " = " + "_me.Point('" + name1 + "')\n")

def declare_particles(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    name2 = "particle_" + name1

    self.symbol_table2.update({name1: name2})
    self.type2.update({name1: "particle"})
    self.bodies.update({name1: name2})
    self.write(name2 + " = " + "_me.Particle('" + name1 + "', " + "_me.Point('" +
                name1 + "_pt" + "'), " + "_sm.Symbol('m'))\n")

def declare_bodies(self, ctx, i, j=None):
    if "{" in ctx.getText():
        if j:
            name1 = ctx.ID().getText().lower() + str(i) + str(j)
        else:
            name1 = ctx.ID().getText().lower() + str(i)
    else:
        name1 = ctx.ID().getText().lower()

    name2 = "body_" + name1
    self.bodies.update({name1: name2})
    masscenter = name2 + "_cm"
    refFrame = name2 + "_f"

    self.symbol_table2.update({name1: name2})
    self.symbol_table2.update({name1 + "o": masscenter})
    self.symbol_table.update({name1 + "1>": refFrame+".x"})
    self.symbol_table.update({name1 + "2>": refFrame+".y"})
    self.symbol_table.update({name1 + "3>": refFrame+".z"})

    self.type2.update({name1: "bodies"})
    self.type2.update({name1+"o": "point"})

    self.write(masscenter + " = " + "_me.Point('" + name1 + "_cm" + "')\n")
    if self.newtonian:
        self.write(masscenter + ".set_vel(" + self.newtonian + ", " + "0)\n")
    self.write(refFrame + " = " + "_me.ReferenceFrame('" + name1 + "_f" + "')\n")
    # We set a dummy mass and inertia here.
    # They will be reset using the setters later in the code anyway.
    self.write(name2 + " = " + "_me.RigidBody('" + name1 + "', " + masscenter + ", " +
                refFrame + ", " + "_sm.symbols('m'), (_me.outer(" + refFrame +
                ".x," + refFrame + ".x)," + masscenter + "))\n")

def inertia_func(self, v1, v2, l, frame):

    if self.type2[v1] == "particle":
        l.append("_me.inertia_of_point_mass(" + self.bodies[v1] + ".mass, " + self.bodies[v1] +
                 ".point.pos_from(" + self.symbol_table2[v2] + "), " + frame + ")")

    elif self.type2[v1] == "bodies":
        # Inertia has been defined about center of mass.
        if self.inertia_point[v1] == v1 + "o":
            # Asking point is cm as well
            if v2 == self.inertia_point[v1]:
                l.append(self.symbol_table2[v1] + ".inertia[0]")

            # Asking point is not cm
            else:
                l.append(self.bodies[v1] + ".inertia[0]" + " + " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[v2] +
                         "), " + frame + ")")

        # Inertia has been defined about another point
        else:
            # Asking point is the defined point
            if v2 == self.inertia_point[v1]:
                l.append(self.symbol_table2[v1] + ".inertia[0]")
            # Asking point is cm
            elif v2 == v1 + "o":
                l.append(self.bodies[v1] + ".inertia[0]" + " - " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[self.inertia_point[v1]] +
                         "), " + frame + ")")
            # Asking point is some other point
            else:
                l.append(self.bodies[v1] + ".inertia[0]" + " - " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[self.inertia_point[v1]] +
                         "), " + frame + ")" + " + " +
                         "_me.inertia_of_point_mass(" + self.bodies[v1] +
                         ".mass, " + self.bodies[v1] + ".masscenter" +
                         ".pos_from(" + self.symbol_table2[v2] +
                         "), " + frame + ")")


def processConstants(self, ctx):
    # Process constant declarations of the type: Constants F = 3, g = 9.81
    name = ctx.ID().getText().lower()
    if "=" in ctx.getText():
        self.symbol_table.update({name: name})
        # self.inputs.update({self.symbol_table[name]: self.getValue(ctx.getChild(2))})
        self.write(self.symbol_table[name] + " = " + "_sm.S(" + self.getValue(ctx.getChild(2)) + ")\n")
        self.type.update({name: "constants"})
        return

    # Constants declarations of the type: Constants A, B
    else:
        if "{" not in ctx.getText():
            self.symbol_table[name] = name
            self.type[name] = "constants"

    # Process constant declarations of the type: Constants C+, D-
    if ctx.getChildCount() == 2:
        # This is set for declaring nonpositive=True and nonnegative=True
        if ctx.getChild(1).getText() == "+":
            self.sign[name] = "+"
        elif ctx.getChild(1).getText() == "-":
            self.sign[name] = "-"
    else:
        if "{" not in ctx.getText():
            self.sign[name] = "o"

    # Process constant declarations of the type: Constants K{4}, a{1:2, 1:2}, b{1:2}
    if "{" in ctx.getText():
        if ":" in ctx.getText():
            num1 = int(ctx.INT(0).getText())
            num2 = int(ctx.INT(1).getText()) + 1
        else:
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1

        if ":" in ctx.getText():
            if "," in ctx.getText():
                num3 = int(ctx.INT(2).getText())
                num4 = int(ctx.INT(3).getText()) + 1
                for i in range(num1, num2):
                    for j in range(num3, num4):
                        self.symbol_table[name + str(i) + str(j)] = name + str(i) + str(j)
                        self.type[name + str(i) + str(j)] = "constants"
                        self.var_list.append(name + str(i) + str(j))
                        self.sign[name + str(i) + str(j)] = "o"
            else:
                for i in range(num1, num2):
                    self.symbol_table[name + str(i)] = name + str(i)
                    self.type[name + str(i)] = "constants"
                    self.var_list.append(name + str(i))
                    self.sign[name + str(i)] = "o"

        elif "," in ctx.getText():
            for i in range(1, int(ctx.INT(0).getText()) + 1):
                for j in range(1, int(ctx.INT(1).getText()) + 1):
                    self.symbol_table[name] = name + str(i) + str(j)
                    self.type[name + str(i) + str(j)] = "constants"
                    self.var_list.append(name + str(i) + str(j))
                    self.sign[name + str(i) + str(j)] = "o"

        else:
            for i in range(num1, num2):
                self.symbol_table[name + str(i)] = name + str(i)
                self.type[name + str(i)] = "constants"
                self.var_list.append(name + str(i))
                self.sign[name + str(i)] = "o"

    if "{" not in ctx.getText():
        self.var_list.append(name)


def writeConstants(self, ctx):
    l1 = list(filter(lambda x: self.sign[x] == "o", self.var_list))
    l2 = list(filter(lambda x: self.sign[x] == "+", self.var_list))
    l3 = list(filter(lambda x: self.sign[x] == "-", self.var_list))
    try:
        if self.settings["complex"] == "on":
            real = ", real=True"
        elif self.settings["complex"] == "off":
            real = ""
    except Exception:
        real = ", real=True"

    if l1:
        a = ", ".join(l1) + " = " + "_sm.symbols(" + "'" +\
            " ".join(l1) + "'" + real + ")\n"
        self.write(a)
    if l2:
        a = ", ".join(l2) + " = " + "_sm.symbols(" + "'" +\
            " ".join(l2) + "'" + real + ", nonnegative=True)\n"
        self.write(a)
    if l3:
        a = ", ".join(l3) + " = " + "_sm.symbols(" + "'" + \
            " ".join(l3) + "'" + real + ", nonpositive=True)\n"
        self.write(a)
    self.var_list = []


def processVariables(self, ctx):
    # Specified F = x*N1> + y*N2>
    name = ctx.ID().getText().lower()
    if "=" in ctx.getText():
        text = name + "'"*(ctx.getChildCount()-3)
        self.write(text + " = " + self.getValue(ctx.expr()) + "\n")
        return

    # Process variables of the type: Variables qA, qB
    if ctx.getChildCount() == 1:
        self.symbol_table[name] = name
        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
            self.type.update({name: self.getValue(ctx.parentCtx.getChild(0))})

        self.var_list.append(name)
        self.sign[name] = 0

    # Process variables of the type: Variables x', y''
    elif "'" in ctx.getText() and "{" not in ctx.getText():
        if ctx.getText().count("'") > self.maxDegree:
            self.maxDegree = ctx.getText().count("'")
        for i in range(ctx.getChildCount()):
            self.sign[name + strfunc(i)] = i
            self.symbol_table[name + "'"*i] = name + strfunc(i)
            if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                self.type.update({name + "'"*i: self.getValue(ctx.parentCtx.getChild(0))})
            self.var_list.append(name + strfunc(i))

    elif "{" in ctx.getText():
        # Process variables of the type: Variables x{3}, y{2}

        if "'" in ctx.getText():
            dash_count = ctx.getText().count("'")
            if dash_count > self.maxDegree:
                self.maxDegree = dash_count

        if ":" in ctx.getText():
            # Variables C{1:2, 1:2}
            if "," in ctx.getText():
                num1 = int(ctx.INT(0).getText())
                num2 = int(ctx.INT(1).getText()) + 1
                num3 = int(ctx.INT(2).getText())
                num4 = int(ctx.INT(3).getText()) + 1
            # Variables C{1:2}
            else:
                num1 = int(ctx.INT(0).getText())
                num2 = int(ctx.INT(1).getText()) + 1

        # Variables C{1,3}
        elif "," in ctx.getText():
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1
            num3 = 1
            num4 = int(ctx.INT(1).getText()) + 1
        else:
            num1 = 1
            num2 = int(ctx.INT(0).getText()) + 1

        for i in range(num1, num2):
            try:
                for j in range(num3, num4):
                    try:
                        for z in range(dash_count+1):
                            self.symbol_table.update({name + str(i) + str(j) + "'"*z: name + str(i) + str(j) + strfunc(z)})
                            if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                                self.type.update({name + str(i) + str(j) +  "'"*z: self.getValue(ctx.parentCtx.getChild(0))})
                            self.var_list.append(name + str(i) + str(j) + strfunc(z))
                            self.sign.update({name + str(i) + str(j) + strfunc(z): z})
                            if dash_count > self.maxDegree:
                                self.maxDegree = dash_count
                    except Exception:
                        self.symbol_table.update({name + str(i) + str(j): name + str(i) + str(j)})
                        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                            self.type.update({name + str(i) + str(j): self.getValue(ctx.parentCtx.getChild(0))})
                        self.var_list.append(name + str(i) + str(j))
                        self.sign.update({name + str(i) + str(j): 0})
            except Exception:
                try:
                    for z in range(dash_count+1):
                        self.symbol_table.update({name + str(i) + "'"*z: name + str(i) + strfunc(z)})
                        if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                            self.type.update({name + str(i) +  "'"*z: self.getValue(ctx.parentCtx.getChild(0))})
                        self.var_list.append(name + str(i) + strfunc(z))
                        self.sign.update({name + str(i) + strfunc(z): z})
                        if dash_count > self.maxDegree:
                            self.maxDegree = dash_count
                except Exception:
                    self.symbol_table.update({name + str(i): name + str(i)})
                    if self.getValue(ctx.parentCtx.getChild(0)) in ("variable", "specified", "motionvariable", "motionvariable'"):
                        self.type.update({name + str(i): self.getValue(ctx.parentCtx.getChild(0))})
                    self.var_list.append(name + str(i))
                    self.sign.update({name + str(i): 0})

def writeVariables(self, ctx):
    #print(self.sign)
    #print(self.symbol_table)
    if self.var_list:
        for i in range(self.maxDegree+1):
            if i == 0:
                j = ""
                t = ""
            else:
                j = str(i)
                t = ", "
            l = []
            for k in list(filter(lambda x: self.sign[x] == i, self.var_list)):
                if i == 0:
                    l.append(k)
                if i == 1:
                    l.append(k[:-1])
                if i > 1:
                    l.append(k[:-2])
            a = ", ".join(list(filter(lambda x: self.sign[x] == i, self.var_list))) + " = " +\
                "_me.dynamicsymbols(" + "'" + " ".join(l) + "'" + t + j + ")\n"
            l = []
            self.write(a)
        self.maxDegree = 0
    self.var_list = []

def processImaginary(self, ctx):
    name = ctx.ID().getText().lower()
    self.symbol_table[name] = name
    self.type[name] = "imaginary"
    self.var_list.append(name)


def writeImaginary(self, ctx):
    a = ", ".join(self.var_list) + " = " + "_sm.symbols(" + "'" + \
        " ".join(self.var_list) + "')\n"
    b = ", ".join(self.var_list) + " = " + "_sm.I\n"
    self.write(a)
    self.write(b)
    self.var_list = []

if AutolevListener:
    class MyListener(AutolevListener):  # type: ignore
        def __init__(self, include_numeric=False):
            # Stores data in tree nodes(tree annotation). Especially useful for expr reconstruction.
            self.tree_property = {}

            # Stores the declared variables, constants etc as they are declared in Autolev and SymPy
            # {"<Autolev symbol>": "<SymPy symbol>"}.
            self.symbol_table = collections.OrderedDict()

            # Similar to symbol_table. Used for storing Physical entities like Frames, Points,
            # Particles, Bodies etc
            self.symbol_table2 = collections.OrderedDict()

            # Used to store nonpositive, nonnegative etc for constants and number of "'"s (order of diff)
            # in variables.
            self.sign = {}

            # Simple list used as a store to pass around variables between the 'process' and 'write'
            # methods.
            self.var_list = []

            # Stores the type of a declared variable (constants, variables, specifieds etc)
            self.type = collections.OrderedDict()

            # Similar to self.type. Used for storing the type of Physical entities like Frames, Points,
            # Particles, Bodies etc
            self.type2 = collections.OrderedDict()

            # These lists are used to distinguish matrix, numeric and vector expressions.
            self.matrix_expr = []
            self.numeric_expr = []
            self.vector_expr = []
            self.fr_expr = []

            self.output_code = []

            # Stores the variables and their rhs for substituting upon the Autolev command EXPLICIT.
            self.explicit = collections.OrderedDict()

            # Write code to import common dependencies.
            self.output_code.append("import sympy.physics.mechanics as _me\n")
            self.output_code.append("import sympy as _sm\n")
            self.output_code.append("import math as m\n")
            self.output_code.append("import numpy as _np\n")
            self.output_code.append("\n")

            # Just a store for the max degree variable in a line.
            self.maxDegree = 0

            # Stores the input parameters which are then used for codegen and numerical analysis.
            self.inputs = collections.OrderedDict()
            # Stores the variables which appear in Output Autolev commands.
            self.outputs = []
            # Stores the settings specified by the user. Ex: Complex on/off, Degrees on/off
            self.settings = {}
            # Boolean which changes the behaviour of some expression reconstruction
            # when parsing Input Autolev commands.
            self.in_inputs = False
            self.in_outputs = False

            # Stores for the physical entities.
            self.newtonian = None
            self.bodies = collections.OrderedDict()
            self.constants = []
            self.forces = collections.OrderedDict()
            self.q_ind = []
            self.q_dep = []
            self.u_ind = []
            self.u_dep = []
            self.kd_eqs = []
            self.dependent_variables = []
            self.kd_equivalents = collections.OrderedDict()
            self.kd_equivalents2 = collections.OrderedDict()
            self.kd_eqs_supplied = None
            self.kane_type = "no_args"
            self.inertia_point = collections.OrderedDict()
            self.kane_parsed = False
            self.t = False

            # PyDy ode code will be included only if this flag is set to True.
            self.include_numeric = include_numeric

        def write(self, string):
            self.output_code.append(string)

        def getValue(self, node):
            return self.tree_property[node]

        def setValue(self, node, value):
            self.tree_property[node] = value

        def getSymbolTable(self):
            return self.symbol_table

        def getType(self):
            return self.type

        def exitVarDecl(self, ctx):
            # This event method handles variable declarations. The parse tree node varDecl contains
            # one or more varDecl2 nodes. Eg varDecl for 'Constants a{1:2, 1:2}, b{1:2}' has two varDecl2
            # nodes(one for a{1:2, 1:2} and one for b{1:2}).

            # Variable declarations are processed and stored in the event method exitVarDecl2.
            # This stored information is used to write the final SymPy output code in the exitVarDecl event method.

            # determine the type of declaration
            if self.getValue(ctx.varType()) == "constant":
                writeConstants(self, ctx)
            elif self.getValue(ctx.varType()) in\
            ("variable", "motionvariable", "motionvariable'", "specified"):
                writeVariables(self, ctx)
            elif self.getValue(ctx.varType()) == "imaginary":
                writeImaginary(self, ctx)

        def exitVarType(self, ctx):
            # Annotate the varType tree node with the type of the variable declaration.
            name = ctx.getChild(0).getText().lower()
            if name[-1] == "s" and name != "bodies":
                self.setValue(ctx, name[:-1])
            else:
                self.setValue(ctx, name)

        def exitVarDecl2(self, ctx):
            # Variable declarations are processed and stored in the event method exitVarDecl2.
            # This stored information is used to write the final SymPy output code in the exitVarDecl event method.
            # This is the case for constants, variables, specifieds etc.

            # This isn't the case for all types of declarations though. For instance
            # Frames A, B, C, N cannot be defined on one line in SymPy. So we do not append A, B, C, N
            # to a var_list or use exitVarDecl. exitVarDecl2 directly writes out to the file.

            # determine the type of declaration
            if self.getValue(ctx.parentCtx.varType()) == "constant":
                processConstants(self, ctx)

            elif self.getValue(ctx.parentCtx.varType()) in \
            ("variable", "motionvariable", "motionvariable'", "specified"):
                processVariables(self, ctx)

            elif self.getValue(ctx.parentCtx.varType()) == "imaginary":
                processImaginary(self, ctx)

            elif self.getValue(ctx.parentCtx.varType()) in ("frame", "newtonian", "point", "particle", "bodies"):
                if "{" in ctx.getText():
                    if ":" in ctx.getText() and "," not in ctx.getText():
                        num1 = int(ctx.INT(0).getText())
                        num2 = int(ctx.INT(1).getText()) + 1
                    elif ":" not in ctx.getText() and "," in ctx.getText():
                        num1 = 1
                        num2 = int(ctx.INT(0).getText()) + 1
                        num3 = 1
                        num4 = int(ctx.INT(1).getText()) + 1
                    elif ":" in ctx.getText() and "," in ctx.getText():
                        num1 = int(ctx.INT(0).getText())
                        num2 = int(ctx.INT(1).getText()) + 1
                        num3 = int(ctx.INT(2).getText())
                        num4 = int(ctx.INT(3).getText()) + 1
                    else:
                        num1 = 1
                        num2 = int(ctx.INT(0).getText()) + 1
                else:
                    num1 = 1
                    num2 = 2
                for i in range(num1, num2):
                    try:
                        for j in range(num3, num4):
                            declare_phy_entities(self, ctx, self.getValue(ctx.parentCtx.varType()), i, j)
                    except Exception:
                        declare_phy_entities(self, ctx, self.getValue(ctx.parentCtx.varType()), i)
        # ================== Subrules of parser rule expr (Start) ====================== #

        def exitId(self, ctx):
            # Tree annotation for ID which is a labeled subrule of the parser rule expr.
            # A_C
            python_keywords = ["and", "as", "assert", "break", "class", "continue", "def", "del", "elif", "else", "except",\
            "exec", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "not", "or", "pass", "print",\
            "raise", "return", "try", "while", "with", "yield"]

            if ctx.ID().getText().lower() in python_keywords:
                warnings.warn("Python keywords must not be used as identifiers. Please refer to the list of keywords at https://docs.python.org/2.5/ref/keywords.html",
                SyntaxWarning)

            if "_" in ctx.ID().getText() and ctx.ID().getText().count('_') == 1:
                e1, e2 = ctx.ID().getText().lower().split('_')
                try:
                    if self.type2[e1] == "frame":
                        e1 = self.symbol_table2[e1]
                    elif self.type2[e1] == "bodies":
                        e1 = self.symbol_table2[e1] + "_f"
                    if self.type2[e2] == "frame":
                        e2 = self.symbol_table2[e2]
                    elif self.type2[e2] == "bodies":
                        e2 = self.symbol_table2[e2] + "_f"

                    self.setValue(ctx, e1 + ".dcm(" + e2 + ")")
                except Exception:
                    self.setValue(ctx, ctx.ID().getText().lower())
            else:
                # Reserved constant Pi
                if ctx.ID().getText().lower() == "pi":
                    self.setValue(ctx, "_sm.pi")
                    self.numeric_expr.append(ctx)

                # Reserved variable T (for time)
                elif ctx.ID().getText().lower() == "t":
                    self.setValue(ctx, "_me.dynamicsymbols._t")
                    if not self.in_inputs and not self.in_outputs:
                        self.t = True

                else:
                    idText = ctx.ID().getText().lower() + "'"*(ctx.getChildCount() - 1)
                    if idText in self.type.keys() and self.type[idText] == "matrix":
                        self.matrix_expr.append(ctx)
                    if self.in_inputs:
                        try:
                            self.setValue(ctx, self.symbol_table[idText])
                        except Exception:
                            self.setValue(ctx, idText.lower())
                    else:
                        try:
                            self.setValue(ctx, self.symbol_table[idText])
                        except Exception:
                            pass

        def exitInt(self, ctx):
            # Tree annotation for int which is a labeled subrule of the parser rule expr.
            int_text = ctx.INT().getText()
            self.setValue(ctx, int_text)
            self.numeric_expr.append(ctx)

        def exitFloat(self, ctx):
            # Tree annotation for float which is a labeled subrule of the parser rule expr.
            floatText = ctx.FLOAT().getText()
            self.setValue(ctx, floatText)
            self.numeric_expr.append(ctx)

        def exitAddSub(self, ctx):
            # Tree annotation for AddSub which is a labeled subrule of the parser rule expr.
            # The subrule is expr = expr (+|-) expr
            if ctx.expr(0) in self.matrix_expr or ctx.expr(1) in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.expr(0) in self.vector_expr or ctx.expr(1) in self.vector_expr:
                self.vector_expr.append(ctx)
            if ctx.expr(0) in self.numeric_expr and ctx.expr(1) in self.numeric_expr:
                self.numeric_expr.append(ctx)
            self.setValue(ctx, self.getValue(ctx.expr(0)) + ctx.getChild(1).getText() +
                          self.getValue(ctx.expr(1)))

        def exitMulDiv(self, ctx):
            # Tree annotation for MulDiv which is a labeled subrule of the parser rule expr.
            # The subrule is expr = expr (*|/) expr
            try:
                if ctx.expr(0) in self.vector_expr and ctx.expr(1) in self.vector_expr:
                    self.setValue(ctx, "_me.outer(" + self.getValue(ctx.expr(0)) + ", " +
                                  self.getValue(ctx.expr(1)) + ")")
                else:
                    if ctx.expr(0) in self.matrix_expr or ctx.expr(1) in self.matrix_expr:
                        self.matrix_expr.append(ctx)
                    if ctx.expr(0) in self.vector_expr or ctx.expr(1) in self.vector_expr:
                        self.vector_expr.append(ctx)
                    if ctx.expr(0) in self.numeric_expr and ctx.expr(1) in self.numeric_expr:
                        self.numeric_expr.append(ctx)
                    self.setValue(ctx, self.getValue(ctx.expr(0)) + ctx.getChild(1).getText() +
                                  self.getValue(ctx.expr(1)))
            except Exception:
                pass

        def exitNegativeOne(self, ctx):
            # Tree annotation for negativeOne which is a labeled subrule of the parser rule expr.
            self.setValue(ctx, "-1*" + self.getValue(ctx.getChild(1)))
            if ctx.getChild(1) in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.getChild(1) in self.numeric_expr:
                self.numeric_expr.append(ctx)

        def exitParens(self, ctx):
            # Tree annotation for parens which is a labeled subrule of the parser rule expr.
            # The subrule is expr = '(' expr ')'
            if ctx.expr() in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.expr() in self.vector_expr:
                self.vector_expr.append(ctx)
            if ctx.expr() in self.numeric_expr:
                self.numeric_expr.append(ctx)
            self.setValue(ctx, "(" + self.getValue(ctx.expr()) + ")")

        def exitExponent(self, ctx):
            # Tree annotation for Exponent which is a labeled subrule of the parser rule expr.
            # The subrule is expr = expr ^ expr
            if ctx.expr(0) in self.matrix_expr or ctx.expr(1) in self.matrix_expr:
                self.matrix_expr.append(ctx)
            if ctx.expr(0) in self.vector_expr or ctx.expr(1) in self.vector_expr:
                self.vector_expr.append(ctx)
            if ctx.expr(0) in self.numeric_expr and ctx.expr(1) in self.numeric_expr:
                self.numeric_expr.append(ctx)
            self.setValue(ctx, self.getValue(ctx.expr(0)) + "**" + self.getValue(ctx.expr(1)))

        def exitExp(self, ctx):
            s = ctx.EXP().getText()[ctx.EXP().getText().index('E')+1:]
            if "-" in s:
                s = s[0] + s[1:].lstrip("0")
            else:
                s = s.lstrip("0")
            self.setValue(ctx, ctx.EXP().getText()[:ctx.EXP().getText().index('E')] +
                          "*10**(" + s + ")")

        def exitFunction(self, ctx):
            # Tree annotation for function which is a labeled subrule of the parser rule expr.

            # The difference between this and FunctionCall is that this is used for non standalone functions
            # appearing in expressions and assignments.
            # Eg:
            # When we come across a standalone function say Expand(E, n:m) then it is categorized as FunctionCall
            # which is a parser rule in itself under rule stat. exitFunctionCall() takes care of it and writes to the file.
            #
            # On the other hand, while we come across E_diff = D(E, y), we annotate the tree node
            # of the function D(E, y) with the SymPy equivalent in exitFunction().
            # In this case it is the method exitAssignment() that writes the code to the file and not exitFunction().

            ch = ctx.getChild(0)
            func_name = ch.getChild(0).getText().lower()

            # Expand(y, n:m) *
            if func_name == "expand":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    # _sm.Matrix([i.expand() for i in z]).reshape(z.shape[0], z.shape[1])
                    self.setValue(ctx, "_sm.Matrix([i.expand() for i in " + expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + expr + ")" + "." + "expand()")

            # Factor(y, x) *
            elif func_name == "factor":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([_sm.factor(i, " + self.getValue(ch.expr(1)) + ") for i in " +
                                  expr + "])" + ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "_sm.factor(" + "(" + expr + ")" +
                                  ", " + self.getValue(ch.expr(1)) + ")")

            # D(y, x)
            elif func_name == "d":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.diff(" + self.getValue(ch.expr(1)) + ") for i in " +
                                  expr + "])" + ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    if ch.getChildCount() == 8:
                        frame = self.symbol_table2[ch.expr(2).getText().lower()]
                        self.setValue(ctx, "(" + expr + ")" + "." + "diff(" + self.getValue(ch.expr(1)) +
                                      ", " + frame + ")")
                    else:
                        self.setValue(ctx, "(" + expr + ")" + "." + "diff(" +
                                      self.getValue(ch.expr(1)) + ")")

            # Dt(y)
            elif func_name == "dt":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.vector_expr:
                    text = "dt("
                else:
                    text = "diff(_sm.Symbol('t')"
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i." + text +
                                  ") for i in " + expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    if ch.getChildCount() == 6:
                        frame = self.symbol_table2[ch.expr(1).getText().lower()]
                        self.setValue(ctx, "(" + expr + ")" + "." + "dt(" +
                                      frame + ")")
                    else:
                        self.setValue(ctx, "(" + expr + ")" + "." + text + ")")

            # Explicit(EXPRESS(IMPLICIT>,C))
            elif func_name == "explicit":
                if ch.expr(0) in self.vector_expr:
                    self.vector_expr.append(ctx)
                expr = self.getValue(ch.expr(0))
                if self.explicit.keys():
                    explicit_list = []
                    for i in self.explicit.keys():
                        explicit_list.append(i + ":" + self.explicit[i])
                    self.setValue(ctx, "(" + expr + ")" + ".subs({" + ", ".join(explicit_list) + "})")
                else:
                    self.setValue(ctx, expr)

            # Taylor(y, 0:2, w=a, x=0)
            # TODO: Currently only works with symbols. Make it work for dynamicsymbols.
            elif func_name == "taylor":
                exp = self.getValue(ch.expr(0))
                order = self.getValue(ch.expr(1).expr(1))
                x = (ch.getChildCount()-6)//2
                l = []
                for i in range(x):
                    index = 2 + i
                    child = ch.expr(index)
                    l.append(".series(" + self.getValue(child.getChild(0)) +
                             ", " + self.getValue(child.getChild(2)) +
                             ", " + order + ").removeO()")
                self.setValue(ctx, "(" + exp + ")" + "".join(l))

            # Evaluate(y, a=x, b=2)
            elif func_name == "evaluate":
                expr = self.getValue(ch.expr(0))
                l = []
                x = (ch.getChildCount()-4)//2
                for i in range(x):
                    index = 1 + i
                    child = ch.expr(index)
                    l.append(self.getValue(child.getChild(0)) + ":" +
                             self.getValue(child.getChild(2)))

                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.subs({" + ",".join(l) + "}) for i in " +
                                  expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    if self.explicit:
                        explicit_list = []
                        for i in self.explicit.keys():
                            explicit_list.append(i + ":" + self.explicit[i])
                        self.setValue(ctx, "(" + expr + ")" + ".subs({" + ",".join(explicit_list) +
                                      "}).subs({" + ",".join(l) + "})")
                    else:
                        self.setValue(ctx, "(" + expr + ")" + ".subs({" + ",".join(l) + "})")

            # Polynomial([a, b, c], x)
            elif func_name == "polynomial":
                self.setValue(ctx, "_sm.Poly(" + self.getValue(ch.expr(0)) + ", " +
                              self.getValue(ch.expr(1)) + ")")

            # Roots(Poly, x, 2)
            # Roots([1; 2; 3; 4])
            elif func_name == "roots":
                self.matrix_expr.append(ctx)
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.setValue(ctx, "[i.evalf() for i in " + "_sm.solve(" +
                                  "_sm.Poly(" + expr + ", " + "x),x)]")
                else:
                    self.setValue(ctx, "[i.evalf() for i in " + "_sm.solve(" +
                                  expr + ", " + self.getValue(ch.expr(1)) + ")]")

            # Transpose(A), Inv(A)
            elif func_name in ("transpose", "inv", "inverse"):
                self.matrix_expr.append(ctx)
                if func_name == "transpose":
                    e = ".T"
                elif func_name in ("inv", "inverse"):
                    e = "**(-1)"
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + e)

            # Eig(A)
            elif func_name == "eig":
                # "_sm.Matrix([i.evalf() for i in " +
                self.setValue(ctx, "_sm.Matrix([i.evalf() for i in (" +
                              self.getValue(ch.expr(0)) + ").eigenvals().keys()])")

            # Diagmat(n, m, x)
            # Diagmat(3, 1)
            elif func_name == "diagmat":
                self.matrix_expr.append(ctx)
                if ch.getChildCount() == 6:
                    l = []
                    for i in range(int(self.getValue(ch.expr(0)))):
                        l.append(self.getValue(ch.expr(1)) + ",")

                    self.setValue(ctx, "_sm.diag(" + ("".join(l))[:-1] + ")")

                elif ch.getChildCount() == 8:
                    # _sm.Matrix([x if i==j else 0 for i in range(n) for j in range(m)]).reshape(n, m)
                    n = self.getValue(ch.expr(0))
                    m = self.getValue(ch.expr(1))
                    x = self.getValue(ch.expr(2))
                    self.setValue(ctx, "_sm.Matrix([" + x + " if i==j else 0 for i in range(" +
                                  n + ") for j in range(" + m + ")]).reshape(" + n + ", " + m + ")")

            # Cols(A)
            # Cols(A, 1)
            # Cols(A, 1, 2:4, 3)
            elif func_name in ("cols", "rows"):
                self.matrix_expr.append(ctx)
                if func_name == "cols":
                    e1 = ".cols"
                    e2 = ".T."
                else:
                    e1 = ".rows"
                    e2 = "."
                if ch.getChildCount() == 4:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + e1)
                elif ch.getChildCount() == 6:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" +
                                  e1[:-1] + "(" + str(int(self.getValue(ch.expr(1))) - 1) + ")")
                else:
                    l = []
                    for i in range(4, ch.getChildCount()):
                        try:
                            if ch.getChild(i).getChildCount() > 1 and ch.getChild(i).getChild(1).getText() == ":":
                                for j in range(int(ch.getChild(i).getChild(0).getText()),
                                int(ch.getChild(i).getChild(2).getText())+1):
                                    l.append("(" + self.getValue(ch.getChild(2)) + ")" + e2 +
                                             "row(" + str(j-1) + ")")
                            else:
                                l.append("(" + self.getValue(ch.getChild(2)) + ")" + e2 +
                                         "row(" + str(int(ch.getChild(i).getText())-1) + ")")
                        except Exception:
                            pass
                    self.setValue(ctx, "_sm.Matrix([" + ",".join(l) + "])")

            # Det(A) Trace(A)
            elif func_name in ["det", "trace"]:
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + "." +
                              func_name + "()")

            # Element(A, 2, 3)
            elif func_name == "element":
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + "[" +
                              str(int(self.getValue(ch.expr(1)))-1) + "," +
                              str(int(self.getValue(ch.expr(2)))-1) + "]")

            elif func_name in \
            ["cos", "sin", "tan", "cosh", "sinh", "tanh", "acos", "asin", "atan",
            "log", "exp", "sqrt", "factorial", "floor", "sign"]:
                self.setValue(ctx, "_sm." + func_name + "(" + self.getValue(ch.expr(0)) + ")")

            elif func_name == "ceil":
                self.setValue(ctx, "_sm.ceiling" + "(" + self.getValue(ch.expr(0)) + ")")

            elif func_name == "sqr":
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) +
                              ")" + "**2")

            elif func_name == "log10":
                self.setValue(ctx, "_sm.log" +
                              "(" + self.getValue(ch.expr(0)) + ", 10)")

            elif func_name == "atan2":
                self.setValue(ctx, "_sm.atan2" + "(" + self.getValue(ch.expr(0)) + ", " +
                              self.getValue(ch.expr(1)) + ")")

            elif func_name in ["int", "round"]:
                self.setValue(ctx, func_name +
                              "(" + self.getValue(ch.expr(0)) + ")")

            elif func_name == "abs":
                self.setValue(ctx, "_sm.Abs(" + self.getValue(ch.expr(0)) + ")")

            elif func_name in ["max", "min"]:
                # max(x, y, z)
                l = []
                for i in range(1, ch.getChildCount()):
                    if ch.getChild(i) in self.tree_property.keys():
                        l.append(self.getValue(ch.getChild(i)))
                    elif ch.getChild(i).getText() in [",", "(", ")"]:
                        l.append(ch.getChild(i).getText())
                self.setValue(ctx, "_sm." + ch.getChild(0).getText().capitalize() + "".join(l))

            # Coef(y, x)
            elif func_name == "coef":
                #A41_A53=COEF([RHS(U4);RHS(U5)],[U1,U2,U3])
                if ch.expr(0) in self.matrix_expr and ch.expr(1) in self.matrix_expr:
                    icount = jcount = 0
                    for i in range(ch.expr(0).getChild(0).getChildCount()):
                        try:
                            ch.expr(0).getChild(0).getChild(i).getRuleIndex()
                            icount+=1
                        except Exception:
                            pass
                    for j in range(ch.expr(1).getChild(0).getChildCount()):
                        try:
                            ch.expr(1).getChild(0).getChild(j).getRuleIndex()
                            jcount+=1
                        except Exception:
                            pass
                    l = []
                    for i in range(icount):
                        for j in range(jcount):
                            # a41_a53[i,j] = u4.expand().coeff(u1)
                            l.append(self.getValue(ch.expr(0).getChild(0).expr(i)) + ".expand().coeff("
                                     + self.getValue(ch.expr(1).getChild(0).expr(j)) + ")")
                    self.setValue(ctx, "_sm.Matrix([" + ", ".join(l) + "]).reshape(" + str(icount) + ", " + str(jcount) + ")")
                else:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) +
                                  ")" + ".expand().coeff(" + self.getValue(ch.expr(1)) + ")")

            # Exclude(y, x) Include(y, x)
            elif func_name in ("exclude", "include"):
                if func_name == "exclude":
                    e = "0"
                else:
                    e = "1"
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.collect(" + self.getValue(ch.expr(1)) + "])" +
                                  ".coeff(" + self.getValue(ch.expr(1)) + "," + e + ")" + "for i in " + expr + ")" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + expr +
                                  ")" + ".collect(" + self.getValue(ch.expr(1)) + ")" +
                                  ".coeff(" + self.getValue(ch.expr(1)) + "," + e + ")")

            # RHS(y)
            elif func_name == "rhs":
                self.setValue(ctx, self.explicit[self.getValue(ch.expr(0))])

            # Arrange(y, n, x) *
            elif func_name == "arrange":
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.collect(" + self.getValue(ch.expr(2)) +
                                  ")" + "for i in " + expr + "])"+
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + expr +
                                  ")" + ".collect(" + self.getValue(ch.expr(2)) + ")")

            # Replace(y, sin(x)=3)
            elif func_name == "replace":
                l = []
                for i in range(1, ch.getChildCount()):
                    try:
                        if ch.getChild(i).getChild(1).getText() == "=":
                            l.append(self.getValue(ch.getChild(i).getChild(0)) +
                                     ":" + self.getValue(ch.getChild(i).getChild(2)))
                    except Exception:
                        pass
                expr = self.getValue(ch.expr(0))
                if ch.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                    self.matrix_expr.append(ctx)
                    self.setValue(ctx, "_sm.Matrix([i.subs({" + ",".join(l) + "}) for i in " +
                                  expr + "])" +
                                  ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])")
                else:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" +
                                  ".subs({" + ",".join(l) + "})")

            # Dot(Loop>, N1>)
            elif func_name == "dot":
                l = []
                num = (ch.expr(1).getChild(0).getChildCount()-1)//2
                if ch.expr(1) in self.matrix_expr:
                    for i in range(num):
                        l.append("_me.dot(" + self.getValue(ch.expr(0)) + ", " + self.getValue(ch.expr(1).getChild(0).expr(i)) + ")")
                    self.setValue(ctx, "_sm.Matrix([" + ",".join(l) + "]).reshape(" + str(num) + ", " + "1)")
                else:
                    self.setValue(ctx, "_me.dot(" + self.getValue(ch.expr(0)) + ", " + self.getValue(ch.expr(1)) + ")")
            # Cross(w_A_N>, P_NA_AB>)
            elif func_name == "cross":
                self.vector_expr.append(ctx)
                self.setValue(ctx, "_me.cross(" + self.getValue(ch.expr(0)) + ", " + self.getValue(ch.expr(1)) + ")")

            # Mag(P_O_Q>)
            elif func_name == "mag":
                self.setValue(ctx, self.getValue(ch.expr(0)) + "." + "magnitude()")

            # MATRIX(A, I_R>>)
            elif func_name == "matrix":
                if self.type2[ch.expr(0).getText().lower()] == "frame":
                    text = ""
                elif self.type2[ch.expr(0).getText().lower()] == "bodies":
                    text = "_f"
                self.setValue(ctx, "(" + self.getValue(ch.expr(1)) + ")" + ".to_matrix(" +
                              self.symbol_table2[ch.expr(0).getText().lower()] + text + ")")

            # VECTOR(A, ROWS(EIGVECS,1))
            elif func_name == "vector":
                if self.type2[ch.expr(0).getText().lower()] == "frame":
                    text = ""
                elif self.type2[ch.expr(0).getText().lower()] == "bodies":
                    text = "_f"
                v = self.getValue(ch.expr(1))
                f = self.symbol_table2[ch.expr(0).getText().lower()] + text
                self.setValue(ctx, v + "[0]*" + f + ".x +" + v + "[1]*" + f + ".y +" +
                              v + "[2]*" + f + ".z")

            # Express(A2>, B)
            # Here I am dealing with all the Inertia commands as I expect the users to use Inertia
            # commands only with Express because SymPy needs the Reference frame to be specified unlike Autolev.
            elif func_name == "express":
                self.vector_expr.append(ctx)
                if self.type2[ch.expr(1).getText().lower()] == "frame":
                    frame = self.symbol_table2[ch.expr(1).getText().lower()]
                else:
                    frame = self.symbol_table2[ch.expr(1).getText().lower()] + "_f"
                if ch.expr(0).getText().lower() == "1>>":
                    self.setValue(ctx, "_me.inertia(" + frame + ", 1, 1, 1)")

                elif '_' in ch.expr(0).getText().lower() and ch.expr(0).getText().lower().count('_') == 2\
                and ch.expr(0).getText().lower()[0] == "i" and ch.expr(0).getText().lower()[-2:] == ">>":
                    v1 = ch.expr(0).getText().lower()[:-2].split('_')[1]
                    v2 = ch.expr(0).getText().lower()[:-2].split('_')[2]
                    l = []
                    inertia_func(self, v1, v2, l, frame)
                    self.setValue(ctx, " + ".join(l))

                elif ch.expr(0).getChild(0).getChild(0).getText().lower() == "inertia":
                    if ch.expr(0).getChild(0).getChildCount() == 4:
                        l = []
                        v2 = ch.expr(0).getChild(0).ID(0).getText().lower()
                        for v1 in self.bodies:
                            inertia_func(self, v1, v2, l, frame)
                        self.setValue(ctx, " + ".join(l))

                    else:
                        l = []
                        l2 = []
                        v2 = ch.expr(0).getChild(0).ID(0).getText().lower()
                        for i in range(1, (ch.expr(0).getChild(0).getChildCount()-2)//2):
                            l2.append(ch.expr(0).getChild(0).ID(i).getText().lower())
                        for v1 in l2:
                            inertia_func(self, v1, v2, l, frame)
                        self.setValue(ctx, " + ".join(l))

                else:
                    self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + ".express(" +
                                  self.symbol_table2[ch.expr(1).getText().lower()] + ")")
            # CM(P)
            elif func_name == "cm":
                if self.type2[ch.expr(0).getText().lower()] == "point":
                    text = ""
                else:
                    text = ".point"
                if ch.getChildCount() == 4:
                    self.setValue(ctx, "_me.functions.center_of_mass(" + self.symbol_table2[ch.expr(0).getText().lower()] +
                                  text + "," + ", ".join(self.bodies.values()) + ")")
                else:
                    bodies = []
                    for i in range(1, (ch.getChildCount()-1)//2):
                        bodies.append(self.symbol_table2[ch.expr(i).getText().lower()])
                    self.setValue(ctx, "_me.functions.center_of_mass(" + self.symbol_table2[ch.expr(0).getText().lower()] +
                                  text + "," + ", ".join(bodies) + ")")

            # PARTIALS(V_P1_E>,U1)
            elif func_name == "partials":
                speeds = []
                for i in range(1, (ch.getChildCount()-1)//2):
                    if self.kd_equivalents2:
                        speeds.append(self.kd_equivalents2[self.symbol_table[ch.expr(i).getText().lower()]])
                    else:
                        speeds.append(self.symbol_table[ch.expr(i).getText().lower()])
                v1, v2, v3 = ch.expr(0).getText().lower().replace(">","").split('_')
                if self.type2[v2] == "point":
                    point = self.symbol_table2[v2]
                elif self.type2[v2] == "particle":
                    point = self.symbol_table2[v2] + ".point"
                frame = self.symbol_table2[v3]
                self.setValue(ctx, point + ".partial_velocity(" + frame + ", " + ",".join(speeds) + ")")

            # UnitVec(A1>+A2>+A3>)
            elif func_name == "unitvec":
                self.setValue(ctx, "(" + self.getValue(ch.expr(0)) + ")" + ".normalize()")

            # Units(deg, rad)
            elif func_name == "units":
                if ch.expr(0).getText().lower() == "deg" and ch.expr(1).getText().lower() == "rad":
                    factor = 0.0174533
                elif ch.expr(0).getText().lower() == "rad" and ch.expr(1).getText().lower() == "deg":
                    factor = 57.2958
                self.setValue(ctx, str(factor))
            # Mass(A)
            elif func_name == "mass":
                l = []
                try:
                    ch.ID(0).getText().lower()
                    for i in range((ch.getChildCount()-1)//2):
                        l.append(self.symbol_table2[ch.ID(i).getText().lower()] + ".mass")
                    self.setValue(ctx, "+".join(l))
                except Exception:
                    for i in self.bodies.keys():
                        l.append(self.bodies[i] + ".mass")
                    self.setValue(ctx, "+".join(l))

            # Fr() FrStar()
            # _me.KanesMethod(n, q_ind, u_ind, kd, velocity_constraints).kanes_equations(pl, fl)[0]
            elif func_name in ["fr", "frstar"]:
                if not self.kane_parsed:
                    if self.kd_eqs:
                        for i in self.kd_eqs:
                            self.q_ind.append(self.symbol_table[i.strip().split('-')[0].replace("'","")])
                            self.u_ind.append(self.symbol_table[i.strip().split('-')[1].replace("'","")])

                    for i in range(len(self.kd_eqs)):
                        self.kd_eqs[i] = self.symbol_table[self.kd_eqs[i].strip().split('-')[0]] + " - " +\
                                         self.symbol_table[self.kd_eqs[i].strip().split('-')[1]]

                    # Do all of this if kd_eqs are not specified
                    if not self.kd_eqs:
                        self.kd_eqs_supplied = False
                        self.matrix_expr.append(ctx)
                        for i in self.type.keys():
                            if self.type[i] == "motionvariable":
                                if self.sign[self.symbol_table[i.lower()]] == 0:
                                    self.q_ind.append(self.symbol_table[i.lower()])
                                elif self.sign[self.symbol_table[i.lower()]] == 1:
                                    name = "u_" + self.symbol_table[i.lower()]
                                    self.symbol_table.update({name: name})
                                    self.write(name + " = " + "_me.dynamicsymbols('" + name + "')\n")
                                    if self.symbol_table[i.lower()] not in self.dependent_variables:
                                        self.u_ind.append(name)
                                        self.kd_equivalents.update({name: self.symbol_table[i.lower()]})
                                    else:
                                        self.u_dep.append(name)
                                        self.kd_equivalents.update({name: self.symbol_table[i.lower()]})

                        for i in self.kd_equivalents.keys():
                            self.kd_eqs.append(self.kd_equivalents[i] + "-" + i)

                    if not self.u_ind and not self.kd_eqs:
                        self.u_ind = self.q_ind.copy()
                        self.q_ind = []

                # deal with velocity constraints
                if self.dependent_variables:
                    for i in self.dependent_variables:
                        self.u_dep.append(i)
                        if i in self.u_ind:
                            self.u_ind.remove(i)


                self.u_dep[:] = [i for i in self.u_dep if i not in self.kd_equivalents.values()]

                force_list = []
                for i in self.forces.keys():
                    force_list.append("(" + i + "," + self.forces[i] + ")")
                if self.u_dep:
                    u_dep_text = ", u_dependent=[" + ", ".join(self.u_dep) + "]"
                else:
                    u_dep_text = ""
                if self.dependent_variables:
                    velocity_constraints_text = ", velocity_constraints = velocity_constraints"
                else:
                    velocity_constraints_text = ""
                if ctx.parentCtx not in self.fr_expr:
                    self.write("kd_eqs = [" + ", ".join(self.kd_eqs) + "]\n")
                    self.write("forceList = " + "[" + ", ".join(force_list) + "]\n")
                    self.write("kane = _me.KanesMethod(" + self.newtonian + ", " + "q_ind=[" +
                            ",".join(self.q_ind) + "], " + "u_ind=[" +
                            ", ".join(self.u_ind) + "]" + u_dep_text + ", " +
                            "kd_eqs = kd_eqs" + velocity_constraints_text + ")\n")
                    self.write("fr, frstar = kane." + "kanes_equations([" +
                                ", ".join(self.bodies.values()) + "], forceList)\n")
                    self.fr_expr.append(ctx.parentCtx)
                self.kane_parsed = True
                self.setValue(ctx, func_name)

        def exitMatrices(self, ctx):
            # Tree annotation for Matrices which is a labeled subrule of the parser rule expr.

            # MO = [a, b; c, d]
            # we generate _sm.Matrix([a, b, c, d]).reshape(2, 2)
            # The reshape values are determined by counting the "," and ";" in the Autolev matrix

            # Eg:
            # [1, 2, 3; 4, 5, 6; 7, 8, 9; 10, 11, 12]
            # semicolon_count = 3 and rows = 3+1 = 4
            # comma_count = 8 and cols = 8/rows + 1 = 8/4 + 1 = 3

            # TODO** Parse block matrices
            self.matrix_expr.append(ctx)
            l = []
            semicolon_count = 0
            comma_count = 0
            for i in range(ctx.matrix().getChildCount()):
                child = ctx.matrix().getChild(i)
                if child == AutolevParser.ExprContext:
                    l.append(self.getValue(child))
                elif child.getText() == ";":
                    semicolon_count += 1
                    l.append(",")
                elif child.getText() == ",":
                    comma_count += 1
                    l.append(",")
                else:
                    try:
                        try:
                            l.append(self.getValue(child))
                        except Exception:
                            l.append(self.symbol_table[child.getText().lower()])
                    except Exception:
                        l.append(child.getText().lower())
            num_of_rows = semicolon_count + 1
            num_of_cols = (comma_count//num_of_rows) + 1

            self.setValue(ctx, "_sm.Matrix(" + "".join(l) + ")" + ".reshape(" +
                          str(num_of_rows) + ", " + str(num_of_cols) + ")")

        def exitVectorOrDyadic(self, ctx):
            self.vector_expr.append(ctx)
            ch = ctx.vec()

            if ch.getChild(0).getText() == "0>":
                self.setValue(ctx, "0")

            elif ch.getChild(0).getText() == "1>>":
                self.setValue(ctx, "1>>")

            elif "_" in ch.ID().getText() and ch.ID().getText().count('_') == 2:
                vec_text = ch.getText().lower()
                v1, v2, v3 = ch.ID().getText().lower().split('_')

                if v1 == "p":
                    if self.type2[v2] == "point":
                        e2 = self.symbol_table2[v2]
                    elif self.type2[v2] == "particle":
                        e2 = self.symbol_table2[v2] + ".point"
                    if self.type2[v3] == "point":
                        e3 = self.symbol_table2[v3]
                    elif self.type2[v3] == "particle":
                        e3 = self.symbol_table2[v3] + ".point"
                    get_vec = e3 + ".pos_from(" + e2 + ")"
                    self.setValue(ctx, get_vec)

                elif v1 in ("w", "alf"):
                    if v1 == "w":
                        text = ".ang_vel_in("
                    elif v1 == "alf":
                        text = ".ang_acc_in("
                    if self.type2[v2] == "bodies":
                        e2 = self.symbol_table2[v2] + "_f"
                    elif self.type2[v2] == "frame":
                        e2 = self.symbol_table2[v2]
                    if self.type2[v3] == "bodies":
                        e3 = self.symbol_table2[v3] + "_f"
                    elif self.type2[v3] == "frame":
                        e3 = self.symbol_table2[v3]
                    get_vec = e2 + text + e3 + ")"
                    self.setValue(ctx, get_vec)

                elif v1 in ("v", "a"):
                    if v1 == "v":
                        text = ".vel("
                    elif v1 == "a":
                        text = ".acc("
                    if self.type2[v2] == "point":
                        e2 = self.symbol_table2[v2]
                    elif self.type2[v2] == "particle":
                        e2 = self.symbol_table2[v2] + ".point"
                    get_vec = e2 + text + self.symbol_table2[v3] + ")"
                    self.setValue(ctx, get_vec)

                else:
                    self.setValue(ctx, vec_text.replace(">", ""))

            else:
                vec_text = ch.getText().lower()
                name = self.symbol_table[vec_text]
                self.setValue(ctx, name)

        def exitIndexing(self, ctx):
            if ctx.getChildCount() == 4:
                try:
                    int_text = str(int(self.getValue(ctx.getChild(2))) - 1)
                except Exception:
                    int_text = self.getValue(ctx.getChild(2)) + " - 1"
                self.setValue(ctx, ctx.ID().getText().lower() + "[" + int_text + "]")
            elif ctx.getChildCount() == 6:
                try:
                    int_text1 = str(int(self.getValue(ctx.getChild(2))) - 1)
                except Exception:
                    int_text1 = self.getValue(ctx.getChild(2)) + " - 1"
                try:
                    int_text2 = str(int(self.getValue(ctx.getChild(4))) - 1)
                except Exception:
                    int_text2 = self.getValue(ctx.getChild(2)) + " - 1"
                self.setValue(ctx, ctx.ID().getText().lower() + "[" + int_text1 + ", " + int_text2 + "]")


        # ================== Subrules of parser rule expr (End) ====================== #

        def exitRegularAssign(self, ctx):
            # Handle assignments of type ID = expr
            if ctx.equals().getText() in ["=", "+=", "-=", "*=", "/="]:
                equals = ctx.equals().getText()
            elif ctx.equals().getText() == ":=":
                equals = " = "
            elif ctx.equals().getText() == "^=":
                equals = "**="

            try:
                a = ctx.ID().getText().lower() + "'"*ctx.diff().getText().count("'")
            except Exception:
                a = ctx.ID().getText().lower()

            if a in self.type.keys() and self.type[a] in ("motionvariable", "motionvariable'") and\
            self.type[ctx.expr().getText().lower()] in ("motionvariable", "motionvariable'"):
                b = ctx.expr().getText().lower()
                if "'" in b and "'" not in a:
                    a, b = b, a
                if not self.kane_parsed:
                    self.kd_eqs.append(a + "-" + b)
                    self.kd_equivalents.update({self.symbol_table[a]:
                                                self.symbol_table[b]})
                    self.kd_equivalents2.update({self.symbol_table[b]:
                                                    self.symbol_table[a]})

            if a in self.symbol_table.keys() and a in self.type.keys() and self.type[a] in ("variable", "motionvariable"):
                self.explicit.update({self.symbol_table[a]: self.getValue(ctx.expr())})

            else:
                if ctx.expr() in self.matrix_expr:
                    self.type.update({a: "matrix"})

                try:
                    b = self.symbol_table[a]
                except KeyError:
                    self.symbol_table[a] = a

                if "_" in a and a.count("_") == 1:
                    e1, e2 = a.split('_')
                    if e1 in self.type2.keys() and self.type2[e1] in ("frame", "bodies")\
                    and e2 in self.type2.keys() and self.type2[e2] in ("frame", "bodies"):
                        if self.type2[e1] == "bodies":
                            t1 = "_f"
                        else:
                            t1 = ""
                        if self.type2[e2] == "bodies":
                            t2 = "_f"
                        else:
                            t2 = ""

                        self.write(self.symbol_table2[e2] + t2 + ".orient(" + self.symbol_table2[e1] +
                                   t1 + ", 'DCM', " + self.getValue(ctx.expr()) + ")\n")
                    else:
                        self.write(self.symbol_table[a] + " " + equals + " " +
                                    self.getValue(ctx.expr()) + "\n")
                else:
                    self.write(self.symbol_table[a] + " " + equals + " " +
                                self.getValue(ctx.expr()) + "\n")

        def exitIndexAssign(self, ctx):
            # Handle assignments of type ID[index] = expr
                if ctx.equals().getText() in ["=", "+=", "-=", "*=", "/="]:
                    equals = ctx.equals().getText()
                elif ctx.equals().getText() == ":=":
                    equals = " = "
                elif ctx.equals().getText() == "^=":
                    equals = "**="

                text = ctx.ID().getText().lower()
                self.type.update({text: "matrix"})
                # Handle assignments of type ID[2] = expr
                if ctx.index().getChildCount() == 1:
                    if ctx.index().getChild(0).getText() == "1":
                        self.type.update({text: "matrix"})
                        self.symbol_table.update({text: text})
                        self.write(text + " = " + "_sm.Matrix([[0]])\n")
                        self.write(text + "[0] = " + self.getValue(ctx.expr()) + "\n")
                    else:
                        # m = m.row_insert(m.shape[0], _sm.Matrix([[0]]))
                        self.write(text + " = " + text +
                                   ".row_insert(" + text + ".shape[0]" + ", " + "_sm.Matrix([[0]])" + ")\n")
                        self.write(text + "[" + text + ".shape[0]-1" + "] = " + self.getValue(ctx.expr()) + "\n")

                # Handle assignments of type ID[2, 2] = expr
                elif ctx.index().getChildCount() == 3:
                    l = []
                    try:
                        l.append(str(int(self.getValue(ctx.index().getChild(0)))-1))
                    except Exception:
                        l.append(self.getValue(ctx.index().getChild(0)) + "-1")
                    l.append(",")
                    try:
                        l.append(str(int(self.getValue(ctx.index().getChild(2)))-1))
                    except Exception:
                        l.append(self.getValue(ctx.index().getChild(2)) + "-1")
                    self.write(self.symbol_table[ctx.ID().getText().lower()] +
                               "[" + "".join(l) + "]" + " " + equals + " " + self.getValue(ctx.expr()) + "\n")

        def exitVecAssign(self, ctx):
            # Handle assignments of the type vec = expr
            ch = ctx.vec()
            vec_text = ch.getText().lower()

            if "_" in ch.ID().getText():
                num = ch.ID().getText().count('_')

                if num == 2:
                    v1, v2, v3 = ch.ID().getText().lower().split('_')

                    if v1 == "p":
                        if self.type2[v2] == "point":
                            e2 = self.symbol_table2[v2]
                        elif self.type2[v2] == "particle":
                            e2 = self.symbol_table2[v2] + ".point"
                        if self.type2[v3] == "point":
                            e3 = self.symbol_table2[v3]
                        elif self.type2[v3] == "particle":
                            e3 = self.symbol_table2[v3] + ".point"
                        # ab.set_pos(na, la*a.x)
                        self.write(e3 + ".set_pos(" + e2 + ", " + self.getValue(ctx.expr()) + ")\n")

                    elif v1 in ("w", "alf"):
                        if v1 == "w":
                            text = ".set_ang_vel("
                        elif v1 == "alf":
                            text = ".set_ang_acc("
                        # a.set_ang_vel(n, qad*a.z)
                        if self.type2[v2] == "bodies":
                            e2 = self.symbol_table2[v2] + "_f"
                        else:
                            e2 = self.symbol_table2[v2]
                        if self.type2[v3] == "bodies":
                            e3 = self.symbol_table2[v3] + "_f"
                        else:
                            e3 = self.symbol_table2[v3]
                        self.write(e2 + text + e3 + ", " + self.getValue(ctx.expr()) + ")\n")

                    elif v1 in ("v", "a"):
                        if v1 == "v":
                            text = ".set_vel("
                        elif v1 == "a":
                            text = ".set_acc("
                        if self.type2[v2] == "point":
                            e2 = self.symbol_table2[v2]
                        elif self.type2[v2] == "particle":
                            e2 = self.symbol_table2[v2] + ".point"
                        self.write(e2 + text + self.symbol_table2[v3] +
                                   ", " + self.getValue(ctx.expr()) + ")\n")
                    elif v1 == "i":
                        if v2 in self.type2.keys() and self.type2[v2] == "bodies":
                            self.write(self.symbol_table2[v2] + ".inertia = (" + self.getValue(ctx.expr()) +
                            ", " + self.symbol_table2[v3] + ")\n")
                            self.inertia_point.update({v2: v3})
                        elif v2 in self.type2.keys() and self.type2[v2] == "particle":
                            self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")
                        else:
                            self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")
                    else:
                        self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")

                elif num == 1:
                    v1, v2 = ch.ID().getText().lower().split('_')

                    if v1 in ("force", "torque"):
                        if self.type2[v2] in ("point", "frame"):
                            e2 = self.symbol_table2[v2]
                        elif self.type2[v2] == "particle":
                            e2 = self.symbol_table2[v2] + ".point"
                        self.symbol_table.update({vec_text: ch.ID().getText().lower()})

                        if e2 in self.forces.keys():
                            self.forces[e2] = self.forces[e2] + " + " + self.getValue(ctx.expr())
                        else:
                            self.forces.update({e2: self.getValue(ctx.expr())})
                        self.write(ch.ID().getText().lower() + " = " + self.forces[e2] + "\n")

                    else:
                        name = ch.ID().getText().lower()
                        self.symbol_table.update({vec_text: name})
                        self.write(ch.ID().getText().lower() + " = " + self.getValue(ctx.expr()) + "\n")
                else:
                    name = ch.ID().getText().lower()
                    self.symbol_table.update({vec_text: name})
                    self.write(name + " " + ctx.getChild(1).getText() + " " + self.getValue(ctx.expr()) + "\n")
            else:
                name = ch.ID().getText().lower()
                self.symbol_table.update({vec_text: name})
                self.write(name + " " + ctx.getChild(1).getText() + " " + self.getValue(ctx.expr()) + "\n")

        def enterInputs2(self, ctx):
            self.in_inputs = True

        # Inputs
        def exitInputs2(self, ctx):
            # Stores numerical values given by the input command which
            # are used for codegen and numerical analysis.
            if ctx.getChildCount() == 3:
                try:
                    self.inputs.update({self.symbol_table[ctx.id_diff().getText().lower()]: self.getValue(ctx.expr(0))})
                except Exception:
                    self.inputs.update({ctx.id_diff().getText().lower(): self.getValue(ctx.expr(0))})
            elif ctx.getChildCount() == 4:
                try:
                    self.inputs.update({self.symbol_table[ctx.id_diff().getText().lower()]:
                    (self.getValue(ctx.expr(0)), self.getValue(ctx.expr(1)))})
                except Exception:
                    self.inputs.update({ctx.id_diff().getText().lower():
                    (self.getValue(ctx.expr(0)), self.getValue(ctx.expr(1)))})

            self.in_inputs = False

        def enterOutputs(self, ctx):
            self.in_outputs = True
        def exitOutputs(self, ctx):
            self.in_outputs = False

        def exitOutputs2(self, ctx):
            try:
                if "[" in ctx.expr(1).getText():
                    self.outputs.append(self.symbol_table[ctx.expr(0).getText().lower()] +
                                        ctx.expr(1).getText().lower())
                else:
                    self.outputs.append(self.symbol_table[ctx.expr(0).getText().lower()])

            except Exception:
                pass

        # Code commands
        def exitCodegen(self, ctx):
            # Handles the CODE() command ie the solvers and the codgen part.
            # Uses linsolve for the algebraic solvers and nsolve for non linear solvers.

            if ctx.functionCall().getChild(0).getText().lower() == "algebraic":
                matrix_name = self.getValue(ctx.functionCall().expr(0))
                e = []
                d = []
                for i in range(1, (ctx.functionCall().getChildCount()-2)//2):
                    a = self.getValue(ctx.functionCall().expr(i))
                    e.append(a)

                for i in self.inputs.keys():
                    d.append(i + ":" + self.inputs[i])
                self.write(matrix_name + "_list" + " = " + "[]\n")
                self.write("for i in " + matrix_name + ":  " + matrix_name +
                           "_list" + ".append(i.subs({" + ", ".join(d) + "}))\n")
                self.write("print(_sm.linsolve(" + matrix_name + "_list" + ", " + ",".join(e) + "))\n")

            elif ctx.functionCall().getChild(0).getText().lower() == "nonlinear":
                e = []
                d = []
                guess = []
                for i in range(1, (ctx.functionCall().getChildCount()-2)//2):
                    a = self.getValue(ctx.functionCall().expr(i))
                    e.append(a)
                #print(self.inputs)
                for i in self.inputs.keys():
                    if i in self.symbol_table.keys():
                        if type(self.inputs[i]) is tuple:
                            j, z = self.inputs[i]
                        else:
                            j = self.inputs[i]
                            z = ""
                        if i not in e:
                            if z == "deg":
                                d.append(i + ":" + "_np.deg2rad(" + j + ")")
                            else:
                                d.append(i + ":" + j)
                        else:
                            if z == "deg":
                                guess.append("_np.deg2rad(" + j + ")")
                            else:
                                guess.append(j)

                self.write("matrix_list" + " = " + "[]\n")
                self.write("for i in " + self.getValue(ctx.functionCall().expr(0)) + ":")
                self.write("matrix_list" + ".append(i.subs({" + ", ".join(d) + "}))\n")
                self.write("print(_sm.nsolve(matrix_list," + "(" + ",".join(e) + ")" +
                           ",(" + ",".join(guess) + ")" + "))\n")

            elif ctx.functionCall().getChild(0).getText().lower() in ["ode", "dynamics"] and self.include_numeric:
                if self.kane_type == "no_args":
                    for i in self.symbol_table.keys():
                        try:
                            if self.type[i] == "constants" or self.type[self.symbol_table[i]] == "constants":
                                self.constants.append(self.symbol_table[i])
                        except Exception:
                            pass
                    q_add_u = self.q_ind + self.q_dep + self.u_ind + self.u_dep
                    x0 = []
                    for i in q_add_u:
                        try:
                            if i in self.inputs.keys():
                                if type(self.inputs[i]) is tuple:
                                    if self.inputs[i][1] == "deg":
                                        x0.append(i + ":" + "_np.deg2rad(" + self.inputs[i][0] + ")")
                                    else:
                                        x0.append(i + ":" + self.inputs[i][0])
                                else:
                                    x0.append(i + ":" + self.inputs[i])
                            elif self.kd_equivalents[i] in self.inputs.keys():
                                if type(self.inputs[self.kd_equivalents[i]]) is tuple:
                                    x0.append(i + ":" + self.inputs[self.kd_equivalents[i]][0])
                                else:
                                    x0.append(i + ":" + self.inputs[self.kd_equivalents[i]])
                        except Exception:
                            pass

                    # numerical constants
                    numerical_constants = []
                    for i in self.constants:
                        if i in self.inputs.keys():
                            if type(self.inputs[i]) is tuple:
                                numerical_constants.append(self.inputs[i][0])
                            else:
                                numerical_constants.append(self.inputs[i])

                    # t = linspace
                    t_final = self.inputs["tfinal"]
                    integ_stp = self.inputs["integstp"]

                    self.write("from pydy.system import System\n")
                    const_list = []
                    if numerical_constants:
                        for i in range(len(self.constants)):
                            const_list.append(self.constants[i] + ":" + numerical_constants[i])
                    specifieds = []
                    if self.t:
                        specifieds.append("_me.dynamicsymbols('t')" + ":" + "lambda x, t: t")

                    for i in self.inputs:
                        if i in self.symbol_table.keys() and self.symbol_table[i] not in\
                        self.constants + self.q_ind + self.q_dep + self.u_ind + self.u_dep:
                            specifieds.append(self.symbol_table[i] + ":" + self.inputs[i])

                    self.write("sys = System(kane, constants = {" + ", ".join(const_list) + "},\n" +
                               "specifieds={" + ", ".join(specifieds) + "},\n" +
                               "initial_conditions={" + ", ".join(x0) + "},\n" +
                               "times = _np.linspace(0.0, " + str(t_final) + ", " + str(t_final) +
                               "/" + str(integ_stp) + "))\n\ny=sys.integrate()\n")

                    # For outputs other than qs and us.
                    other_outputs = []
                    for i in self.outputs:
                        if i not in q_add_u:
                            if "[" in i:
                                other_outputs.append((i[:-3] + i[-2], i[:-3] + "[" + str(int(i[-2])-1) + "]"))
                            else:
                                other_outputs.append((i, i))

                    for i in other_outputs:
                        self.write(i[0] + "_out" + " = " + "[]\n")
                    if other_outputs:
                        self.write("for i in y:\n")
                        self.write("    q_u_dict = dict(zip(sys.coordinates+sys.speeds, i))\n")
                        for i in other_outputs:
                            self.write(" "*4 + i[0] + "_out" + ".append(" + i[1] + ".subs(q_u_dict)" +
                                    ".subs(sys.constants).evalf())\n")

        # Standalone function calls (used for dual functions)
        def exitFunctionCall(self, ctx):
            # Basically deals with standalone function calls ie functions which are not a part of
            # expressions and assignments. Autolev Dual functions can both appear in standalone
            # function calls and also on the right hand side as part of expr or assignment.

            # Dual functions are indicated by a * in the comments below

            # Checks if the function is a statement on its own
            if ctx.parentCtx.getRuleIndex() == AutolevParser.RULE_stat:
                func_name = ctx.getChild(0).getText().lower()
                # Expand(E, n:m) *
                if func_name == "expand":
                    # If the first argument is a pre declared variable.
                    expr = self.getValue(ctx.expr(0))
                    symbol = self.symbol_table[ctx.expr(0).getText().lower()]
                    if ctx.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                        self.write(symbol + " = " + "_sm.Matrix([i.expand() for i in " + expr + "])" +
                                   ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])\n")
                    else:
                        self.write(symbol + " = " + symbol + "." + "expand()\n")

                # Factor(E, x) *
                elif func_name == "factor":
                    expr = self.getValue(ctx.expr(0))
                    symbol = self.symbol_table[ctx.expr(0).getText().lower()]
                    if ctx.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                        self.write(symbol + " = " + "_sm.Matrix([_sm.factor(i," + self.getValue(ctx.expr(1)) +
                                   ") for i in " + expr + "])" +
                                   ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])\n")
                    else:
                        self.write(expr + " = " + "_sm.factor(" + expr + ", " +
                                   self.getValue(ctx.expr(1)) + ")\n")

                # Solve(Zero, x, y)
                elif func_name == "solve":
                    l = []
                    l2 = []
                    num = 0
                    for i in range(1, ctx.getChildCount()):
                        if ctx.getChild(i).getText() == ",":
                            num+=1
                        try:
                            l.append(self.getValue(ctx.getChild(i)))
                        except Exception:
                            l.append(ctx.getChild(i).getText())

                        if i != 2:
                            try:
                                l2.append(self.getValue(ctx.getChild(i)))
                            except Exception:
                                pass

                    for i in l2:
                        self.explicit.update({i: "_sm.solve" + "".join(l) + "[" + i + "]"})

                    self.write("print(_sm.solve" + "".join(l) + ")\n")

                # Arrange(y, n, x) *
                elif func_name == "arrange":
                    expr = self.getValue(ctx.expr(0))
                    symbol = self.symbol_table[ctx.expr(0).getText().lower()]

                    if ctx.expr(0) in self.matrix_expr or (expr in self.type.keys() and self.type[expr] == "matrix"):
                        self.write(symbol + " = " + "_sm.Matrix([i.collect(" + self.getValue(ctx.expr(2)) +
                                   ")" + "for i in " + expr + "])" +
                                   ".reshape((" + expr + ").shape[0], " + "(" + expr + ").shape[1])\n")
                    else:
                        self.write(self.getValue(ctx.expr(0)) + ".collect(" +
                                   self.getValue(ctx.expr(2)) + ")\n")

                # Eig(M, EigenValue, EigenVec)
                elif func_name == "eig":
                    self.symbol_table.update({ctx.expr(1).getText().lower(): ctx.expr(1).getText().lower()})
                    self.symbol_table.update({ctx.expr(2).getText().lower(): ctx.expr(2).getText().lower()})
                    # _sm.Matrix([i.evalf() for i in (i_s_so).eigenvals().keys()])
                    self.write(ctx.expr(1).getText().lower() + " = " +
                               "_sm.Matrix([i.evalf() for i in " +
                               "(" + self.getValue(ctx.expr(0)) + ")" + ".eigenvals().keys()])\n")
                    # _sm.Matrix([i[2][0].evalf() for i in (i_s_o).eigenvects()]).reshape(i_s_o.shape[0], i_s_o.shape[1])
                    self.write(ctx.expr(2).getText().lower() + " = " +
                               "_sm.Matrix([i[2][0].evalf() for i in " + "(" + self.getValue(ctx.expr(0)) + ")" +
                               ".eigenvects()]).reshape(" + self.getValue(ctx.expr(0)) + ".shape[0], " +
                               self.getValue(ctx.expr(0)) + ".shape[1])\n")

                # Simprot(N, A, 3, qA)
                elif func_name == "simprot":
                    # A.orient(N, 'Axis', qA, N.z)
                    if self.type2[ctx.expr(0).getText().lower()] == "frame":
                        frame1 = self.symbol_table2[ctx.expr(0).getText().lower()]
                    elif self.type2[ctx.expr(0).getText().lower()] == "bodies":
                        frame1 = self.symbol_table2[ctx.expr(0).getText().lower()] + "_f"
                    if self.type2[ctx.expr(1).getText().lower()] == "frame":
                        frame2 = self.symbol_table2[ctx.expr(1).getText().lower()]
                    elif self.type2[ctx.expr(1).getText().lower()] == "bodies":
                        frame2 = self.symbol_table2[ctx.expr(1).getText().lower()] + "_f"
                    e2 = ""
                    if ctx.expr(2).getText()[0] == "-":
                        e2 = "-1*"
                    if ctx.expr(2).getText() in ("1", "-1"):
                        e = frame1 + ".x"
                    elif ctx.expr(2).getText() in ("2", "-2"):
                        e = frame1 + ".y"
                    elif ctx.expr(2).getText() in ("3", "-3"):
                        e = frame1 + ".z"
                    else:
                        e = self.getValue(ctx.expr(2))
                        e2 = ""

                    if "degrees" in self.settings.keys() and self.settings["degrees"] == "off":
                        value = self.getValue(ctx.expr(3))
                    else:
                        if ctx.expr(3) in self.numeric_expr:
                            value = "_np.deg2rad(" + self.getValue(ctx.expr(3)) + ")"
                        else:
                            value = self.getValue(ctx.expr(3))
                    self.write(frame2 + ".orient(" + frame1 +
                               ", " + "'Axis'" + ", " + "[" + value +
                               ", " + e2 + e + "]" + ")\n")

                # Express(A2>, B) *
                elif func_name == "express":
                    if self.type2[ctx.expr(1).getText().lower()] == "bodies":
                        f = "_f"
                    else:
                        f = ""

                    if '_' in ctx.expr(0).getText().lower() and ctx.expr(0).getText().count('_') == 2:
                        vec = ctx.expr(0).getText().lower().replace(">", "").split('_')
                        v1 = self.symbol_table2[vec[1]]
                        v2 = self.symbol_table2[vec[2]]
                        if vec[0] == "p":
                            self.write(v2 + ".set_pos(" + v1 + ", " + "(" + self.getValue(ctx.expr(0)) +
                                    ")" + ".express(" + self.symbol_table2[ctx.expr(1).getText().lower()] + f + "))\n")
                        elif vec[0] == "v":
                            self.write(v1 + ".set_vel(" + v2 + ", " + "(" + self.getValue(ctx.expr(0)) +
                                    ")" + ".express(" + self.symbol_table2[ctx.expr(1).getText().lower()] + f + "))\n")
                        elif vec[0] == "a":
                            self.write(v1 + ".set_acc(" + v2 + ", " + "(" + self.getValue(ctx.expr(0)) +
                                    ")" + ".express(" + self.symbol_table2[ctx.expr(1).getText().lower()] + f + "))\n")
                        else:
                            self.write(self.getValue(ctx.expr(0)) + " = " + "(" + self.getValue(ctx.expr(0)) + ")" + ".express(" +
                                        self.symbol_table2[ctx.expr(1).getText().lower()] + f + ")\n")
                    else:
                        self.write(self.getValue(ctx.expr(0)) + " = " + "(" + self.getValue(ctx.expr(0)) + ")" + ".express(" +
                                    self.symbol_table2[ctx.expr(1).getText().lower()] + f + ")\n")

                # Angvel(A, B)
                elif func_name == "angvel":
                    self.write("print(" + self.symbol_table2[ctx.expr(1).getText().lower()] +
                               ".ang_vel_in(" + self.symbol_table2[ctx.expr(0).getText().lower()] + "))\n")

                # v2pts(N, A, O, P)
                elif func_name in ("v2pts", "a2pts", "v2pt", "a1pt"):
                    if func_name == "v2pts":
                        text = ".v2pt_theory("
                    elif func_name == "a2pts":
                        text = ".a2pt_theory("
                    elif func_name == "v1pt":
                        text = ".v1pt_theory("
                    elif func_name == "a1pt":
                        text = ".a1pt_theory("
                    if self.type2[ctx.expr(1).getText().lower()] == "frame":
                        frame = self.symbol_table2[ctx.expr(1).getText().lower()]
                    elif self.type2[ctx.expr(1).getText().lower()] == "bodies":
                        frame = self.symbol_table2[ctx.expr(1).getText().lower()] + "_f"
                    expr_list = []
                    for i in range(2, 4):
                        if self.type2[ctx.expr(i).getText().lower()] == "point":
                            expr_list.append(self.symbol_table2[ctx.expr(i).getText().lower()])
                        elif self.type2[ctx.expr(i).getText().lower()] == "particle":
                            expr_list.append(self.symbol_table2[ctx.expr(i).getText().lower()] + ".point")

                    self.write(expr_list[1] + text + expr_list[0] +
                               "," + self.symbol_table2[ctx.expr(0).getText().lower()] + "," +
                               frame + ")\n")

                # Gravity(g*N1>)
                elif func_name == "gravity":
                    for i in self.bodies.keys():
                        if self.type2[i] == "bodies":
                            e = self.symbol_table2[i] + ".masscenter"
                        elif self.type2[i] == "particle":
                            e = self.symbol_table2[i] + ".point"
                        if e in self.forces.keys():
                            self.forces[e] = self.forces[e] + self.symbol_table2[i] +\
                                             ".mass*(" + self.getValue(ctx.expr(0)) + ")"
                        else:
                            self.forces.update({e: self.symbol_table2[i] +
                                               ".mass*(" + self.getValue(ctx.expr(0)) + ")"})
                        self.write("force_" + i + " = " + self.forces[e] + "\n")

                # Explicit(EXPRESS(IMPLICIT>,C))
                elif func_name == "explicit":
                    if ctx.expr(0) in self.vector_expr:
                        self.vector_expr.append(ctx)
                    expr = self.getValue(ctx.expr(0))
                    if self.explicit.keys():
                        explicit_list = []
                        for i in self.explicit.keys():
                            explicit_list.append(i + ":" + self.explicit[i])
                        if '_' in ctx.expr(0).getText().lower() and ctx.expr(0).getText().count('_') == 2:
                            vec = ctx.expr(0).getText().lower().replace(">", "").split('_')
                            v1 = self.symbol_table2[vec[1]]
                            v2 = self.symbol_table2[vec[2]]
                            if vec[0] == "p":
                                self.write(v2 + ".set_pos(" + v1 + ", " + "(" + expr +
                                        ")" + ".subs({" + ", ".join(explicit_list) + "}))\n")
                            elif vec[0] == "v":
                                self.write(v2 + ".set_vel(" + v1 + ", " + "(" + expr +
                                        ")" + ".subs({" + ", ".join(explicit_list) + "}))\n")
                            elif vec[0] == "a":
                                self.write(v2 + ".set_acc(" + v1 + ", " + "(" + expr +
                                        ")" + ".subs({" + ", ".join(explicit_list) + "}))\n")
                            else:
                                self.write(expr + " = " + "(" + expr + ")" + ".subs({" + ", ".join(explicit_list) + "})\n")
                        else:
                            self.write(expr + " = " + "(" + expr + ")" + ".subs({" + ", ".join(explicit_list) + "})\n")

                # Force(O/Q, -k*Stretch*Uvec>)
                elif func_name in ("force", "torque"):

                    if "/" in ctx.expr(0).getText().lower():
                        p1 = ctx.expr(0).getText().lower().split('/')[0]
                        p2 = ctx.expr(0).getText().lower().split('/')[1]
                        if self.type2[p1] in ("point", "frame"):
                            pt1 = self.symbol_table2[p1]
                        elif self.type2[p1] == "particle":
                            pt1 = self.symbol_table2[p1] + ".point"
                        if self.type2[p2] in ("point", "frame"):
                            pt2 = self.symbol_table2[p2]
                        elif self.type2[p2] == "particle":
                            pt2 = self.symbol_table2[p2] + ".point"
                        if pt1 in self.forces.keys():
                            self.forces[pt1] = self.forces[pt1] + " + -1*("+self.getValue(ctx.expr(1)) + ")"
                            self.write("force_" + p1 + " = " + self.forces[pt1] + "\n")
                        else:
                            self.forces.update({pt1: "-1*("+self.getValue(ctx.expr(1)) + ")"})
                            self.write("force_" + p1 + " = " + self.forces[pt1] + "\n")
                        if pt2 in self.forces.keys():
                            self.forces[pt2] = self.forces[pt2] + "+ " + self.getValue(ctx.expr(1))
                            self.write("force_" + p2 + " = " + self.forces[pt2] + "\n")
                        else:
                            self.forces.update({pt2: self.getValue(ctx.expr(1))})
                            self.write("force_" + p2 + " = " + self.forces[pt2] + "\n")

                    elif ctx.expr(0).getChildCount() == 1:
                        p1 = ctx.expr(0).getText().lower()
                        if self.type2[p1] in ("point", "frame"):
                            pt1 = self.symbol_table2[p1]
                        elif self.type2[p1] == "particle":
                            pt1 = self.symbol_table2[p1] + ".point"
                        if pt1 in self.forces.keys():
                            self.forces[pt1] = self.forces[pt1] + "+ -1*(" + self.getValue(ctx.expr(1)) + ")"
                        else:
                            self.forces.update({pt1: "-1*(" + self.getValue(ctx.expr(1)) + ")"})

                # Constrain(Dependent[qB])
                elif func_name == "constrain":
                    if ctx.getChild(2).getChild(0).getText().lower() == "dependent":
                        self.write("velocity_constraints = [i for i in dependent]\n")
                    x = (ctx.expr(0).getChildCount()-2)//2
                    for i in range(x):
                        self.dependent_variables.append(self.getValue(ctx.expr(0).expr(i)))

                # Kane()
                elif func_name == "kane":
                    if ctx.getChildCount() == 3:
                        self.kane_type = "no_args"

        # Settings
        def exitSettings(self, ctx):
            # Stores settings like Complex on/off, Degrees on/off etc in self.settings.
            try:
                self.settings.update({ctx.getChild(0).getText().lower():
                                     ctx.getChild(1).getText().lower()})
            except Exception:
                pass

        def exitMassDecl2(self, ctx):
            # Used for declaring the masses of particles and rigidbodies.
            particle = self.symbol_table2[ctx.getChild(0).getText().lower()]
            if ctx.getText().count("=") == 2:
                if ctx.expr().expr(1) in self.numeric_expr:
                    e = "_sm.S(" + self.getValue(ctx.expr().expr(1)) + ")"
                else:
                    e = self.getValue(ctx.expr().expr(1))
                self.symbol_table.update({ctx.expr().expr(0).getText().lower(): ctx.expr().expr(0).getText().lower()})
                self.write(ctx.expr().expr(0).getText().lower() + " = " + e + "\n")
                mass = ctx.expr().expr(0).getText().lower()
            else:
                try:
                    if ctx.expr() in self.numeric_expr:
                        mass = "_sm.S(" + self.getValue(ctx.expr()) + ")"
                    else:
                        mass = self.getValue(ctx.expr())
                except Exception:
                    a_text = ctx.expr().getText().lower()
                    self.symbol_table.update({a_text: a_text})
                    self.type.update({a_text: "constants"})
                    self.write(a_text + " = " + "_sm.symbols('" + a_text + "')\n")
                    mass = a_text

            self.write(particle + ".mass = " + mass + "\n")

        def exitInertiaDecl(self, ctx):
            inertia_list = []
            try:
                ctx.ID(1).getText()
                num = 5
            except Exception:
                num = 2
            for i in range((ctx.getChildCount()-num)//2):
                try:
                    if ctx.expr(i) in self.numeric_expr:
                        inertia_list.append("_sm.S(" + self.getValue(ctx.expr(i)) + ")")
                    else:
                        inertia_list.append(self.getValue(ctx.expr(i)))
                except Exception:
                    a_text = ctx.expr(i).getText().lower()
                    self.symbol_table.update({a_text: a_text})
                    self.type.update({a_text: "constants"})
                    self.write(a_text + " = " + "_sm.symbols('" + a_text + "')\n")
                    inertia_list.append(a_text)

            if len(inertia_list) < 6:
                for i in range(6-len(inertia_list)):
                    inertia_list.append("0")
            # body_a.inertia = (_me.inertia(body_a, I1, I2, I3, 0, 0, 0), body_a_cm)
            try:
                frame = self.symbol_table2[ctx.ID(1).getText().lower()]
                point = self.symbol_table2[ctx.ID(0).getText().lower().split('_')[1]]
                body = self.symbol_table2[ctx.ID(0).getText().lower().split('_')[0]]
                self.inertia_point.update({ctx.ID(0).getText().lower().split('_')[0]
                                          : ctx.ID(0).getText().lower().split('_')[1]})
                self.write(body + ".inertia" + " = " + "(_me.inertia(" + frame + ", " +
                           ", ".join(inertia_list) + "), " + point + ")\n")

            except Exception:
                body_name = self.symbol_table2[ctx.ID(0).getText().lower()]
                body_name_cm = body_name + "_cm"
                self.inertia_point.update({ctx.ID(0).getText().lower(): ctx.ID(0).getText().lower() + "o"})
                self.write(body_name + ".inertia" + " = " + "(_me.inertia(" + body_name + "_f" + ", " +
                           ", ".join(inertia_list) + "), " + body_name_cm + ")\n")
