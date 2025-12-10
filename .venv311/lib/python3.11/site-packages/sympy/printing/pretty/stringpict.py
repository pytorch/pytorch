"""Prettyprinter by Jurjen Bos.
(I hate spammers: mail me at pietjepuk314 at the reverse of ku.oc.oohay).
All objects have a method that create a "stringPict",
that can be used in the str method for pretty printing.

Updates by Jason Gedge (email <my last name> at cs mun ca)
    - terminal_string() method
    - minor fixes and changes (mostly to prettyForm)

TODO:
    - Allow left/center/right alignment options for above/below and
      top/center/bottom alignment options for left/right
"""

import shutil

from .pretty_symbology import hobj, vobj, xsym, xobj, pretty_use_unicode, line_width, center
from sympy.utilities.exceptions import sympy_deprecation_warning

_GLOBAL_WRAP_LINE = None

class stringPict:
    """An ASCII picture.
    The pictures are represented as a list of equal length strings.
    """
    #special value for stringPict.below
    LINE = 'line'

    def __init__(self, s, baseline=0):
        """Initialize from string.
        Multiline strings are centered.
        """
        self.s = s
        #picture is a string that just can be printed
        self.picture = stringPict.equalLengths(s.splitlines())
        #baseline is the line number of the "base line"
        self.baseline = baseline
        self.binding = None

    @staticmethod
    def equalLengths(lines):
        # empty lines
        if not lines:
            return ['']

        width = max(line_width(line) for line in lines)
        return [center(line, width) for line in lines]

    def height(self):
        """The height of the picture in characters."""
        return len(self.picture)

    def width(self):
        """The width of the picture in characters."""
        return line_width(self.picture[0])

    @staticmethod
    def next(*args):
        """Put a string of stringPicts next to each other.
        Returns string, baseline arguments for stringPict.
        """
        #convert everything to stringPicts
        objects = []
        for arg in args:
            if isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)

        #make a list of pictures, with equal height and baseline
        newBaseline = max(obj.baseline for obj in objects)
        newHeightBelowBaseline = max(
            obj.height() - obj.baseline
            for obj in objects)
        newHeight = newBaseline + newHeightBelowBaseline

        pictures = []
        for obj in objects:
            oneEmptyLine = [' '*obj.width()]
            basePadding = newBaseline - obj.baseline
            totalPadding = newHeight - obj.height()
            pictures.append(
                oneEmptyLine * basePadding +
                obj.picture +
                oneEmptyLine * (totalPadding - basePadding))

        result = [''.join(lines) for lines in zip(*pictures)]
        return '\n'.join(result), newBaseline

    def right(self, *args):
        r"""Put pictures next to this one.
        Returns string, baseline arguments for stringPict.
        (Multiline) strings are allowed, and are given a baseline of 0.

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("10").right(" + ",stringPict("1\r-\r2",1))[0])
             1
        10 + -
             2

        """
        return stringPict.next(self, *args)

    def left(self, *args):
        """Put pictures (left to right) at left.
        Returns string, baseline arguments for stringPict.
        """
        return stringPict.next(*(args + (self,)))

    @staticmethod
    def stack(*args):
        """Put pictures on top of each other,
        from top to bottom.
        Returns string, baseline arguments for stringPict.
        The baseline is the baseline of the second picture.
        Everything is centered.
        Baseline is the baseline of the second picture.
        Strings are allowed.
        The special value stringPict.LINE is a row of '-' extended to the width.
        """
        #convert everything to stringPicts; keep LINE
        objects = []
        for arg in args:
            if arg is not stringPict.LINE and isinstance(arg, str):
                arg = stringPict(arg)
            objects.append(arg)

        #compute new width
        newWidth = max(
            obj.width()
            for obj in objects
            if obj is not stringPict.LINE)

        lineObj = stringPict(hobj('-', newWidth))

        #replace LINE with proper lines
        for i, obj in enumerate(objects):
            if obj is stringPict.LINE:
                objects[i] = lineObj

        #stack the pictures, and center the result
        newPicture = [center(line, newWidth) for obj in objects for line in obj.picture]
        newBaseline = objects[0].height() + objects[1].baseline
        return '\n'.join(newPicture), newBaseline

    def below(self, *args):
        """Put pictures under this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of top picture

        Examples
        ========

        >>> from sympy.printing.pretty.stringpict import stringPict
        >>> print(stringPict("x+3").below(
        ...       stringPict.LINE, '3')[0]) #doctest: +NORMALIZE_WHITESPACE
        x+3
        ---
         3

        """
        s, baseline = stringPict.stack(self, *args)
        return s, self.baseline

    def above(self, *args):
        """Put pictures above this picture.
        Returns string, baseline arguments for stringPict.
        Baseline is baseline of bottom picture.
        """
        string, baseline = stringPict.stack(*(args + (self,)))
        baseline = len(string.splitlines()) - self.height() + self.baseline
        return string, baseline

    def parens(self, left='(', right=')', ifascii_nougly=False):
        """Put parentheses around self.
        Returns string, baseline arguments for stringPict.

        left or right can be None or empty string which means 'no paren from
        that side'
        """
        h = self.height()
        b = self.baseline

        # XXX this is a hack -- ascii parens are ugly!
        if ifascii_nougly and not pretty_use_unicode():
            h = 1
            b = 0

        res = self

        if left:
            lparen = stringPict(vobj(left, h), baseline=b)
            res = stringPict(*lparen.right(self))
        if right:
            rparen = stringPict(vobj(right, h), baseline=b)
            res = stringPict(*res.right(rparen))

        return ('\n'.join(res.picture), res.baseline)

    def leftslash(self):
        """Precede object by a slash of the proper size.
        """
        # XXX not used anywhere ?
        height = max(
            self.baseline,
            self.height() - 1 - self.baseline)*2 + 1
        slash = '\n'.join(
            ' '*(height - i - 1) + xobj('/', 1) + ' '*i
            for i in range(height)
        )
        return self.left(stringPict(slash, height//2))

    def root(self, n=None):
        """Produce a nice root symbol.
        Produces ugly results for big n inserts.
        """
        # XXX not used anywhere
        # XXX duplicate of root drawing in pretty.py
        #put line over expression
        result = self.above('_'*self.width())
        #construct right half of root symbol
        height = self.height()
        slash = '\n'.join(
            ' ' * (height - i - 1) + '/' + ' ' * i
            for i in range(height)
        )
        slash = stringPict(slash, height - 1)
        #left half of root symbol
        if height > 2:
            downline = stringPict('\\ \n \\', 1)
        else:
            downline = stringPict('\\')
        #put n on top, as low as possible
        if n is not None and n.width() > downline.width():
            downline = downline.left(' '*(n.width() - downline.width()))
            downline = downline.above(n)
        #build root symbol
        root = downline.right(slash)
        #glue it on at the proper height
        #normally, the root symbel is as high as self
        #which is one less than result
        #this moves the root symbol one down
        #if the root became higher, the baseline has to grow too
        root.baseline = result.baseline - result.height() + root.height()
        return result.left(root)

    def render(self, * args, **kwargs):
        """Return the string form of self.

           Unless the argument line_break is set to False, it will
           break the expression in a form that can be printed
           on the terminal without being broken up.
         """
        if _GLOBAL_WRAP_LINE is not None:
            kwargs["wrap_line"] = _GLOBAL_WRAP_LINE

        if kwargs["wrap_line"] is False:
            return "\n".join(self.picture)

        if kwargs["num_columns"] is not None:
            # Read the argument num_columns if it is not None
            ncols = kwargs["num_columns"]
        else:
            # Attempt to get a terminal width
            ncols = self.terminal_width()

        if ncols <= 0:
            ncols = 80

        # If smaller than the terminal width, no need to correct
        if self.width() <= ncols:
            return type(self.picture[0])(self)

        """
        Break long-lines in a visually pleasing format.
        without overflow indicators | with overflow indicators
        |   2  2        3     |     |   2  2        3    ↪|
        |6*x *y  + 4*x*y  +   |     |6*x *y  + 4*x*y  +  ↪|
        |                     |     |                     |
        |     3    4    4     |     |↪      3    4    4   |
        |4*y*x  + x  + y      |     |↪ 4*y*x  + x  + y    |
        |a*c*e + a*c*f + a*d  |     |a*c*e + a*c*f + a*d ↪|
        |*e + a*d*f + b*c*e   |     |                     |
        |+ b*c*f + b*d*e + b  |     |↪ *e + a*d*f + b*c* ↪|
        |*d*f                 |     |                     |
        |                     |     |↪ e + b*c*f + b*d*e ↪|
        |                     |     |                     |
        |                     |     |↪ + b*d*f            |
        """

        overflow_first = ""
        if kwargs["use_unicode"] or pretty_use_unicode():
            overflow_start = "\N{RIGHTWARDS ARROW WITH HOOK} "
            overflow_end   = " \N{RIGHTWARDS ARROW WITH HOOK}"
        else:
            overflow_start = "> "
            overflow_end   = " >"

        def chunks(line):
            """Yields consecutive chunks of line_width ncols"""
            prefix = overflow_first
            width, start = line_width(prefix + overflow_end), 0
            for i, x in enumerate(line):
                wx = line_width(x)
                # Only flush the screen when the current character overflows.
                # This way, combining marks can be appended even when width == ncols.
                if width + wx > ncols:
                    yield prefix + line[start:i] + overflow_end
                    prefix = overflow_start
                    width, start = line_width(prefix + overflow_end), i
                width += wx
            yield prefix + line[start:]

        # Concurrently assemble chunks of all lines into individual screens
        pictures = zip(*map(chunks, self.picture))

        # Join lines of each screen into sub-pictures
        pictures = ["\n".join(picture) for picture in pictures]

        # Add spacers between sub-pictures
        return "\n\n".join(pictures)

    def terminal_width(self):
        """Return the terminal width if possible, otherwise return 0.
        """
        size = shutil.get_terminal_size(fallback=(0, 0))
        return size.columns

    def __eq__(self, o):
        if isinstance(o, str):
            return '\n'.join(self.picture) == o
        elif isinstance(o, stringPict):
            return o.picture == self.picture
        return False

    def __hash__(self):
        return super().__hash__()

    def __str__(self):
        return '\n'.join(self.picture)

    def __repr__(self):
        return "stringPict(%r,%d)" % ('\n'.join(self.picture), self.baseline)

    def __getitem__(self, index):
        return self.picture[index]

    def __len__(self):
        return len(self.s)


class prettyForm(stringPict):
    """
    Extension of the stringPict class that knows about basic math applications,
    optimizing double minus signs.

    "Binding" is interpreted as follows::

        ATOM this is an atom: never needs to be parenthesized
        FUNC this is a function application: parenthesize if added (?)
        DIV  this is a division: make wider division if divided
        POW  this is a power: only parenthesize if exponent
        MUL  this is a multiplication: parenthesize if powered
        ADD  this is an addition: parenthesize if multiplied or powered
        NEG  this is a negative number: optimize if added, parenthesize if
             multiplied or powered
        OPEN this is an open object: parenthesize if added, multiplied, or
             powered (example: Piecewise)
    """
    ATOM, FUNC, DIV, POW, MUL, ADD, NEG, OPEN = range(8)

    def __init__(self, s, baseline=0, binding=0, unicode=None):
        """Initialize from stringPict and binding power."""
        stringPict.__init__(self, s, baseline)
        self.binding = binding
        if unicode is not None:
            sympy_deprecation_warning(
                """
                The unicode argument to prettyForm is deprecated. Only the s
                argument (the first positional argument) should be passed.
                """,
                deprecated_since_version="1.7",
                active_deprecations_target="deprecated-pretty-printing-functions")
        self._unicode = unicode or s

    @property
    def unicode(self):
        sympy_deprecation_warning(
            """
            The prettyForm.unicode attribute is deprecated. Use the
            prettyForm.s attribute instead.
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-pretty-printing-functions")
        return self._unicode

    # Note: code to handle subtraction is in _print_Add

    def __add__(self, *others):
        """Make a pretty addition.
        Addition of negative numbers is simplified.
        """
        arg = self
        if arg.binding > prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            #add parentheses for weak binders
            if arg.binding > prettyForm.NEG:
                arg = stringPict(*arg.parens())
            #use existing minus sign if available
            if arg.binding != prettyForm.NEG:
                result.append(' + ')
            result.append(arg)
        return prettyForm(binding=prettyForm.ADD, *stringPict.next(*result))

    def __truediv__(self, den, slashed=False):
        """Make a pretty division; stacked or slashed.
        """
        if slashed:
            raise NotImplementedError("Can't do slashed fraction yet")
        num = self
        if num.binding == prettyForm.DIV:
            num = stringPict(*num.parens())
        if den.binding == prettyForm.DIV:
            den = stringPict(*den.parens())

        if num.binding==prettyForm.NEG:
            num = num.right(" ")[0]

        return prettyForm(binding=prettyForm.DIV, *stringPict.stack(
            num,
            stringPict.LINE,
            den))

    def __mul__(self, *others):
        """Make a pretty multiplication.
        Parentheses are needed around +, - and neg.
        """
        quantity = {
            'degree': "\N{DEGREE SIGN}"
        }

        if len(others) == 0:
            return self  # We aren't actually multiplying... So nothing to do here.

        # add parens on args that need them
        arg = self
        if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
            arg = stringPict(*arg.parens())
        result = [arg]
        for arg in others:
            if arg.picture[0] not in quantity.values():
                result.append(xsym('*'))
            #add parentheses for weak binders
            if arg.binding > prettyForm.MUL and arg.binding != prettyForm.NEG:
                arg = stringPict(*arg.parens())
            result.append(arg)

        len_res = len(result)
        for i in range(len_res):
            if i < len_res - 1 and result[i] == '-1' and result[i + 1] == xsym('*'):
                # substitute -1 by -, like in -1*x -> -x
                result.pop(i)
                result.pop(i)
                result.insert(i, '-')
        if result[0][0] == '-':
            # if there is a - sign in front of all
            # This test was failing to catch a prettyForm.__mul__(prettyForm("-1", 0, 6)) being negative
            bin = prettyForm.NEG
            if result[0] == '-':
                right = result[1]
                if right.picture[right.baseline][0] == '-':
                    result[0] = '- '
        else:
            bin = prettyForm.MUL
        return prettyForm(binding=bin, *stringPict.next(*result))

    def __repr__(self):
        return "prettyForm(%r,%d,%d)" % (
            '\n'.join(self.picture),
            self.baseline,
            self.binding)

    def __pow__(self, b):
        """Make a pretty power.
        """
        a = self
        use_inline_func_form = False
        if b.binding == prettyForm.POW:
            b = stringPict(*b.parens())
        if a.binding > prettyForm.FUNC:
            a = stringPict(*a.parens())
        elif a.binding == prettyForm.FUNC:
            # heuristic for when to use inline power
            if b.height() > 1:
                a = stringPict(*a.parens())
            else:
                use_inline_func_form = True

        if use_inline_func_form:
            #         2
            #  sin  +   + (x)
            b.baseline = a.prettyFunc.baseline + b.height()
            func = stringPict(*a.prettyFunc.right(b))
            return prettyForm(*func.right(a.prettyArgs))
        else:
            #      2    <-- top
            # (x+y)     <-- bot
            top = stringPict(*b.left(' '*a.width()))
            bot = stringPict(*a.right(' '*b.width()))

        return prettyForm(binding=prettyForm.POW, *bot.above(top))

    simpleFunctions = ["sin", "cos", "tan"]

    @staticmethod
    def apply(function, *args):
        """Functions of one or more variables.
        """
        if function in prettyForm.simpleFunctions:
            #simple function: use only space if possible
            assert len(
                args) == 1, "Simple function %s must have 1 argument" % function
            arg = args[0].__pretty__()
            if arg.binding <= prettyForm.DIV:
                #optimization: no parentheses necessary
                return prettyForm(binding=prettyForm.FUNC, *arg.left(function + ' '))
        argumentList = []
        for arg in args:
            argumentList.append(',')
            argumentList.append(arg.__pretty__())
        argumentList = stringPict(*stringPict.next(*argumentList[1:]))
        argumentList = stringPict(*argumentList.parens())
        return prettyForm(binding=prettyForm.ATOM, *argumentList.left(function))
