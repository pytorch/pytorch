from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError

from types import FunctionType


class TableForm:
    r"""
    Create a nice table representation of data.

    Examples
    ========

    >>> from sympy import TableForm
    >>> t = TableForm([[5, 7], [4, 2], [10, 3]])
    >>> print(t)
    5  7
    4  2
    10 3

    You can use the SymPy's printing system to produce tables in any
    format (ascii, latex, html, ...).

    >>> print(t.as_latex())
    \begin{tabular}{l l}
    $5$ & $7$ \\
    $4$ & $2$ \\
    $10$ & $3$ \\
    \end{tabular}

    """

    def __init__(self, data, **kwarg):
        """
        Creates a TableForm.

        Parameters:

            data ...
                            2D data to be put into the table; data can be
                            given as a Matrix

            headings ...
                            gives the labels for rows and columns:

                            Can be a single argument that applies to both
                            dimensions:

                                - None ... no labels
                                - "automatic" ... labels are 1, 2, 3, ...

                            Can be a list of labels for rows and columns:
                            The labels for each dimension can be given
                            as None, "automatic", or [l1, l2, ...] e.g.
                            ["automatic", None] will number the rows

                            [default: None]

            alignments ...
                            alignment of the columns with:

                                - "left" or "<"
                                - "center" or "^"
                                - "right" or ">"

                            When given as a single value, the value is used for
                            all columns. The row headings (if given) will be
                            right justified unless an explicit alignment is
                            given for it and all other columns.

                            [default: "left"]

            formats ...
                            a list of format strings or functions that accept
                            3 arguments (entry, row number, col number) and
                            return a string for the table entry. (If a function
                            returns None then the _print method will be used.)

            wipe_zeros ...
                            Do not show zeros in the table.

                            [default: True]

            pad ...
                            the string to use to indicate a missing value (e.g.
                            elements that are None or those that are missing
                            from the end of a row (i.e. any row that is shorter
                            than the rest is assumed to have missing values).
                            When None, nothing will be shown for values that
                            are missing from the end of a row; values that are
                            None, however, will be shown.

                            [default: None]

        Examples
        ========

        >>> from sympy import TableForm, Symbol
        >>> TableForm([[5, 7], [4, 2], [10, 3]])
        5  7
        4  2
        10 3
        >>> TableForm([list('.'*i) for i in range(1, 4)], headings='automatic')
          | 1 2 3
        ---------
        1 | .
        2 | . .
        3 | . . .
        >>> TableForm([[Symbol('.'*(j if not i%2 else 1)) for i in range(3)]
        ...            for j in range(4)], alignments='rcl')
            .
          . . .
         .. . ..
        ... . ...
        """
        from sympy.matrices.dense import Matrix

        # We only support 2D data. Check the consistency:
        if isinstance(data, Matrix):
            data = data.tolist()
        _h = len(data)

        # fill out any short lines
        pad = kwarg.get('pad', None)
        ok_None = False
        if pad is None:
            pad = " "
            ok_None = True
        pad = Symbol(pad)
        _w = max(len(line) for line in data)
        for i, line in enumerate(data):
            if len(line) != _w:
                line.extend([pad]*(_w - len(line)))
            for j, lj in enumerate(line):
                if lj is None:
                    if not ok_None:
                        lj = pad
                else:
                    try:
                        lj = S(lj)
                    except SympifyError:
                        lj = Symbol(str(lj))
                line[j] = lj
            data[i] = line
        _lines = Tuple(*[Tuple(*d) for d in data])

        headings = kwarg.get("headings", [None, None])
        if headings == "automatic":
            _headings = [range(1, _h + 1), range(1, _w + 1)]
        else:
            h1, h2 = headings
            if h1 == "automatic":
                h1 = range(1, _h + 1)
            if h2 == "automatic":
                h2 = range(1, _w + 1)
            _headings = [h1, h2]

        allow = ('l', 'r', 'c')
        alignments = kwarg.get("alignments", "l")

        def _std_align(a):
            a = a.strip().lower()
            if len(a) > 1:
                return {'left': 'l', 'right': 'r', 'center': 'c'}.get(a, a)
            else:
                return {'<': 'l', '>': 'r', '^': 'c'}.get(a, a)
        std_align = _std_align(alignments)
        if std_align in allow:
            _alignments = [std_align]*_w
        else:
            _alignments = []
            for a in alignments:
                std_align = _std_align(a)
                _alignments.append(std_align)
                if std_align not in ('l', 'r', 'c'):
                    raise ValueError('alignment "%s" unrecognized' %
                        alignments)
        if _headings[0] and len(_alignments) == _w + 1:
            _head_align = _alignments[0]
            _alignments = _alignments[1:]
        else:
            _head_align = 'r'
        if len(_alignments) != _w:
            raise ValueError(
                'wrong number of alignments: expected %s but got %s' %
                (_w, len(_alignments)))

        _column_formats = kwarg.get("formats", [None]*_w)

        _wipe_zeros = kwarg.get("wipe_zeros", True)

        self._w = _w
        self._h = _h
        self._lines = _lines
        self._headings = _headings
        self._head_align = _head_align
        self._alignments = _alignments
        self._column_formats = _column_formats
        self._wipe_zeros = _wipe_zeros

    def __repr__(self):
        from .str import sstr
        return sstr(self, order=None)

    def __str__(self):
        from .str import sstr
        return sstr(self, order=None)

    def as_matrix(self):
        """Returns the data of the table in Matrix form.

        Examples
        ========

        >>> from sympy import TableForm
        >>> t = TableForm([[5, 7], [4, 2], [10, 3]], headings='automatic')
        >>> t
          | 1  2
        --------
        1 | 5  7
        2 | 4  2
        3 | 10 3
        >>> t.as_matrix()
        Matrix([
        [ 5, 7],
        [ 4, 2],
        [10, 3]])
        """
        from sympy.matrices.dense import Matrix
        return Matrix(self._lines)

    def as_str(self):
        # XXX obsolete ?
        return str(self)

    def as_latex(self):
        from .latex import latex
        return latex(self)

    def _sympystr(self, p):
        """
        Returns the string representation of 'self'.

        Examples
        ========

        >>> from sympy import TableForm
        >>> t = TableForm([[5, 7], [4, 2], [10, 3]])
        >>> s = t.as_str()

        """
        column_widths = [0] * self._w
        lines = []
        for line in self._lines:
            new_line = []
            for i in range(self._w):
                # Format the item somehow if needed:
                s = str(line[i])
                if self._wipe_zeros and (s == "0"):
                    s = " "
                w = len(s)
                if w > column_widths[i]:
                    column_widths[i] = w
                new_line.append(s)
            lines.append(new_line)

        # Check heading:
        if self._headings[0]:
            self._headings[0] = [str(x) for x in self._headings[0]]
            _head_width = max(len(x) for x in self._headings[0])

        if self._headings[1]:
            new_line = []
            for i in range(self._w):
                # Format the item somehow if needed:
                s = str(self._headings[1][i])
                w = len(s)
                if w > column_widths[i]:
                    column_widths[i] = w
                new_line.append(s)
            self._headings[1] = new_line

        format_str = []

        def _align(align, w):
            return '%%%s%ss' % (
                ("-" if align == "l" else ""),
                str(w))
        format_str = [_align(align, w) for align, w in
                      zip(self._alignments, column_widths)]
        if self._headings[0]:
            format_str.insert(0, _align(self._head_align, _head_width))
            format_str.insert(1, '|')
        format_str = ' '.join(format_str) + '\n'

        s = []
        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                d = [""] + d
            first_line = format_str % tuple(d)
            s.append(first_line)
            s.append("-" * (len(first_line) - 1) + "\n")
        for i, line in enumerate(lines):
            d = [l if self._alignments[j] != 'c' else
                 l.center(column_widths[j]) for j, l in enumerate(line)]
            if self._headings[0]:
                l = self._headings[0][i]
                l = (l if self._head_align != 'c' else
                     l.center(_head_width))
                d = [l] + d
            s.append(format_str % tuple(d))
        return ''.join(s)[:-1]  # don't include trailing newline

    def _latex(self, printer):
        """
        Returns the string representation of 'self'.
        """
        # Check heading:
        if self._headings[1]:
            new_line = []
            for i in range(self._w):
                # Format the item somehow if needed:
                new_line.append(str(self._headings[1][i]))
            self._headings[1] = new_line

        alignments = []
        if self._headings[0]:
            self._headings[0] = [str(x) for x in self._headings[0]]
            alignments = [self._head_align]
        alignments.extend(self._alignments)

        s = r"\begin{tabular}{" + " ".join(alignments) + "}\n"

        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                d = [""] + d
            first_line = " & ".join(d) + r" \\" + "\n"
            s += first_line
            s += r"\hline" + "\n"
        for i, line in enumerate(self._lines):
            d = []
            for j, x in enumerate(line):
                if self._wipe_zeros and (x in (0, "0")):
                    d.append(" ")
                    continue
                f = self._column_formats[j]
                if f:
                    if isinstance(f, FunctionType):
                        v = f(x, i, j)
                        if v is None:
                            v = printer._print(x)
                    else:
                        v = f % x
                    d.append(v)
                else:
                    v = printer._print(x)
                    d.append("$%s$" % v)
            if self._headings[0]:
                d = [self._headings[0][i]] + d
            s += " & ".join(d) + r" \\" + "\n"
        s += r"\end{tabular}"
        return s
