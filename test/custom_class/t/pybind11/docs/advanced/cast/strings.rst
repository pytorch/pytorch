Strings, bytes and Unicode conversions
######################################

.. note::

    This section discusses string handling in terms of Python 3 strings. For
    Python 2.7, replace all occurrences of ``str`` with ``unicode`` and
    ``bytes`` with ``str``.  Python 2.7 users may find it best to use ``from
    __future__ import unicode_literals`` to avoid unintentionally using ``str``
    instead of ``unicode``.

Passing Python strings to C++
=============================

When a Python ``str`` is passed from Python to a C++ function that accepts
``std::string`` or ``char *`` as arguments, pybind11 will encode the Python
string to UTF-8. All Python ``str`` can be encoded in UTF-8, so this operation
does not fail.

The C++ language is encoding agnostic. It is the responsibility of the
programmer to track encodings. It's often easiest to simply `use UTF-8
everywhere <http://utf8everywhere.org/>`_.

.. code-block:: c++

    m.def("utf8_test",
        [](const std::string &s) {
            cout << "utf-8 is icing on the cake.\n";
            cout << s;
        }
    );
    m.def("utf8_charptr",
        [](const char *s) {
            cout << "My favorite food is\n";
            cout << s;
        }
    );

.. code-block:: python

    >>> utf8_test('🎂')
    utf-8 is icing on the cake.
    🎂

    >>> utf8_charptr('🍕')
    My favorite food is
    🍕

.. note::

    Some terminal emulators do not support UTF-8 or emoji fonts and may not
    display the example above correctly.

The results are the same whether the C++ function accepts arguments by value or
reference, and whether or not ``const`` is used.

Passing bytes to C++
--------------------

A Python ``bytes`` object will be passed to C++ functions that accept
``std::string`` or ``char*`` *without* conversion.  On Python 3, in order to
make a function *only* accept ``bytes`` (and not ``str``), declare it as taking
a ``py::bytes`` argument.


Returning C++ strings to Python
===============================

When a C++ function returns a ``std::string`` or ``char*`` to a Python caller,
**pybind11 will assume that the string is valid UTF-8** and will decode it to a
native Python ``str``, using the same API as Python uses to perform
``bytes.decode('utf-8')``. If this implicit conversion fails, pybind11 will
raise a ``UnicodeDecodeError``.

.. code-block:: c++

    m.def("std_string_return",
        []() {
            return std::string("This string needs to be UTF-8 encoded");
        }
    );

.. code-block:: python

    >>> isinstance(example.std_string_return(), str)
    True


Because UTF-8 is inclusive of pure ASCII, there is never any issue with
returning a pure ASCII string to Python. If there is any possibility that the
string is not pure ASCII, it is necessary to ensure the encoding is valid
UTF-8.

.. warning::

    Implicit conversion assumes that a returned ``char *`` is null-terminated.
    If there is no null terminator a buffer overrun will occur.

Explicit conversions
--------------------

If some C++ code constructs a ``std::string`` that is not a UTF-8 string, one
can perform a explicit conversion and return a ``py::str`` object. Explicit
conversion has the same overhead as implicit conversion.

.. code-block:: c++

    // This uses the Python C API to convert Latin-1 to Unicode
    m.def("str_output",
        []() {
            std::string s = "Send your r\xe9sum\xe9 to Alice in HR"; // Latin-1
            py::str py_s = PyUnicode_DecodeLatin1(s.data(), s.length());
            return py_s;
        }
    );

.. code-block:: python

    >>> str_output()
    'Send your résumé to Alice in HR'

The `Python C API
<https://docs.python.org/3/c-api/unicode.html#built-in-codecs>`_ provides
several built-in codecs.


One could also use a third party encoding library such as libiconv to transcode
to UTF-8.

Return C++ strings without conversion
-------------------------------------

If the data in a C++ ``std::string`` does not represent text and should be
returned to Python as ``bytes``, then one can return the data as a
``py::bytes`` object.

.. code-block:: c++

    m.def("return_bytes",
        []() {
            std::string s("\xba\xd0\xba\xd0");  // Not valid UTF-8
            return py::bytes(s);  // Return the data without transcoding
        }
    );

.. code-block:: python

    >>> example.return_bytes()
    b'\xba\xd0\xba\xd0'


Note the asymmetry: pybind11 will convert ``bytes`` to ``std::string`` without
encoding, but cannot convert ``std::string`` back to ``bytes`` implicitly.

.. code-block:: c++

    m.def("asymmetry",
        [](std::string s) {  // Accepts str or bytes from Python
            return s;  // Looks harmless, but implicitly converts to str
        }
    );

.. code-block:: python

    >>> isinstance(example.asymmetry(b"have some bytes"), str)
    True

    >>> example.asymmetry(b"\xba\xd0\xba\xd0")  # invalid utf-8 as bytes
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xba in position 0: invalid start byte


Wide character strings
======================

When a Python ``str`` is passed to a C++ function expecting ``std::wstring``,
``wchar_t*``, ``std::u16string`` or ``std::u32string``, the ``str`` will be
encoded to UTF-16 or UTF-32 depending on how the C++ compiler implements each
type, in the platform's native endianness. When strings of these types are
returned, they are assumed to contain valid UTF-16 or UTF-32, and will be
decoded to Python ``str``.

.. code-block:: c++

    #define UNICODE
    #include <windows.h>

    m.def("set_window_text",
        [](HWND hwnd, std::wstring s) {
            // Call SetWindowText with null-terminated UTF-16 string
            ::SetWindowText(hwnd, s.c_str());
        }
    );
    m.def("get_window_text",
        [](HWND hwnd) {
            const int buffer_size = ::GetWindowTextLength(hwnd) + 1;
            auto buffer = std::make_unique< wchar_t[] >(buffer_size);

            ::GetWindowText(hwnd, buffer.data(), buffer_size);

            std::wstring text(buffer.get());

            // wstring will be converted to Python str
            return text;
        }
    );

.. warning::

    Wide character strings may not work as described on Python 2.7 or Python
    3.3 compiled with ``--enable-unicode=ucs2``.

Strings in multibyte encodings such as Shift-JIS must transcoded to a
UTF-8/16/32 before being returned to Python.


Character literals
==================

C++ functions that accept character literals as input will receive the first
character of a Python ``str`` as their input. If the string is longer than one
Unicode character, trailing characters will be ignored.

When a character literal is returned from C++ (such as a ``char`` or a
``wchar_t``), it will be converted to a ``str`` that represents the single
character.

.. code-block:: c++

    m.def("pass_char", [](char c) { return c; });
    m.def("pass_wchar", [](wchar_t w) { return w; });

.. code-block:: python

    >>> example.pass_char('A')
    'A'

While C++ will cast integers to character types (``char c = 0x65;``), pybind11
does not convert Python integers to characters implicitly. The Python function
``chr()`` can be used to convert integers to characters.

.. code-block:: python

    >>> example.pass_char(0x65)
    TypeError

    >>> example.pass_char(chr(0x65))
    'A'

If the desire is to work with an 8-bit integer, use ``int8_t`` or ``uint8_t``
as the argument type.

Grapheme clusters
-----------------

A single grapheme may be represented by two or more Unicode characters. For
example 'é' is usually represented as U+00E9 but can also be expressed as the
combining character sequence U+0065 U+0301 (that is, the letter 'e' followed by
a combining acute accent). The combining character will be lost if the
two-character sequence is passed as an argument, even though it renders as a
single grapheme.

.. code-block:: python

    >>> example.pass_wchar('é')
    'é'

    >>> combining_e_acute = 'e' + '\u0301'

    >>> combining_e_acute
    'é'

    >>> combining_e_acute == 'é'
    False

    >>> example.pass_wchar(combining_e_acute)
    'e'

Normalizing combining characters before passing the character literal to C++
may resolve *some* of these issues:

.. code-block:: python

    >>> example.pass_wchar(unicodedata.normalize('NFC', combining_e_acute))
    'é'

In some languages (Thai for example), there are `graphemes that cannot be
expressed as a single Unicode code point
<http://unicode.org/reports/tr29/#Grapheme_Cluster_Boundaries>`_, so there is
no way to capture them in a C++ character type.


C++17 string views
==================

C++17 string views are automatically supported when compiling in C++17 mode.
They follow the same rules for encoding and decoding as the corresponding STL
string type (for example, a ``std::u16string_view`` argument will be passed
UTF-16-encoded data, and a returned ``std::string_view`` will be decoded as
UTF-8).

References
==========

* `The Absolute Minimum Every Software Developer Absolutely, Positively Must Know About Unicode and Character Sets (No Excuses!) <https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/>`_
* `C++ - Using STL Strings at Win32 API Boundaries <https://msdn.microsoft.com/en-ca/magazine/mt238407.aspx>`_
