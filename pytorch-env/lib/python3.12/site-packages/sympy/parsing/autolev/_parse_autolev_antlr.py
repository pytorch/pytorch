from importlib.metadata import version
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


def parse_autolev(autolev_code, include_numeric):
    antlr4 = import_module('antlr4')
    if not antlr4 or not version('antlr4-python3-runtime').startswith('4.11'):
        raise ImportError("Autolev parsing requires the antlr4 Python package,"
                          " provided by pip (antlr4-python3-runtime)"
                          " conda (antlr-python-runtime), version 4.11")
    try:
        l = autolev_code.readlines()
        input_stream = antlr4.InputStream("".join(l))
    except Exception:
        input_stream = antlr4.InputStream(autolev_code)

    if AutolevListener:
        from ._listener_autolev_antlr import MyListener
        lexer = AutolevLexer(input_stream)
        token_stream = antlr4.CommonTokenStream(lexer)
        parser = AutolevParser(token_stream)
        tree = parser.prog()
        my_listener = MyListener(include_numeric)
        walker = antlr4.ParseTreeWalker()
        walker.walk(my_listener, tree)
        return "".join(my_listener.output_code)
