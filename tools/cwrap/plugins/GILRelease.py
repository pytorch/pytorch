from . import CWrapPlugin
from string import Template


class GILRelease(CWrapPlugin):

    OPTION_START = [
        'PyThreadState *_save = NULL;',
        'try {',
    ]

    BEFORE_CALL = 'Py_UNBLOCK_THREADS;'

    AFTER_CALL = 'Py_BLOCK_THREADS;'

    OPTION_END = [
        '} catch (...) {',
        'if (_save) {',
        'Py_BLOCK_THREADS;',
        '}',
        'throw;',
        '}',
    ]

    def process_option_code_template(self, template, option):
        call_idx = template.index('$call')
        template.insert(call_idx, self.BEFORE_CALL)
        template.insert(call_idx + 2, self.AFTER_CALL)
        return self.OPTION_START + template + self.OPTION_END
