import torch


def check_error(desc, fn, *required_substrings):
    try:
        fn()
    except Exception as e:
        error_message = e.args[0]
        print('=' * 80)
        print(desc)
        print('-' * 80)
        print(error_message)
        print('')
        for sub in required_substrings:
            assert sub in error_message
        return
    raise AssertionError("given function ({}) didn't raise an error".format(desc))

check_error(
    'Wrong argument types',
    lambda: torch.FloatStorage(object()),
    'object')

check_error('Unknown keyword argument',
            lambda: torch.FloatStorage(content=1234.),
            'keyword')

check_error('Invalid types inside a sequence',
            lambda: torch.FloatStorage(['a', 'b']),
            'list', 'str')

check_error('Invalid size type',
            lambda: torch.FloatStorage(1.5),
            'float')

check_error('Invalid offset',
            lambda: torch.FloatStorage(torch.FloatStorage(2), 4),
            '2', '4')

check_error('Negative offset',
            lambda: torch.FloatStorage(torch.FloatStorage(2), -1),
            '2', '-1')

check_error('Invalid size',
            lambda: torch.FloatStorage(torch.FloatStorage(3), 1, 5),
            '2', '1', '5')

check_error('Negative size',
            lambda: torch.FloatStorage(torch.FloatStorage(3), 1, -5),
            '2', '1', '-5')

check_error('Invalid index type',
            lambda: torch.FloatStorage(10)['first item'],
            'str')


def assign():
    torch.FloatStorage(10)[1:-1] = '1'
check_error('Invalid value type',
            assign,
            'str')

check_error('resize_ with invalid type',
            lambda: torch.FloatStorage(10).resize_(1.5),
            'float')

check_error('fill_ with invalid type',
            lambda: torch.IntStorage(10).fill_('asdf'),
            'str')

# TODO: frombuffer
