




import caffe2.python.serialized_test.serialized_test_util as serial


def pytest_addoption(parser):
    parser.addoption(
        '-G',
        '--generate-serialized',
        action='store_true',
        dest='generate',
        help='generate output files (default=false, compares to current files)',
    )
    parser.addoption(
        '-O',
        '--output',
        default=serial.DATA_DIR,
        dest='output',
        help='output directory (default: %(default)s)'
    )
    parser.addoption(
        '-D',
        '--disable-serialized-check',
        action='store_true',
        dest='disable',
        help='disable checking serialized tests'
    )
    parser.addoption(
        '-C',
        '--disable-gen-coverage',
        action='store_true',
        dest='disable_coverage',
        help='disable generating coverage markdown file'
    )


def pytest_configure(config):
    generate = config.getoption('generate', default=False)
    output = config.getoption('output', default=serial.DATA_DIR)
    disable = config.getoption('disable', default=False)
    disable_coverage = config.getoption('disable_coverage', default=False)
    serial._output_context.__setattr__('should_generate_output', generate)
    serial._output_context.__setattr__('output_dir', output)
    serial._output_context.__setattr__('disable_serialized_check', disable)
    serial._output_context.__setattr__('disable_gen_coverage', disable_coverage)
