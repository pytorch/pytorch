from collections import namedtuple

TorchNNTestParams = namedtuple(
    'TorchNNTestParams',
    [
        'module_name',
        'module_variant_name',
        'python_constructor_args',
        'cpp_constructor_args',
        'example_inputs',
        'has_parity',
        'python_module_class',
        'cpp_sources',
        'num_attrs_recursive',
        'device',
    ]
)

CppArg = namedtuple('CppArg', ['type', 'value'])

ParityStatus = namedtuple('ParityStatus', ['has_impl_parity', 'has_doc_parity'])

TorchNNModuleMetadata = namedtuple('TorchNNModuleMetadata', ['cpp_default_constructor_args', 'num_attrs_recursive', 'cpp_sources'])
TorchNNModuleMetadata.__new__.__defaults__ = (None, None, '')

'''
This function expects the parity tracker Markdown file to have the following format:

## package1_name

API | Implementation Parity | Doc Parity
------------- | ------------- | -------------
API_Name|No|No
...

## package2_name

API | Implementation Parity | Doc Parity
------------- | ------------- | -------------
API_Name|No|No
...
'''
def parse_parity_tracker_table(file_path):
    parity_tracker_dict = {}

    with open(file_path, 'r') as f:
        all_text = f.read()
        packages = all_text.split('##')
        for package in packages[1:]:
            lines = [line for line in package.split('\n') if line.strip() != '']
            package_name = lines[0].strip(' ')
            if package_name in parity_tracker_dict:
                raise RuntimeError("Duplicated package name `{}` found in {}".format(package_name, file_path))
            else:
                parity_tracker_dict[package_name] = {}
            for api_status in lines[4:]:
                api_name, has_impl_parity_str, has_doc_parity_str = api_status.split('|')
                parity_tracker_dict[package_name][api_name] = ParityStatus(
                    has_impl_parity=(has_impl_parity_str == 'Yes'),
                    has_doc_parity=(has_doc_parity_str == 'Yes'))

    return parity_tracker_dict
