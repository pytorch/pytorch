from collections import namedtuple

TorchNNTestParams = namedtuple(
    'TorchNNTestParams',
    [
        'module_name',
        'module_variant_name',
        'test_instance',
        'cpp_constructor_args',
        'has_parity',
        'device',
    ]
)

CppArg = namedtuple('CppArg', ['type', 'value'])

ParityStatus = namedtuple('ParityStatus', ['has_impl_parity', 'has_doc_parity'])

TorchNNModuleMetadata = namedtuple(
    'TorchNNModuleMetadata',
    [
        'cpp_default_constructor_args',
        'num_attrs_recursive',
        'python_ignored_constructor_args',
        'python_ignored_attrs',
        'python_optional_attribute_to_jit_type',
        'cpp_sources',
    ]
)
TorchNNModuleMetadata.__new__.__defaults__ = (None, None, [], [], {}, '')

'''
This function expects the parity tracker Markdown file to have the following format:

```
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
```

The returned dict has the following format:

```
Dict[package_name]
    -> Dict[api_name]
        -> ParityStatus
```
'''
def parse_parity_tracker_table(file_path):
    def parse_parity_choice(str):
        if str in ['Yes', 'No']:
            return str == 'Yes'
        else:
            raise RuntimeError(
                '{} is not a supported parity choice. The valid choices are "Yes" and "No".'.format(str))

    parity_tracker_dict = {}

    with open(file_path, 'r') as f:
        all_text = f.read()
        packages = all_text.split('##')
        for package in packages[1:]:
            lines = [line.strip() for line in package.split('\n') if line.strip() != '']
            package_name = lines[0]
            if package_name in parity_tracker_dict:
                raise RuntimeError("Duplicated package name `{}` found in {}".format(package_name, file_path))
            else:
                parity_tracker_dict[package_name] = {}
            for api_status in lines[3:]:
                api_name, has_impl_parity_str, has_doc_parity_str = [x.strip() for x in api_status.split('|')]
                parity_tracker_dict[package_name][api_name] = ParityStatus(
                    has_impl_parity=parse_parity_choice(has_impl_parity_str),
                    has_doc_parity=parse_parity_choice(has_doc_parity_str))

    return parity_tracker_dict
