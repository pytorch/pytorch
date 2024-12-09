from collections import namedtuple


ParityStatus = namedtuple("ParityStatus", ["has_impl_parity", "has_doc_parity"])

"""
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
"""


def parse_parity_tracker_table(file_path):
    def parse_parity_choice(str):
        if str in ["Yes", "No"]:
            return str == "Yes"
        else:
            raise RuntimeError(
                f'{str} is not a supported parity choice. The valid choices are "Yes" and "No".'
            )

    parity_tracker_dict = {}

    with open(file_path) as f:
        all_text = f.read()
        packages = all_text.split("##")
        for package in packages[1:]:
            lines = [line.strip() for line in package.split("\n") if line.strip() != ""]
            package_name = lines[0]
            if package_name in parity_tracker_dict:
                raise RuntimeError(
                    f"Duplicated package name `{package_name}` found in {file_path}"
                )
            else:
                parity_tracker_dict[package_name] = {}
            for api_status in lines[3:]:
                api_name, has_impl_parity_str, has_doc_parity_str = (
                    x.strip() for x in api_status.split("|")
                )
                parity_tracker_dict[package_name][api_name] = ParityStatus(
                    has_impl_parity=parse_parity_choice(has_impl_parity_str),
                    has_doc_parity=parse_parity_choice(has_doc_parity_str),
                )

    return parity_tracker_dict
