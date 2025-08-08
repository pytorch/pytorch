import tempfile

import yaml


def create_temp_yaml(content: str | dict) -> str:
    tmp_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml", delete=False)
    if isinstance(content, dict):
        yaml.dump(content, tmp_file)
    else:
        tmp_file.write(content)
    tmp_file.flush()
    return tmp_file.name
