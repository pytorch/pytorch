from torch._C import _compile_graph_to_code_table, _generate_upgraders_graph
from typing import List

def format_bytecode(table):
    # given a nested tuple, convert it to nested list
    def listify(content):
        if not isinstance(content, tuple):
            return content
        return [listify(i) for i in content]

    formatted_table = {}
    for entry in table:
        identifier = entry[0]
        content = entry[1]
        content = listify(content)
        formatted_table[identifier] = content
    return formatted_table

def generate_upgraders_bytecode() -> List:
    yaml_content = []
    upgraders_graph_map = _generate_upgraders_graph()
    for upgrader_name, upgrader_graph in upgraders_graph_map.items():
        bytecode_table = _compile_graph_to_code_table(upgrader_name, upgrader_graph)
        entry = {upgrader_name: format_bytecode(bytecode_table)}
        yaml_content.append(entry)
    return yaml_content

if __name__ == "__main__":
    raise RuntimeError("This file is not meant to be run directly")
