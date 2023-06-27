import inspect
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch._dynamo as torchdynamo

from torch._export.db.case import ExportCase, normalize_inputs
from torch._export.db.examples import all_examples


PWD = Path(__file__).absolute().parent
ROOT = Path(__file__).absolute().parent.parent.parent.parent
SOURCE = ROOT / Path("source")
EXPORTDB_SOURCE = SOURCE / Path("generated") / Path("exportdb")


@dataclass
class _DynamoConfig:
    capture_scalar_outputs: bool = True
    capture_dynamic_output_shape_ops: bool = True
    guard_nn_modules: bool = True
    dynamic_shapes: bool = True
    specialize_int: bool = True
    allow_rnn: bool = True
    verbose: bool = True
    assume_static_by_default: bool = False


def generate_example_rst(example_case: ExportCase):
    """
    Generates the .rst files for all the examples in db/examples/
    """

    model = example_case.model

    example_inputs = example_case.example_inputs
    tags = ", ".join(f":doc:`{tag} <{tag}>`" for tag in example_case.tags)

    source_file = (
        inspect.getfile(model.__class__)
        if isinstance(model, torch.nn.Module)
        else inspect.getfile(model)
    )
    with open(source_file, "r") as file:
        source_code = file.read()
    source_code = re.sub(r"from torch\._export\.db\.case import .*\n", "", source_code)
    source_code = re.sub(r"@export_case\((.|\n)*?\)\n", "", source_code)
    source_code = source_code.replace("\n", "\n    ")
    splitted_source_code = re.split(r"@export_rewrite_case.*\n", source_code)

    assert len(splitted_source_code) in {
        1,
        2,
    }, f"more than one @export_rewrite_case decorator in {source_code}"

    # Generate contents of the .rst file
    title = f"{example_case.name}"
    doc_contents = f"""{title}
{'^' * (len(title))}

.. note::

    Tags: {tags}

    Support Level: {example_case.support_level.name}

Original source code:

.. code-block:: python

    {splitted_source_code[0]}

Result:

.. code-block::

"""

    # Get resulting graph from dynamo trace
    try:
        # pyre-ignore
        with torchdynamo.config.patch(asdict(_DynamoConfig())):
            inputs = normalize_inputs(example_inputs)
            exported_model, _ = torchdynamo.export(
                model,
                *inputs.args,
                aten_graph=True,
                tracing_mode="symbolic",
                **inputs.kwargs,
            )
            graph_output = exported_model.print_readable(False)
            graph_output = re.sub(r"        # File(.|\n)*?\n", "", graph_output)
            graph_output = graph_output.replace("\n", "\n    ")
            output = f"    {graph_output}"
    except torchdynamo.exc.Unsupported as e:
        output = "    Unsupported: " + str(e).split("\n")[0]

    doc_contents += output + "\n"

    if len(splitted_source_code) == 2:
        doc_contents += f"""\n
You can rewrite the example above to something like the following:

.. code-block:: python

{splitted_source_code[1]}

"""

    # with open(os.path.join(EXPORTDB_SOURCE, f"{example_case.name}.rst"), "w") as f:
    #     f.write(doc_contents)

    return doc_contents


def generate_index_rst(example_cases, tag_to_modules, support_level_to_modules):
    """
    Generates the index.rst file
    """

    # support_level_to_modules = {}
    # for _, example_case in sorted(example_cases.items()):
    #     support_modules = support_level_to_modules.setdefault(
    #         example_case.support_level, []
    #     )
    #     support_modules.append(example_case.name)

    support_contents = ""
    for k, v in support_level_to_modules.items():
        support_level = k.name.lower().replace("_", " ").title()
        module_contents = "\n\n".join(v)
        support_contents += f"""
{support_level}
{'-' * (len(support_level))}

{module_contents}
"""

    tag_names = "\n    ".join(t for t in tag_to_modules.keys())

    with open(os.path.join(PWD, "blurb.md"), "r") as file:
        blurb = file.read()

    # Generate contents of the .rst file
    doc_contents = f"""ExportDB
========

{blurb}

.. toctree::
    :maxdepth: 1
    :caption: Tags

    {tag_names}

{support_contents}
"""

    with open(os.path.join(EXPORTDB_SOURCE, "index.rst"), "w") as f:
        f.write(doc_contents)


def generate_tag_rst(tag_to_modules):
    """
    For each tag that shows up in each ExportCase.tag, generate an .rst file
    containing all the examples that have that tag.
    """

    for tag, modules_rst in tag_to_modules.items():
        doc_contents = f"{tag} tag\n{'=' * (len(tag) + 4)}\n"
        doc_contents += "\n\n".join(modules_rst).replace("=", "-")

        with open(os.path.join(EXPORTDB_SOURCE, f"{tag}.rst"), "w") as f:
            f.write(doc_contents)


def generate_rst():
    if not os.path.exists(EXPORTDB_SOURCE):
        os.makedirs(EXPORTDB_SOURCE)

    example_cases = all_examples()
    tag_to_modules = {}
    support_level_to_modules = {}
    for example_case in example_cases.values():

        doc_contents = generate_example_rst(example_case)

        for tag in example_case.tags:
            tag_to_modules.setdefault(tag, []).append(doc_contents)

        support_level_to_modules.setdefault(example_case.support_level, []).append(doc_contents)

    generate_tag_rst(tag_to_modules)
    generate_index_rst(example_cases, tag_to_modules, support_level_to_modules)


if __name__ == "__main__":
    generate_rst()
