#!/usr/bin/env python3






import argparse
from textwrap import dedent
from subprocess import call


def parse_lines(lines):
    # States
    EMPTY = 0
    OP = 1
    MACRO = 2
    parse_state = EMPTY

    # Preprocess the macros
    curr_macro = ""
    macros = {}

    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lower().startswith("macro"):
            assert parse_state == EMPTY
            macro_line = line.split(" ")
            # Support macros that look like attributes
            # e.g. macro - CONV_LIKE
            curr_macro = " ".join(macro_line[1:])
            assert curr_macro not in macros, 'Macro "{}" defined twice.'.format(
                curr_macro
            )
            macros[curr_macro] = []
            parse_state = MACRO
            lines = lines[:index] + lines[index + 1 :]
            continue
        elif line.lower().startswith("endmacro"):
            assert parse_state == MACRO
            parse_state = EMPTY
            lines = lines[:index] + lines[index + 1 :]
            continue
        elif parse_state == MACRO:
            macros[curr_macro].append(line)
            lines = lines[:index] + lines[index + 1 :]
            continue
        index += 1

    index = 0
    while index < len(lines):
        line = lines[index]
        if line in macros:
            lines = lines[:index] + macros[line] + lines[index + 1 :]
            index += len(macros[line]) - 1
        index += 1

    # Now parse the file
    curr_op = ""
    # dict of the form
    #  opName : { attributes: [], ... }
    ops = {}
    # To preserve parsing order for dependencies (for things like init_from)
    op_list = []

    for line in lines:
        if not len(line):
            continue
        if line[0] == "-":
            assert parse_state is OP
            attr = [_.strip() for _ in line[1:].split(":")]
            assert attr[0][0].isupper()
            if len(attr) == 2:  # attribute : type
                ops[curr_op]["attributes"].append((attr[0], attr[1]))
            elif len(attr) == 3:  # attribute : type
                ops[curr_op]["attributes"].append((attr[0], attr[1], attr[2]))
        else:
            op = [l.strip() for l in line.split(":")]
            assert len(op[0].split(" ")) == 1
            parse_state = OP
            curr_op = op[0]
            assert curr_op not in ops
            ops[curr_op] = {}
            op_list.append(curr_op)
            if len(op) > 1:
                ops[curr_op]["init_from"] = [op[1]]
            ops[curr_op]["attributes"] = []
    return ops, op_list


def gen_class(op, op_def):
    attributes = op_def["attributes"]
    attribute_args = []
    default_init = "NeuralNetOperator(NNKind::{op})".format(op=op)
    attribute_init = [default_init]
    attribute_declarations = []
    attribute_getters = []
    attribute_setters = []
    for attr in attributes:
        lower_name = attr[0][0].lower() + attr[0][1:]
        private_name = lower_name + "_"
        default_arg = "" if len(attr) < 3 else " = {}".format(attr[2])
        name = attr[0]
        t = attr[1]
        attr_arg = "{type} {lower_name}".format(
            type=t, lower_name=lower_name + default_arg
        )
        attr_init = "{private_name}({lower_name})".format(
            private_name=private_name, lower_name=lower_name)
        attr_declare = "{type} {private_name};".format(
            type=t, private_name=private_name)
        attr_get = dedent(
            """
              {type} get{name}() const {{
                return {private_name};
              }}
            """.format(
                type=t, name=name, private_name=private_name
            )
        )
        attr_set = dedent(
            """
              void set{name}({type} {lower_name}) {{
                {private_name} = {lower_name};
              }}
            """.format(
                type=t, name=name, private_name=private_name, lower_name=lower_name
            )
        )
        attribute_args.append(attr_arg)
        attribute_init.append(attr_init)
        attribute_declarations.append(attr_declare)
        attribute_getters.append(attr_get)
        attribute_setters.append(attr_set)

    extra_init = ""
    if "init_from" in op_def:
        for other_op in op_def["init_from"]:
            lower_other_op = other_op[0].lower() + other_op[1:]
            other_init = [default_init]
            for attr in attributes:
                lower_name = attr[0][0].lower() + attr[0][1:]
                private_name = lower_name + "_"
                other_init.append(
                    "{private_name}({other_op}.get{name}())".format(
                        name=attr[0], private_name=private_name, other_op=lower_other_op
                    )
                )
            init = dedent(
                """
                  {op}(const {other_op}& {lower_other_op}) :
                      {other_init} {{}}
                """.format(
                    op=op,
                    other_op=other_op,
                    lower_other_op=lower_other_op,
                    other_init=",\n      ".join(other_init),
                )
            )
            extra_init += init

    return dedent(
        """
        class {op} : public NeuralNetOperator {{
         public:
          {op}({attribute_args}) :
              {attribute_init} {{}}
          {extra_init}
          ~{op}() {{}}

          NOMNIGRAPH_DEFINE_NN_RTTI({op});
        {getters}{setters}
         private:
          {attribute_declarations}
        }};

        """.format(
            op=op,
            extra_init=extra_init,
            getters="".join(attribute_getters),
            setters="".join(attribute_setters),
            attribute_args=",\n".join(attribute_args),
            attribute_init=",\n".join(attribute_init),
            attribute_declarations="\n".join(attribute_declarations),
        )
    )


def gen_classes(ops, op_list):
    f = ""
    for op in op_list:
        f += gen_class(op, ops[op])
    return f


def gen_enum(op_list):
    return ",\n".join([op for op in op_list]) + "\n"


def gen_names(op_list):
    f = ""
    for op in op_list:
        f += dedent(
            """
            case NNKind::{name}:
                return \"{name}\";
            """.format(
                name=op
            )
        )
    return f


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate op files.")
    parser.add_argument("--install_dir", help="installation directory")
    parser.add_argument("--source_def", help="ops.def", action="append")
    args = parser.parse_args()
    install_dir = args.install_dir
    sources = args.source_def

    lines = []
    for source in sources:
        with open(source, "rb") as f:
            lines_tmp = f.readlines()
            lines += [l.strip().decode("utf-8") for l in lines_tmp]
    ops, op_list = parse_lines(lines)

    with open(install_dir + "/OpClasses.h", "wb") as f:
        f.write(gen_classes(ops, op_list).encode("utf-8"))
    with open(install_dir + "/OpNames.h", "wb") as f:
        f.write(gen_names(op_list).encode("utf-8"))
    with open(install_dir + "/OpEnum.h", "wb") as f:
        f.write(gen_enum(op_list).encode("utf-8"))

    try:
        cmd = ["clang-format", "-i", install_dir + "/OpClasses.h"]
        call(cmd)
        cmd = ["clang-format", "-i", install_dir + "/OpNames.h"]
        call(cmd)
        cmd = ["clang-format", "-i", install_dir + "/OpEnum.h"]
        call(cmd)
    except Exception:
        pass
