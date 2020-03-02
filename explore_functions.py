import os
import re
from collections import OrderedDict

import yaml
# from github import Github


path = os.path.dirname(os.path.realpath(__file__))
aten_native_yaml = os.path.join(path, "aten/src/ATen/native/native_functions.yaml")
function_test_metadata_yaml = os.path.join(path, "native_functions_test_metadata.yaml")
declarations_yaml = os.path.join(path, "aten/src/ATen/Declarations.cwrap")
under_test = dict()
declarations_th = dict()
native = dict()
allowed_fields = [("aten_ported_cpu", "If operator ported from TH to Aten", "False")]
declarations_cnames = dict()


def represent_ordereddict(dumper, data):
    value = []
    for item_key, item_value in data.items():
        node_key = dumper.represent_data(item_key)
        node_value = dumper.represent_data(item_value)
        value.append((node_key, node_value))
    return yaml.nodes.MappingNode(u"tag:yaml.org,2002:map", value)


yaml.add_representer(OrderedDict, represent_ordereddict)


def test_meta_add_function(fname):
    global under_test
    if fname not in under_test:
        under_test[fname] = dict(func=fname)


def test_meta_modify_functions(fnames, **kwargs):
    global under_test
    if not kwargs:
        return
    if isinstance(fnames, str):
        fnames = [fnames]
    for fname in fnames:
        if fname not in under_test:
            under_test[fname] = dict(func=fname)
        for k, v in kwargs.items():
            under_test[fname][k] = v


def update_test_meta():
    global under_test
    result = list()
    for k in sorted(under_test.keys()):
        result.append(
            OrderedDict(
                sorted(
                    under_test[k].items(), key=lambda t: t[0] if t[0] != "func" else ""
                )
            )
        )
    with open(function_test_metadata_yaml, "w") as outfile:
        yaml.dump(result, outfile)


def get_all_functions():
    global under_test
    global declarations_th
    global declarations_cnames
    global native

    with open(declarations_yaml, "r") as file:
        content = file.read()
        for match in re.finditer(r"\[\[(.*?)\]\]", content, re.S):
            a = yaml.load(match.group(1))
            declarations_th[a["name"]] = a
            if "cname" in a:
                declarations_cnames[a["cname"]] = a
                declarations_cnames[a["cname"] + "_"] = a
            if "options" in a:
                for o in a["options"]:
                    if "cname" in o:
                        declarations_cnames[o["cname"]] = a
                        declarations_cnames[o["cname"] + "_"] = a

    with open(function_test_metadata_yaml, "r") as file:
        for f in yaml.load(file.read()):
            under_test[f["func"]] = f

    with open(aten_native_yaml, "r") as file:
        for f in yaml.load(file.read()):
            m = re.search(r"^([^(.]+)", f["func"])
            if m:
                short_name = m.group(0)
                f["short_name"] = short_name
                if short_name in native:
                    native[short_name].append(f)
                else:
                    native[short_name] = [f]

    total = 0
    for k, v in declarations_th.items():
        inc = 1
        if "backends" in v:
            inc *= len(v["backends"])
        else:
            inc *= 2
        if "options" in v:
            inc *= len(v["options"])
        total += inc
    print("Total declarations:", total)

    test_meta_modify_functions(
        under_test.keys(),
        requires_porting_from_th_cpu=False,
        requires_porting_from_th_cuda=False,
    )

    total = 0
    for k, v in declarations_cnames.items():
        if k in native:
            if "backends" in v:
                if "CPU" in v["backends"]:
                    total += 1
                    test_meta_modify_functions(k, requires_porting_from_th_cpu=True)
                if "CUDA" in v["backends"]:
                    total += 1
                    test_meta_modify_functions(k, requires_porting_from_th_cuda=True)
            else:
                total += 2
                test_meta_modify_functions(
                    k,
                    requires_porting_from_th_cpu=True,
                    requires_porting_from_th_cuda=True,
                )
    print("Total recorded declarations:", total)

    for k, v in native.items():
        for item in v:
            if "dispatch" in item and type(item["dispatch"]) == dict:
                for dk, dv in item["dispatch"].items():
                    if dv.startswith("legacy::cpu::"):
                        test_meta_modify_functions(k, requires_porting_from_th_cpu=True)
                    if dv.startswith("legacy::cuda::"):
                        test_meta_modify_functions(
                            k, requires_porting_from_th_cuda=True
                        )

    cuda_ports = dict()
    for k, v in under_test.items():
        if v.get("requires_porting_from_th_cuda", False) and not v.get(
            "requires_porting_from_th_cuda_issue", 0
        ):
            canonical = k
            if canonical[-1] == "_":
                canonical = canonical[:-1]
            if canonical not in cuda_ports:
                cuda_ports[canonical] = []
            cuda_ports[canonical].append(k)



    # g = Github("603242c08d581078261d26c5f526d6a3920e32c5")
    # repo = g.get_repo("pytorch/pytorch")

    # label_triaged = repo.get_label("triaged")
    # label_porting = repo.get_label("topic: porting" )
    # label_operators = repo.get_label("module: operators" )
    # label_be = repo.get_label("better-engineering" )

    # labels = [label_triaged, label_operators, label_be, label_porting]

    body = "Porting TH operators is essential for code simplicity and performance reasons.\n\nPorting guides and Q&A are available in umbrella issue: #24507\n\nFeel free to add @VitalyFedyunin as a reviewer to get a prioritized review."

    for v in cuda_ports.values():
        title = "Migrate "
        v_q = ["`" + v_ + "`" for v_ in v]
        title += " and ".join(v_q)
        title += " from the TH to Aten (CUDA)"
        # issue = repo.create_issue(title=title, body=body, labels = labels)
        # print("- [ ] #%s %s" % (issue.number, title))

    cpu_ports = dict()
    for k, v in under_test.items():
        if v.get("requires_porting_from_th_cpu", False):
            canonical = k
            if canonical[-1] == "_":
                canonical = canonical[:-1]
            if canonical not in cpu_ports:
                cpu_ports[canonical] = []
            cpu_ports[canonical].append(k)

    for v in cpu_ports.values():
        title = "Migrate "
        v_q = ["`" + v_ + "`" for v_ in v]
        title += " and ".join(v_q)
        title += " from the TH to Aten (CPU)"
        # issue = repo.create_issue(title=title, body=body, labels = labels)
        # print("- [ ] #%s %s" % (issue.number, title))

    # print(json.dumps(cuda_ports,indent=4))


get_all_functions()
update_test_meta()
