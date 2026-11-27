default_rnns = ["cudnn", "aten", "py"]
default_cnns = ["resnet18", "resnet50"]
all_nets = default_rnns + default_cnns


def pytest_generate_tests(metafunc):
    if metafunc.cls.__name__ == "TestBenchNetwork":
        metafunc.parametrize("net_name", all_nets, scope="class")
