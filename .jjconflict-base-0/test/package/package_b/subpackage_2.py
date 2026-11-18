__import__("math", fromlist=[])
__import__("xml.sax.xmlreader")

result = "subpackage_2"


class PackageBSubpackage2Object_0:
    pass


def dynamic_import_test(name: str):
    __import__(name)
