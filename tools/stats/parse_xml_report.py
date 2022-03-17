import xml.etree.ElementTree as ET


def parse_xml_report(report):
    """Convert a test report xml file into a JSON-serializable list of test cases."""
    root = ET.parse(
        report,
        ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)),  # type: ignore [call-arg]
    )

    test_cases = []
    for test_case in root.findall("testcase"):
        test_cases.append(process_xml_element(test_case))

    return test_cases


def process_xml_element(element):
    """Convert a test suite element into a JSON-serializable dict."""
    ret = {}

    # Convert attributes directly into dict elements.
    # e.g.
    #     <testcase name="test_foo" classname="test_bar"></testcase>
    # becomes:
    #     {"name": "test_foo", "classname": "test_bar"}
    ret.update(element.attrib)

    # Convert inner and outer text into special dict elements.
    # e.g.
    #     <testcase>my_inner_text</testcase> my_tail
    # becomes:
    #     {"text": "my_inner_text", "tail": " my_tail"}
    if element.text and element.text.strip():
        ret["text"] = element.text
    if element.tail and element.tail.strip():
        ret["tail"] = element.tail

    # Convert child elements recursively, placing them at a key:
    # e.g.
    #     <testcase>
    #       <foo>hello</foo>
    #     </testcase>
    # becomes
    #    {"foo": {"text": "hello"}}
    for child in element:
        # Special handling for comments.
        if child.tag is ET.Comment:
            ret["comment"] = child.text
        else:
            ret[child.tag] = process_xml_element(child)
    return ret
