def from_buck_visibility(buck_visibility):
    if buck_visibility != ["PUBLIC"]:
        fail("We only support \"PUBLIC\" visibility for now.")
    return ["//visibility:public"]
