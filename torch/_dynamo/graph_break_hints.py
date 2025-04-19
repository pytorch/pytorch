USER_ERROR = [
    "Dynamo has detected that tracing the code will result in an error when running in eager. "
    "Please double check that your code doesn't contain a similar error when actually running eager/uncompiled.",
]
DYNAMO_BUG = [
    "This is likely to be a Dynamo bug. Please report an issue to PyTorch.",
]
DIFFICULT = [
    "This graph break may be difficult to debug. Please report an issue to PyTorch for assistance.",
]
FUNDAMENTAL = [
    "This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through "
    "your code. Consider finding a workaround.",
]
SUPPORTABLE = [
    "It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you "
    "encounter this graph break often and it is causing performance issues.",
]
CAUSED_BY_EARLIER_GRAPH_BREAK = [
    "This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.",
]
