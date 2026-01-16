import pathlib


def test_wrapper_audit_exists_and_mentions_all_wrappers():
    audit_path = pathlib.Path(__file__).resolve().parent.parent.parent / "torch" / "_dynamo" / "pythonify" / "wrapper_audit.md"
    assert audit_path.is_file(), f"missing audit file at {audit_path}"
    text = audit_path.read_text()
    # basic coverage for required wrappers/metadata keywords
    required = [
        "EffectTokensWrapper",
        "AOTDispatchSubclassWrapper",
        "FunctionalizedRngRuntimeWrapper",
        "FakifiedOutWrapper",
        "RuntimeWrapper",
        "AOTDispatchAutograd",
        "AOTDedupeWrapper",
        "AOTSyntheticBaseWrapper",
        "DebugAssertWrapper",
        "tokens",
        "subclass",
        "rng",
        "fakified",
        "dedupe",
        "synthetic",
        "detach",
        "lazy",
    ]
    for term in required:
        assert term.lower() in text.lower(), f"audit missing term: {term}"
