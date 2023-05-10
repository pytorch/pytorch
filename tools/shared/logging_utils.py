def pluralize(count: int, singular_word: str, plural_word: str = None) -> str:
    if count == 1:
        return f"{count} {singular_word}"

    if plural_word is None:
        plural_word = f"{singular_word}s"

    return f"{count} {plural_word}"
