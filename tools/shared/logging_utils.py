def pluralize(count: int, singular_word: str, plural_word: str = "") -> str:
    if count == 1:
        return f"{count} {singular_word}"

    if not plural_word:
        plural_word = f"{singular_word}s"

    return f"{count} {plural_word}"


def duration_to_str(seconds: float) -> str:
    if seconds < 0.00001:
        return "0s"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"
