# SYSTEM IMPORTS
from collections.abc import Sequence


# PYTHON PROJECT IMPORTS


def load_chars_from_file(filepath: str) -> Sequence[str]:
    l: Sequence[str] = list()
    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            line_list: Sequence[str] = list()
            for w in line.rstrip("\n"):
                line_list.append(w)
            l.append(line_list)
    return l


def load_lines_from_file(filepath: str) -> Sequence[str]:
    l: Sequence[str] = None
    with open(filepath, "r", encoding="utf8") as f:
        l = [line.rstrip("\n") for line in f]
    return l

