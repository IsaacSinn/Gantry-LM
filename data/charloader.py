# SYSTEM IMPORTS
from collections.abc import Sequence
from collections import defaultdict

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

# Define special tokens
SPECIAL_TOKENS = set(["<BOS>", "<EOS>", "<EOI>", "<MOVE>", "<SPEED>", "<ROTATE>", "<SC>", "<EC>"])

def parse_tokens_from_file(filepath:str) -> Sequence[tuple[Sequence[str], Sequence[str]]]:
    result = []
    # <EOI> marks end of the input and start of the output

    with open(filepath, "r", encoding="utf8") as f:
        for line in f:
            tokens_before_eoi = []
            tokens_after_eoi = []
            found_eoi = False

            i = 0
            while i < len(line):
                if line[i] == '<':
                    # Check for special token
                    for token in SPECIAL_TOKENS:
                        if line.startswith(token, i):
                            if token == "<EOI>":
                                found_eoi = True
                            elif found_eoi:
                                tokens_after_eoi.append(token)
                            else:
                                tokens_before_eoi.append(token)
                            i += len(token)
                            break
                    else:
                        # If no special token is found, treat '<' as a normal character
                        if found_eoi:
                            tokens_after_eoi.append('<')
                        else:
                            tokens_before_eoi.append('<')
                        i += 1
                else:
                    # Normal character
                    if found_eoi:
                        if line[i] != '\n':
                            tokens_after_eoi.append(line[i])
                    else:
                        if line[i] != '\n':
                            tokens_before_eoi.append(line[i])
                    i += 1

            result.append((tokens_before_eoi, tokens_after_eoi))

    return result