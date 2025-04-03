# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Tuple
from pprint import pprint
import argparse as ap
import numpy as np
import os
import sys
import torch as pt

# PYTHON PROJECT IMPORTS
from data.charloader import parse_tokens_from_file
from transformer import TransformerModel
from vocab import START_TOKEN, END_TOKEN

# CONSTANTS
SPECIAL_TOKENS = set(["<BOS>", "<EOS>", "<EOI>", "<MOVE>", "<SPEED>", "<ROTATE>", "<SC>", "<EC>"])
TOOL_TOKENS = set(["<MOVE>", "<SPEED>", "<ROTATE>"])
MISC_TOKENS = set(["<SC>", "<EC>", "<EOS>"])
token_types = {"<MOVE>": "tool", "<SPEED>": "tool", "<ROTATE>": "tool", "<SC>": "misc", "<EC>": "misc", "<EOS>": "misc"}
tool_weight = 5
misc_weight = 3
normal_weight = 1
OUTPUT_TOKENS = ["<MOVE>", "<SPEED>", "<ROTATE>", "<SC>", "<EC>", "<EOS>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ", "."]

def weight_function(t_i: str) -> int:
    if t_i in TOOL_TOKENS:
        return tool_weight
    elif t_i in MISC_TOKENS:
        return misc_weight
    else:
        return normal_weight

def distance_function(t_i: str, p_i: str) -> float:
    if t_i == p_i:
        return 0
    elif len(t_i) != len(p_i) or token_types[t_i] != token_types[p_i]:
        return 1
    else:
        return 1 - (int(t_i) / int(p_i))

def metric(T: Sequence[str], P: Sequence[str]) -> float:
    n = max(len(T), len(P))

    # Fill out the smaller one with <EOS> tokens
    while len(T) < n:
        T.append("<EOS>")
    
    while len(P) < n:
        P.append("<EOS>")

    numerator = 0
    denominator = 0

    for i in range(n):
        w_i = weight_function(T[i])
        d_i = distance_function(T[i], P[i])
        numerator += w_i * d_i
        denominator += w_i

    return 1 - (numerator / denominator)

def train_transformer() -> TransformerModel:
    train_data: Sequence[Sequence[str]] = parse_tokens_from_file("./data/dev")

    print(train_data[0])

    saved_model_path = None
    if os.path.exists("./weights/transformer_1.model"):
        saved_model_path = "./weights/transformer_1.model"

    return TransformerModel(train_data, saved_model_path=saved_model_path, num_epochs=1)

def dev_transformer(m: TransformerModel) -> Tuple[int, int]:
    dev_data: Sequence[tuple[Sequence[str], Sequence[str]]] = parse_tokens_from_file("./data/dev")

    metric_sum: int = 0
    total: int = 0

    # We can summarize metric by averaging the metric of each line

    for line in dev_data:
        input, output = line
        output.append("<EOS>")
        input_indices = [m.vocab.numberize(c) for c in [START_TOKEN] + input]
        target_indices = [m.vocab.numberize(c) for c in output]

        input_tensor = pt.tensor(input_indices).unsqueeze(1)
        target_tensor = pt.tensor(target_indices).unsqueeze(1)

        q = m.start()
        eoi = False
        sample_out = []

        for c_input in input_tensor:
            logits = m.forward(input_tensor, target_tensor[:-1])
            p = pt.argmax(logits[-1], dim=-1).item()
            sample_out.append(OUTPUT_TOKENS[p])

            if OUTPUT_TOKENS[p] == "<EOI>":
                eoi = True
                continue

        metric_sum += metric(output, sample_out)
        total += max(len(output), len(sample_out))

    return metric_sum, total

def main() -> None:
    m: TransformerModel = train_transformer()
    metric_total, total = dev_transformer(m)
    print(f"WTAS: {metric_total}/{total} = {metric_total/total:.2f}")

if __name__ == "__main__":
    main()