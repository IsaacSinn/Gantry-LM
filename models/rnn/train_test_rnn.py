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
from baseline_rnn import RNN
from vocab.vocab import START_TOKEN, END_TOKEN

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

def train_rnn() -> RNN:
    train_data: Sequence[Sequence[str]] = parse_tokens_from_file("./data/dev")

    saved_model_path = None
    if os.path.exists("./weights/rnn_1.model"):
        saved_model_path = "./weights/rnn_1.model"

    return RNN(train_data, saved_model_path=saved_model_path, num_epochs=10)

def dev_rnn(m: RNN) -> Tuple[int, int]:
    dev_data: Sequence[tuple[Sequence[str], Sequence[str]]] = parse_tokens_from_file("./data/dev")

    metric_sum: int = 0
    total: int = 0

    # We can summarize metric by averaging the metric of each line

    for line in dev_data:
        input, output = line
        output.append("<EOS>")
        q = m.start()

        eoi = False
        sample_out = []

        for c_input in ([START_TOKEN] + input):
            q, p = m.step(q, m.vocab.numberize(c_input))

            if c_input == "<EOI>":
                eoi = True
                continue

            if eoi:
                # Max index 
                p = pt.argmax(p)
                sample_out.append(OUTPUT_TOKENS[p])

        metric_sum += metric(output, sample_out)
        total += max(len(output), len(sample_out))

    return metric_sum, total


def main() -> None:
    m: RNN = train_rnn()
    metric_total, total = dev_rnn(m)
    print(f"WTAS: {metric_total}/{total} = {metric_total/total:.2f}")

if __name__ == "__main__":
    main()