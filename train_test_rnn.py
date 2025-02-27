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
from data.charloader import load_chars_from_file
from baseline_rnn import RNN
from vocab import START_TOKEN, END_TOKEN

def train_rnn() -> RNN:
    train_data: Sequence[Sequence[str]] = load_chars_from_file("./data/chatgpt_train")

    saved_model_path = None
    if os.path.exists("./weights/rnn_1.model"):
        saved_model_path = "./weights/rnn_1.model"

    return RNN(train_data, saved_model_path=saved_model_path, num_epochs=10)

def dev_rnn(m: RNN) -> Tuple[int, int]:
    dev_data: Sequence[Sequence[str]] = load_chars_from_file("./data/chatgpt_dev")

    num_correct: int = 0
    total: int = 0
    for dev_line in dev_data:
        q = m.start()

        for c_input, c_actual in zip([START_TOKEN] + dev_line,
                                      dev_line + [END_TOKEN]):
            q, p = m.step(q, m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(p.argmax())

            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total


def main() -> None:
    m: RNN = train_rnn()
    num_correct, total = dev_rnn(m)
    print(f"Accuracy: {num_correct}/{total} = {num_correct/total:.2f}")

if __name__ == "__main__":
    main()