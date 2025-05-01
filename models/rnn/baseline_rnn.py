'''
baseline rnn model to control a XY gantry with natural language commands
'''

import torch as pt
from typing import Type, Tuple, Sequence, Mapping
from vocab.vocab import Vocab, START_TOKEN, END_TOKEN
from tqdm import tqdm
import random

RNNType: Type = Type["RNN"]
StateType: Type = pt.Tensor

output_tokens: int = 18
OUTPUT_TOKENS = ["<MOVE>", "<SPEED>", "<ROTATE>", "<SC>", "<EC>", "<EOS>", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", " ", "."]

class RNN(pt.nn.Module):
    def __init__(self: RNNType, data: Sequence[Tuple[Sequence[str], Sequence[str]]], saved_model_path: str = None, num_epochs: int = 1) -> None:
        super().__init__()

        self.vocab = Vocab()

        self.vocab.add(START_TOKEN)

        # Init vocab
        b = False
        for task, command in data:
            
            for c in task:
                self.vocab.add(c)
            for c in command:
                self.vocab.add(c)

        # |vocab| -> 64 -> |OUTPUT_VOCAB| 
        
        self.rnn = pt.nn.RNNCell(len(self.vocab), 64)
        self.fc = pt.nn.Linear(64, 18)

        self.error = pt.nn.CrossEntropyLoss()
        self.optimizer = pt.optim.Adam(self.parameters(), lr=1e-3)
    
        if saved_model_path is None:
            for epoch in range(num_epochs):

                random.shuffle(data)
                train_chars = 0

                for line in tqdm(data, desc=f"epoch {epoch}"):

                    self.optimizer.zero_grad()
                    loss = 0.

                    q = self.start()

                    for c_in, c_out in zip([START_TOKEN] + line[0], line[1] + [END_TOKEN]):
                        train_chars += 1
                        q, p = self.step(q, self.vocab.numberize(c_in))
                        
                        ground_truth = pt.tensor([OUTPUT_TOKENS.index(c_out)])

                        loss += self.error(p, ground_truth)
                    
                    loss.backward()

                    self.optimizer.step()

                    pt.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                pt.save(self.state_dict(), f"./weights/rnn_{epoch}.model")
        else:
            print(f"loading model from {saved_model_path}")
            self.load_state_dict(pt.load(saved_model_path, weights_only=True))

    
    def start(self: RNNType) -> StateType:
        return pt.zeros(1, 64)

    def forward(self: RNNType, input_word: int, hidden) -> pt.Tensor:
        one_hot = pt.zeros(1, len(self.vocab))
        one_hot[0, input_word] = 1

        hidden = self.rnn(one_hot, hidden)
        
        output_logits = self.fc(hidden)
        
        # output_logits: [batch, vocab_size]
        return output_logits, hidden

    def step(self: RNNType, state: StateType, x: int) -> Tuple[StateType, Mapping[str, float]]:
        logits, state = self.forward(x, state)

        return state, pt.log_softmax(logits, dim=0)