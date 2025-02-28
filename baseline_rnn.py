'''
baseline rnn model to control a XY gantry with natural language commands
'''

import torch as pt
from typing import Type, Tuple, Sequence, Mapping
from vocab import Vocab, START_TOKEN, END_TOKEN
from tqdm import tqdm
import random

RNNType: Type = Type["RNN"]
StateType: Type = pt.Tensor

class RNN(pt.nn.Module):
    def __init__(self: RNNType, data: Sequence[Sequence[str]], saved_model_path: str = None, num_epochs: int = 1) -> None:
        super().__init__()

        self.vocab = Vocab()

        self.vocab.add(START_TOKEN)

        for line in data: 
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)

        # |vocab| -> 64 -> |vocab| 
        
        self.rnn = pt.nn.RNNCell(len(self.vocab), 64)
        self.fc = pt.nn.Linear(64, len(self.vocab))

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

                    for c_in, c_out in zip([START_TOKEN] + line, line + [END_TOKEN]):
                        train_chars += 1
                        q, p = self.step(q, self.vocab.numberize(c_in))

                        print(c_in, c_out)
                        print(self.vocab.numberize(c_out))
                        
                        ground_truth = pt.tensor([self.vocab.numberize(c_out)])

                        # print dimension of p and ground truth
                        print(p.shape, ground_truth.shape)

                        loss += self.error(p, ground_truth)
                    
                    loss.backward()

                    self.optimizer.step()

                    pt.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                pt.save(self.state_dict(), f"./rnn_{epoch}.model")
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

        logits = logits.squeeze(0)

        return state, pt.log_softmax(logits, dim=0)