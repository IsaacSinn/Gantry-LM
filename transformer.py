'''
Transformer model to control a XY gantry with natural language commands
'''

import torch as pt
from torch import nn
from typing import Sequence, Tuple
from vocab import Vocab, START_TOKEN, END_TOKEN
from tqdm import tqdm
import random

class TransformerModel(nn.Module):
    def __init__(self, data: Sequence[Tuple[Sequence[str], Sequence[str]]], saved_model_path: str = None, num_epochs: int = 1) -> None:
        super().__init__()

        self.vocab = Vocab()
        self.vocab.add(START_TOKEN)
        self.vocab.add(END_TOKEN)

        # Initialize vocab
        for task, command in data:
            for c in task:
                self.vocab.add(c)
            for c in command:
                self.vocab.add(c)

        self.vocab_size = len(self.vocab)
        self.embedding_dim = 64
        self.num_heads = 4
        self.num_layers = 2
        self.output_tokens = 18

        # Embedding layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.positional_encoding = self._generate_positional_encoding(100, self.embedding_dim)

        # Transformer layers
        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dim_feedforward=256,
            dropout=0.1,
            activation='relu'
        )

        # Output layer
        self.fc_out = nn.Linear(self.embedding_dim, self.output_tokens)

        # Loss and optimizer
        self.error = nn.CrossEntropyLoss()
        self.optimizer = pt.optim.Adam(self.parameters(), lr=1e-3)

        if saved_model_path is None:
            for epoch in range(num_epochs):
                random.shuffle(data)
                train_chars = 0

                for line in tqdm(data, desc=f"epoch {epoch}"):
                    self.optimizer.zero_grad()
                    loss = 0.

                    input_seq = [START_TOKEN] + line[0]
                    target_seq = line[1] + [END_TOKEN]

                    input_indices = pt.tensor([self.vocab.numberize(c) for c in input_seq]).unsqueeze(1)
                    target_indices = pt.tensor([self.vocab.numberize(c) for c in target_seq]).unsqueeze(1)

                    input_emb = self._embed_with_positional_encoding(input_indices)
                    target_emb = self._embed_with_positional_encoding(target_indices[:-1])

                    output = self.transformer(
                        src=input_emb,
                        tgt=target_emb,
                        src_key_padding_mask=None,
                        tgt_key_padding_mask=None,
                        memory_key_padding_mask=None
                    )

                    logits = self.fc_out(output)
                    ground_truth = target_indices[1:].squeeze(1)

                    loss = self.error(logits.view(-1, self.output_tokens), ground_truth)
                    loss.backward()
                    self.optimizer.step()

                pt.save(self.state_dict(), f"./transformer_{epoch}.model")
        else:
            print(f"loading model from {saved_model_path}")
            self.load_state_dict(pt.load(saved_model_path, weights_only=True))

    def _generate_positional_encoding(self, max_len: int, d_model: int) -> pt.Tensor:
        pe = pt.zeros(max_len, d_model)
        position = pt.arange(0, max_len, dtype=pt.float).unsqueeze(1)
        div_term = pt.exp(pt.arange(0, d_model, 2).float() * (-pt.log(pt.tensor(10000.0)) / d_model))
        pe[:, 0::2] = pt.sin(position * div_term)
        pe[:, 1::2] = pt.cos(position * div_term)
        return pe.unsqueeze(0)

    def _embed_with_positional_encoding(self, indices: pt.Tensor) -> pt.Tensor:
        embeddings = self.embedding(indices).squeeze(1)
        seq_len = embeddings.size(0)
        return embeddings + self.positional_encoding[:, :seq_len, :]

    def forward(self, input_seq: Sequence[str], target_seq: Sequence[str]) -> pt.Tensor:
        input_indices = pt.tensor([self.vocab.numberize(c) for c in input_seq]).unsqueeze(1)
        target_indices = pt.tensor([self.vocab.numberize(c) for c in target_seq]).unsqueeze(1)

        input_emb = self._embed_with_positional_encoding(input_indices)
        target_emb = self._embed_with_positional_encoding(target_indices[:-1])

        output = self.transformer(
            src=input_emb,
            tgt=target_emb,
            src_key_padding_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )

        logits = self.fc_out(output)
        return logits