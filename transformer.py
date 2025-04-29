'''
Transformer model to control a XY gantry with natural language commands
'''

import torch as pt
from torch import nn
from typing import Sequence, Tuple
from vocab import Vocab, START_TOKEN, END_TOKEN
from tqdm import tqdm
from v_builder import vocab_builder
from data.charloader import charloader_generator
import random

class TransformerModel(nn.Module):
    def __init__(self, data: Sequence[Tuple[Sequence[str], Sequence[str]]], saved_model_path: str = None, num_epochs: int = 1) -> None:
        super().__init__()

        self.vocab = None

        # Load vocab from file if it exists
        try:
            with open('vocab.pkl', 'rb') as f:
                print("Loading vocab from file.")
                self.vocab = pt.load(f)
        except FileNotFoundError:
            print("Vocab file not found. Creating a new vocab.")
            self.vocab = vocab_builder()

        self.vocab_size = len(self.vocab)
        self.embedding_dim = 256
        self.num_heads = 4
        self.num_layers = 6
        self.hidden_dim = 1024
        self.max_seq_len = 2048
        self.output_tokens = self.vocab_size

        # Embeddings
        self.positional_embedding = nn.Embedding(self.max_seq_len, self.embedding_dim)
        self.token_embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # Init Transformer
        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            num_encoder_layers=self.num_layers,
            num_decoder_layers=self.num_layers,
            dim_feedforward=self.hidden_dim,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )

        self.output_layer = nn.Linear(self.embedding_dim, self.output_tokens)

    def forward(self, src: pt.Tensor, tgt: pt.Tensor) -> pt.Tensor:
        batch_size, seq_len = src.size()
        _, tgt_len = tgt.size()

        device = src.device

        # Turn IDs into embeddings
        src_embedding = self.token_embedding(src) + self.positional_embedding(pt.arange(seq_len, device=src.device).unsqueeze(0))
        tgt_embedding = self.token_embedding(tgt) + self.positional_embedding(pt.arange(tgt_len, device=tgt.device).unsqueeze(0))

        # Create a mask for the target sequence
        target_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(device)

        # Pass through the transformer
        output = self.transformer(
            src_embedding,
            tgt_embedding,
            tgt_mask=target_mask
        )

        logits = self.output_layer(output)

        return logits
    
    def train_one_epoch(self, optimizer, loss_function, data_loader, device):
        self.train()

        total_loss = 0.0

        for src, tgt in data_loader:
            src = src.to(device)
            tgt = tgt.to(device)

            # Align target sequence
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            
            # Flatten for loss function
            logits = self(src, tgt_input)
            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = tgt_output.view(-1)
            
            # Calc loss and backprop
            loss = loss_function(flat_logits, flat_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        return average_loss
    









