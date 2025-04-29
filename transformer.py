'''
Transformer model to control a XY gantry with natural language commands
'''

import torch as pt
from torch import nn
from typing import Sequence, Tuple
from vocab import Vocab, START_TOKEN, END_TOKEN, UNK_TOKEN, PADDING_TOKEN
from tqdm import tqdm
from v_builder import vocab_builder
from data.charloader import charloader_generator
import random
import pickle

class TransformerModel(nn.Module):
    def __init__(self, saved_model_path: str = None) -> None:
        super().__init__()

        self.vocab = None

        # Load vocab from file if it exists
        try:
            with open('vocab.pkl', 'rb') as f:
                print("Loading vocab from file.")
                self.vocab = pickle.load(f)
        except FileNotFoundError:
            print("Vocab file not found. Creating a new vocab.")
            self.vocab = vocab_builder()

        print("UNK token index:", self.vocab.word_to_num[UNK_TOKEN])
        print("Vocab size:", len(self.vocab))

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
        '''
        Forward pass through the transformer model.
        src: source sequence (input), (batch_size, seq_len), src[i][j] = token index of j-th token in i-th sequence
        tgt: target sequence (output)
        '''
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

        print("Logits shape:", logits.shape)

        return logits
    
    def train_one_epoch(self, optimizer, loss_function, data_loader, device) -> float:
        '''
        Train the model for one epoch.
        optimizer: optimizer to use for training
        loss_function: loss function to use for training
        data_loader: data loader to use for training
        device: device to use for training (CPU or GPU)
        '''
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
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_targets = tgt_output.reshape(-1)
            
            # Calc loss and backprop
            loss = loss_function(flat_logits, flat_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(data_loader)
        return average_loss
    
    def train_model(self, optimizer, loss_function, filepath, device, num_epochs=1000, batch_size=8) -> None:
        '''
        Train the model on the given data.
        optimizer: optimizer to use for training
        ...
        '''
        epoch_count = 0

        for epoch in range(num_epochs):
            total_loss = 0
            batch = []

            full_dataset = list(charloader_generator(filepath))
            random.shuffle(full_dataset)

            for prompt_tokens, completion_tokens in full_dataset:

                # Need to pass these as vocab ID's, including UNK's

                print(len(prompt_tokens), len(completion_tokens))  

                prompt_ids = [self.vocab.numberize(token) for token in prompt_tokens][:self.max_seq_len]
                completion_ids = [self.vocab.numberize(token) for token in completion_tokens][:self.max_seq_len]

                batch.append((prompt_ids, completion_ids))

                # Train on this batch
                if len(batch) == batch_size:
                    src_batch, tgt_batch = self.collate_batch(batch, device)

                    print("Sending a batch")

                    avg_loss = self.train_one_epoch(optimizer, loss_function, [(src_batch, tgt_batch)], device)
                    total_loss += avg_loss

                    batch = []

            # Leftover data
            if len(batch) > 0:
                src_batch, tgt_batch = self.collate_batch(batch, device)
                avg_loss = self.train_one_epoch(optimizer, loss_function, [(src_batch, tgt_batch)], device)
                total_loss += avg_loss

            epoch_count += 1
            if epoch_count % 25 == 0:
                # Save entire model
                pt.save(self, f"model_epoch_{epoch_count}.pt")

        print(f"Epoch {epoch+1}/{num_epochs} | Total Loss: {total_loss:.4f}")

    def collate_batch(self, batch: Sequence[Tuple[Sequence[int], Sequence[int]]], device: str) -> Tuple[pt.Tensor, pt.Tensor]:
        '''
        Collate a batch of data into a tensor that can be passed to the model.
        Pads the sequences to the same length and converts them to tensors.
        '''

        src_batch = []
        tgt_batch = []

        for prompt_ids, completion_ids in batch:
            src_batch.append(pt.tensor(prompt_ids, dtype=pt.long))
            tgt_batch.append(pt.tensor(completion_ids, dtype=pt.long))

        src_padded = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=self.vocab.numberize(PADDING_TOKEN))
        tgt_padded = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=self.vocab.numberize(PADDING_TOKEN))

        return src_padded.to(device), tgt_padded.to(device)









