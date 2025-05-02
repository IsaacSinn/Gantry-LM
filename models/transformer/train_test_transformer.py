'''
To Run inference on this model, use the following command:
python models/transformer/train_test_transformer.py

!Note! Some imports may need to be changed depending on your directory structure.
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch as pt
from transformer import TransformerModel
from vocab.vocab import Vocab, START_TOKEN, END_TOKEN, UNK_TOKEN, PADDING_TOKEN
from data.charloader import charloader_generator
from tqdm import tqdm
from typing import Sequence, Tuple
from torch import nn

def train_transformer(model: TransformerModel, filepath: str, optimizer, loss_func, device) -> None:
    print("Starting to train the model...")
    model.train_model(
        optimizer=optimizer,
        loss_function=loss_func,
        filepath=filepath,
        device=device,
        num_epochs= 500,
        batch_size=8
    )
    print("Finished training the model!")

def test_transformer(model: TransformerModel, filepath, device, batch_size: int = 32) -> None:
    print("Starting to test the model...")
    model.eval()

    total_loss = 0.0
    num_entries = 0
    loss_function = nn.CrossEntropyLoss(ignore_index=model.vocab.numberize(PADDING_TOKEN))

    batch = []

    with pt.no_grad():
        for prompt_tokens, completion_tokens in charloader_generator(filepath):
            prompt_ids = [model.vocab.numberize(token) for token in prompt_tokens]
            completion_ids = [model.vocab.numberize(token) for token in completion_tokens]

            batch.append((prompt_ids, completion_ids))

            if len(batch) == batch_size:
                src_batch, tgt_batch = model.collate_batch(batch, device)

                tgt_input = tgt_batch[:, :-1]
                tgt_output = tgt_batch[:, 1:]

                logits = model(src_batch, tgt_input)

                flat_logits = logits.view(-1, logits.size(-1))
                flat_targets = tgt_output.view(-1)

                loss = loss_function(flat_logits, flat_targets)

                total_loss += loss.item()
                num_entries += len(batch)
                batch = []

        # Handle leftover data
        if len(batch) > 0:
            src_batch, tgt_batch = model.collate_batch(batch, device)

            tgt_input = tgt_batch[:, :-1]
            tgt_output = tgt_batch[:, 1:]

            logits = model(src_batch, tgt_input)

            flat_logits = logits.view(-1, logits.size(-1))
            flat_targets = tgt_output.view(-1)

            loss = loss_function(flat_logits, flat_targets)

            total_loss += loss.item()
            num_entries += len(batch)

        # Print Loss
        print(f"Test Loss: {total_loss / num_entries:.4f}")
        return total_loss, num_entries
    
def inference(model: TransformerModel, prompt: str, device) -> str:
    return model.generate(prompt, device, max_len=100)

def main():
    print("Starting")
    device = 'cuda' if pt.cuda.is_available() else 'cpu'

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "transformer_params.pt")

    if os.path.exists(model_path):
        print(f"Loading saved model from {model_path}")
        transformer = TransformerModel()
        transformer.load_state_dict(pt.load(model_path, map_location=device))
        transformer = transformer.to(device)
    else:
        print("No saved model found. Initializing new model and starting training.")
        transformer = TransformerModel()
        transformer = transformer.to(device)

        optimizer = pt.optim.Adam(transformer.parameters(), lr=0.001)
        loss_function = nn.CrossEntropyLoss(reduction='none', ignore_index=transformer.vocab.numberize(PADDING_TOKEN))

        train_file = 'data/train_2200.jsonl'
        train_transformer(transformer, train_file, optimizer, loss_function, device)

    # Inference loop
    while True:
        prompt = input("Enter a prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break

        # Generate completion
        completion = inference(transformer, prompt, device)
        print(f"Generated completion: {completion}")

if __name__ == "__main__":
    main()