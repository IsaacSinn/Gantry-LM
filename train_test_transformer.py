import torch as pt
from transformer import TransformerModel
from vocab import Vocab, START_TOKEN, END_TOKEN, UNK_TOKEN, PADDING_TOKEN
from data.charloader import charloader_generator
from tqdm import tqdm
from typing import Sequence, Tuple
from torch import nn

def train_transformer(model: TransformerModel, filepath: str, optimizer, loss_func, device) -> None:
    print("Starting to train the model...")
    model.train_model(
        optimizer=optimizer,
        loss_func=loss_func,
        filepath=filepath,
        device=device,
        num_epochs= 50,
        batch_size=32
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

def main():
    device = 'cuda' if pt.cuda.is_available() else 'cpu'

    transformer = TransformerModel()
    transformer = transformer.to(device)

    optimizer = pt.optim.Adam(transformer.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss(ignore_index=transformer.vocab.numberize(PADDING_TOKEN))

    train_file = 'data/dev_100.jsonl'
    test_file = 'data/dev_sample.jsonl'

    # Train the model
    train_transformer(transformer, train_file, optimizer, loss_function)

    # Test the model
    test_transformer(transformer, test_file, device)

if __name__ == "__main__":
    main()