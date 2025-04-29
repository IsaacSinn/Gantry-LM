from vocab import Vocab, START_TOKEN, END_TOKEN, UNK_TOKEN, PADDING_TOKEN

import string
import pickle

# Characters you should include
CHARACTER_SET = (
    list(string.ascii_letters) +  # a-z, A-Z
    list(string.digits) +          # 0-9
    list(" .,!?-_:/'()[]{}\"") +    # common punctuation and G-code useful symbols
    ["+", "=", "*", "&", "%", "$", "#", "@", "<", ">", "\\", "|", "^", "~"] +  # rare but safe extras
    [" ", "\n"]                    # space and newline (critical for G-code structure)
)

def vocab_builder():
    vocab = Vocab()
    for char in CHARACTER_SET:
        vocab.add(char)

    vocab.add(START_TOKEN)
    vocab.add(END_TOKEN)
    vocab.add(UNK_TOKEN)
    vocab.add(PADDING_TOKEN)
 
    return vocab

def main():
    vocab = vocab_builder()

    # Save vocab object to a pkl file
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

if __name__ == "__main__":
    main()


