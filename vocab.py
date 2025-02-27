# SYSTEM IMPORTS
from collections.abc import Collection, Iterable, Mapping, MutableSet, Sequence, Set
from typing import Type, Tuple


# PYTHON PROJECT IMPORTS


# Types declared in this module
VocabType: Type = Type["Vocab"]


# Constants declared in this module
START_TOKEN: str = "<BOS>"
END_TOKEN: str = "<EOS>"
UNK_TOKEN: str = "<UNK>"


class Vocab(MutableSet):
    """Set-like data structure that can change words into numbers and back."""

    def __init__(self: VocabType) -> None:
        words: Set[str] = {START_TOKEN, END_TOKEN, UNK_TOKEN}
        self.num_to_word: Sequence[str] = list(words)    
        self.word_to_num: Mapping[str, int] = {word:num for num, word in enumerate(self.num_to_word)}

    def add(self: VocabType,
            word: str
            ) -> None:
        if word not in self:
            num: int = len(self.num_to_word)
            self.num_to_word.append(word)
            self.word_to_num[word] = num

    def discard(self: VocabType,
                word: str
                ) -> None:
        raise NotImplementedError()

    def update(self: VocabType,
               words: Collection[str]
               ) -> None:
        self |= words

    def __contains__(self: VocabType,
                     word: str
                     ) -> bool:
        return word in self.word_to_num

    def __len__(self: VocabType) -> int:
        return len(self.num_to_word)

    def __iter__(self: VocabType) -> Iterable[str]:
        return iter(self.num_to_word)

    def numberize(self: VocabType,
                  word: str
                  ) -> int:
        """Convert a word into a number."""
        if word in self.word_to_num:
            return self.word_to_num[word]
        else: 
            return self.word_to_num[UNK_TOKEN]

    def denumberize(self: VocabType,
                    num: int
                    ) -> str:
        """Convert a number into a word."""
        return self.num_to_word[num]

