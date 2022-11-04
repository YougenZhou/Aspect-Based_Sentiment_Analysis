import pickle


class Vocab(object):
    def __init__(self, counter, specials):
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.li_toks = list(specials)
        for tok in specials:
            del counter[tok]
        word_and_frequences = sorted(counter.items(), key=lambda tup: tup[0])
        word_and_frequences.sort(key=lambda tup: tup[1], reverse=True)
        for word, _ in word_and_frequences:
            self.li_toks.append(word)
        self.toks2index = {tok: i for i, tok in enumerate(self.li_toks)}

    def __len__(self):
        return len(self.li_toks)

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            print('Saving vocab to:', vocab_path)
            pickle.dump(self, f)
