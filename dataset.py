from torchtext.data import Field, Dataset, BucketIterator, BPTTIterator
from torchtext.datasets import LanguageModelingDataset
import spacy
spacy_en = spacy.load('en')

def tokenizer2(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

if __name__ == '__main__':
    W2V = Field()
    TEXT = Field(sequential=True, tokenize=tokenizer2, lower=True, is_target=False)
    lang = LanguageModelingDataset(path='./dataset/prideand.txt', text_field=TEXT)
    print(lang[0].text[:10])
    TEXT.build_vocab(lang, min_freq=3)
    vocab = TEXT.vocab
    vocab.load_vectors('glove.6B.100d')
    counter = vocab.freqs
    print( vocab.vectors.shape )
    train_loader = BPTTIterator(dataset=lang, batch_size=64, bptt_len=50)
    train_loader.create_batches()
    for idx, data_ in enumerate(train_loader):
        if idx == 1: break
        print(idx)
        print(data_.text[1])
        for i in data_.text[1]:
            print(vocab.itos[i], end=' ')
        print(data_.target[0])
        print('\n')
        for i in data_.target[0]:
            print(vocab.itos[i], end=' ')
        print('\n')
