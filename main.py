import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchtext.data import Field, Dataset, BucketIterator, BPTTIterator
from torchtext.datasets import LanguageModelingDataset
import nltk
from nltk.corpus import stopwords
import spacy
spacy_en = spacy.load('en')

from lstm_lm import LSTMLM

MODEL_SAVE_PATH = './model/pr_lm_model.pt'

GPU = torch.cuda.is_available()
EMBEDDING_DIM = 100
NUM_LAYERS = 2
HIDDEN_DIM = 50
BATCH_SIZE = 128
EPOCH_NUM = 100000
lr = 0.1

def to_onehot(target, n_dims=None):
   bptt_size, batch_size = target.shape[0], target.shape[1]
   target_onehot = torch.zeros([bptt_size, batch_size, n_dims], dtype=torch.long)
   src_target = target.unsqueeze(2)
   target_onehot.scatter_(2, src_target, 1)
   return target_onehot

def repackage_hidden(h):
   if isinstance(h, torch.Tensor):
      return h.detach()
   else:
      return tuple(repackage_hidden(v) for v in h)

def train(epoch, model, device, train_loader, optimizer, vocab_size):
   model.train()
   hidden = model.init_hidden(BATCH_SIZE)
   criterion = nn.CrossEntropyLoss()
   for ep in range(epoch):
      total_loss = 0.
      for batch_idx, (data) in enumerate(train_loader):
         text, target = data.text.to(device), data.target.to(device)
         #target = to_onehot(target, vocab_size)
         hidden = repackage_hidden(hidden)
         model.zero_grad()
         output, hidden = model(text, hidden)
         loss = criterion(output.view(-1, vocab_size), target.view(-1,1).squeeze())
         loss.backward()
         torch.nn.utils.clip_grad_norm(model.parameters(), 0.5)
         for i, p in enumerate( model.parameters() ):
            if p.grad is not None:
               p.data.add_(-lr*p.grad.data)

         total_loss += loss.item()
      print(total_loss/float(batch_idx))



def test(model, device, test_loader):
   model.eval()


def tokenizer2(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def main():
   nltk.download('stopwords')
   en_stopwords = stopwords.words('english')

   TEXT = Field(sequential=True, tokenize=tokenizer2, lower=True, stop_words=en_stopwords)
   lang = LanguageModelingDataset(path='./dataset/prideand.txt', text_field=TEXT)
   TEXT.build_vocab(lang, min_freq=3)
   vocab = TEXT.vocab
   vocab.load_vectors('glove.6B.100d')
   vocab_size = vocab.vectors.shape[0]

   device = torch.device('cuda' if GPU else 'cpu')
   model = LSTMLM(vocab_size, EMBEDDING_DIM, NUM_LAYERS, HIDDEN_DIM, GPU).to(device)
   optimizer = optim.SGD(model.parameters(), lr=0.01)
   model.init_hidden(128)

   model.set_embed_parameter(vocab.vectors)

   train_loader = BPTTIterator(dataset=lang, batch_size=BATCH_SIZE, bptt_len=30)
   train(EPOCH_NUM, model, device, train_loader, optimizer, vocab_size)
   torch.save(model.state_dict(), MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()
