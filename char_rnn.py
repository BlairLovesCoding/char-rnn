import codecs
import unidecode
import string
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


# read and un-unicode-encoding data
all_characters = string.printable
n_characters = len(all_characters)


# turn any potential unicode characters into plain ASCII
def read_file(filename, encoding_way):
    if encoding_way == "ascii":
        f = unidecode.unidecode(open(filename).read())
    else:
        op = codecs.open(filename, 'r', 'ascii', 'ignore')
        f = unidecode.unidecode(op.read())
    return f, len(f)


# turn a string into a tensor
def char_tensor(str):
    tens = torch.zeros(len(str)).long()
    for c in range(len(str)):
        try:
            tens[c] = all_characters.index(str[c])
        except:
            continue
    return tens


# build up rnn model
# There are three layers:
# one linear layer that encodes the input character into an internal state,
# one GRU/LSTM layer (which may itself have multiple layers) that operates on that internal state and a hidden state,
# and a decoder layer that outputs the probability distribution.
class charRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="gru", n_layers=2):
        super(charRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, inp, hidden):
        batch_size = inp.size(0)
        encoded = self.encoder(inp)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def forward2(self, inp, hidden):
        encoded = self.encoder(inp.view(1, -1))
        output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))


# parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--encoding_way', type=str, default="ascii")
argparser.add_argument('--model', type=str, default="gru")
argparser.add_argument('--n_epochs', type=int, default=2000)
argparser.add_argument('--hidden_size', type=int, default=100)
argparser.add_argument('--n_layers', type=int, default=2)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
args = argparser.parse_args()

file, file_len = read_file(args.filename, args.encoding_way)


# assemble a pair of input and target tensors from a random chunk
# The input will be all characters up to the last, and the target will be all characters from the first.
def random_training_set(chunk_len, batch_size):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    # split inputs(big string of data) into chunks
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    return inp, target


# split test set
test_inp = torch.LongTensor(1, args.chunk_len)
test_target = torch.LongTensor(1, args.chunk_len)
for bi in range(1):
    start_index = random.randint(0, file_len - args.chunk_len)
    end_index = start_index + args.chunk_len + 1
    chunk = file[start_index:end_index]
    test_inp[bi] = char_tensor(chunk[:-1])
    test_target[bi] = char_tensor(chunk[1:])
test_inp = Variable(test_inp)
test_target = Variable(test_target)


def train(inp, target):
    hidden = decoder.init_hidden(args.batch_size)
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data[0] / args.chunk_len


def test(inp, target):
    hidden = decoder.init_hidden(1)
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(1, -1), target[:,c])

    return loss.data[0] / args.chunk_len


def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


# initialize models and start training
decoder = charRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()


train_losses = []
test_losses = []

try:
    print("Training for %d epochs..." % args.n_epochs)
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        loss = train(*random_training_set(args.chunk_len, args.batch_size))
        train_losses.append(loss)

        tloss = test(test_inp, test_target)
        test_losses.append(tloss)

    print("Final test loss: %f" % tloss)
    print("Saving...")
    save()

except KeyboardInterrupt:
    print("Saving before quit...")
    save()

plt.figure()
plt.plot(train_losses, c='blue', label='train loss')
plt.plot(test_losses, c='orange', label='test loss')
plt.xlabel("epoch number")
plt.ylabel("cross entropy loss")
plt.title("loss vs. epoch")
plt.legend(loc='upper right')
plt.show()