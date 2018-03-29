# char-rnn.pytorch

A PyTorch implementation of [char-rnn](https://github.com/karpathy/char-rnn) for character-level text generation. This is modified with respect to [the Practical PyTorch series](https://github.com/spro/practical-pytorch/blob/master/char-rnn-generation/char-rnn-generation.ipynb).

## Training

Download [dataset](https://cs.stanford.edu/people/karpathy/char-rnn/) (from the original char-rnn) as `[filename].txt`.  Or bring your own dataset &mdash; it should be a plain text file (preferably ASCII).

Run `char_rnn.py` with the dataset filename to train and save the network:

```
> python3 char_rnn.py [filename].txt

Training for 2000 epochs...
(... 30 minutes (without gpu) later ...)
Saved as [filename].pt
```
After training the model will be saved as `[filename].pt` and will be used to generate text predictions afterwards.

### Training options

```
Usage: char_rnn.py [filename] [options]

Options:
--encoding_way     Whether ASCII or not                ascii
--model            Whether to use LSTM or GRU units    gru
--n_epochs         Number of epochs to train           2000
--hidden_size      Hidden size of network              50
--n_layers         Number of network layers            2
--learning_rate    Learning rate                       0.01
--chunk_len        Length of training chunks           200
--batch_size       Number of examples per batch        100
```

## Generation

Run `predict.py` with the saved model from training.

```
> python3 predict.py [filename].pt [options]

(... some chunks of characters generated ...)
```

### Generation options
```
Usage: predict.py [filename] [options]

Options:
-p, --prime_str      String to prime generation with       A
-l, --predict_len    Length of prediction                  1000
-t, --temperature    Temperature (higher is more chaotic)  0.8
```

