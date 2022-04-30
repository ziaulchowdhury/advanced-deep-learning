#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable
import os

from tqdm import tqdm

from helpers import char_tensor, read_file, time_since, all_characters, n_characters
from model import CharRNN
from generate import generate
import time
import random

def random_training_set(chunk_len, batch_size, cuda, file_len, file):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = char_tensor(chunk[:-1])
        target[bi] = char_tensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    if cuda:
        inp = inp.cuda()
        target = target.cuda()
    return inp, target

def train(inp, target, cuda, chunk_len, batch_size, criterion, decoder, decoder_optimizer):
    hidden = decoder.init_hidden(batch_size)
    if cuda:
        hidden = hidden.cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[:,c], hidden)
        loss += criterion(output.view(batch_size, -1), target[:,c])

    loss.backward()
    decoder_optimizer.step()

    return loss.data / chunk_len

def save(decoder, filename):
    save_filename = os.path.splitext(os.path.basename(filename))[0] + '.pt'
    torch.save(decoder, save_filename)
    print('Saved as %s' % save_filename)


def train_save_model(
        filename, model='gru', n_epochs=2000, print_every=100, hidden_size=100,
        n_layers=2, learning_rate=0.01, chunk_len=200, batch_size=100, 
        shuffle=True, cuda=True):
    
    # chunk_len = torch.tensor(chunk_len)
    
    if cuda:
        print("Using CUDA")
        # chunk_len.cuda()

    file, file_len = read_file(filename)
    print(f'file length: {file_len}')
    
    # Initialize models and start training
    
    decoder = CharRNN(
        n_characters,
        hidden_size,
        n_characters,
        model=model,
        n_layers=n_layers,
    )
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    if cuda:
        decoder.cuda()
    
    start = time.time()
    # all_losses = []
    loss_avg = 0
    
    try:
        print("Training for %d epochs..." % n_epochs)
        for epoch in tqdm(range(1, n_epochs + 1)):
            inp, target = random_training_set(chunk_len, batch_size, cuda, file_len, file)
            loss = train(inp, target, cuda, chunk_len, batch_size, criterion, decoder, decoder_optimizer)
            loss_avg += loss
    
            if epoch % print_every == 0:
                print('[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
                print(generate(decoder, 'Wh', 100, cuda=cuda), '\n')
    
        print("Saving...")
        save(decoder, filename)
    
    except KeyboardInterrupt:
        print("Saving before quit...")
        save(decoder, filename)
