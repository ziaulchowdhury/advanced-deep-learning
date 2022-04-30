# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 01:02:11 2022

@author: ziaul
"""

import torch

from train import train_save_model
from generate import generate


def predict_texts_with_prime_strs(decoder, prime_strs=[], predict_len=100):
    for prime_str in prime_strs:        
        predicted_random_text = generate(decoder, prime_str=prime_str, predict_len=predict_len, cuda=True)
        print(f'\nPredicted text for "{prime_str}":\n  {predicted_random_text}')
    

if __name__ == '__main__':
    
    train_model = False
    
    if train_model:
        filename = '../dataset/tinyshakespeare.txt'
        train_save_model(filename)
    
    
    decoder = torch.load('./tinyshakespeare.pt')
    
    #
    # Task 2
    #
    prime_strs_task2 = ['2 b3n', 'bg09Z', 'xyz']
    predict_texts_with_prime_strs(decoder, prime_strs_task2)
    
    #
    # Task 3
    #
    prime_strs_task3 = ['The', 'What is', 'Shall I give', 'X087hNYB BHN BYFVuhsdbs']
    predict_texts_with_prime_strs(decoder, prime_strs_task3)
    