import codecs
import os
import collections
from six.moves import cPickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories='auto')
"""
Implement a class object that should have the following functions:

1) object initialization:
This function should be able to take arguments of data directory, batch size and sequence length.
The initialization should be able to process data, load preprocessed data and create training and 
validation mini batches.

2)helper function to preprocess the text data:
This function should be able to do:
    a)read the txt input data using encoding='utf-8'
    b)
        b1)create self.char that is a tuple contains all unique character appeared in the txt input.
        b2)create self.vocab_size that is the number of unique character appeared in the txt input.
        b3)create self.vocab that is a dictionary that the key is every unique character and its value is a unique integer label.
    c)split training and validation data.
    d)save your self.char as pickle (pkl) file that you may use later.
    d)map all characters of training and validation data to their integer label and save as 'npy' files respectively.

3)helper function to load preprocessed data

4)helper functions to create training and validation mini batches


"""
class TextLoader():
    def __init__(self):
        self.char = []
        self.vocab_size = 0
        self.vocab = {} # a dictionary that the key is every unique character and its value is a unique integer label
        self.inverse_vocab = {} # a dictionary that the value is every unique character and its key is a unique integer label
        
        
        
        
    def char_to_int(self,text):
        integers = []
        for t in text:
            integers.append(self.vocab[t])
        return integers
    
    def int_to_char(self,integers):
        texts = []
        for i in integers:
            texts.append(self.inverse_vocab[i])
        return texts  
    
    def read_data(self, dir):
        corpus = open(dir,encoding='utf-8').read()
        
        # char is a tuple contains all unique character appeared in the txt input
        for character in corpus:
            if character not in self.char:
                self.char.append(character)
                
        self.vocab_size = len(self.char)
        for i, j in zip(self.char,range(self.vocab_size)):
            self.vocab[i] = j 
            self.inverse_vocab[j] = i
        
        # turn char data to int format
        data = np.asarray(self.char_to_int(corpus))
        data_onehot = onehotencoder.fit_transform(np.reshape(data,(-1,1))).toarray()
        
        # split training & validation data
        a = int(data.shape[0]*0.9)
        train_data, val_data = data[:a], data[a:]
        train_data_onehot, val_data_onehot = data_onehot[:a], data_onehot[a:]
        
        # save data to npy
        np.save('train.npy', train_data)
        np.save('val.npy', val_data)
        np.save('train_onehot.npy', train_data_onehot)
        np.save('val_onehot.npy', val_data_onehot)
        
        # write vocab dict to a file
        output = open('vocab.pkl', 'wb')
        cPickle.dump(self.vocab, output)
        output.close()
        
        # write vocab dict to a file
        output = open('inverse_vocab.pkl', 'wb')
        cPickle.dump(self.inverse_vocab, output)
        output.close()
        
#     def load_data(self,train_dir, val_dir, vocab_dir):
#         train = np.load(train_dir)
#         val = np.load(val_dir)
#         # read vocab dict back from the file
#         pkl_file = open(vocab_dir, 'rb')
#         self.vocab = cPickle.load(pkl_file)
#         pkl_file.close()
        
#         self.char = list(self.vocab.keys())
#         self.vocab_size = 
        
#         return train, val 


            
      