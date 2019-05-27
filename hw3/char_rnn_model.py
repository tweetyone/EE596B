import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

"""
TO: Define your char rnn model here

You will define two functions inside the class object:

1) __init__(self, args_1, args_2, ... ,args_n):

    The initialization function receives all hyperparameters as arguments.

    Some necessary arguments will be: batch size, sequence_length, vocabulary size (number of unique characters), rnn size,
    number of layers, whether use dropout, learning rate, use embedding or one hot encoding,
    and whether in training or testing,etc.

    You will also define the tensorflow operations here. (placeholder, rnn model, loss function, training operation, etc.)


2) sample(self, sess, char, vocab, n, starting_string):
    
    Once you finish training, you will use this function to generate new text

    args:
        sess: tensorflow session
        char: a tuple that contains all unique characters appeared in the text data
        vocab: the dictionary that contains the pair of unique character and its assoicated integer label.
        n: a integer that indicates how many characters you want to generate
        starting string: a string that is the initial part of your new text. ex: 'The '

    return:
        a string that contains the genereated text

"""
class Model():
    
    def __init__(self, batch_size, seq_len, rnn_size, num_layers, learning_rate, vocab_size, keep_prob):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.lr = learning_rate
        self.vocab_size = vocab_size
        self.keep_prob = keep_prob
        
        
        # a lstm multi-layer net which predict the next character of each character

        tf.reset_default_graph()
        # input [batch_size,seq_len,1]   each cell input 1 character, lstm have seq_len cells
        self.X = tf.placeholder(tf.float64,[None,self.seq_len,1],name='X')
        # output [batch_size, seq_len]  for each seq_len cells, generate the out put
        self.Y = tf.placeholder(tf.int64,[None,self.seq_len],name='Y')

        def LSTM(x):   

            # construct multi-layer LSTM cell
            cells = []
            for _ in range(self.num_layers): 
                cell = tf.contrib.rnn.LSTMCell(num_units = self.rnn_size)
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
                cells.append(cell)
            stacked_rnn_cell = tf.contrib.rnn.MultiRNNCell(cells)

            initial_state = stacked_rnn_cell.zero_state(self.batch_size, tf.float64)

            # link the cells to build LSTM
            # get output of each cell --outputs
            outputs, final_state = tf.nn.dynamic_rnn(stacked_rnn_cell, x, initial_state=initial_state)

            #fully connected layer to change dimension to [batch-size,voca_size]  
            logits = tf.contrib.layers.fully_connected(inputs=outputs, num_outputs=self.vocab_size, activation_fn=None)

            logits =tf.reshape(logits,(batch_size,seq_len,67))
            return logits 


        #predicted labels
        self.logits = LSTM(self.X)

        self.pred = tf.argmax(self.logits,2)

        self.correct_pred = tf.equal(self.pred,self.Y)
        self.num_correct = tf.reduce_sum(tf.cast(self.correct_pred,tf.float64))

        # loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,labels=self.Y),name='loss')

        # optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.optimizer.minimize(self.loss)
    

    def sample(self,sess, vocab, inverse_vocab, n, start):
        
        # covert the input to index
        split = [vocab[i] for i in start]
           
        for i in range(n):
            length = len(split)
            
            #get the last seq_len elements of the sequence
            piece = split[length - seq_len : length]
            
            batch_x = [piece]
            batch_x = np.reshape(batch_x,[1,self.seq_len,1])

            next_chars = sess.run(logits, feed_dict={X:batch_x})[:,-1,:]
            val = np.argmax(next_chars)
            split.append(val) # generate new input 

        # change the prediction from idx to character
        string_back = []
        for i in range(len(split)):
            val = split[i] 
            string_back.append(inverse_vocab[val])
            
        
        return "".join(string_back)
        