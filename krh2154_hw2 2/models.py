"""
COMS 4705 Natural Language Processing Spring 2021
Kathy McKeown
Homework 2: Emotion Classification with Neural Networks - Models File

Kate Harwood
krh2154
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseNetwork(nn.Module):
    def __init__(self, embeddings):
        super(DenseNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.embeddings = embeddings
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings.float())
        self.embedding_layer.weight.requires_grad = False

        # Define input and output dimensions
        hidden_dim = 100
        num_classes = 4
        word_vector_length = self.embeddings.size(1)

        # Create the dense layers
        self.layer_1 = nn.Linear(word_vector_length, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Take the sum of all word embeddings in a sentence; and
        # TODO: 3) Feed the result into 2-layer feedforward network which produces a 4-vector of values,
        # TODO: one for each class

        # Sum word embeddings in sentences (the pooling layer)
        pooled_embeddings = torch.sum(self.embedding_layer(x), axis=1)

        # Run the training data through the network
        out = F.relu(self.layer_2(F.relu((self.layer_1(pooled_embeddings.float())))))
        return out


class RecurrentNetwork(nn.Module):
    def __init__(self, embeddings):
        super(RecurrentNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        # TODO: Here, create any layers and attributes your network needs.
        self.embeddings = embeddings
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings.float())
        self.embedding_layer.weight.requires_grad = False

        # Choose hidden dim size and num layers
        hidden_dim = 128
        num_layers = 2
        output_dim = 4

        # Create RNN
        self.rnn = nn.RNN(self.embeddings.size(1), hidden_dim, num_layers, batch_first=True)

        # Linear layer at end of RNN before output
        self.linear = nn.Linear(hidden_dim, output_dim)
    

    # x is a PaddedSequence for an RNN
    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # TODO: Fill in the forward pass of your neural network.
        # TODO: (The backward pass will be performed by PyTorch magic for you!)
        # TODO: Your architecture should...
        # TODO: 1) Put the words through an Embedding layer (which was initialized with the pretrained embeddings);
        # TODO: 2) Feed the sequence of embeddings through a 2-layer RNN; and
        # TODO: 3) Feed the last output state into a dense layer to become a 4-vector of values, one for each class
        
        embedding_output = self.embedding_layer(x)

        # Pack the padded sentence sequences        
        # Note: I was using torch.count_nonzero here, but that is only available for
        # torch versions later than 1.7, and with python 3.6 the latest torch version
        # I could install was torch version 1.4. I think this line would be slightly more
        # efficient with the use of torch.count_nonzero, and not take as much time to run.
        sequence_lengths = np.count_nonzero(embedding_output, axis=1)
        sequence_lengths = sequence_lengths[:,0]
        packed_input = nn.utils.rnn.pack_padded_sequence(embedding_output, sequence_lengths, batch_first=True, enforce_sorted=False)

        # Run the input through the RNN
        packed_out, (last_hidden_out, deep_hidden_out) = self.rnn(packed_input)

        # Pass the last hidden state output from the RNN sequence through the dense layer
        final_out = F.relu(self.linear(last_hidden_out))

        return final_out


# TODO: If you do any extensions that require you to change your models, make a copy and change it here instead.
# TODO: PyTorch unfortunately requires us to have your original class definitions in order to load your saved
# TODO: dense and recurrent models in order to grade their performance.
class ExperimentalNetwork(nn.Module):
    def __init__(self, embeddings):
        super(ExperimentalNetwork, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embeddings = embeddings

        # Here we are experimenting with learning the word embeddings from scratch to see if there is any effect.
        # I ultimately had to remove this functionality to get all the models to run in under 10 minutes.
        # self.embedding_layer = nn.Embedding(self.embeddings.size(0), self.embeddings.size(1))
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings.float())
        self.embedding_layer.weight.requires_grad = False


        # Define input and output dimensions
        hidden_dim = 100
        num_classes = 4
        word_vector_length = self.embeddings.size(1)

        # Create the dense layers
        self.layer_1 = nn.Linear(word_vector_length, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, num_classes)


    def forward(self, x):
        ########## YOUR CODE HERE ##########
        # Sum word embeddings in sentences (the pooling layer)
        pooled_embeddings = torch.sum(self.embedding_layer(x), axis=1)

        # Run the training data through the network
        out = self.layer_2(F.relu((self.layer_1(pooled_embeddings.float()))))
        return out


# extension-grading
class ExperimentalNetwork2(nn.Module):
    def __init__(self, embeddings):
        super(ExperimentalNetwork2, self).__init__()

        ########## YOUR CODE HERE ##########
        self.embeddings = embeddings
        self.embedding_layer = nn.Embedding.from_pretrained(embeddings.float())
        self.embedding_layer.weight.requires_grad = False

        # Define input, output dimensions and CNN params
        num_classes = 4
        word_vector_length = self.embeddings.size(1)
        filter_sizes = 2 # We want to look at two words together at a time
        stride = 1
        conv_output_dim = 190 # Number of features to map
        padding = 0 # Having no padding ended up working the best

        # Create the 1D convolution layer
        self.conv_layer = nn.Conv1d(word_vector_length, conv_output_dim, filter_sizes, stride, padding)

        # Create the dropout layer so we can reduce overfitting
        dropout_probability = 0.5
        self.dropout_layer = nn.Dropout(dropout_probability)

        # Create the dense layer
        self.dense_linear_layer = nn.Linear(conv_output_dim, num_classes)

    # extension-grading
    def forward(self, x):
        ########## YOUR CODE HERE ##########

        # Run the input through the embedding layer and reshape
        # to (batch, word_vector, sequence) for the convolutional layer
        embeddings = self.embedding_layer(x).float().permute(0,2,1)

        # Run the data through the convolutional part of the network
        conv_out = self.conv_layer(embeddings)

        # Pass through max pooling layer to eliminate the sequence length dimension,
        # only taking the most salient features of the sentence (as humans do themselves)
        pool_out = F.max_pool1d(F.relu(conv_out), conv_out.shape[2])

        # Reshape so it is in the form the dense linear layer expects
        pool_out = pool_out.squeeze(dim=2)

        # Finally, run the output through the dropout and dense linear layers
        out = self.dense_linear_layer(self.dropout_layer(pool_out))

        return out





