import torch.nn as nn

from recurrent_blocks import StackedRNNEncoder


class VanillaRNNNameModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) #embedding layer
        self.encoder = StackedRNNEncoder(embed_dim, hidden_size, num_layers, dropout) # RNN Encoder
        self.fc = nn.Linear(hidden_size, vocab_size) #output layer
        self.model_name = "Vanilla RNN"

    def forward(self, x, lengths):
        emb = self.embedding(x)
        _, final = self.encoder(emb, lengths)
        return self.fc(final)

    def architecture_description(self):
        return (
            "Character embedding -> stacked tanh RNN encoder -> final hidden state "
            "-> linear layer -> next-character probabilities"
        )
