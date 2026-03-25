import torch.nn as nn

from recurrent_blocks import BiLSTMEncoder

class BLSTMNameModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)       # Converts character indices to dense vectors
        self.encoder = BiLSTMEncoder(embed_dim, hidden_size, num_layers, dropout)
        self.fc = nn.Linear(2 * hidden_size, vocab_size) # map concatenated hidden state to vocabulary logits
        self.model_name = "Bidirectional LSTM"

    def forward(self, x, lengths):
        emb = self.embedding(x) #bidirectional LSTM encoding
        _, final = self.encoder(emb, lengths)
        return self.fc(final)

    def architecture_description(self):
        return (
            "Character embedding -> stacked bidirectional LSTM encoder over the current prefix "
            "-> concatenate final forward/backward states -> linear layer -> next-character probabilities"
        )
