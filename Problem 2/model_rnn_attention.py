import torch
import torch.nn as nn

from recurrent_blocks import StackedRNNEncoder, AdditiveAttention


class RNNAttentionNameModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) #Embedding Layer
        self.encoder = StackedRNNEncoder(embed_dim, hidden_size, num_layers, dropout) # RNN Encoder
        self.attn = AdditiveAttention(hidden_size)  # Attention Layer
        self.fc = nn.Linear(2 * hidden_size, vocab_size) #Output layer
        self.model_name = "RNN + Attention"

    def forward(self, x, lengths):
        emb = self.embedding(x)
        outputs, final = self.encoder(emb, lengths)
        context, _ = self.attn(final, outputs, lengths)  # context i.e, weighted combination of all hidden states
        fused = torch.cat([final, context], dim=-1) #combine final state + context
        return self.fc(fused)  #predict next charracter

    def architecture_description(self):
        return (
            "Character embedding -> stacked tanh RNN encoder -> additive attention over all "
            "prefix hidden states using final hidden state as query -> concatenate [final state, context] "
            "-> linear layer -> next-character probabilities"
        )
