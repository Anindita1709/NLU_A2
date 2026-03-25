import torch
import torch.nn as nn


class BasicRNNCell(nn.Module):
    #Implements a single RNN cell from scratch.

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.x2h = nn.Linear(input_size, hidden_size)  #input to hidden transformation
        self.h2h = nn.Linear(hidden_size, hidden_size) #hidden to hidden transformation
    
    #compute h_t = tanh(Wx * x_t(current input) + Wh * h_{t-1}(previous hidden input))

    def forward(self, x_t, h_prev):
        return torch.tanh(self.x2h(x_t) + self.h2h(h_prev))


#Implement a single LSTM cell from scratch    
    
class BasicLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.x2h = nn.Linear(input_size, 4 * hidden_size)     # Combine all 4 gates in one matrix
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)

    def forward(self, x_t, state):
        h_prev, c_prev = state
        gates = self.x2h(x_t) + self.h2h(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)  #split into 4 parts
        #apply activation
        i = torch.sigmoid(i) #input gate
        f = torch.sigmoid(f) #forget gate
        g = torch.tanh(g)    #candidate cell state
        o = torch.sigmoid(o)  #output gate
        c_t = f * c_prev + i * g  #update cell state
        h_t = o * torch.tanh(c_t) #compute hidden state
        return h_t, c_t

# Multi-layer RNN encoder using BasicRNNCell

class StackedRNNEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([          # Stack multiple RNN layers
            BasicRNNCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout) #dropout between layers

    def forward(self, x, lengths):
        # x: [B, T, D]
        B, T, _ = x.shape
        # Initialize hidden states for each layer
        h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(T):
            inp = x[:, t, :]
            for l, cell in enumerate(self.layers):
                new_h = cell(inp, h[l])
                active = (lengths > t).float().unsqueeze(-1) # Mask to ignore padded positions
                h[l] = active * new_h + (1.0 - active) * h[l]
                # Pass to next layer
                inp = self.dropout(h[l]) if l < self.num_layers - 1 else h[l]
            outputs.append(inp.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        last = outputs[torch.arange(B, device=x.device), lengths - 1]
        return outputs, last

#Implements Bidirectional LSTM
class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Forward LSTM layers
        self.fw_layers = nn.ModuleList([
            BasicLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        #backward LSTM layers
        self.bw_layers = nn.ModuleList([
            BasicLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def _run_direction(self, x, lengths, reverse=False):
        B, T, _ = x.shape
        layers = self.bw_layers if reverse else self.fw_layers # Select forward or backward layers
         # Initialize hidden and cell states
        h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        outs = [None] * T
        time_indices = range(T - 1, -1, -1) if reverse else range(T)

        for step_idx, t in enumerate(time_indices):
            inp = x[:, t, :]
            valid = (lengths > step_idx).float().unsqueeze(-1)
            for l, cell in enumerate(layers):
                new_h, new_c = cell(inp, (h[l], c[l]))
                h[l] = valid * new_h + (1.0 - valid) * h[l]
                c[l] = valid * new_c + (1.0 - valid) * c[l]
                inp = self.dropout(h[l]) if l < self.num_layers - 1 else h[l]
            outs[t] = inp.unsqueeze(1)

        outputs = torch.cat(outs, dim=1)  #Outputs are concatenated
        last = h[-1]
        return outputs, last

    def forward(self, x, lengths):
        fw_out, fw_last = self._run_direction(x, lengths, reverse=False)
        bw_out, bw_last = self._run_direction(x, lengths, reverse=True)
        outputs = torch.cat([fw_out, bw_out], dim=-1)
        final = torch.cat([fw_last, bw_last], dim=-1)
        return outputs, final

#Computes attention over sequence
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False) # Transform keys
        self.W_q = nn.Linear(hidden_size, hidden_size, bias=False) # Transform query
        self.v = nn.Linear(hidden_size, 1, bias=False)   #score projection

    def forward(self, query, keys, lengths):
        # query: [B, H], keys: [B, T, H]
        # Compute attention scores
        scores = self.v(torch.tanh(self.W_h(keys) + self.W_q(query).unsqueeze(1))).squeeze(-1)
        T = keys.size(1)
        # Mask padding positions
        mask = torch.arange(T, device=keys.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)   # Softmax to attention weights
        context = torch.bmm(attn.unsqueeze(1), keys).squeeze(1)   ## Weighted sum of hidden states
        return context, attn
