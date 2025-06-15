import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, embed_dim]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x
    

class SequentialEncoderTF(nn.Module):
    def __init__(self, walk_lengths, input_dim, embed_dim, feedforward_dim, num_heads, num_layers, dropout):
        super(SequentialEncoderTF, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=max(walk_lengths))
        self.encoders = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model = embed_dim, nhead = num_heads, dim_feedforward = feedforward_dim, dropout = dropout, batch_first = True) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, k, seq_len, feat_dim = x.shape
        mask_tot = (x == -1e9).all(dim=-1).float().unsqueeze(-1)
        x = x.view(-1, seq_len, feat_dim)
        mask = (x == -1e9).all(dim=-1)
        x = self.linear(x)
        x = self.pos_encoder(x)
        for layer in self.encoders:
            x = x + layer(self.norm(x), src_key_padding_mask=mask)
        x = x.view(batch_size, k, seq_len, -1)
        x = self.dropout(x)
        x = torch.sum(x * (1 - mask_tot), dim=2) / torch.sum(1 - mask_tot, dim=2)
        return x
    

class SequentialEncoderGRU(nn.Module):
    def __init__(self, input_dim, embed_dim, num_layers, dropout):
        super(SequentialEncoderGRU, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, k, seq_len, feat_dim = x.shape
        mask_tot = (x == -1e9).all(dim=-1).float().unsqueeze(-1)
        x = x.view(-1, seq_len, feat_dim)
        x = self.linear(x)
        x = self.norm(x)
        output, _ = self.gru(x)
        output = output.view(batch_size, k, seq_len, -1)
        output = self.dropout(output)
        output = torch.sum(output * (1 - mask_tot), dim=2) / torch.sum(1 - mask_tot, dim=2)
        return output


class HypergraphPatternMachine(nn.Module):
    def __init__(self, embed_dim, walk_lengths,
                 semantic_dim, semantic_encoder_feedforward_dim, semantic_encoder_num_heads, semantic_encoder_num_layers, 
                 anonymous_dim, anonymous_encoder_num_layers,
                 pattern_identifier_feedforward_dim, pattern_identifier_num_heads, pattern_identifier_num_layers,  
                 dropout, weight_an, weight_ae, num_classes):
        super(HypergraphPatternMachine, self).__init__()
        self.sp_encoder = SequentialEncoderTF(walk_lengths, semantic_dim, embed_dim, semantic_encoder_feedforward_dim, semantic_encoder_num_heads, semantic_encoder_num_layers, dropout)
        self.anp_encoder = SequentialEncoderGRU(anonymous_dim, embed_dim, anonymous_encoder_num_layers, dropout)
        self.aep_encoder = SequentialEncoderGRU(anonymous_dim, embed_dim, anonymous_encoder_num_layers, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.pattern_identifier = nn.ModuleList(
            [nn.TransformerEncoderLayer(d_model = embed_dim, nhead = pattern_identifier_num_heads, dim_feedforward = pattern_identifier_feedforward_dim, dropout = dropout) for _ in range(pattern_identifier_num_layers)]
        )
        self.wn = weight_an
        self.we = weight_ae
        self.task_head = nn.Linear(embed_dim, num_classes, bias = True)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, s, an, ae):
        s = self.sp_encoder(s)
        an = self.anp_encoder(an)
        ae = self.aep_encoder(ae)
        x = s + self.wn * an + self.we * ae
        for layer in self.pattern_identifier:
            x = x + layer(self.norm(x))
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.dropout(x)
        x = self.task_head(x)
        return x.squeeze(1)
        
        