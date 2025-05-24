import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleTemporalAttention(nn.Module):
    def __init__(self, hidden_dim, num_scales=8, temperature=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales
        self.temperature = temperature
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ) for _ in range(num_scales)
        ])
        
        self.scale_weights = nn.Sequential(
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )
        
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        q = self.query(x[:, -1, :]).unsqueeze(1)
        k = self.key(x)
        v = self.value(x)
        
        attn_base = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(self.hidden_dim * self.temperature)
        attn_base = F.softmax(attn_base, dim=-1)
        
        multi_scale_features = []
        for i in range(self.num_scales):
            scale_size = max(1, seq_len // (2**i))
            
            if scale_size == 1:
                scaled_x = self.scale_projections[i](torch.mean(x, dim=1, keepdim=True))
            else:
                padded_x = F.pad(x, (0, 0, 0, scale_size - seq_len % scale_size if seq_len % scale_size != 0 else 0))
                reshaped_x = padded_x.unfold(1, scale_size, max(1, scale_size//2))
                scaled_x = self.scale_projections[i](torch.mean(reshaped_x, dim=3))
            
            multi_scale_features.append(scaled_x)
        
        scale_importance = self.scale_weights(x[:, -1, :])
        
        combined_context = None
        for i, feature in enumerate(multi_scale_features):
            feature_attn = F.softmax(torch.matmul(q, feature.transpose(1, 2)) / math.sqrt(self.hidden_dim), dim=-1)
            context_i = torch.matmul(feature_attn, feature).squeeze(1)
            
            if combined_context is None:
                combined_context = scale_importance[:, i:i+1] * context_i
            else:
                combined_context += scale_importance[:, i:i+1] * context_i
        
        base_context = torch.matmul(attn_base, v).squeeze(1)
        
        final_context = self.output_projection(base_context + combined_context)
        
        return final_context

class BiGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, output_dim):
        super().__init__()
        self.num_layers = len(hidden_layer_sizes)
        self.bigru_layers = nn.ModuleList()

        self.bigru_layers.append(nn.GRU(input_dim, hidden_layer_sizes[0], batch_first=True, bidirectional=True))
        
        for i in range(1, self.num_layers):
                self.bigru_layers.append(nn.GRU(hidden_layer_sizes[i-1] * 2, hidden_layer_sizes[i], batch_first=True, bidirectional=True))
        
        self.attention = MultiScaleTemporalAttention(hidden_layer_sizes[-1] * 2)
                
        hidden_size = hidden_layer_sizes[-1] * 2
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.linear = nn.Linear(hidden_size, output_dim)

    def forward(self, input_seq):
        bigru_out = input_seq
        for bigru in self.bigru_layers:
            bigru_out, _ = bigru(bigru_out)
        
        context = self.attention(bigru_out)
        
        context = self.ffn(context) + context
        
        predict = self.linear(context)
        return predict 