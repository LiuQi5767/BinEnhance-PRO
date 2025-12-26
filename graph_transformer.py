import torch
import torch.nn as nn
import dgl
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn


class RGATLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_heads=8, num_bases=-1, negative_slope=0.2, activation=None,
                 is_input_layer=False, is_output_layer=False, bias=None):
        super(RGATLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.batch_norm = nn.BatchNorm1d(in_feat)
        self.negative_slope = negative_slope
        self.activation = activation
        self.is_input_layer = is_input_layer
        self.is_output_layer = is_output_layer
        self.bias = bias
        self.head_out_dim = out_feat // num_heads

        # sanity check for num_bases
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # Weight matrices for attention mechanism
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))

        # Attention coefficients for each relation type if num_bases < num_rels
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # Attention mechanism parameters
        self.attn_l = nn.Parameter(torch.Tensor(self.num_heads, self.head_out_dim))
        self.attn_r = nn.Parameter(torch.Tensor(self.num_heads, self.head_out_dim))

        # Bias term if bias is enabled
        if self.bias:
            # self.bias = nn.Parameter(torch.Tensor(self.out_feat * self.num_heads))
            self.bias = nn.Parameter(torch.Tensor(self.out_feat))
            # nn.init.zeros_(self.bias_param)

        # Batch normalization for node features
        self.batch_norm = nn.BatchNorm1d(out_feat)

        # Initialize weights and bias for attention mechanism
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('leaky_relu', self.negative_slope))

        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('leaky_relu', negative_slope))
        nn.init.xavier_uniform_(self.attn_l, gain=nn.init.calculate_gain('leaky_relu', negative_slope))
        nn.init.xavier_uniform_(self.attn_r, gain=nn.init.calculate_gain('leaky_relu', negative_slope))

        if self.bias:
            # nn.init.zeros_(self.bias)
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('leaky_relu', negative_slope))

    def forward(self, g):
        
        # g = g.to(self.device)

        if self.num_bases < self.num_rels:
            # Generate all weights from bases and then linearly combine them
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight  # Use the same weight for all relations

        
        if self.is_input_layer:
            def message_func(edges):
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] * self.in_feat + edges.src['features']
                return {'msg': embed[index]}
        else:
            def message_func(edges): 
                w = weight[edges.data['rel_type'].to(weight.device)]  # [num_edges, in_feat, out_feat] [10040, 128, 128]
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze(1)  # [num_edges, out_feat] [10040, 128]

                # Reshape msg and dst h
                msg = msg.view(-1, self.num_heads, self.head_out_dim)  # [num_edges, num_heads, head_out_dim] 
                dst_h = edges.dst['h'].view(-1, self.num_heads, self.head_out_dim)  # [num_edges, num_heads, head_out_dim]

                # Compute attention scores 
                attn_score_l = (msg * self.attn_l).sum(dim=-1)  # [num_edges, num_heads]
                attn_score_r = (dst_h * self.attn_r).sum(dim=-1)  # [num_edges, num_heads]

                attn_score = F.leaky_relu(attn_score_l + attn_score_r, negative_slope=self.negative_slope) 
                attn_score = torch.softmax(attn_score, dim=1)  # [num_edges, num_heads]

                attn_score = attn_score.unsqueeze(-1)  # [num_edges, num_heads, 1]
                msg = msg * attn_score  # Broadcasting to [num_edges, num_heads, head_out_dim]
                msg = msg.view(-1, self.num_heads * self.head_out_dim)  # [num_edges, num_heads * head_out_dim]

                return {'msg': msg}

        # Define apply function to update node features
        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            h = self.batch_norm(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        # Perform message passing and node feature update
        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)

        return g