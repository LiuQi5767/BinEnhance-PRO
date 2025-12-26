import time
import dgl
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import Identity
from graph_transformer import RGATLayer
from dgl.nn.pytorch import GATConv

def comp_jaccard_sim_weight(i1, i2):
    i1 = set(i1)
    i2 = set(i2)
    i1_un_i2 = i1.union(i2)
    score2 = len(i1_un_i2)
    if score2 == 0:
        return 0.01
    i1_in_i2 = i1.intersection(i2)
    score1 = len(i1_in_i2)
    sim = score1 / score2
    return sim

def calculate_var_similarity(func_names, basic_dict, gpu=1):
    p_sims = []
    n_sims = []
    for func_name in func_names:
        if func_name[0] in basic_dict:
            a_func_dict = basic_dict[func_name[0]]
            p_sim = 0.01 if func_name[1] not in basic_dict else comp_jaccard_sim_weight(a_func_dict,
                                                                                        basic_dict[func_name[1]])
            n_sims_batch = [0.01 if func_name[2][i] not in basic_dict else comp_jaccard_sim_weight(a_func_dict, basic_dict[func_name[2][i]]) for i in range(len(func_name[2]))]

        else:
            p_sim = 0.01
            n_sims_batch = [0.01] * (len(func_name[2]))
        p_sims.append(p_sim)
        n_sims.append(n_sims_batch)
    return torch.tensor(p_sims, device=gpu), torch.stack([torch.tensor(n, device=gpu) for n in n_sims], dim=0).transpose(0, 1)


class RGAT_Model(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_rels, num_heads=8, num_bases=-1):
        super(RGAT_Model, self).__init__()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_heads = num_heads
        self.num_bases = num_bases
        self.negative_slope = 0.2
        self.num_hidden_layers = 4
        self.build_model()
        self.cos1 = nn.CosineSimilarity(dim=1, eps=1e-10)
        self.cos2 = nn.CosineSimilarity(dim=2, eps=1e-10)
        self.combine_sims = nn.Linear(4, 1)
        self.sim_activate = nn.Tanh()
        self.residual = 1
        if self.residual:
            if self.in_dim != self.h_dim:
                self.res_fc = nn.Linear(self.in_dim, self.h_dim, bias=False)
            else:
                self.res_fc = Identity()

    def build_model(self):
        self.layers = nn.ModuleList()
        i2h = self.build_input_layer()
        self.layers.append(i2h)
        # hidden to hidden layers
        for _ in range(self.num_hidden_layers-2):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        h2o = self.build_output_layer()
        self.layers.append(h2o)

    def build_input_layer(self):
        return RGATLayer(self.in_dim, self.h_dim, self.num_rels, self.num_heads, self.num_bases,
                         activation=F.leaky_relu, negative_slope=self.negative_slope)

    def build_hidden_layer(self):
        return RGATLayer(self.h_dim, self.h_dim, self.num_rels, self.num_heads, self.num_bases,
                         activation=F.leaky_relu, negative_slope=self.negative_slope)

    def build_output_layer(self):
        return RGATLayer(self.h_dim, self.out_dim, self.num_rels, self.num_heads, self.num_bases,
                         activation=F.leaky_relu, negative_slope=self.negative_slope)

    def get_graph_embedding(self, g, index):
        g_list = dgl.unbatch(g)
        embed_h = g_list[0].ndata['h'][int(index[0])].unsqueeze(0)
        if self.residual:
            embed_f = g_list[0].ndata['feature'][int(index[0])].unsqueeze(0)
        for i in range(1, len(g_list)):
            embed_h = torch.cat((embed_h, g_list[i].ndata['h'][int(index[i])].unsqueeze(0)), dim=0)
            if self.residual:
                embed_f = torch.cat((embed_f, g_list[i].ndata['feature'][int(index[i])].unsqueeze(0)), dim=0)
        if self.residual:
            embed_r = self.res_fc(embed_f)
            embed = embed_h + embed_r
        else:
            embed = embed_h
        return embed

    def forward(self, batch, batch_size, f_strings, global_vars, external_funcs):
        a_g, p_g, n_gs, funcNames = batch

        a_g = a_g.to(self.device)
        p_g = p_g.to(self.device)
        a_g.ndata['h'] = a_g.ndata['feature'].to(self.device)
        p_g.ndata['h'] = p_g.ndata['feature'].to(self.device)

        for layer in self.layers:
            # a_g = layer(a_g, edge_type_a)
            # p_g = layer(p_g, edge_type_p)
            a_g = layer(a_g)
            p_g = layer(p_g)

        for layer in self.layers:
            n_gs[0] = n_gs[0].to(self.device)
            n_gs[0].ndata['h'] = n_gs[0].ndata['feature'].to(self.device)
            n_gs[0] = layer(n_gs[0])
            n_embeds = self.get_graph_embedding(n_gs[0], np.zeros(batch_size).tolist()).unsqueeze(0)
            # n_embeds = self.get_graph_embedding(n_gs[0], np.zeros(batch_size).tolist())
        for i in range(1, len(n_gs)):
            n_gs[i] = n_gs[i].to(self.device)
            n_gs[i].ndata['h'] = n_gs[i].ndata['feature'].to(self.device)
            for layer in self.layers:
                # n_gs[i] = layer(n_gs[i], edge_type_n)
                n_gs[i] = layer(n_gs[i])
            n_embeds = torch.cat(
                (n_embeds, self.get_graph_embedding(n_gs[i], np.zeros(batch_size).tolist()).unsqueeze(0)), dim=0)
            # n_embeds = self.get_graph_embedding(n_gs[i], np.zeros(batch_size).tolist())

        a_embed = self.get_graph_embedding(a_g, np.zeros(batch_size).tolist())
        p_embed = self.get_graph_embedding(p_g, np.zeros(batch_size).tolist())
        cos_dist_p = self.cos1(a_embed, p_embed)
        cos_dist_ns = self.cos2(a_embed.expand_as(n_embeds), n_embeds)

        s_sim_p, s_sim_n = calculate_var_similarity(funcNames, f_strings, self.device)
        var_sim_p, var_sim_n = calculate_var_similarity(funcNames, global_vars, self.device)
        ef_sim_p, ef_sim_n = calculate_var_similarity(funcNames, external_funcs, self.device)

        # Combine similarities
        sim_p = torch.cat(
            (cos_dist_p.unsqueeze(1), s_sim_p.unsqueeze(1), var_sim_p.unsqueeze(1), ef_sim_p.unsqueeze(1)), dim=1)

        cos_dist_ns_unsqueezed = cos_dist_ns.unsqueeze(2) 
        s_sim_n_unsqueezed = s_sim_n.unsqueeze(2)  
        var_sim_n_unsqueezed = var_sim_n.unsqueeze(2)  
        ef_sim_n_unsqueezed = ef_sim_n.unsqueeze(2)  

        sim_n = torch.cat((cos_dist_ns_unsqueezed, s_sim_n_unsqueezed, var_sim_n_unsqueezed, ef_sim_n_unsqueezed), dim=2)

        # Apply activation function after combining
        sim_p = self.sim_activate(self.combine_sims(sim_p))

        sim_n_new = self.sim_activate(self.combine_sims(sim_n[0])).unsqueeze(0)
        for i in range(1, len(sim_n)):
            sim_n_new = torch.cat((sim_n_new, self.sim_activate(self.combine_sims(sim_n[i])).unsqueeze(0)), dim=0)
        all_sims = torch.cat([sim_p.unsqueeze(0), sim_n_new], dim=0).squeeze(2)

        # Compute InfoNCE Loss
        tau = 0.07  
        exp_sims = torch.exp(all_sims / tau)
        sum_exp_sims  = torch.sum(exp_sims, dim=0)  
        infoNCE_loss  = -torch.log(exp_sims[0] / sum_exp_sims).mean()
        return sim_p, sim_n, infoNCE_loss

    def forward_once(self, g, device):
        g = g.to(device)
        g.ndata['h'] = g.ndata['feature'].to(device)
        for layer in self.layers:
            g = layer(g)
        embed = g.ndata['h'][0]
        if self.residual:
            embed_r = self.res_fc(g.ndata['feature'][0])
            embed = g.ndata['h'][0] + embed_r
        return embed