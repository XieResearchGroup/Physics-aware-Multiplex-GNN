import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_add_pool, radius, knn
from torch_geometric.utils import remove_self_loops, to_networkx

from layers import Global_MessagePassing, Local_MessagePassing, Local_MessagePassing_s, \
    BesselBasisLayer, SphericalBasisLayer, MLP

class Config(object):
    def __init__(self, dataset, dim, n_layer, cutoff_l, cutoff_g, mode, knns:int):
        self.dataset = dataset
        self.dim = dim
        if mode == "backbone":
            self.out_dim = 12
        else:
            self.out_dim = 15
        self.n_layer = n_layer
        self.cutoff_l = cutoff_l
        self.cutoff_g = cutoff_g
        self.knns = knns

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class PAMNet(nn.Module):
    def __init__(self, config: Config, num_spherical=7, num_radial=6, envelope_exponent=5, time_dim=16):
        super(PAMNet, self).__init__()

        self.dataset = config.dataset
        self.dim = config.dim
        self.time_dim = time_dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g
        self.atom_dim = config.out_dim - 3 # 4 atom_types + 1 c4_prime flag + 4 residue types (AGCU) - 3 coordinates
        self.knns = config.knns
        self.non_mutable_edges:dict = None
        
        assert self.dim % 2 == 0, "The dimension of the embeddings must be even."

        self.init_linear = nn.Linear(3, self.dim//2, bias=False) # MLP([3, self.dim])
        self.atom_properties = nn.Linear(self.atom_dim, self.dim//2, bias=False)
        radial_bessels = 16
        self.attn = nn.MultiheadAttention(self.dim + self.time_dim, num_heads=4)

        self.rbf_g = BesselBasisLayer(radial_bessels, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(radial_bessels, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        # Add 3 to rbf to process the edge attributes with type of the edge
        self.mlp_rbf_g = MLP([radial_bessels + 3, self.dim+self.time_dim])
        self.mlp_rbf_l = MLP([radial_bessels + 3, self.dim+self.time_dim])
        self.mlp_sbf1 = MLP([num_spherical * num_radial, self.dim+self.time_dim])
        self.mlp_sbf2 = MLP([num_spherical * num_radial, self.dim+self.time_dim])

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.global_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.global_layer.append(Global_MessagePassing(self.dim+self.time_dim, config.out_dim))

        self.local_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.local_layer.append(Local_MessagePassing(self.dim+self.time_dim, config.out_dim))

        self.out_linear = nn.Linear(2*config.out_dim+self.time_dim, config.out_dim)
        # self.out_linear = nn.Linear(config.out_dim+self.time_dim, config.out_dim)

        self.softmax = nn.Softmax(dim=-1)

    def get_edge_info(self, edge_index, pos):
        edge_index, _ = remove_self_loops(edge_index)
        j, i = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        return edge_index, dist

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        mask_j = idx_j1_pair != idx_j2_pair  # Remove j == j' triplets.
        idx_i_pair, idx_j1_pair, idx_j2_pair = idx_i_pair[mask_j], idx_j1_pair[mask_j], idx_j2_pair[mask_j]

        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair
    
    def get_non_redundant_edges(self, edge_index, edge_attr, device):
        edge_index = edge_index.t().tolist()
        new_edge_index = []
        new_edge_attr = []
        for i, j in edge_index:
            if (i, j) not in self.non_mutable_edges:
                new_edge_index.append([i, j])
                new_edge_attr.append(edge_attr[i])
        return torch.tensor(new_edge_index, device=device).t(), torch.stack(new_edge_attr, dim=0)
    
    def merge_edge_attr(self, data, shape):
        edge_attr = torch.zeros(shape, device=data.edge_attr.device).float()
        edge_attr[:, 2] = 1
        return torch.cat((edge_attr, data.edge_attr), dim=0)
    
    def get_interaction_edges(self, data, cutoff):
        atom_names = data.x[:, -4:]
        atoms_argmax = torch.argmax(atom_names, dim=1)
        c2_atoms = torch.where(atoms_argmax==1)[0]
        c4_or_c6 = torch.where(atoms_argmax==2)[0]
        n_atoms = torch.where(atoms_argmax==3)[0]
        
        base_atoms = torch.cat((c2_atoms, c4_or_c6, n_atoms), dim=0)

        pos = data.x[base_atoms, :3].contiguous()
        batch = data.batch[base_atoms]
        row, col = knn(pos, pos, self.knns, batch, batch)
        edge_index_knn = torch.stack([row, col], dim=0)
        edge_index_knn, dist_knn = self.get_edge_info(edge_index_knn, pos)
        cutoff_thr = torch.ones_like(dist_knn, device=dist_knn.device) * cutoff
        mask = dist_knn <= cutoff_thr
        edges = edge_index_knn[:, mask]
        edges[0, :] = base_atoms[edges[0, :]]
        edges[1, :] = base_atoms[edges[1, :]]
        
        edge_attr = self.merge_edge_attr(data, (edges.size(1),3))
        edge_indeces = torch.cat((edges, data.edge_index), dim=1) # TODO: should we remove redundant edges?
        # edge_indeces, edge_attr = self.get_non_redundant_edges(edge_indeces, edge_attr, device=data.edge_attr.device)
        return edge_indeces, edge_attr

    def sequence_local_attention(self, x, batch, atoms_chunks1:int = 32, atoms_chunks2:int = 32):
        out = [] # Optimization: extend x to size divisible by atoms_chunks2, apply x.view(-1, atoms_chunks2, self.dim) and then reshape back
        
        for i in range(batch.max().item()+1): # iterate over batches to prevent computing attention between atoms in different graphs
            atoms = x[batch==i]
            # iterate over diagonal blocks of the matrix and compute attention between atoms in the same graph
            for j in range(0, atoms.size(0), atoms_chunks1):
                x_j = atoms[j:j+atoms_chunks1]
                x_k = atoms[j:j+atoms_chunks2]
                x_jk, _ = self.attn(x_j, x_k, x_k)
                out.append(x_jk)
        return torch.cat(out, dim=0)

    def forward(self, data, t=None):
        x_raw = data.x.contiguous()
        batch = data.batch # This parameter assigns an index to each node in the graph, indicating which graph it belongs to.

        # self.non_mutable_edges = {} # save the indices of edges given by the sequence and 2D structure
        # for i, j in data.edge_index.t().tolist():
        #     self.non_mutable_edges[(i, j)] = True

        x_raw = x_raw.unsqueeze(-1) if x_raw.dim() == 1 else x_raw
        x = x_raw[:, 3:]  # one-hot encoded atom types
        time_emb = self.time_mlp(t)
        pos = x_raw[:,:3].contiguous()
        x_pos = self.init_linear(pos) # coordinates embeddings
        x_prop = self.atom_properties(x) # atom properties embeddings
        x = torch.cat([x_pos, x_prop, time_emb], dim=1)
        x = x + self.sequence_local_attention(x, batch)        
        

        edge_index_g, edge_g_attr = self.get_interaction_edges(data, self.cutoff_g)
        edge_index_g, dist_g = self.get_edge_info(edge_index_g, pos)


        edge_index_l, edge_l_attr = self.get_interaction_edges(data, self.cutoff_l)
        edge_index_l, dist_l = self.get_edge_info(edge_index_l, pos)
        
        idx_i, idx_j, idx_k, idx_kj, idx_ji, idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = self.indices(edge_index_l, num_nodes=x.size(0))
        
        # Compute two-hop angles in local layer
        pos_ji, pos_kj = pos[idx_j] - pos[idx_i], pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.linalg.cross(pos_ji, pos_kj).norm(dim=-1)
        angle2 = torch.atan2(b, a)

        # Compute one-hop angles in local layer
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair, pos_jj_pair = pos_j1_pair - pos_i_pair, pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.linalg.cross(pos_ji_pair, pos_jj_pair).norm(dim=-1)
        angle1 = torch.atan2(b, a)

        # Get rbf and sbf embeddings
        rbf_l = self.rbf_l(dist_l)
        rbf_g = self.rbf_g(dist_g)
        sbf1 = self.sbf(dist_l, angle1, idx_jj_pair)
        sbf2 = self.sbf(dist_l, angle2, idx_kj)
        

        rbf_l = torch.cat((rbf_l, edge_l_attr), dim=1)
        rbf_g = torch.cat((rbf_g, edge_g_attr), dim=1)
        edge_attr_rbf_l = self.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.mlp_rbf_g(rbf_g)
        edge_attr_sbf1 = self.mlp_sbf1(sbf1)
        edge_attr_sbf2 = self.mlp_sbf2(sbf2)

        # Message Passing Modules
        out_global = []
        out_local = []
        att_score_global = []
        att_score_local = []
        
        for layer in range(self.n_layer):
            x, out_g, att_score_g = self.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            out_global.append(out_g)
            att_score_global.append(att_score_g)

            x, out_l, att_score_l = self.local_layer[layer](x, edge_attr_rbf_l, edge_attr_sbf2, edge_attr_sbf1, \
                                                    idx_kj, idx_ji, idx_jj_pair, idx_ji_pair, edge_index_l)
            out_local.append(out_l)
            att_score_local.append(att_score_l)
        
        # Fusion Module
        att_score = torch.cat((torch.cat(att_score_global, 0), torch.cat(att_score_local, 0)), -1)
        # att_score = torch.cat(att_score_local, 0)
        att_score = F.leaky_relu(att_score, 0.2)
        att_weight = self.softmax(att_score)

        out = torch.cat((torch.cat(out_global, 0), torch.cat(out_local, 0)), -1)
        # out = torch.cat(out_local, 0)
        out = (out * att_weight)
        out = out.sum(dim=0)
        out = torch.cat((out, time_emb), dim=1)
        out = self.out_linear(out)

        return out


class PAMNet_s(nn.Module):
    def __init__(self, config: Config, num_spherical=7, num_radial=6, envelope_exponent=5):
        super(PAMNet_s, self).__init__()

        self.dataset = config.dataset
        self.dim = config.dim
        self.n_layer = config.n_layer
        self.cutoff_l = config.cutoff_l
        self.cutoff_g = config.cutoff_g

        self.embeddings = nn.Parameter(torch.ones((5, self.dim)))

        self.rbf_g = BesselBasisLayer(16, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(16, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        self.mlp_rbf_g = MLP([16, self.dim])
        self.mlp_rbf_l = MLP([16, self.dim])    
        self.mlp_sbf = MLP([num_spherical * num_radial, self.dim])

        self.global_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.global_layer.append(Global_MessagePassing(config))

        self.local_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.local_layer.append(Local_MessagePassing_s(config))

        self.softmax = nn.Softmax(dim=-1)

        self.init()

    def init(self):
        stdv = math.sqrt(3)
        self.embeddings.data.uniform_(-stdv, stdv)

    def indices(self, edge_index, num_nodes):
        row, col = edge_index

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        
        adj_t_col = adj_t[col]

        num_pairs = adj_t_col.set_value(None).sum(dim=1).to(torch.long)
        idx_i_pair = row.repeat_interleave(num_pairs)
        idx_j1_pair = col.repeat_interleave(num_pairs)
        idx_j2_pair = adj_t_col.storage.col()

        mask_j = idx_j1_pair != idx_j2_pair  # Remove j == j' triplets.
        idx_i_pair, idx_j1_pair, idx_j2_pair = idx_i_pair[mask_j], idx_j1_pair[mask_j], idx_j2_pair[mask_j]

        idx_ji_pair = adj_t_col.storage.row()[mask_j]
        idx_jj_pair = adj_t_col.storage.value()[mask_j]

        return idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair

    def forward(self, data):
        if self.dataset != "QM9":
            raise ValueError("Invalid dataset. The current PAMNet_s is only for QM9 experiments.")
        
        x_raw = data.x
        edge_index_l = data.edge_index
        pos = data.pos
        batch = data.batch
        x = torch.index_select(self.embeddings, 0, x_raw.long())
        
        # Compute pairwise distances in local layer
        edge_index_l, _ = remove_self_loops(edge_index_l)
        j_l, i_l = edge_index_l
        dist_l = (pos[i_l] - pos[j_l]).pow(2).sum(dim=-1).sqrt()

        # Compute pairwise distances in global layer
        row, col = radius(pos, pos, self.cutoff_g, batch, batch, max_num_neighbors=500)
        edge_index_g = torch.stack([row, col], dim=0)
        edge_index_g, _ = remove_self_loops(edge_index_g)
        j_g, i_g = edge_index_g
        dist_g = (pos[i_g] - pos[j_g]).pow(2).sum(dim=-1).sqrt()

        idx_i_pair, idx_j1_pair, idx_j2_pair, idx_jj_pair, idx_ji_pair = self.indices(edge_index_l, num_nodes=x.size(0))

        # Compute one-hop angles in local layer
        pos_i_pair = pos[idx_i_pair]
        pos_j1_pair = pos[idx_j1_pair]
        pos_j2_pair = pos[idx_j2_pair]
        pos_ji_pair, pos_jj_pair = pos_j1_pair - pos_i_pair, pos_j2_pair - pos_j1_pair
        a = (pos_ji_pair * pos_jj_pair).sum(dim=-1)
        b = torch.cross(pos_ji_pair, pos_jj_pair).norm(dim=-1)
        angle = torch.atan2(b, a)

        # Get rbf and sbf embeddings
        rbf_l = self.rbf_l(dist_l)
        rbf_g = self.rbf_g(dist_g)
        sbf = self.sbf(dist_l, angle, idx_jj_pair)

        edge_attr_rbf_l = self.mlp_rbf_l(rbf_l)
        edge_attr_rbf_g = self.mlp_rbf_g(rbf_g)
        edge_attr_sbf = self.mlp_sbf(sbf)

        # Message Passing Modules
        out_global = []
        out_local = []
        att_score_global = []
        att_score_local = []
        
        for layer in range(self.n_layer):
            x, out_g, att_score_g = self.global_layer[layer](x, edge_attr_rbf_g, edge_index_g)
            out_global.append(out_g)
            att_score_global.append(att_score_g)
            
            x, out_l, att_score_l = self.local_layer[layer](x, edge_attr_rbf_l, edge_attr_sbf, \
                                                            idx_jj_pair, idx_ji_pair, edge_index_l)
            out_local.append(out_l)
            att_score_local.append(att_score_l)
        
        # Fusion Module
        att_score = torch.cat((torch.cat(att_score_global, 0), torch.cat(att_score_local, 0)), -1)
        att_score = F.leaky_relu(att_score, 0.2)
        att_weight = self.softmax(att_score)

        out = torch.cat((torch.cat(out_global, 0), torch.cat(out_local, 0)), -1)
        out = (out * att_weight).sum(dim=-1)
        out = out.sum(dim=0).unsqueeze(-1)
        out = global_add_pool(out, batch)

        return out.view(-1)