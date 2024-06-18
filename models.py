import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import knn
from torch_geometric.utils import remove_self_loops
from rinalmo.pretrained import get_pretrained_model

from layers import Global_MessagePassing, Local_MessagePassing, \
    BesselBasisLayer, SphericalBasisLayer, MLP

from constants import REV_RESIDUES

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
    
class SequenceModule(nn.Module):
    def __init__(self, dim):
        super(SequenceModule, self).__init__()
        self.rinalmo, self.alphabet = get_pretrained_model(model_name="giga-v1")
        self.out_embedding = nn.Linear(1280, dim, bias=False)
        self.emb_act = nn.ReLU()

    def forward(self, seqs, device):
        # RiNALMo - RiboNucleic Acid Language Model
        # Future TODO:
        # Combine the embeddings with 3D structure coordinates in attention blocks
        # tokens: 0- begin, 1 - pad, 2 - end
        self.rinalmo.eval()
        tokens = torch.tensor(self.alphabet.batch_tokenize(seqs), dtype=torch.int64, device=device)
        flat_tokens = tokens.flatten()
        nt_positions = torch.where(flat_tokens > 4)[0]
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.rinalmo(tokens)

        out = self.out_embedding(outputs["representation"])
        out = self.emb_act(out)
        # out = out + outputs["representation"]
        out = out.reshape((-1, out.size(2)))
        return out[nt_positions]

class SequenceStructureModule(nn.Module):
    def __init__(self, dim, n_layers:int=6, nhead:int=8):
        super(SequenceStructureModule, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(dim))

    def forward(self, seq_emb, x_struct, batch):
        x = torch.cat((seq_emb, x_struct), dim=1)
        attn_mask = torch.zeros(x.size(0), x.size(0), device=x.device)
        attn_pos = torch.where(batch[:, None] == batch[None, :])
        attn_mask[attn_pos] = 1
        out = self.transformer_encoder(x, mask=attn_mask.bool())
        return out

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
        self.seq_emb_dim = config.dim
        
        assert self.dim % 2 == 0, "The dimension of the embeddings must be even."

        self.total_dim = self.dim + self.atom_dim + self.seq_emb_dim + self.time_dim
        self.init_linear = MLP([3, self.dim])
        # self.atom_properties = nn.Linear(self.atom_dim, self.dim//2, bias=False)
        radial_bessels = 16
        # self.attn = nn.MultiheadAttention(self.dim + self.time_dim, num_heads=4)

        self.sequence_module = SequenceModule(self.seq_emb_dim)
        self.seq_struct_module = SequenceStructureModule(self.seq_emb_dim + self.dim, n_layers=6, nhead=8)

        self.rbf_g = BesselBasisLayer(radial_bessels, self.cutoff_g, envelope_exponent)
        self.rbf_l = BesselBasisLayer(radial_bessels, self.cutoff_l, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, self.cutoff_l, envelope_exponent)

        # Add 3 to rbf to process the edge attributes with type of the edge
        self.mlp_rbf_g = MLP([radial_bessels + 3, self.total_dim])
        self.mlp_rbf_l = MLP([radial_bessels + 3, self.total_dim])
        self.mlp_sbf1 = MLP([num_spherical * num_radial, self.total_dim])
        self.mlp_sbf2 = MLP([num_spherical * num_radial, self.total_dim])

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(self.dim),
            nn.Linear(self.dim, self.time_dim),
            nn.GELU(),
            nn.Linear(self.time_dim, self.time_dim),
        )

        self.global_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.global_layer.append(Global_MessagePassing(self.total_dim, self.total_dim))

        self.local_layer = torch.nn.ModuleList()
        for _ in range(config.n_layer):
            self.local_layer.append(Local_MessagePassing(self.total_dim, self.total_dim))

        # self.out_linear = nn.Linear(2*config.out_dim+self.time_dim, config.out_dim)
        self.struct_emb = nn.Linear(2*self.total_dim, config.dim)
        self.out_linear = nn.Linear(self.seq_emb_dim + self.dim, config.out_dim)

        self.softmax = nn.Softmax(dim=-1)

    def get_edge_info(self, edge_index, edge_attr, pos):
        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        j, i = edge_index
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()
        return edge_index, edge_attr, dist

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
        """
        Shape is extended to dimension 3 to add the edge type. The classes are 0 (interactios like 2D structure), 1 (covalent bondings),
        and 2 (interactions found by knn)
        """
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

    
    def merge_seq_embeddings(self, seq_emb, x):
        seq_emb = seq_emb.repeat_interleave(5, dim=0) # the embeddings are only for N atoms
        p = torch.argmax(x[:, :4], dim=1)
        p_pos = torch.where(p==3)[0] # Find P atoms
        # find places where difference between p_pos is different than 5
        # i.e. P atom is missing
        diff = p_pos[1:] - p_pos[:-1]
        diff = torch.where(diff != 5)[0]
        to_drop = p_pos[diff] + 5
        if len(seq_emb) - len(x) != len(to_drop):
            to_drop = torch.cat((torch.tensor([0], device=x.device), to_drop))
        assert len(seq_emb) - len(x) == len(to_drop), f"len(x)={len(x)}, len(seq_emb)={len(seq_emb)}, len(to_drop)={len(to_drop)}, {to_drop}"
        valid_positions = torch.zeros(seq_emb.size(0), device=x.device)
        valid_positions[to_drop] = 1
        valid_positions = torch.where(valid_positions==0)[0]
        seq_emb = seq_emb[valid_positions]
        return torch.cat((x, seq_emb), dim=1), seq_emb

    def forward(self, data, seqs, t=None):
        x_raw = data.x.contiguous()
        batch = data.batch # This parameter assigns an index to each node in the graph, indicating which graph it belongs to.

        x_raw = x_raw.unsqueeze(-1) if x_raw.dim() == 1 else x_raw
        x = x_raw[:, 3:]  # one-hot encoded atom types
        seq_emb = self.sequence_module(seqs, x.device)
        seq_x, seq_emb = self.merge_seq_embeddings(seq_emb, x)
        time_emb = self.time_mlp(t)
        pos = x_raw[:,:3].contiguous()
        x_pos = self.init_linear(pos) # coordinates embeddings
        # x_prop = self.atom_properties(x) # atom properties embeddings
        x = torch.cat([x_pos, seq_x, time_emb], dim=1)

        row, col = knn(pos, pos, self.knns, batch, batch)
        edge_index_knn = torch.stack([row, col], dim=0)
        edge_index_knn, _, dist_knn = self.get_edge_info(edge_index_knn, edge_attr=None, pos=pos)

        # Compute pairwise distances in global layer
        tensor_g = torch.ones_like(dist_knn, device=dist_knn.device) * self.cutoff_g
        mask_g = dist_knn <= tensor_g
        edge_index_g = edge_index_knn[:, mask_g]
        
        # edge_index_g = self.get_non_redundant_edges(edge_index_g)
        edge_g_attr = self.merge_edge_attr(data, (edge_index_g.size(1),3))
        edge_index_g = torch.cat((edge_index_g, data.edge_index), dim=1)
        edge_index_g, edge_g_attr, dist_g = self.get_edge_info(edge_index_g, edge_attr=edge_g_attr, pos=pos)


        # Compute pairwise distances in local layer
        tensor_l = torch.ones_like(dist_knn, device=dist_knn.device) * self.cutoff_l
        mask_l = dist_knn <= tensor_l
        edge_index_l = edge_index_knn[:, mask_l]
        
        # edge_index_l = self.get_non_redundant_edges(edge_index_l)
        edge_l_attr = self.merge_edge_attr(data, (edge_index_l.size(1),3))
        edge_index_l = torch.cat((edge_index_l, data.edge_index), dim=1)
        edge_index_l, edge_l_attr, dist_l = self.get_edge_info(edge_index_l, edge_attr=edge_l_attr, pos=pos)
        
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
        out = self.struct_emb(out)
        out = self.seq_struct_module(seq_emb, out, batch)
        out = self.out_linear(out)
        out = F.relu(out)
        
        return out
    
    def fine_tuning(self):
        # freeze all layers
        for param in self.parameters():
            param.requires_grad = False
        # unfreeze the last layer
        for param in self.struct_emb.parameters():
            param.requires_grad = True
        # initialize last layer from scratch
        # self.out_linear.reset_parameters()