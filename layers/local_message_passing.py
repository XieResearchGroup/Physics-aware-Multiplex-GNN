import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter

from layers import MLP, Res


class Local_MessagePassing(torch.nn.Module):
    def __init__(self, config):
        super(Local_MessagePassing, self).__init__()
        self.dim = config.dim

        self.mlp_x1 = MLP([self.dim, self.dim])
        self.mlp_m_ji = MLP([3 * self.dim, self.dim])
        self.mlp_m_kj = MLP([3 * self.dim, self.dim])
        self.mlp_sbf = MLP([self.dim, self.dim, self.dim])
        self.lin_rbf = nn.Linear(self.dim, self.dim, bias=False)

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)

        self.lin_rbf_out = nn.Linear(self.dim, self.dim, bias=False)
        self.mlp_x2 = MLP([self.dim, self.dim])
        
        self.mlp_out = MLP([self.dim, self.dim, self.dim, self.dim])
        self.W_out = nn.Linear(self.dim, 1)
        self.W = nn.Parameter(torch.Tensor(self.dim, 1))

        self.init()

    def init(self):
        glorot(self.W)

    def forward(self, x, rbf, sbf2, sbf1, idx_kj, idx_ji, idx_jj_pair, idx_ji_pair, edge_index):
        j, i = edge_index
        idx = torch.cat((idx_kj, idx_jj_pair), 0)
        idx_scatter = torch.cat((idx_ji, idx_ji_pair), 0)
        sbf = torch.cat((sbf2, sbf1), 0)

        res_x = x
        x = self.mlp_x1(x)

        # Message Block
        m = torch.cat([x[i], x[j], rbf], dim=-1)
        m_ji = self.mlp_m_ji(m)
        m_neighbor = self.mlp_m_kj(m) * self.lin_rbf(rbf)
        m_other = m_neighbor[idx] * self.mlp_sbf(sbf)
        m_other = scatter(m_other, idx_scatter, dim=0, dim_size=m.size(0), reduce='add')
        m = m_ji + m_other

        m = self.lin_rbf_out(rbf) * m
        x = x + scatter(m, i, dim=0, dim_size=x.size(0), reduce='add')
        x = self.mlp_x2(x)

        # Update Block
        x = self.res1(x) + res_x
        x = self.res2(x)
        x = self.res3(x)

        out = self.mlp_out(x)
        att_score = out.matmul(self.W).unsqueeze(0)
        out = self.W_out(out).unsqueeze(0)

        return x, out, att_score


class Local_MessagePassing_s(torch.nn.Module):
    def __init__(self, config):
        super(Local_MessagePassing_s, self).__init__()
        self.dim = config.dim

        self.mlp_x1 = MLP([self.dim, self.dim])
        self.mlp_m_ji = MLP([3 * self.dim, self.dim])
        self.mlp_m_jj = MLP([3 * self.dim, self.dim])
        self.mlp_sbf = MLP([self.dim, self.dim, self.dim])
        self.lin_rbf = nn.Linear(self.dim, self.dim, bias=False)

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)

        self.lin_rbf_out = nn.Linear(self.dim, self.dim, bias=False)
        self.mlp_x2 = MLP([self.dim, self.dim])

        self.mlp_out = MLP([self.dim, self.dim, self.dim, self.dim])
        self.W_out = nn.Linear(self.dim, 1)
        self.W = nn.Parameter(torch.Tensor(self.dim, 1))

        self.init()

    def init(self):
        glorot(self.W)

    def forward(self, x, rbf, sbf, idx_jj_pair, idx_ji_pair, edge_index):
        j, i = edge_index
        
        res_x = x
        x = self.mlp_x1(x)

        # Message Block
        m = torch.cat([x[i], x[j], rbf], dim=-1)
        m_ji = self.mlp_m_ji(m)
        m_neighbor = self.mlp_m_jj(m) * self.lin_rbf(rbf)
        m_other = m_neighbor[idx_jj_pair] * self.mlp_sbf(sbf)
        m_other = scatter(m_other, idx_ji_pair, dim=0, dim_size=m.size(0), reduce='add')
        m = m_ji + m_other

        m = self.lin_rbf_out(rbf) * m
        x = x + scatter(m, i, dim=0, dim_size=x.size(0), reduce='add')
        x = self.mlp_x2(x)

        # Update Block
        x = self.res1(x) + res_x
        x = self.res2(x)
        x = self.res3(x)

        out = self.mlp_out(x)
        att_score = out.matmul(self.W).unsqueeze(0)
        out = self.W_out(out).unsqueeze(0)

        return x, out, att_score

