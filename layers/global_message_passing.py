import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot

from layers import MLP, Res


class Global_MessagePassing(MessagePassing):
    def __init__(self, config):
        super(Global_MessagePassing, self).__init__()
        self.dim = config.dim

        self.mlp_x1 = MLP([self.dim, self.dim])
        self.mlp_x2 = MLP([self.dim, self.dim])

        self.res1 = Res(self.dim)
        self.res2 = Res(self.dim)
        self.res3 = Res(self.dim)

        self.mlp_m = MLP([self.dim * 3, self.dim])
        self.W_edge_attr = nn.Linear(self.dim, self.dim, bias=False)

        self.mlp_out = MLP([self.dim, self.dim, self.dim, self.dim])
        self.W_out = nn.Linear(self.dim, 1)
        self.W = nn.Parameter(torch.Tensor(self.dim, 1))

        self.init()

    def init(self):
        glorot(self.W)

    def forward(self, x, edge_attr, edge_index):
        res_x = x
        x = self.mlp_x1(x)

        # Message Block
        x = x + self.propagate(edge_index, x=x, num_nodes=x.size(0), edge_attr=edge_attr)
        x = self.mlp_x2(x)

        # Update Block
        x = self.res1(x) + res_x
        x = self.res2(x)
        x = self.res3(x)

        out = self.mlp_out(x)
        att_score = out.matmul(self.W).unsqueeze(0)
        out = self.W_out(out).unsqueeze(0)

        return x, out, att_score

    def message(self, x_i, x_j, edge_attr, edge_index, num_nodes):
        m = torch.cat((x_i, x_j, edge_attr), -1)
        m = self.mlp_m(m)

        return m * self.W_edge_attr(edge_attr)

    def update(self, aggr_out):

        return aggr_out
