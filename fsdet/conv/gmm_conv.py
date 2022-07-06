from cv2 import repeat
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from .inits import reset, glorot, zeros
import pdb as pdb

EPS = 1e-15


class GMMConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 dim,
                 kernel_size,
                 bias=True,
                 **kwargs):
        super(GMMConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size

        # print("*******************self.lin", in_channels, out_channels, kernel_size)

        self.lin = torch.nn.Linear(in_channels,
                                   out_channels * kernel_size,
                                   bias=False)
        self.mu = Parameter(torch.Tensor(kernel_size, dim))
        self.sigma = Parameter(torch.Tensor(kernel_size, dim))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.mu)
        glorot(self.sigma)
        zeros(self.bias)
        reset(self.lin)

    def forward(self, x, edge_index, pseudo):
        # pdb.set_trace()
        # print("INSIDE GMM_CONV x.shape *************************", x.shape)
        x = x.unsqueeze(-1) if x.dim() == 1 else x

        # x = x.unsqueeze(0)

        # pdb.set_trace()

        # print("INSIDE GMM_CONV edge_index.shape *************************", edge_index.shape)
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo
        out = self.lin(x)
        # out = out.view(-1, self.kernel_size, self.out_channels).contiguous()
        # print("INSIDE GMM_CONV out *************************", out.shape)

        # print("INSIDE GMM_CONV pseudo *************************", pseudo.shape)
        out = self.propagate(edge_index, x=out, pseudo=pseudo)

        if self.bias is not None:
            out = out + self.bias
        return out

    def message(self, x_j, pseudo):
        (E, D), K = pseudo.size(), self.mu.size(0)

        # E 4096
        #  D 2
        # K  25

        # pdb.set_trace()

        # print("INSIDE GMM_CONV x_j *************************", x_j.shape)
        # print("INSIDE GMM_CONV pseudo 2 *************************", pseudo.shape)
        # B, _, _ = x_j.shape
        gaussian = -0.5 * (pseudo.view(E, 1, D) - self.mu.view(1, K, D))**2
        gaussian = gaussian / (EPS + self.sigma.view(1, K, D)**2)
        gaussian = torch.exp(gaussian.sum(dim=-1, keepdim=True))  # [E, K, 1]  4096x25x1

        # B = x_j.shape[0]
        # x_j = x_j.view(B, E, self.kernel_size, -1)
        x_j = x_j.view(E, self.kernel_size, -1)    #   4096x25x1024


        # print("DUY gaussian *************************", gaussian.shape)
        # gaussian = gaussian.unsqueeze(0).repeat(B, 1, 1).transpose(1,2).contiguous() # [B, K, E]
 
        # print("*********************************xj", x_j.shape)
        # print("*********************************gaussian", gaussian.shape)
        # print("*********************** (x_j * gaussian)", (x_j * gaussian).shape)
        # print("*********************** (x_j * gaussian).sum(dim=1)", (x_j * gaussian).sum(dim=1).shape)
        # pdb.set_trace()
        return (x_j * gaussian).sum(dim=1)
        # result = torch.bmm(gaussian, x_j).sum(dim=1) 
        # print(result.shape)
        # return result

    def __repr__(self):
        return '{}({}, {}, kernel_size={})'.format(self.__class__.__name__,
                                                   self.in_channels,
                                                   self.out_channels,
                                                   self.kernel_size)
