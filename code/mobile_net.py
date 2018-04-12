import torch
from collections import OrderedDict
import pdb


class mobile_net_model(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.build_model()
        return

    def conv2d_m(self, F, M, K, S, N):
        """
        F is num input features
        M is num input channels
        K is kernel size, assumed kx=ky
        S is stride
        N is num of output channels
        """
        # Need in_channels = out_channels = groups
        # for the case of depthwise separable conv
        P = (K + S * F - S - F) // 2
        conv = torch.nn.Sequential(
            # Depthwise
            torch.nn.Conv2d(M, M, K, S, padding=P, groups=M),
            torch.nn.BatchNorm2d(M),
            torch.nn.ReLU(),
            # PointWise
            torch.nn.Conv2d(M, N, 1, 1),
            torch.nn.BatchNorm2d(N),
            torch.nn.ReLU()
        )

        return conv

    def build_model(self):
        F = self.config['inp_dim'][0]
        layers = [32, 64, (128, 2), 128, (256, 2), 256, (512, 2),
                  512, 512, 512, 512, 512, (1024, 2), 1024]
        num_classes = self.config['num_classes']
        self.mobile_net = torch.nn.Sequential()
        self.mobile_net.add_module('l1_conv_s2', torch.nn.Conv2d(3, 32, 3))
        self.mobile_net.add_module('bn1', torch.nn.BatchNorm2d(32))
        for ind, l in enumerate(layers[:-1]):
            N = layers[ind+1]
            if type(l) == tuple:

                self.mobile_net.add_module('l'+str(ind)+'_convm_s2',
                                           self.conv2d_m(F, l[0], 3, 2, N))
            else:
                if type(N) == tuple:
                    N = N[0]
                self.mobile_net.add_module('l'+str(ind)+'_convm',
                                           self.conv2d_m(F, l, 3, 1, N))

        self.mobile_net.add_module('l_avg', torch.nn.AvgPool2d(7))
        self.mobile_net.add_module('l_fc', torch.nn.Linear(1024, num_classes))
        # self.mobile_net = torch.nn.Sequential(OrderedDict([
        #     ('l1_conv_s2', torch.nn.Conv2d(3, 32, 3, 2)),
        #     ('bn1', torch.nn.BatchNorm2d(32)),
        #     ('l2_convm', self.conv2d_m(F//2, 32, 3, 1, 64)),
        #     ('l3_convm_s2', self.conv2d_m(F//2, 64, 3, 2, 128)),
        #     ('l4_convm', self.conv2d_m(F//4, 128, 3, 2, 128)),
        #     ('l5_convm_s2', self.conv2d_m(F//4, 128, ))
        # ])
        # )
        return

    def forward(self, inp):
        return self.mobile_net(inp)
