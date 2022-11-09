import torch
from se2cnn.utils.rotation_matrix import MultiRotationOperatorMatrix
from torch import nn
import torch.nn.functional as F
import math as m


class R2ToSE2Conv(nn.Module):

    def __init__(self, input_dim, output_dim, num_theta, kernel_size, padding=0, stride=1, bias=True, diskMask=True, groups=1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_theta = num_theta
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.diskMask = diskMask
        self.groups = groups

        # The layer parameters
        self.kernel = nn.Parameter(torch.zeros(output_dim, int(input_dim / groups), kernel_size, kernel_size))
        self.kernel = nn.init.normal_(self.kernel, 0, 2. / m.sqrt(input_dim * kernel_size * kernel_size / groups))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

        # Rotated filter bank
        self.register_buffer('RR', torch.from_numpy(
            MultiRotationOperatorMatrix(kernel_size, kernel_size, num_theta, diskMask=diskMask)).type_as(
            self.kernel))  # [num_theta, X * Y])

    def kernel_stack_conv2d(self):
        kernel_stack = torch.einsum('oix,tyx->otiy', self.kernel.flatten(-2, -1), self.RR).unflatten(-1,
                                                                                                     [self.kernel_size,
                                                                                                      self.kernel_size])
        kernel_stack_conv2d = kernel_stack.flatten(0, 1)  # [Cout*num_theta, Cin, x, y]
        return kernel_stack_conv2d

    def forward(self, x):
        x = F.conv2d(x, self.kernel_stack_conv2d(), None, self.stride, self.padding, groups=self.groups)  # [B, Cout*num_theta, X, Y]
        x = x.unflatten(1, [self.output_dim, self.num_theta])  # [ B, Cout, num_theta, X, Y]
        if self.bias is not None:
            x = x + self.bias[None,:,None, None,None]
        return x


class SE2ToSE2Conv(nn.Module):

    def __init__(self, input_dim, output_dim, num_theta, kernel_size, padding=0, stride=1, bias=True, diskMask=True, groups=1):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_theta = num_theta
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.diskMask = diskMask
        self.groups = groups

        # The layer parameters
        self.kernel = nn.Parameter(torch.zeros(output_dim, int(input_dim / groups), num_theta, kernel_size, kernel_size))
        self.kernel = nn.init.normal_(self.kernel, 0, 2. / m.sqrt(input_dim * num_theta * kernel_size * kernel_size / groups))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

        # Rotated filter bank
        self.register_buffer('RR', torch.from_numpy(
            MultiRotationOperatorMatrix(kernel_size, kernel_size, num_theta, diskMask=diskMask)).type_as(
            self.kernel))  # [num_theta, X * Y])

    def kernel_stack_conv2d(self):
        # Let's use a for input rotation axis, b for output rotation axis:
        kernel_stack = torch.einsum('oiax,byx->obiay', self.kernel.flatten(-2, -1), self.RR).unflatten(-1, [
            self.kernel_size, self.kernel_size])
        for b in range(self.num_theta):
            kernel_stack[:, b, ...] = torch.roll(kernel_stack[:, b, ...], b, 2)
        kernel_stack_conv2d = kernel_stack.flatten(0, 1)  # [Cout*num_theta, Cin, num_theta, x, y]
        kernel_stack_conv2d = kernel_stack_conv2d.flatten(1, 2)  # [Cout*num_theta, Cin*num_theta, x, y]
        return kernel_stack_conv2d

    def forward(self, x):
        x = F.conv2d(x.flatten(1, 2), self.kernel_stack_conv2d(), None, self.stride, self.padding, groups=self.groups)  # [B, Cout*num_theta, X, Y]
        x = x.unflatten(1, [self.output_dim, self.num_theta])  # [ B, Cout, num_theta, X, Y]
        if self.bias is not None:
            x = x + self.bias[None,:,None, None,None]
        return x


class SE2ToR2Conv(nn.Module):

    def __init__(self, input_dim, output_dim, num_theta, kernel_size, padding=0, stride=1, bias=True, diskMask=True):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_theta = num_theta
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.diskMask = diskMask

        # The layer parameters
        self.kernel = nn.Parameter(torch.zeros(output_dim, input_dim, num_theta, kernel_size, kernel_size))
        self.kernel = nn.init.normal_(self.kernel, 0, 2. / m.sqrt(input_dim * num_theta * kernel_size * kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.bias = None

        # Rotated filter bank
        self.register_buffer('RR', torch.from_numpy(
            MultiRotationOperatorMatrix(kernel_size, kernel_size, num_theta, diskMask=diskMask)).type_as(
            self.kernel))  # [num_theta, X * Y])

    def kernel_stack_conv2d(self):
        # Let's use a for input rotation axis, b for output rotation axis:
        kernel_stack = torch.einsum('oiax,byx->obiay', self.kernel.flatten(-2, -1), self.RR).unflatten(-1, [
            self.kernel_size, self.kernel_size])
        for b in range(self.num_theta):
            kernel_stack[:, b, ...] = torch.roll(kernel_stack[:, b, ...], b, 2)
        # Symmetrize (mean pool before convolution)
        kernel_stack = torch.mean(kernel_stack, (1))  # [Cout, Cin, num_theta x, y]
        kernel_stack_conv2d = kernel_stack.flatten(1, 2)  # [Cout, Cin * num_theta, x, y]
        return kernel_stack_conv2d

    def forward(self, x):
        x = F.conv2d(x.flatten(1, 2), self.kernel_stack_conv2d(), None, self.stride, self.padding)  # [B, Cout, X, Y]
        if self.bias is not None:
            x = x + self.bias[None, :, None, None]
        return x

class SE2ToR2Projection(nn.Module):

    def __init__(self, method):
        super().__init__()

        self.method = method

    def forward(self, x):
        if self.method == "mean":
            x = torch.mean(x, 2)
        elif self.method == "max":
            x, _ = torch.max(x, 2)
        else:
            raise Exception("Fiber pooling method not known")
        return x


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # help(z2_se2n)
    # help(se2n_se2n)
    # help(spatial_max_pool)
    # help(rotate_lifting_kernels)
    # help(rotate_gconv_kernels)


    # input_dim = 3
    # output_dim = 15
    # output_dim2 = 13
    # num_theta = 8
    # kernel_size = 7
    #
    # layer = LiftingLayer(input_dim, output_dim, num_theta, kernel_size, diskMask=True)
    # layer2 = LiftingLayer(output_dim, output_dim, num_theta, kernel_size, diskMask=True)
    # x = torch.randn([16,input_dim,151,151])
    # y = F.relu(layer(x))
    # z = F.relu(layer2(y[:,:,0]))
    # print(torch.std(x), torch.std(y), torch.std(z), y.shape)
    #
    # for i in range(num_theta):
    #     plt.imshow(layer.kernel_stack[0,i,0].detach().numpy())
    #     plt.show()

    input_dim = 3
    output_dim = 15
    output_dim2 = 13
    num_theta = 8
    kernel_size = 7
    x = torch.randn([16, input_dim, 151, 151])
    layer1 = R2ToSE2Conv(input_dim, output_dim, num_theta, kernel_size)
    layer2 = SE2ToSE2Conv(output_dim, output_dim2, num_theta, kernel_size)
    y = F.relu(layer1(x))
    print(y.shape)
    print(layer2.kernel_stack.shape)
    # fig, axis = plt.subplots(nrows=num_theta, ncols=num_theta, figsize=(3, 8))
    # for b in range(num_theta):
    #     for a in range(num_theta):
    #         axis[b][a].imshow(layer2.kernel_stack[0,b,0,a].detach().numpy())
    #         axis[b][a].get_xaxis().set_visible(False)
    #         axis[b][a].get_yaxis().set_visible(False)
    # plt.show()
    z = F.relu(layer2(y))
    print(torch.std(x), torch.std(y), torch.std(z), y.shape, z.shape)

    num_theta = 8
    layer2irrep1 = RegularToType1(num_theta)
    out = layer2irrep1(z)
    print(z.shape, out.shape)

    proj_layer = SE2ToR2Projection("max")
    out2 = proj_layer(z)
    print(out2.shape)

    output_dim3 = 78
    proj_conv = SE2ToR2Conv(output_dim2, output_dim3, num_theta, kernel_size)
    out3 = proj_conv(z)
    print(out3.shape)

    for i in range(num_theta):
        plt.imshow(proj_conv.kernel_stack[0,0,i].detach().numpy())
        plt.show()