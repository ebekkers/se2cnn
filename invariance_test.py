import torch
import torchvision

from se2cnn.nn.layers import R2ToSE2Conv, SE2ToSE2Conv, SE2ToR2Conv, SE2ToR2Projection, RegularToType1

kernel_size = 5
spatial_dim = (kernel_size - 1) * 3 + 1
batch_size = 1
input_dim = 3
hidden_dim = 3
output_dim = 3
input = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)

num_theta = 8
layer1 = R2ToSE2Conv(input_dim, hidden_dim, num_theta, kernel_size)
layer2 = SE2ToSE2Conv(hidden_dim, 2 * hidden_dim, num_theta, kernel_size)
# layer3 = torch.nn.Sequential(SE2ToSE2Conv(2 * hidden_dim, output_dim, num_theta, kernel_size), SE2ToR2Projection("mean"))
layer3 = SE2ToR2Conv(2 * hidden_dim, output_dim, num_theta, kernel_size, bias=False)
# layer3 = SE2ToR2Projection("mean")



x = input
x = layer1(x)
x = layer2(x)
x = layer3(x)
print(x.shape)


x_rot = torchvision.transforms.functional.rotate(input,90)
x_rot = layer1(x_rot)
x_rot = layer2(x_rot)
x_rot = layer3(x_rot)

print(x[:,:].squeeze(), x_rot[:,:].squeeze())





kernel_size = 5
spatial_dim = (kernel_size - 1) * 2 + 1
batch_size = 1
input_dim = 3
hidden_dim = 3
output_dim = 3
input = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)


layer_irrep = RegularToType1(num_theta)


x = input
x = layer1(x)
x = layer2(x)
v = layer_irrep(x)

theta_degree = 180
x_rot = torchvision.transforms.functional.rotate(input,theta_degree)
x_rot = layer1(x_rot)
x_rot = layer2(x_rot)
v_rot = layer_irrep(x_rot)

import math as m
theta = (theta_degree/360) * 2 * m.pi
rot_2d = torch.tensor([[m.cos(theta), -m.sin(theta)], [m.sin(theta), m.cos(theta)]])
v_unrot = torch.einsum('io,bci...->bco...',rot_2d, v_rot)
print(v.shape)
print(v[0,0,:,0,0], v_rot[0,0,:,0,0])
print(v[0,0,:,0,0], v_unrot[0,0,:,0,0])
