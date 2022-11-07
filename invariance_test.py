import torch
import torchvision

from se2cnn.nn.layers import R2toSE2Conv, SE2toSE2Conv, SE2toR2Conv, SE2toR2Projection

kernel_size = 5
spatial_dim = (kernel_size - 1) * 3 + 1
batch_size = 1
input_dim = 3
hidden_dim = 3
output_dim = 3
input = torch.randn(batch_size, input_dim, spatial_dim, spatial_dim)

num_theta = 8
layer1 = R2toSE2Conv(input_dim, hidden_dim, num_theta, kernel_size)
layer2 = SE2toSE2Conv(hidden_dim, 2 * hidden_dim, num_theta, kernel_size)
# layer3 = torch.nn.Sequential(SE2toSE2Conv(2 * hidden_dim, output_dim, num_theta, kernel_size), SE2toR2Projection("mean"))
layer3 = SE2toR2Conv(2 * hidden_dim, output_dim, num_theta, kernel_size, bias=False)
# layer3 = SE2toR2Projection("mean")

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


