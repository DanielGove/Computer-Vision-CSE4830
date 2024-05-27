import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, aggregation="sum"):
    super(VectorConv2d, self).__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = (kernel_size, kernel_size)
    self.stride = stride
    self.padding = padding
    self.aggregation = aggregation

    # Define the weights for the magnitudes and phases of the filters
    self.filter_x = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
    self.filter_y = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))

    if bias:
      self.bias = nn.Parameter(torch.randn(out_channels))
    else:
      self.bias = None

  def forward(self, x):
    assert x.dim() == 5 and x.shape[-1] == 2, "Input to VectorConv2d should be 5D with last dim 2."

    magnitude = x[..., 0]
    phase = x[..., 1]

    # Convert from polar to cartesian
    x_cart = magnitude * torch.cos(phase)
    y_cart = magnitude * torch.sin(phase)

    # Rotate the filters
    rotated_filter_x = self.rotate_filters(self.filter_x)
    rotated_filter_y = self.rotate_filters(self.filter_y)
    bias = self.bias.repeat(4)

    # Apply convolution in cartesian coordinates
    real_conv = F.conv2d(x_cart, rotated_filter_x, bias, self.stride, self.padding) - \
                F.conv2d(y_cart, rotated_filter_y, None, self.stride, self.padding)
    imag_conv = F.conv2d(x_cart, rotated_filter_y, None, self.stride, self.padding) + \
                F.conv2d(y_cart, rotated_filter_x, None, self.stride, self.padding)

    # Reshape to isolate rotations to their own axis
    batch_size, _, height, width = real_conv.shape
    x_conv = real_conv.view(batch_size, self.out_channels, 4, height, width)
    y_conv = imag_conv.view(batch_size, self.out_channels, 4, height, width)

    if self.aggregation == "sum":
      # Add the outputs in cartesian coordinates
      output_x = torch.sum(x_conv, dim=2)
      output_y = torch.sum(y_conv, dim=2)

      # Convert the results to polar coordinates
      output_mag = torch.sqrt(output_x**2 + output_y**2)
      output_phase = torch.atan2(output_y, output_x)

      vectorized_output = torch.stack([output_mag, output_phase], dim=-1)
    else:
      # Convert the results to polar coordinates
      output_mag = torch.sqrt(x_conv**2 + y_conv**2)
      output_phase = torch.atan2(y_conv, x_conv)

      # Apply Max Pooling, drop excess rotation channels
      max_magnitude, max_indices = torch.max(output_mag, dim=2, keepdim=True)
      max_phase = torch.gather(output_phase, 2, max_indices)

      vectorized_output = torch.stack([max_magnitude.squeeze(2), max_phase.squeeze(2)], dim=-1)
    
    return vectorized_output

  def rotate_filters(self, filters):
    rotated_filters = torch.stack([
      filters,
      filters.transpose(2, 3).flip(2),
      filters.flip(2).flip(3),
      filters.transpose(2, 3).flip(3)
    ], dim=0).view(-1, *filters.shape[1:])
    return rotated_filters

class VectorTransformConv2d(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, aggregation="pool"):
    super(VectorTransformConv2d, self).__init__()

    # Parameters for the convolution operation
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = (kernel_size, kernel_size)
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.groups = groups
    self.aggregation = aggregation

    # Learnable parameters: weights and bias
    self.filters = nn.Parameter(torch.randn(out_channels, in_channels, *self.kernel_size))
    self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None

    # Phase iteration: angular spacing between rotations
    self.phase_values = nn.Parameter(torch.tensor([0, torch.pi/2, torch.pi, 3*torch.pi/2]), requires_grad=False)

  def forward(self, x):
    # Apply convolution with each rotated filter and stack the outputs
    output = F.conv2d(x, self.rotate_filters(self.filters), self.bias.repeat(4), self.stride, self.padding, self.dilation, self.groups)
      
    # Reshape output to separate filters from rotations
    batch_size, _, height, width = output.shape
    output_mag = output.view(batch_size, self.out_channels, 4, height, width)
    output_phase = self.phase_values.repeat(self.out_channels).view(1, self.out_channels, 4, 1, 1).expand(batch_size, -1, -1, height, width)

    if self.aggregation == "sum":
      # Convert into cartesian coordinates
      x_conv = output * torch.cos(output_phase)
      y_conv = output * torch.sin(output_phase)

      # Add the outputs in cartesian coordinates
      output_x = torch.sum(x_conv, dim=2)
      output_y = torch.sum(y_conv, dim=2)

      # Convert the results to polar coordinates
      output_mag = torch.sqrt(output_x**2 + output_y**2)
      output_phase = torch.atan2(output_y, output_x)

      vectorized_output = torch.stack([output_mag, output_phase], dim=-1)
    else:
      # Apply Max Pooling, drop excess rotation channels
      max_magnitude, max_indices = torch.max(output_mag, dim=2, keepdim=True)
      max_phase = torch.gather(output_phase, 2, max_indices)

      vectorized_output = torch.stack([max_magnitude.squeeze(2), max_phase.squeeze(2)], dim=-1)
    
    return vectorized_output

  def rotate_filters(self, filters):
    # Rotate the weight tensor by 0, 90, 180, and 270 degrees
    rotated_filters = torch.stack([
      filters,
      filters.transpose(2, 3).flip(2),
      filters.flip(2).flip(3),
      filters.transpose(2, 3).flip(3)
    ], dim=0).view(-1, *filters.shape[1:])
    return rotated_filters

class VectorMaxPool2d(nn.Module):
  def __init__(self, kernel_size, stride=None, padding=0):
    super(VectorMaxPool2d, self).__init__()
    # Initialize parameters similar to nn.MaxPool2d
    self.kernel_size = kernel_size
    self.stride = stride if stride is not None else kernel_size
    self.padding = padding

  def forward(self, x):
    # Extract magnitude and phase
    magnitude = x[..., 0]
    phase = x[..., 1]

    # Apply max pooling to the magnitude part
    pooled_magnitude, indices = F.max_pool2d(magnitude, self.kernel_size, self.stride, self.padding, return_indices=True)

    # Use the indices from max pooling on magnitude to gather corresponding phases
    # Phase tensor reshaping for gathering
    phase_flat = phase.view(phase.shape[0], phase.shape[1], -1) # Flatten height and width for indexing
    pooled_phase = torch.gather(phase_flat, 2, indices.view(phase.shape[0], phase.shape[1], -1))
    pooled_phase = pooled_phase.view_as(pooled_magnitude)

    pooled_output = torch.stack([pooled_magnitude, pooled_phase], dim=-1)
    return pooled_output

class VectorBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(VectorBatchNorm2d, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features)

    def forward(self, x):
        magnitude, phase = x[..., 0], x[..., 1]
        magnitude = self.batch_norm(magnitude)
        return torch.stack([magnitude, phase], dim=-1)

class Vector2Magnitude(nn.Module):
  def __init__(self):
    super(Vector2Magnitude, self).__init__()

  def forward(self, x):
    assert x.size(-1) == 2, "Input must be a vector feature map with last dim = 2"
    return x[..., 0]

def vector_relu(x):
  magnitude, phase = x[..., 0], x[..., 1]
  magnitude = F.relu(magnitude)
  return torch.stack([magnitude, phase], dim=-1)

class VectorRelu(nn.Module):
  def __init__(self):
    super(VectorRelu, self).__init__()
  
  def forward(self, x):
    magnitude = x[..., 0]
    phase = x[..., 0]
    magnitude = F.relu(magnitude - 1) + 1
    return torch.stack([magnitude, phase], dim=-1)