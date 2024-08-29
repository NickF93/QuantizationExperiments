import torch

def linear_q_with_scale_and_zero_point(tensor, scale, zero_point, dtype = torch.int8):

    scaled_and_shifted_tensor = tensor / scale + zero_point

    rounded_tensor = torch.round(scaled_and_shifted_tensor)

    q_min = torch.iinfo(dtype).min
    q_max = torch.iinfo(dtype).max

    q_tensor = rounded_tensor.clamp(q_min,q_max).to(dtype)
    
    return q_tensor

def linear_dequantizer(quantized_tensor, scale, zero_point):
    dequantized_tensor = scale * (quantized_tensor.float() - zero_point)
    return dequantized_tensor