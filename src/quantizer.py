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

def q_min_max(dtype=torch.int8):
    return torch.iinfo(dtype).min, torch.iinfo(dtype).max

def get_q_scale_and_zero_point(tensor, dtype=torch.int8):
    
    q_min, q_max = q_min_max(dtype)
    r_min, r_max = tensor.min().item(), tensor.max().item()

    scale = (r_max - r_min) / (q_max - q_min)

    zero_point = q_min - (r_min / scale)

    # clip the zero_point to fall in [quantized_min, quantized_max]
    if zero_point < q_min:
        zero_point = q_min
    elif zero_point > q_max:
        zero_point = q_max
    else:
        # round and cast to int
        zero_point = int(round(zero_point))
    
    return scale, zero_point

def linear_quantizer(tensor, dtype=torch.int8):
    scale, zero_point = get_q_scale_and_zero_point(tensor, 
                                                   dtype=dtype)
    
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor,
                                                          scale, 
                                                          zero_point, 
                                                          dtype=dtype)
    
    return quantized_tensor, scale , zero_point

def get_q_scale_symmetric(tensor: torch.Tensor, dtype: torch.dtype = torch.int8):
    r_max = tensor.abs().max().item()
    q_max = torch.iinfo(dtype).max
    return r_max / q_max

def linear_q_symmetric(tensor, dtype=torch.int8):
    scale = get_q_scale_symmetric(tensor)
    quantized_tensor = linear_q_with_scale_and_zero_point(tensor, scale=scale, zero_point=0, dtype=dtype)

    return quantized_tensor, scale