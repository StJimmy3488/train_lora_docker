import torch
from safetensors.torch import load_file, save_file
from collections import OrderedDict

meta = OrderedDict()
meta['format'] = "pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reduce_weight(weight, target_size):
    weight = weight.to(device, torch.float32)
    original_shape = weight.shape

    if len(original_shape) == 1:
        # For 1D tensors, simply truncate
        return weight[:target_size]

    if original_shape[0] <= target_size:
        return weight

    # Reshape the tensor to 2D
    flattened = weight.reshape(original_shape[0], -1)

    # Perform SVD
    U, S, V = torch.svd(flattened)

    # Reduce the dimensions
    reduced = torch.mm(U[:target_size, :], torch.diag(S)).mm(V.t())

    # Reshape back to the original shape with reduced first dimension
    new_shape = (target_size,) + original_shape[1:]
    return reduced.reshape(new_shape)


def reduce_bias(bias, target_size):
    bias = bias.to(device, torch.float32)
    return bias[:target_size]


# Load your original state dict
state_dict = load_file(
    "/home/jaret/Dev/models/hf/PixArt-Sigma-XL-2-512_MS_t5large_raw/transformer/diffusion_pytorch_model.orig.safetensors")

# Create a new state dict for the reduced model
new_state_dict = {}

for key, value in state_dict.items():
    value = value.to(device, torch.float32)

    if 'weight' in key or 'scale_shift_table' in key:
        if value.shape[0] == 1152:
            if len(value.shape) == 4:
                orig_shape = value.shape
                output_shape = (512, orig_shape[1], orig_shape[2], orig_shape[3])  # reshape to (1152, -1)
                # reshape to (1152, -1)
                value = value.view(value.shape[0], -1)
                value = reduce_weight(value, 512)
                value = value.view(output_shape)
            else:
                # value = reduce_weight(value.t(), 576).t().contiguous()
                value = reduce_weight(value, 512)
                pass
        elif value.shape[0] == 4608:
            if len(value.shape) == 4:
                orig_shape = value.shape
                output_shape = (2048, orig_shape[1], orig_shape[2], orig_shape[3])
                value = value.view(value.shape[0], -1)
                value = reduce_weight(value, 2048)
                value = value.view(output_shape)
            else:
                value = reduce_weight(value, 2048)
        elif value.shape[0] == 6912:
            if len(value.shape) == 4:
                orig_shape = value.shape
                output_shape = (3072, orig_shape[1], orig_shape[2], orig_shape[3])
                value = value.view(value.shape[0], -1)
                value = reduce_weight(value, 3072)
                value = value.view(output_shape)
            else:
                value = reduce_weight(value, 3072)

        if len(value.shape) > 1 and value.shape[
            1] == 1152 and 'attn2.to_k.weight' not in key and 'attn2.to_v.weight' not in key:
            value = reduce_weight(value.t(), 512).t().contiguous()  # Transpose before and after reduction
            pass
        elif len(value.shape) > 1 and value.shape[1] == 4608:
            value = reduce_weight(value.t(), 2048).t().contiguous()   # Transpose before and after reduction
            pass

    elif 'bias' in key:
        if value.shape[0] == 1152:
            value = reduce_bias(value, 512)
        elif value.shape[0] == 4608:
            value = reduce_bias(value, 2048)
        elif value.shape[0] == 6912:
            value = reduce_bias(value, 3072)

    new_state_dict[key] = value

# Move all to CPU and convert to float16
for key, value in new_state_dict.items():
    new_state_dict[key] = value.cpu().to(torch.float16)

# Save the new state dict
save_file(new_state_dict,
          "/home/jaret/Dev/models/hf/PixArt-Sigma-XL-2-512_MS_t5large_raw/transformer/diffusion_pytorch_model.safetensors",
          metadata=meta)

print("Done!")