import torch
import os
import json

def absmax_quantize(X):
    # Calculate scale
    scale = 127 / torch.max(torch.abs(X))

    # Quantize
    X_quant = (scale * X).round()

    # Dequantize
    X_dequant = X_quant / scale

    return X_quant.to(torch.int8), scale

state_dict = torch.load("lenet5_mnist.pth", map_location="cpu")

output_dir = "/Users/sarvesh/Desktop/lenet5-cpp-inference/weights"
os.makedirs(output_dir, exist_ok=True)

for name, tensor in state_dict.items():
    values = tensor.detach().cpu()
    if "weight" in name:
        quantized_tensor,scale=absmax_quantize(tensor)
    else:
        quantized_tensor=values
        scale=1
    quantized_tensor=quantized_tensor.numpy()
    # Create the full path
    bin_path = os.path.join(output_dir, name + ".bin")

    # Dump the raw memory to that path
    quantized_tensor.tofile(bin_path)
    scale_val= scale.item() if hasattr(scale, 'item') else scale
    metadata = {
        "shape": list(quantized_tensor.shape),
        "scale": scale_val
    }
    # Create the JSON path
    json_path = os.path.join(output_dir, name + ".json")

    # Write the dictionary to the file
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    
