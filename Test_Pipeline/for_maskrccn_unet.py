import os
import pickle
import re
import numpy as np
from collections import Counter

import torch 
from CNN_MLPredictor import classification_gpu_usage

# activation function preparation
# List of activation functions and their positional encodings
print("the activation function are positionally encoded safely")
activations = ['ELU', 'GELU', 'LeakyReLU', 'Mish', 'PReLU', 'ReLU', 'SELU', 'SiLU', 'Softplus', 'Tanh']

# Positional encoding function (already defined in your code)
def positional_encoding_2d(num_states):
    positions = []
    for i in range(num_states):
        position = (np.sin(i * np.pi / num_states), np.cos(i * np.pi / num_states))
        positions.append(position)
    return np.array(positions)

# Generate positional encodings
positional_encodings = positional_encoding_2d(len(activations))
activation_to_encoding = {activation: positional_encodings[i] for i, activation in enumerate(activations)}

# Function to get the encoding for a given activation function
def get_activation_encoding(activation):
    if activation == "ReLU6":
        activation = "ReLU"
    return activation_to_encoding[activation]

def most_frequent_activation_function(activation_functions):
    activation_counter = Counter(activation_functions)
    most_common_activation, _ = activation_counter.most_common(1)[0]
    return most_common_activation

def extract_model_info(out_file, batch_size):
    import re
    from collections import Counter

    try:
        with open(out_file, 'r') as f:
            lines = f.readlines()

        if not lines:
            return [], 'None', 0, 0, 0, 0.0, 0.0, 0.0, 0.0, {
                'conv2d': 0, 'batchnorm2d': 0, 'dropout': 0,
                'adaptive_avg_pool2d': 0, 'linear': 0, 'softmax': 0
            }

        activations_params = []
        activation_functions_list = []
        total_params = 0
        total_activations = 0
        depth = 0

        input_size_mb = 0.0
        forward_backward_size_mb = 0.0
        params_size_mb = 0.0
        estimated_total_size_mb = 0.0

        layer_counts = {
            'conv2d': 0,
            'batchnorm2d': 0,
            'dropout': 0,
            'adaptive_avg_pool2d': 0,
            'linear': 0,
            'softmax': 0
        }

        # Equivalences
        conv2d_equivalents = {'Conv2d', 'ConvTranspose2d', 'Conv2dNormActivation'}
        bn_names         = {'BatchNorm2d', 'SyncBatchNorm', 'FrozenBatchNorm2d'}
        dropout_names    = {'Dropout', 'Dropout2d'}
        adaptive_pool    = {'AdaptiveAvgPool2d'}
        linear_names     = {'Linear', 'LazyLinear'}
        softmax_names    = {'Softmax', 'LogSoftmax'}
        activation_names = {'ELU','GELU','LeakyReLU','Mish','PReLU','ReLU','SELU','SiLU','Softplus','Tanh','ReLU6'}

        # Composites (counts heuristic); containers to skip
        composite_tokens = {'BasicBlock', 'Bottleneck', 'UnetDecoderBlock'}
        skip_containers  = {
            'ModuleList', 'Sequential', 'BackboneWithFPN', 'RegionProposalNetwork', 'RoIHeads',
            'FeaturePyramidNetwork', 'GeneralizedRCNNTransform', 'AnchorGenerator', 'MaskRCNNHeads',
            'FastRCNNPredictor', 'RPNHead', 'MultiScaleRoIAlign', 'MaskRCNNPredictor', 'LastLevelMaxPool',
            'ResNetEncoder', 'UnetDecoder', 'SegmentationHead', 'Identity', 'Activation'
        }

        # Regex
        type_colon_re   = re.compile(r'([A-Za-z0-9]+)\s*:\s*\d+-\d+')
        shapes_re       = re.compile(r'\[([^\[\]]+)\]')
        total_params_re = re.compile(r'^Total params:\s*([\d,]+)')
        input_mb_re     = re.compile(r'^Input size \(MB\):\s*([\d.]+)')
        fwbw_mb_re      = re.compile(r'^Forward/backward pass size \(MB\):\s*([\d.]+)')
        params_mb_re    = re.compile(r'^Params size \(MB\):\s*([\d.]+)')
        est_total_mb_re = re.compile(r'^Estimated Total Size \(MB\):\s*([\d.]+)')

        def parse_shape_tokens(s):
            toks = [t.strip() for t in s.split(',')]
            out = []
            for t in toks:
                try: out.append(int(t))
                except ValueError: return []
            return out

        def extract_in_out_shapes(line):
            groups = shapes_re.findall(line)
            if not groups: return [], []
            in_shape  = parse_shape_tokens(groups[0])
            out_shape = parse_shape_tokens(groups[-1])  # LAST bracket = Output Shape
            return (in_shape if in_shape else []), (out_shape if out_shape else [])

        def n_activations_from_out_shape(out_shape):
            if not out_shape: return 0
            if len(out_shape) >= 4:  # [B,C,H,W]
                C, H, W = out_shape[-3], out_shape[-2], out_shape[-1]
                return C * H * W
            if len(out_shape) == 3:  # [B,C,L]
                C, L = out_shape[-2], out_shape[-1]
                return C * L
            if len(out_shape) == 2:  # [N,F]
                return out_shape[-1]
            return 0

        for raw in lines:
            line = raw.strip()

            # Footer
            m = total_params_re.match(line)
            if m: total_params = int(m.group(1).replace(',', '')); continue
            m = input_mb_re.match(line)
            if m: input_size_mb = float(m.group(1)); continue
            m = fwbw_mb_re.match(line)
            if m: forward_backward_size_mb = float(m.group(1)); continue
            m = params_mb_re.match(line)
            if m: params_size_mb = float(m.group(1)); continue
            m = est_total_mb_re.match(line)
            if m: estimated_total_size_mb = float(m.group(1)); continue

            # Identify op first (even for recursive lines)
            tmatch = type_colon_re.search(line)
            if not tmatch: continue
            op_type = tmatch.group(1)
            is_recursive = '(recursive)' in line

            # Count conv-equivalents also on recursive rows
            is_conv_equiv = op_type in conv2d_equivalents
            if is_conv_equiv:
                layer_counts['conv2d'] += 1; depth += 1
            elif op_type in bn_names:
                layer_counts['batchnorm2d'] += 1
            elif op_type in dropout_names:
                layer_counts['dropout'] += 1
            elif op_type in adaptive_pool:
                layer_counts['adaptive_avg_pool2d'] += 1
            elif op_type in linear_names:
                layer_counts['linear'] += 1
            elif op_type in softmax_names:
                layer_counts['softmax'] += 1

            # Skip containers for counts; composites expanded heuristically
            if any(tok + ':' in line for tok in skip_containers):
                pass
            elif op_type in composite_tokens:
                # Heuristic counts only
                if op_type == 'BasicBlock':
                    layer_counts['conv2d'] += 2; layer_counts['batchnorm2d'] += 2; depth += 2
                elif op_type == 'Bottleneck':
                    layer_counts['conv2d'] += 3; layer_counts['batchnorm2d'] += 3; depth += 3
                elif op_type == 'UnetDecoderBlock':
                    layer_counts['conv2d'] += 3; layer_counts['batchnorm2d'] += 2; depth += 3
                # Downsample heuristic
                in_shape, out_shape = extract_in_out_shapes(line)
                if in_shape and out_shape:
                    ch_in  = in_shape[-3] if len(in_shape) >= 3 else None
                    ch_out = out_shape[-3] if len(out_shape) >= 3 else None
                    sp_in  = in_shape[-2:] if len(in_shape) >= 2 else None
                    sp_out = out_shape[-2:] if len(out_shape) >= 2 else None
                    if (ch_in and ch_out and ch_in != ch_out) or (sp_in and sp_out and sp_in != sp_out):
                        layer_counts['conv2d'] += 1; layer_counts['batchnorm2d'] += 1; depth += 1

            # Track activation kind
            if op_type in activation_names:
                activation_functions_list.append('ReLU' if op_type == 'ReLU6' else op_type)

            # Activations: DO NOT add for recursive rows (avoid double counting)
            if not is_recursive:
                _, out_shape = extract_in_out_shapes(line)
                nacts = n_activations_from_out_shape(out_shape)
                if nacts > 0:
                    tag = 'conv2d' if is_conv_equiv else op_type.lower()
                    activations_params.append((tag, nacts * batch_size, 0))
                    total_activations += nacts

        activation_function = (Counter(activation_functions_list).most_common(1)[0][0]
                               if activation_functions_list else "ReLU")

        return (
            activations_params,
            activation_function,
            depth,
            total_params,
            total_activations,
            input_size_mb,
            forward_backward_size_mb,
            params_size_mb,
            estimated_total_size_mb,
            layer_counts
        )

    except Exception as e:
        print(f"Error processing {out_file}: {e}")
        return [], 'None', 0, 0, 0, 0.0, 0.0, 0.0, 0.0, {
            'conv2d': 0, 'batchnorm2d': 0, 'dropout': 0,
            'adaptive_avg_pool2d': 0, 'linear': 0, 'softmax': 0
        }
    
def process_model_files(directory, model_file):
    # Load the pre-trained model from the pickled file
    # Load the model from the .ckpt file
    checkpoint_path = model_file
    output_size = 6  # Set the output size according to your task

    # Load the model with its weights from the checkpoint
    model = classification_gpu_usage.load_from_checkpoint(checkpoint_path, output_size=output_size)

    # Print the state_dict to see if the weights are loaded (you can print keys or size)
    state_dict = model.state_dict()
    for param_tensor in list(state_dict.keys())[:10]:
        print(f"Layer: {param_tensor}, Size: {state_dict[param_tensor].size()}")

    print("=======================================")

    for filename in os.listdir(directory):

        if filename.endswith('.model'):
            match = re.search(r'_(\d+)\.model$', filename)
            if not match:
                print(f"Batch size not found in file name: {filename}")
                continue
            batch_size = int(match.group(1))
            file_path = os.path.join(directory, filename)

            features = extract_model_info(file_path, batch_size)

            model.eval()

            input_features = prepare_features_for_model(features, batch_size)
            input_features = torch.tensor(input_features, dtype=torch.float32).view(1, -1)

            print(input_features)

            with torch.no_grad():
                logits = model(input_features)
                predictions = torch.argmax(logits, dim=1)

            print(filename, "Predictions:", predictions)

            # print("=======================================")
            # activations = input_features[0][0]
            # parameters = input_features[0][1]
            # bs = input_features[0][2]
            # gradients = parameters

            # horus_formula_estimation = (activations * bs + parameters) + (bs * gradients)
            # horus_in_bytes = horus_formula_estimation * 4
            # horus_estimations_MB = horus_in_bytes / (1024 ** 2)

            # print("Horus Formual Estimation: ", horus_estimations_MB, activations, parameters, bs)
            print("=======================================")

def prepare_features_for_model(features, batch_size):
    activations_params, activation_function, depth, total_params, total_activations, input_size_mb, forward_backward_size_mb, params_size_mb, estimated_total_size_mb, layer_counts = features
    activation_encoding = get_activation_encoding(activation_function)
    feature_list_mlp = [
        total_activations,                  # F1
        total_activations * batch_size,     # F2
        total_params,                       # F3
        batch_size,                         # F4
        layer_counts['conv2d'],             # F5
        layer_counts['batchnorm2d'],        # F6
        layer_counts['dropout'],            # F7
        activation_encoding[0],             # F8
        activation_encoding[1]              # F9
    ]
    return feature_list_mlp

# Usage example
if __name__ == "__main__":
    directory = 'new_models'
    model_file = '../Analysis/00-Cleaned-NoteBooks/002-MLP-based-estimators/cnn_mlp_8g.ckpt'
    process_model_files(directory, model_file)
