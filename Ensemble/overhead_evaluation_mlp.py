import torch
import time
from fvcore.nn import FlopCountAnalysis
from utils import read_yaml
from models.mlp_models import EnsembleModel
from dataloaders.dataloaders4mlp import (
    transformer_data4mlp,
    cnn_data4mlp,
    mlp_data4mlp,
)

def get_dataloader(datatype, config):
    if datatype == "transformer":
        return transformer_data4mlp(config)
    elif datatype == "cnn":
        return cnn_data4mlp(config)
    elif datatype == "mlp":
        return mlp_data4mlp(config)
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

def evaluate_mlp(datatype, config_path="config.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = read_yaml(config_path)

    train_loader, _, _, class_counts = get_dataloader(datatype, config)
    x, _ = next(iter(train_loader))
    input_size = x.shape[-1]
    x = x.to(device)

    model = EnsembleModel(
        model_list=[1, 2, 3, 4, 5, 6, 7],
        input_size=input_size,
        output_size=class_counts,
        max_neurons=8,
        min_neurons=4,
        learning_rate=config["learning_rate"]
    ).to(device).eval()

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # Timed inference
    timings = []
    with torch.no_grad():
        for _ in range(100):
            start = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            end = time.time()
            timings.append((end - start) * 1000)

    avg_time_ms = sum(timings) / len(timings)
    std_time_ms = (sum((t - avg_time_ms) ** 2 for t in timings) / len(timings)) ** 0.5
    print(f"✅ Inference Time: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms")

    # VRAM usage
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x)
    mem_used_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"✅ Peak VRAM usage: {mem_used_mb:.2f} MB")

    # FLOPs
    try:
        flops = FlopCountAnalysis(model, x)
        print(f"✅ FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
    except Exception as e:
        print(f"⚠️ FLOPs estimation failed: {e}")

    # Parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Total Parameters: {total:,}")
    print(f"✅ Trainable Parameters: {trainable:,}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datatype", required=True, choices=["transformer", "cnn", "mlp"])
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    evaluate_mlp(args.datatype, args.config)
