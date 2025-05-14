import torch
import time
from fvcore.nn import FlopCountAnalysis
from utils import read_yaml
from models.transformer_models import TransformerEnsemble
from dataloaders.dataloaders4transformer import (
    transformer_data4transformer,
    cnn_data4transformer,
    mlp_data4transformer,
)

def get_dataloader(datatype, config):
    if datatype == "transformer":
        return transformer_data4transformer(config)
    elif datatype == "cnn":
        return cnn_data4transformer(config)
    elif datatype == "mlp":
        return mlp_data4transformer(config)
    else:
        raise ValueError(f"Unsupported datatype: {datatype}")

def evaluate_transformer(datatype, config_path="config.yaml", arch_path="transformers_architectures.yaml"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = read_yaml(config_path)
    archs = read_yaml(arch_path)["model_configs"]

    train_loader, _, _, info = get_dataloader(datatype, config)
    x, z, _ = next(iter(train_loader))
    x, z = x.to(device), z.to(device)

    model = TransformerEnsemble(
        model_configs=archs,
        num_features=x.shape[-1],
        num_classes=info["class_counts"],
        learning_rate=config["learning_rate"],
        max_seq_len=info["max_seq_len"],
        extra_fetures_num=z.shape[-1],
    ).to(device).eval()

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, z)

    # Timed inference
    timings = []
    with torch.no_grad():
        # 100 times running and then averaging over them
        for _ in range(100):
            start = time.time()
            _ = model(x, z)
            torch.cuda.synchronize()
            end = time.time()
            timings.append((end - start) * 1000)

    min_time_ms = min(timings)
    max_time_ms = max(timings)
    avg_time_ms = sum(timings) / len(timings)
    std_time_ms = (sum((t - avg_time_ms) ** 2 for t in timings) / len(timings)) ** 0.5
    print(f"✅ Inference Time | \nMin: {min_time_ms:.2f} ms \n Max: {max_time_ms:.2f} \nAverage: {avg_time_ms:.2f} ± {std_time_ms:.2f} ms")

    # VRAM usage
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(x, z)
    mem_used_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"✅ Peak VRAM usage: {mem_used_mb:.2f} MB")

    # FLOPs
    try:
        flops = FlopCountAnalysis(model, (x, z))
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
    parser.add_argument("-a", "--arch", default="transformers_architectures.yaml")
    args = parser.parse_args()
    evaluate_transformer(args.datatype, args.config, args.arch)
