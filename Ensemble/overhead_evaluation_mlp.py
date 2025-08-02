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
    import statistics

    config = read_yaml(config_path)
    train_loader, _, _, class_counts = get_dataloader(datatype, config)
    x, _ = next(iter(train_loader))
    input_size = x.shape[-1]

    model = EnsembleModel(
        model_list=[1, 2, 3, 4, 5, 6, 7],
        input_size=input_size,
        output_size=class_counts,
        max_neurons=8,
        min_neurons=4,
        learning_rate=config["learning_rate"]
    ).eval()  # initial model on CPU

    results = {}

    def run_inference(label, device):
        model_local = model.to(device)
        x_local = x.to(device)

        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model_local(x_local)

        # Timed inference
        timings = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = model_local(x_local)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                timings.append((end - start) * 1000)  # ms

        stats = {
            "min": min(timings),
            "max": max(timings),
            "avg": statistics.mean(timings),
            "std": statistics.stdev(timings)
        }

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model_local(x_local)
            stats["vram_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results[label] = stats

    # Run both GPU and CPU inference
    if torch.cuda.is_available():
        run_inference("GPU", torch.device("cuda"))
    run_inference("CPU", torch.device("cpu"))

    # Print summary
    for label in results:
        stats = results[label]
        print(f"\n🧠 {label} Inference Stats:")
        print(f"  Min:     {stats['min']:.2f} ms")
        print(f"  Max:     {stats['max']:.2f} ms")
        print(f"  Avg:     {stats['avg']:.2f} ± {stats['std']:.2f} ms")
        if "vram_mb" in stats:
            print(f"  VRAM:    {stats['vram_mb']:.2f} MB")

    # FLOPs
    try:
        flops = FlopCountAnalysis(model, x)
        print(f"\n✅ FLOPs: {flops.total():.2f} ({flops.total() / 1e9:.2f} GFLOPs)")
    except Exception as e:
        print(f"\n⚠️ FLOPs estimation failed: {e}")

    # Parameter count
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n✅ Parameters: {total:,} total | {trainable:,} trainable")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datatype", required=True, choices=["transformer", "cnn", "mlp"])
    parser.add_argument("-c", "--config", default="config.yaml")
    args = parser.parse_args()
    evaluate_mlp(args.datatype, args.config)
