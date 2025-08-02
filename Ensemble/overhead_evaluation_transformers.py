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
    import statistics

    config = read_yaml(config_path)
    archs = read_yaml(arch_path)["model_configs"]
    train_loader, _, _, info = get_dataloader(datatype, config)
    x, z, _ = next(iter(train_loader))
    num_features = x.shape[-1]
    extra_features = z.shape[-1]
    num_classes = info["class_counts"]
    max_seq_len = info["max_seq_len"]

    results = {}

    def run_inference(label, device):
        model = TransformerEnsemble(
            model_configs=archs,
            num_features=num_features,
            num_classes=num_classes,
            learning_rate=config["learning_rate"],
            max_seq_len=max_seq_len,
            extra_fetures_num=extra_features,
        ).to(device).eval()
        x_device, z_device = x.to(device), z.to(device)

        # Warm-up
        with torch.no_grad():
            for _ in range(10):
                _ = model(x_device, z_device)

        # Timed inference
        timings = []
        with torch.no_grad():
            for _ in range(100):
                start = time.time()
                _ = model(x_device, z_device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.time()
                timings.append((end - start) * 1000)

        stats = {
            "min": min(timings),
            "max": max(timings),
            "avg": statistics.mean(timings),
            "std": statistics.stdev(timings)
        }

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            with torch.no_grad():
                _ = model(x_device, z_device)
            stats["vram_mb"] = torch.cuda.max_memory_allocated() / (1024 ** 2)

        results[label] = stats

        return model, (x_device, z_device)

    # Run both GPU and CPU inference
    if torch.cuda.is_available():
        model_gpu, input_gpu = run_inference("GPU", torch.device("cuda"))
    model_cpu, input_cpu = run_inference("CPU", torch.device("cpu"))

    # Print results
    for label in results:
        stats = results[label]
        print(f"\n🧠 {label} Inference Stats:")
        print(f"  Min:     {stats['min']:.2f} ms")
        print(f"  Max:     {stats['max']:.2f} ms")
        print(f"  Avg:     {stats['avg']:.2f} ± {stats['std']:.2f} ms")
        if "vram_mb" in stats:
            print(f"  VRAM:    {stats['vram_mb']:.2f} MB")

    # FLOPs (run on CPU model/input for consistency)
    try:
        flops = FlopCountAnalysis(model_cpu, input_cpu)
        print(f"\n✅ FLOPs: {flops.total():.2f} ({flops.total() / 1e9:.2f} GFLOPs)")
    except Exception as e:
        print(f"\n⚠️ FLOPs estimation failed: {e}")

    # Parameter count
    total = sum(p.numel() for p in model_cpu.parameters())
    trainable = sum(p.numel() for p in model_cpu.parameters() if p.requires_grad)
    print(f"\n✅ Parameters: {total:,} total | {trainable:,} trainable")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--datatype", required=True, choices=["transformer", "cnn", "mlp"])
    parser.add_argument("-c", "--config", default="config.yaml")
    parser.add_argument("-a", "--arch", default="transformers_architectures.yaml")
    args = parser.parse_args()
    evaluate_transformer(args.datatype, args.config, args.arch)
