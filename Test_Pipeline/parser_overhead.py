import time
import statistics
from pathlib import Path
import importlib.util

# Load function from file
def load_function(file_path, function_name):
    spec = importlib.util.spec_from_file_location("mod", file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, function_name)

# Parameters
cnn_mlp_parser_path = "01-Tester_cnn_mlp.py"
trans_mlp_parser_path = "02-Tester_trans_mlp.py"
function_name = "extract_model_info"

# Example inputs
cnn_files = [("cnn_models/efficientnet_32.model", 32)]
trans_files = [("Trans_models/gpt2_xl_bs:2_sl:512.txt", 32, 512)]

repetitions = 100

# Load parser functions
cnn_parser = load_function(cnn_mlp_parser_path, function_name)
trans_parser = load_function(trans_mlp_parser_path, function_name)

def benchmark_parser(fn, args_list, repetitions):
    results = []
    for args in args_list:
        runtimes = []
        for _ in range(repetitions):
            start = time.perf_counter()
            fn(*args)
            end = time.perf_counter()
            runtimes.append((end - start) * 1000)  # ms
        mean_rt = statistics.mean(runtimes)
        std_rt  = statistics.stdev(runtimes) if len(runtimes) > 1 else 0.0
        results.append({
            "input": args,
            "min": min(runtimes),
            "max": max(runtimes),
            "mean": mean_rt,
            "std": std_rt,
        })
    return results

# Run benchmarks
cnn_results = benchmark_parser(cnn_parser, cnn_files, repetitions)
trans_results = benchmark_parser(trans_parser, trans_files, repetitions)

# Write to Markdown
md_lines = [
    "# Parser Overhead Report",
    f"\nRan each parser {repetitions} times\n",
    "## CNN/MLP Parser\n"
]

for res in cnn_results:
    md_lines.append(f"**Input:** {res['input']}")
    md_lines.append(f"- Min: {res['min']:.2f} ms")
    md_lines.append(f"- Max: {res['max']:.2f} ms")
    md_lines.append(f"- Mean: {res['mean']:.2f} ms")
    md_lines.append(f"- Std: {res['std']:.2f} ms")
    md_lines.append(f"- Mean ± Std: {res['mean']:.2f} ± {res['std']:.2f} ms\n")

md_lines.append("## Transformer/MLP Parser\n")
for res in trans_results:
    md_lines.append(f"**Input:** {res['input']}")
    md_lines.append(f"- Min: {res['min']:.2f} ms")
    md_lines.append(f"- Max: {res['max']:.2f} ms")
    md_lines.append(f"- Mean: {res['mean']:.2f} ms")
    md_lines.append(f"- Std: {res['std']:.2f} ms")
    md_lines.append(f"- Mean ± Std: {res['mean']:.2f} ± {res['std']:.2f} ms\n")

# Save output
Path("parser_overhead.md").write_text("\n".join(md_lines))
print("Benchmark complete. Results saved to parser_overhead.md.")