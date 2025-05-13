import subprocess

# Output Markdown file
output_file = "Overheads.md"

# Define the scripts you want to run
scripts = [
    "overhead_evaluation_mlp.py",
    "overhead_evaluation_transformers.py"
]

# Define the datatypes to iterate over
datatypes = ["mlp", "cnn", "transformer"]

# Run all combinations
with open(output_file, "w") as f:
    f.write("# Overhead Evaluation Results\n\n")

    for script in scripts:
        for dtype in datatypes:
            experiment_name = f"{script} with --datatype {dtype}"
            cmd = ["python", script, "-d", dtype]

            f.write(f"## {experiment_name}\n")
            f.write(f"**Command:** `{' '.join(cmd)}`\n\n")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                f.write("**Output:**\n\n```\n")
                f.write(result.stdout)
                f.write("\n```\n\n")
            except subprocess.CalledProcessError as e:
                f.write("**Error:**\n\n```\n")
                f.write(e.stderr or str(e))
                f.write("\n```\n\n")