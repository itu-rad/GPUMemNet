import pandas as pd
import matplotlib.pyplot as plt

def shorten(name, max_len=14):
    return name if len(name) <= max_len else name[:max_len] + "…"



# Load the Excel file
file_path = "fake_tensor.xlsx"  # Replace with your Excel file path
df = pd.read_excel(file_path)

df["Filename"] = df["Filename"].str.replace(".out", "", regex=False)

print(df[df["Difference"] > 8000])

# Sample models as per your earlier instructions
sampled_models = df.sample(n=35, random_state=None)  # Adjust n as needed

# Set up the plot with model-batch names on the x-axis and scatter points for memory values
plt.figure(figsize=(14, 5))

selected_df = sampled_models


selected_df["Short Name"] = selected_df["Filename"].apply(lambda x: shorten(x))


# Scatter plot for Max GPU Memory (MiB) with Model-Batch on x-axis
plt.scatter(
    selected_df["Short Name"], selected_df["Max GPU Memory (MiB)"]/1000,
    color='#000000', label='Actual GPU Memory', alpha=0.6, s=170
)

# Scatter plot for Last Numeric Value with Model-Batch on x-axis
plt.scatter(
    selected_df["Short Name"], selected_df["Last Numeric Value"]/1000,
    color='#939393', label='Estimation with FakeTensor', alpha=0.6, s=100
)

# Draw a horizontal line at y=40000
plt.axhline(y=40, color='black', linestyle='-.', linewidth=2, label='A100 GPU Memory = 40GB')

plt.xticks(rotation=55, ha='right', fontsize=16)  # Set tick font size
plt.yticks(fontsize=20)

# Labels and title with font sizes adjusted
plt.xlabel("Models", fontsize=22)
plt.ylabel("GPU Memory (GB)", fontsize=22)
# plt.title("Actual GPU Memory requirement and Estimation with the FakeTensor", fontsize=14)
plt.legend(fontsize=18)

# Save the plot to PDF
output_pdf_path = "ft_gb.pdf"
plt.savefig(output_pdf_path, format='pdf', bbox_inches='tight')

column_name = "Difference"

df[column_name] = df[column_name].abs()
df = df[df[column_name] < 500000]

print("max difference: ", df[column_name].max())
print("min difference: ", df[column_name].min())
print("mean difference: ", df[column_name].mean())

# Plot histogram with font sizes adjusted
plt.figure(figsize=(14, 5))
counts, bins, patches = plt.hist(df[column_name], bins=50, color='#939393', edgecolor='black')

# Increase x and y axis tick font sizes for the second plot
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

# Annotate each bin with the count
i = 0
for count, patch in zip(counts, patches):
    if i % 2 != 0:
        i += 1
        # print(i)
        continue

    # print(i)
    plt.text(patch.get_x() + patch.get_width() / 2, count, int(count), 
             ha='center', va='bottom', fontsize=14, rotation=45)

    i += 1
# Labels and title with font sizes adjusted
plt.xlabel("Absolute Difference Values (MB)", fontsize=20)
plt.ylabel("Frequency", fontsize=20)
# plt.title(f"Distribution of {column_name}", fontsize=18)

# plt.savefig("difference_distribution.pdf", bbox_inches='tight')