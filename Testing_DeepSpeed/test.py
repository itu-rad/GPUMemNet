from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage_1_and_2 import estimate_zero2_model_states_mem_needs_all_cold
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_cold

from timm import create_model

# Define a vision model, e.g., ResNet50
model = create_model('resnet50', pretrained=False)



# Replace with your GPU and node configurations
memory_estimate = estimate_zero2_model_states_mem_needs_all_live(
    model=model,
    num_gpus_per_node=1,  # Number of GPUs you plan to use per node
    num_nodes=1           # Number of nodes
)
print("Estimated memory (ZeRO Stage 2, live):", memory_estimate, "bytes")


memory_estimate = estimate_zero3_model_states_mem_needs_all_live(
    model=model,
    num_gpus_per_node=1,  # Number of GPUs per node
    num_nodes=1           # Number of nodes
)
print("Estimated memory (ZeRO Stage 3, live):", memory_estimate, "bytes")


# Estimate with parameter count only (e.g., ResNet50 has ~25 million parameters)
total_params = 25e6
memory_estimate = estimate_zero2_model_states_mem_needs_all_cold(
    total_params=total_params,
    num_gpus_per_node=1,  # Number of GPUs per node
    num_nodes=1           # Number of nodes
)
print("Estimated memory (ZeRO Stage 2, cold):", memory_estimate, "bytes")



# Estimate with parameter count only (e.g., ResNet50 has ~25 million parameters)
total_params = 25e6
memory_estimate = estimate_zero2_model_states_mem_needs_all_cold(
    total_params=total_params,
    num_gpus_per_node=1,  # Number of GPUs per node
    num_nodes=1           # Number of nodes
)
print("Estimated memory (ZeRO Stage 2, cold):", memory_estimate, "bytes")


memory_estimate_mb = memory_estimate / (1024 ** 2)
print("Estimated memory in MB:", memory_estimate_mb, "MB")
