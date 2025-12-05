"""
Example: How to enable the new model features in your training script.

Add these lines to your scripts/train_model.py args dictionary to enable new features.
"""

# Example 1: Enable post-GRU stack (2 layers)
args_with_post_gru = {
    # ... all your existing args ...
    "post_gru_num_layers": 2,      # Enable 2 Linear-LayerNorm-Dropout blocks
    "post_gru_dropout": 0.1,       # Dropout rate for post-GRU stack
}

# Example 2: Use LayerNorm instead of day-specific input
args_with_layernorm = {
    # ... all your existing args ...
    "use_day_specific_input": False,  # Disable day-specific linear + softsign
    "use_input_layernorm": True,      # Enable shared LayerNorm before GRU
}

# Example 3: Combine both features
args_combined = {
    # ... all your existing args ...
    "post_gru_num_layers": 2,
    "post_gru_dropout": 0.1,
    "use_day_specific_input": False,
    "use_input_layernorm": True,
}

# IMPORTANT: If you don't add these parameters, the model uses baseline defaults:
# - post_gru_num_layers = 0 (disabled)
# - use_day_specific_input = True (baseline)
# - use_input_layernorm = False (baseline)
# This means the model will be IDENTICAL to the baseline!

