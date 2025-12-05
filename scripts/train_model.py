
from pathlib import Path
from neural_decoder.neural_decoder_trainer import trainModel

# Name for this run
modelName = "speechBaseline4"

# Resolve project root (one level up from scripts/)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Where to save logs / checkpoints
output_dir = REPO_ROOT / "logs" / "speech_logs" / modelName
output_dir.mkdir(parents=True, exist_ok=True)

# Where your formatted dataset pickle lives (adjust if you saved it elsewhere)
dataset_path = REPO_ROOT / "data" / "ptDecoder_ctc.pkl"

args = {}
args["outputDir"]       = str(output_dir)
args["datasetPath"]     = str(dataset_path)
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
# ============================================================================
# HYPERPARAMETER TUNING (Iteration 7) - TARGET: <20% CER
# ============================================================================
# Iteration 6 = 21.14% CER (best so far, but minimal improvement from Iter 4)
#
# Strategy: Model architecture change + larger capacity
# Changes:
#   1. GELU activation in post-GRU stack (model.py change) - better non-linearity
#   2. Larger hidden dim (256 → 384) - more model capacity
#   3. Keep strong augmentation from Iteration 6
#   4. Back to 10000 batches (faster training)
# ============================================================================

args['lrStart'] = 0.05  # Same as Iteration 4
args['lrEnd'] = 0.02    # Back to Iteration 4 (0.01 didn't help much)
args['nUnits'] = 384    # TUNED: Larger hidden dim (was 256) -> 768 effective with bidirectional
args['nBatch'] = 10000  # Back to Iteration 4 (faster)
args['nLayers'] = 5     # Same as Iteration 4
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2   # Same as Iteration 4
# Data augmentation parameters - keep strong augmentation from Iteration 6
args['whiteNoiseSD'] = 1.0  # Slightly reduced from 1.2 (balance with larger model)
args['constantOffsetSD'] = 0.3  # Slightly reduced from 0.4
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5  # REVERTED to Iteration 4

# NEW FEATURES: Improved architecture with residual connections and better regularization
# Option 1: Enable post-GRU stack with residual connections (recommended: 1-2 layers)
args['post_gru_num_layers'] = 2  # Number of Linear-LayerNorm-Dropout blocks after GRU
args['post_gru_dropout'] = 0.2   # REVERTED to Iteration 4
args['gradient_clip'] = 1.0      # Gradient clipping for training stability

# ITERATION 4: Bidirectional GRU + Learning Rate Warmup with Cosine Annealing
# Bidirectional GRU captures both past and future context in neural signals
# This works synergistically with the residual post-GRU stack
args['bidirectional'] = True  # Enable bidirectional GRU (doubles hidden dim to 512)

# Learning rate warmup + cosine annealing for better training stability
# Warmup: Linear increase from ~0 to lrStart over warmup_batches
# Main: Cosine decay from lrStart to lrEnd
args['use_warmup_cosine'] = True  # Enable warmup + cosine annealing schedule
args['warmup_batches'] = 500   # REVERTED to Iteration 4

# Option 2: Use LayerNorm instead of day-specific input (alternative to day-specific params)
# args['use_day_specific_input'] = False  # Disable day-specific linear + softsign
# args['use_input_layernorm'] = True      # Enable shared LayerNorm before GRU

# Option 3: Combine both features
# args['post_gru_num_layers'] = 2
# args['post_gru_dropout'] = 0.1
# args['use_day_specific_input'] = False
# args['use_input_layernorm'] = True

from neural_decoder.neural_decoder_trainer import trainModel
trainModel(args)
