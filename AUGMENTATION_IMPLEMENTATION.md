# Data Augmentation Implementation Summary

## Overview
This document summarizes the data augmentation refactoring and new features added to the neural sequence decoder.

## What Was Implemented

### 1. Unified Augmentation Module (`NeuralAugmentations`)
- **Location**: `src/neural_decoder/augmentations.py`
- **Purpose**: Consolidates all data augmentations into a single, configurable module
- **Features**:
  - White noise augmentation (configurable std)
  - Baseline shift augmentation (configurable std)
  - Time masking (new)
  - Feature masking (new)
  - Training-only application (automatically disabled during eval)

### 2. Existing Augmentations (Made Configurable)
- **White Noise**: Standard deviation configurable via `white_noise_std` (default: 0.8)
- **Baseline Shift**: Standard deviation configurable via `baseline_shift_std` (default: 0.2)
- **Backward Compatibility**: Old parameter names (`whiteNoiseSD`, `constantOffsetSD`) still work

### 3. New Augmentations

#### Time Masking
- Randomly masks contiguous time steps across all features
- Parameters:
  - `time_mask_prob`: Probability of applying per sample (default: 0.0, disabled)
  - `time_mask_max_width`: Maximum mask width in time steps (default: 0)
- Applied only during training

#### Feature Masking
- Randomly masks contiguous feature channels across all time steps
- Parameters:
  - `feature_mask_prob`: Probability of applying per sample (default: 0.0, disabled)
  - `feature_mask_max_width`: Maximum mask width in channels (default: 0)
- Applied only during training

### 4. Integration Points

#### Training Script (`scripts/train_model.py`)
- New parameters can be added to `args` dictionary
- Examples provided in comments
- Backward compatible - existing scripts work unchanged

#### Trainer (`src/neural_decoder/neural_decoder_trainer.py`)
- Augmentations initialized from config
- Automatically enabled/disabled based on model training mode
- Configuration logged at startup

## Usage Examples

### Basic Usage (Backward Compatible)
```python
args = {
    'whiteNoiseSD': 0.8,      # Old name still works
    'constantOffsetSD': 0.2,  # Old name still works
}
```

### Enable Time Masking
```python
args = {
    'whiteNoiseSD': 0.8,
    'constantOffsetSD': 0.2,
    'time_mask_prob': 0.1,        # 10% chance per sample
    'time_mask_max_width': 10,    # Max 10 time steps
}
```

### Enable Feature Masking
```python
args = {
    'whiteNoiseSD': 0.8,
    'constantOffsetSD': 0.2,
    'feature_mask_prob': 0.1,        # 10% chance per sample
    'feature_mask_max_width': 5,     # Max 5 channels
}
```

### Enable Both Masking Types
```python
args = {
    'whiteNoiseSD': 0.8,
    'constantOffsetSD': 0.2,
    'time_mask_prob': 0.1,
    'time_mask_max_width': 10,
    'feature_mask_prob': 0.1,
    'feature_mask_max_width': 5,
}
```

## Future Experiment Hooks

The codebase has been structured with clear hooks for future experiments:

### 1. Optimizer Changes
- **Location**: `neural_decoder_trainer.py` around line 165
- **Hook**: Commented section showing where to swap optimizers
- **Examples**: AdamW, different epsilon values, learning rate schedules

### 2. Log Transforms
- **Location**: `neural_decoder_trainer.py` around line 82
- **Hook**: Commented section for data preprocessing
- **Example**: `X = torch.log(X + epsilon)` before normalization

### 3. Context-Dependent Outputs
- **Location**: `neural_decoder_trainer.py` around line 200
- **Hook**: Commented section for model forward pass modifications
- **Examples**: Diphones, context windows

### 4. CTC Loss Variants
- **Location**: `neural_decoder_trainer.py` around line 160
- **Hook**: Commented section for alternative loss functions
- **Examples**: Focal loss, alignment-free losses

## Backward Compatibility

✅ **Fully backward compatible**:
- Old parameter names (`whiteNoiseSD`, `constantOffsetSD`) still work
- Default values match original baseline behavior
- New augmentations disabled by default
- Existing training scripts work without modification

## Testing

The implementation has been tested:
- ✅ Augmentation module instantiation
- ✅ Forward pass with all augmentation types
- ✅ Training/eval mode switching
- ✅ Configuration summary generation
- ✅ Shape preservation

## Code Quality

- ✅ Comprehensive docstrings
- ✅ Type hints where appropriate
- ✅ Clear parameter validation
- ✅ Logging of configuration
- ✅ Modular design for easy extension

