import os
import pickle
import time

from edit_distance import SequenceMatcher
import hydra
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .model import GRUDecoder
from .dataset import SpeechDataset
from .augmentations import NeuralAugmentations


def getDatasetLoaders(
    datasetName,
    batchSize,
):
    with open(datasetName, "rb") as handle:
        loadedData = pickle.load(handle)

    def _padding(batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    train_ds = SpeechDataset(loadedData["train"], transform=None)
    test_ds = SpeechDataset(loadedData["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=batchSize,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_padding,
    )

    return train_loader, test_loader, loadedData

def trainModel(args):
    os.makedirs(args["outputDir"], exist_ok=True)
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])
    
    # Enforce GPU usage - raise error if CUDA is not available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for training. Please ensure PyTorch is installed with CUDA support.")
    
    device = "cuda"
    
    # Verify GPU actually works by testing a simple operation
    try:
        test_tensor = torch.zeros(1).to(device)
        _ = torch.matmul(test_tensor, test_tensor)
        del test_tensor
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    except RuntimeError as e:
        raise RuntimeError(f"CUDA device failed to initialize: {e}. Please ensure your GPU is compatible with the installed PyTorch version.")

    with open(args["outputDir"] + "/args", "wb") as file:
        pickle.dump(args, file)

    # ========================================================================
    # DATA LOADING
    # ========================================================================
    trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )
    
    # ========================================================================
    # DATA PREPROCESSING: Log Transform
    # ========================================================================
    # Log transformation can help normalize neural data distributions
    # Applied before model processing: log(X + epsilon)
    use_log_transform = args.get("use_log_transform", False)
    log_transform_epsilon = args.get("log_transform_epsilon", 1e-6)
    
    if use_log_transform:
        print(f"✓ Log transform enabled (epsilon={log_transform_epsilon})")
        print("  Applied as: log(X - min(X) + epsilon) to handle negative values")
    else:
        print("  Log transform: disabled (baseline)")

    # Extract new optional parameters with backward-compatible defaults
    post_gru_num_layers = args.get("post_gru_num_layers", 0)
    post_gru_dropout = args.get("post_gru_dropout", 0.2)  # Increased default for better regularization
    use_day_specific_input = args.get("use_day_specific_input", True)
    use_input_layernorm = args.get("use_input_layernorm", False)
    gradient_clip = args.get("gradient_clip", 1.0)  # Gradient clipping for stability
    
    # Extract augmentation parameters with backward-compatible defaults
    # Defaults match current baseline behavior
    white_noise_std = args.get("whiteNoiseSD", args.get("white_noise_std", 0.8))
    baseline_shift_std = args.get("constantOffsetSD", args.get("baseline_shift_std", 0.2))
    time_mask_prob = args.get("time_mask_prob", 0.0)  # Disabled by default (iteration 2 config)
    time_mask_max_width = args.get("time_mask_max_width", 0)  # Disabled by default
    feature_mask_prob = args.get("feature_mask_prob", 0.0)  # Disabled by default
    feature_mask_max_width = args.get("feature_mask_max_width", 0)  # Disabled by default
    
    # Print model configuration for debugging
    bidirectional = args.get("bidirectional", False)
    
    print("\n" + "="*60)
    print("Model Configuration:")
    print(f"  Bidirectional GRU: {bidirectional} (False = baseline)")
    if bidirectional:
        print(f"    → Hidden dim effectively: {args['nUnits']} × 2 = {args['nUnits'] * 2}")
    print(f"  Post-GRU layers: {post_gru_num_layers} (0 = disabled, baseline)")
    print(f"  Post-GRU dropout: {post_gru_dropout}")
    print(f"  Use day-specific input: {use_day_specific_input} (True = baseline)")
    print(f"  Use input LayerNorm: {use_input_layernorm} (False = baseline)")
    print(f"  Gradient clipping: {gradient_clip} (0 = disabled)")
    if post_gru_num_layers == 0 and use_day_specific_input and not use_input_layernorm and not bidirectional:
        print("  ⚠ WARNING: Using baseline configuration (no new features enabled)")
    print("="*60 + "\n")
    
    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=bidirectional,
        post_gru_num_layers=post_gru_num_layers,
        post_gru_dropout=post_gru_dropout,
        use_day_specific_input=use_day_specific_input,
        use_input_layernorm=use_input_layernorm,
    ).to(device)
    
    # Verify model architecture and count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    if bidirectional:
        print(f"✓ Bidirectional GRU enabled")
        print(f"  Effective hidden dim: {args['nUnits']} × 2 = {args['nUnits'] * 2}")
    if post_gru_num_layers > 0:
        assert model.post_gru_stack is not None, "Post-GRU stack should be created"
        post_gru_params = sum(p.numel() for p in model.post_gru_blocks.parameters())
        print(f"✓ Post-GRU stack created with {post_gru_num_layers} blocks ({post_gru_params:,} parameters)")
        print(f"  Using residual connections and dropout={post_gru_dropout}")
    if not use_day_specific_input and use_input_layernorm:
        assert model.input_layernorm is not None, "Input LayerNorm should be created"
        print(f"✓ Input LayerNorm enabled (replacing day-specific input)")
    print()
    
    # Initialize augmentation module
    augmentations = NeuralAugmentations(
        white_noise_std=white_noise_std,
        baseline_shift_std=baseline_shift_std,
        time_mask_prob=time_mask_prob,
        time_mask_max_width=time_mask_max_width,
        feature_mask_prob=feature_mask_prob,
        feature_mask_max_width=feature_mask_max_width,
        device=device,
    ).to(device)
    augmentations.train()  # Enable training mode for augmentations
    
    print("="*60)
    print("Data Augmentation Configuration:")
    print(f"  {augmentations.get_config_summary()}")
    print("="*60 + "\n")

    # ========================================================================
    # LOSS FUNCTION
    # ========================================================================
    # Current: Standard CTC loss
    # Future experiments hook: Alternative loss functions
    #   - Could swap in: Connectionist Temporal Classification variants
    #   - Alignment-free sequence losses
    #   - Focal loss for class imbalance
    #   - Example: loss_ctc = FocalCTCLoss(...) or AlignmentFreeLoss(...)
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    
    # ========================================================================
    # OPTIMIZER
    # ========================================================================
    # Current: Adam optimizer (original baseline)
    # Future experiments hook: Alternative optimizers
    #   - AdamW (weight decay decoupled)
    #   - Different epsilon values
    #   - Example: optimizer = torch.optim.AdamW(..., weight_decay=args["l2_decay"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args["lrStart"],
        betas=(0.9, 0.999),
        eps=0.1,
        weight_decay=args["l2_decay"],
    )
    
    # ========================================================================
    # LEARNING RATE SCHEDULE
    # ========================================================================
    # Two options:
    # 1. Default: Linear decay from lrStart to lrEnd (baseline)
    # 2. Warmup + Cosine Annealing: Better for complex architectures
    use_warmup_cosine = args.get("use_warmup_cosine", False)
    warmup_batches = args.get("warmup_batches", 500)
    
    if use_warmup_cosine:
        # Warmup + Cosine Annealing Schedule
        # Phase 1 (Warmup): Linear increase from ~0 to lrStart
        # Phase 2 (Main): Cosine decay from lrStart to lrEnd
        print(f"✓ Using Warmup ({warmup_batches} batches) + Cosine Annealing schedule")
        print(f"  LR: 0 → {args['lrStart']} (warmup) → {args['lrEnd']} (cosine)")
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,  # Start at lrStart * 1e-4 ≈ 0
            end_factor=1.0,     # End at lrStart
            total_iters=warmup_batches,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args["nBatch"] - warmup_batches,
            eta_min=args["lrEnd"],
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_batches],
        )
    else:
        # Default: Linear decay (original baseline behavior)
        print(f"  Using Linear LR decay: {args['lrStart']} → {args['lrEnd']}")
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=args["lrEnd"] / args["lrStart"],
            total_iters=args["nBatch"],
        )

    # --train--
    testLoss = []
    testCER = []
    startTime = time.time()
    for batch in range(args["nBatch"]):
        model.train()
        augmentations.train()  # Enable augmentations during training

        X, y, X_len, y_len, dayIdx = next(iter(trainLoader))
        X, y, X_len, y_len, dayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            dayIdx.to(device),
        )

        # Apply log transform if enabled (before augmentations)
        # Log transform: log(1 + X) to handle negative values, or log(X - min + epsilon) for centered data
        if use_log_transform:
            # Option 1: log1p (log(1 + X)) - handles negative values but shifts distribution
            # Option 2: Shift to positive then log - preserves relative differences better
            X_min = X.min()
            if X_min < 0:
                # Shift to make all values positive, then log
                X = torch.log(X - X_min + log_transform_epsilon)
            else:
                # Already positive, just add epsilon
                X = torch.log(X + log_transform_epsilon)

        # Apply augmentations (only during training)
        X = augmentations(X)

        # ====================================================================
        # MODEL FORWARD PASS
        # ====================================================================
        # Future experiments hook: Context-dependent outputs
        #   - Diphones instead of single phonemes
        #   - Context window around each prediction
        #   - Example: pred = model.forward_with_context(X, dayIdx, context_window=5)
        pred = model.forward(X, dayIdx)

        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for training stability (especially important with deeper models)
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        scheduler.step()

        # print(endTime - startTime)

        # Eval
        if batch % 100 == 0:
            with torch.no_grad():
                model.eval()
                augmentations.eval()  # Disable augmentations during evaluation
                allLoss = []
                total_edit_distance = 0
                total_seq_length = 0
                for X, y, X_len, y_len, testDayIdx in testLoader:
                    X, y, X_len, y_len, testDayIdx = (
                        X.to(device),
                        y.to(device),
                        X_len.to(device),
                        y_len.to(device),
                        testDayIdx.to(device),
                    )
                    
                    # Apply log transform if enabled (during evaluation too)
                    if use_log_transform:
                        X_min = X.min()
                        if X_min < 0:
                            X = torch.log(X - X_min + log_transform_epsilon)
                        else:
                            X = torch.log(X + log_transform_epsilon)

                    pred = model.forward(X, testDayIdx)
                    loss = loss_ctc(
                        torch.permute(pred.log_softmax(2), [1, 0, 2]),
                        y,
                        ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
                        y_len,
                    )
                    loss = torch.sum(loss)
                    allLoss.append(loss.cpu().detach().numpy())

                    adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
                        torch.int32
                    )
                    for iterIdx in range(pred.shape[0]):
                        decodedSeq = torch.argmax(
                            torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                            dim=-1,
                        )  # [num_seq,]
                        decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                        decodedSeq = decodedSeq.cpu().detach().numpy()
                        decodedSeq = np.array([i for i in decodedSeq if i != 0])

                        trueSeq = np.array(
                            y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                        )

                        matcher = SequenceMatcher(
                            a=trueSeq.tolist(), b=decodedSeq.tolist()
                        )
                        total_edit_distance += matcher.distance()
                        total_seq_length += len(trueSeq)

                avgDayLoss = np.sum(allLoss) / len(testLoader)
                cer = total_edit_distance / total_seq_length

                endTime = time.time()
                print(
                    f"batch {batch}, ctc loss: {avgDayLoss:>7f}, cer: {cer:>7f}, time/batch: {(endTime - startTime)/100:>7.3f}"
                )
                startTime = time.time()

            if len(testCER) > 0 and cer < np.min(testCER):
                torch.save(model.state_dict(), args["outputDir"] + "/modelWeights")
            testLoss.append(avgDayLoss)
            testCER.append(cer)

            tStats = {}
            tStats["testLoss"] = np.array(testLoss)
            tStats["testCER"] = np.array(testCER)

            with open(args["outputDir"] + "/trainingStats", "wb") as file:
                pickle.dump(tStats, file)


def loadModel(modelDir, nInputLayers=24, device=None):
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. GPU is required for model loading. Please ensure PyTorch is installed with CUDA support.")
        device = "cuda"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for model loading.")
    
    modelWeightPath = modelDir + "/modelWeights"
    with open(modelDir + "/args", "rb") as handle:
        args = pickle.load(handle)

    # Extract new optional parameters with backward-compatible defaults
    post_gru_num_layers = args.get("post_gru_num_layers", 0)
    post_gru_dropout = args.get("post_gru_dropout", 0.1)
    use_day_specific_input = args.get("use_day_specific_input", True)
    use_input_layernorm = args.get("use_input_layernorm", False)

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        device=device,
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        bidirectional=args["bidirectional"],
        post_gru_num_layers=post_gru_num_layers,
        post_gru_dropout=post_gru_dropout,
        use_day_specific_input=use_day_specific_input,
        use_input_layernorm=use_input_layernorm,
    ).to(device)

    model.load_state_dict(torch.load(modelWeightPath, map_location=device))
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config")
def main(cfg):
    cfg.outputDir = os.getcwd()
    trainModel(cfg)

if __name__ == "__main__":
    main()