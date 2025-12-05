import torch
from torch import nn

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
    """
    GRU-based decoder for neural sequence decoding (Brain-to-Text).
    
    Architecture:
    - Gaussian smoothing of input features
    - Day-specific linear transformation + softsign (or optional LayerNorm)
    - Unfold operation (stride/kernel windowing)
    - Multi-layer GRU stack
    - Optional post-GRU Linear-LayerNorm-Dropout stack
    - Final linear layer to phoneme classes + CTC blank
    
    Args:
        neural_dim: Number of input neural features (channels)
        n_classes: Number of phoneme classes
        hidden_dim: Hidden dimension of GRU layers
        layer_dim: Number of GRU layers
        nDays: Number of days (for day-specific parameters)
        dropout: Dropout rate for GRU layers
        device: Device to run on ('cuda' or 'cpu')
        strideLen: Stride length for unfold operation
        kernelLen: Kernel length for unfold operation
        gaussianSmoothWidth: Width parameter for Gaussian smoothing
        bidirectional: Whether to use bidirectional GRU
        post_gru_num_layers: Number of post-GRU Linear-LayerNorm-Dropout blocks (0 = disabled)
        post_gru_dropout: Dropout rate for post-GRU stack
        use_day_specific_input: If True, use day-specific linear + softsign (default: True)
        use_input_layernorm: If True and use_day_specific_input=False, use LayerNorm before GRU
    """
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        post_gru_num_layers=0,
        post_gru_dropout=0.2,
        use_day_specific_input=True,
        use_input_layernorm=False,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.post_gru_num_layers = post_gru_num_layers
        self.post_gru_dropout = post_gru_dropout
        self.use_day_specific_input = use_day_specific_input
        self.use_input_layernorm = use_input_layernorm
        
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        
        # Day-specific input processing (default behavior)
        if self.use_day_specific_input:
            self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
            self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
            for x in range(nDays):
                self.dayWeights.data[x, :, :] = torch.eye(neural_dim)
        else:
            # Register dummy parameters to maintain compatibility
            self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
            self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        
        # Optional input LayerNorm (alternative to day-specific processing)
        if not self.use_day_specific_input and self.use_input_layernorm:
            # LayerNorm over the feature dimension after unfold
            # Input to GRU after unfold has shape [batch, time, features * kernelLen]
            self.input_layernorm = nn.LayerNorm(neural_dim * self.kernelLen)
        else:
            self.input_layernorm = None

        # GRU layers
        self.gru_decoder = nn.GRU(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers (legacy, kept for backward compatibility)
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # Post-GRU stack: Linear -> GELU -> LayerNorm -> Dropout (configurable)
        # This stack processes GRU outputs before final classification
        # Uses residual connections to preserve information flow
        # GELU activation added for better non-linear transformations
        if self.post_gru_num_layers > 0:
            gru_output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
            self.post_gru_blocks = nn.ModuleList()
            
            for i in range(self.post_gru_num_layers):
                # Each block: Linear -> GELU -> LayerNorm -> Dropout
                # Will use residual connection in forward pass
                block = nn.ModuleDict({
                    'linear': nn.Linear(gru_output_dim, gru_output_dim),
                    'activation': nn.GELU(),  # Added: non-linearity for better transformations
                    'layernorm': nn.LayerNorm(gru_output_dim),
                    'dropout': nn.Dropout(self.post_gru_dropout)
                })
                # Initialize Linear layer with slightly larger gain since we have activation
                nn.init.orthogonal_(block['linear'].weight, gain=0.2)
                nn.init.zeros_(block['linear'].bias)
                self.post_gru_blocks.append(block)
            
            self.post_gru_stack = True  # Flag to indicate stack exists
        else:
            self.post_gru_stack = None
            self.post_gru_blocks = None

        # Final classification layer
        # Input dimension depends on whether post-GRU stack is used
        if self.bidirectional:
            fc_input_dim = hidden_dim * 2
        else:
            fc_input_dim = hidden_dim
            
        self.fc_decoder_out = nn.Linear(fc_input_dim, n_classes + 1)  # +1 for CTC blank

    def forward(self, neuralInput, dayIdx):
        """
        Forward pass through the GRU decoder.
        
        Args:
            neuralInput: Input tensor of shape [batch, time, features]
            dayIdx: Day indices for each sample in batch, shape [batch]
            
        Returns:
            seq_out: Logits tensor of shape [batch, time_out, n_classes + 1]
                    where time_out = (time - kernelLen) / strideLen + 1
        """
        # Gaussian smoothing
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # Input preprocessing: day-specific linear + softsign (default) or LayerNorm
        if self.use_day_specific_input:
            # Default behavior: day-specific linear transformation + softsign
            dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
            transformedNeural = torch.einsum(
                "btd,bdk->btk", neuralInput, dayWeights
            ) + torch.index_select(self.dayBias, 0, dayIdx)
            transformedNeural = self.inputLayerNonlinearity(transformedNeural)
        else:
            # Alternative: pass through unchanged (or will apply LayerNorm after unfold)
            transformedNeural = neuralInput

        # Stride/kernel windowing (unfold operation)
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )
        
        # Optional input LayerNorm (applied after unfold, before GRU)
        if self.input_layernorm is not None:
            stridedInputs = self.input_layernorm(stridedInputs)

        # GRU layers
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # Post-GRU stack: Linear -> GELU -> LayerNorm -> Dropout with residual connections
        # Residual connections help preserve GRU representations and improve training
        if self.post_gru_stack is not None:
            residual = hid
            for block in self.post_gru_blocks:
                # Apply block: Linear -> GELU -> LayerNorm -> Dropout
                out = block['linear'](residual)
                out = block['activation'](out)  # GELU activation
                out = block['layernorm'](out)
                out = block['dropout'](out)
                # Residual connection: helps preserve information and stabilize training
                residual = residual + out
            hid = residual

        # Final classification layer
        seq_out = self.fc_decoder_out(hid)
        return seq_out
