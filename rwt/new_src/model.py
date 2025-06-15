"""
Transformer-based Classifier for Random Walk Text Classification.

Optimized with FlashAttention for better performance.
"""

import torch
import torch.nn as nn
from typing import Optional

from logger import get_logger

logger = get_logger(__name__)

# Try to use FlashAttention for better performance
use_flash_attn = True
try:
    from flash_attn.modules.mha import FlashSelfAttention
    logger.info("✅ Using FlashAttention for self-attention layers")
except ImportError:
    logger.warning("⚠️ FlashAttention not available, using standard attention")
    use_flash_attn = False


class FlashTransformerSelfAttention(nn.Module):
    """Adapter to match nn.MultiheadAttention API to FlashSelfAttention."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.batch_first = True
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        # Required attributes for PyTorch TransformerEncoder compatibility
        self._qkv_same_embed_dim = True
        self.in_proj_weight = None  # Not used by FlashAttention
        self.in_proj_bias = None
        
        # Projection layers
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # FlashSelfAttention instance
        self.flash_attn = FlashSelfAttention(
            causal=False,
            attention_dropout=dropout
        )
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass compatible with nn.MultiheadAttention."""
        B, S, E = query.size()
        
        # Project to packed qkv: (B, S, 3, H, D)
        qkv = self.qkv_proj(query).view(B, S, 3, self.num_heads, self.head_dim)
        
        # FlashAttention expects float16
        orig_dtype = qkv.dtype
        qkv = qkv.to(torch.float16)
        
        # Apply FlashSelfAttention
        attn_output = self.flash_attn(qkv, causal=False).to(orig_dtype)
        
        # Reshape (B, S, H, D) -> (B, S, E)
        attn_output = attn_output.view(B, S, E)
        
        # Final projection
        attn_output = self.out_proj(attn_output)
        
        # Return tuple to match nn.MultiheadAttention API
        return attn_output, None


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len, 1, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encodings to input embeddings.
        
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            
        Returns:
            x + positional encodings
        """
        seq_len = x.size(1)
        pos_enc = self.pe[:seq_len].transpose(0, 1)  # (1, seq_len, d_model)
        return x + pos_enc


class Classifier(nn.Module):
    """
    Transformer-based classifier for random walk text classification.
    
    Features:
    - Sinusoidal positional encoding
    - Optional FlashAttention for better performance
    - Multiple aggregation methods (mean, max, first, last, cls)
    - Support for contrastive learning via embeddings output
    """
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        max_len: int = 512,
        agg_method: str = "mean",
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 7,
        vocab_size: int = 30522,
        layer_norm_eps: float = 1e-5,
    ):
        """
        Initialize the classifier.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer layers
            max_len: Maximum sequence length
            agg_method: Aggregation method (mean, max, first, last, cls)
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            num_classes: Number of output classes
            vocab_size: Vocabulary size
            layer_norm_eps: Layer norm epsilon
        """
        super().__init__()
        
        self.d_model = d_model
        self.agg_method = agg_method
        self.num_classes = num_classes
        
        # Input embedding and positional encoding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Create transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=True,
        )
        
        # Replace attention with FlashAttention if available
        if use_flash_attn:
            encoder_layer.self_attn = FlashTransformerSelfAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout
            )
        
        # Create transformer encoder
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_encoder_layers
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _aggregate(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate sequence representations.
        
        Args:
            x: Sequence representations (batch_size, seq_len, d_model)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Aggregated representation (batch_size, d_model)
        """
        if self.agg_method == "mean":
            if attention_mask is not None:
                # Masked mean pooling
                mask = attention_mask.unsqueeze(-1).expand_as(x).float()
                x_masked = x * mask
                seq_lengths = mask.sum(dim=1).clamp(min=1e-9)
                return x_masked.sum(dim=1) / seq_lengths
            else:
                return x.mean(dim=1)
                
        elif self.agg_method == "max":
            if attention_mask is not None:
                # Masked max pooling
                mask = attention_mask.unsqueeze(-1).expand_as(x).float()
                x_masked = x * mask + (1 - mask) * (-1e9)
                return x_masked.max(dim=1)[0]
            else:
                return x.max(dim=1)[0]
                
        elif self.agg_method in ["first", "cls"]:
            return x[:, 0]
            
        elif self.agg_method == "last":
            if attention_mask is not None:
                # Get last valid token
                lengths = attention_mask.sum(dim=1) - 1
                batch_indices = torch.arange(x.size(0), device=x.device)
                return x[batch_indices, lengths]
            else:
                return x[:, -1]
                
        else:
            raise ValueError(f"Unknown aggregation method: {self.agg_method}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
        **kwargs
    ):
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            labels: Ground truth labels (batch_size,) - for loss calculation
            return_embeddings: Whether to return embeddings for contrastive learning
            
        Returns:
            Dictionary containing:
            - logits: Classification logits (batch_size, num_classes)
            - embeddings: Aggregated embeddings (if return_embeddings=True)
            - loss: CrossEntropy loss (if labels provided)
        """
        # Input embedding + positional encoding
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        
        # Create key padding mask for transformer
        # True = ignore, False = attend
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        
        # Apply transformer encoder
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # Aggregate sequence representation
        pooled_output = self._aggregate(x, attention_mask)
        
        # Classification
        logits = self.classifier(self.dropout(pooled_output))
        
        # Prepare output
        output = {"logits": logits}
        
        # Add embeddings if requested (for contrastive learning)
        if return_embeddings:
            output["embeddings"] = pooled_output
        
        # Calculate loss if labels provided
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            output["loss"] = loss_fn(logits, labels)
        
        return output
    
    def get_num_params(self) -> dict:
        """Get number of parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": total_params - trainable_params
        }