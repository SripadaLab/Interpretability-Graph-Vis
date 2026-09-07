"""Cross-layer transcoder (CLT) architecture — Anthropic circuit tracing.

Reference: https://transformer-circuits.pub/2025/attribution-graphs/methods.html#building-architecture

A CLT replaces MLP computations with sparsely active features that read from
one residual-stream layer and write to all subsequent MLP outputs:

  a^ℓ = JumpReLU(W_enc^ℓ  x^ℓ)
  ŷ^ℓ = Σ_{ℓ'=1..ℓ}  W_dec^{ℓ'→ℓ}  a^{ℓ'}

Training minimizes reconstruction MSE + a tanh sparsity penalty on decoder
norms × activations. The *replacement model* swaps MLP outputs for ŷ; the
*local* replacement model additionally freezes attention patterns and
normalization denominators from the underlying forward pass so feature–feature
edges become linear attributions.

This module documents the math and exposes thin helpers. Production Gemma-2B
weights are loaded through circuit-tracer (pretrained PLTs/CLTs on HuggingFace);
training a full CLT from scratch is expensive (~210 H100-hours for 2M features).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CLTArchitectureSpec:
    """Architectural hyperparameters matching the paper's CLT definition."""

    n_layers: int
    d_model: int
    d_features_per_layer: int
    activation: str = "jump_relu"
    # Residual read / MLP write hook names (TransformerLens / Gemma-2)
    feature_input_hook: str = "hook_resid_mid"
    feature_output_hook: str = "hook_mlp_out"

    @property
    def total_features(self) -> int:
        return self.n_layers * self.d_features_per_layer


# Gemma-2-2B has 26 layers; published CLTs: 426K and 2.5M total features.
GEMMA_2_2B_CLT_426K = CLTArchitectureSpec(
    n_layers=26,
    d_model=2304,
    d_features_per_layer=426_000 // 26,  # ≈16.4k; actual packs vary by release
)

GEMMA_2_2B_CLT_2_5M = CLTArchitectureSpec(
    n_layers=26,
    d_model=2304,
    d_features_per_layer=2_500_000 // 26,
)

# Convenience HuggingFace identifiers used by circuit-tracer.
GEMMA_PRESETS = {
    "gemma": {
        "model": "google/gemma-2-2b",
        "transcoders": "gemma",  # PLT shortcut → mntss/gemma-scope-transcoders
        "description": "Gemma-2-2B + GemmaScope per-layer transcoders",
    },
    "gemma-clt-426k": {
        "model": "google/gemma-2-2b",
        "transcoders": "mntss/clt-gemma-2-2b-426k",
        "description": "Gemma-2-2B + 426K cross-layer transcoder",
    },
    "gemma-clt-2.5m": {
        "model": "google/gemma-2-2b",
        "transcoders": "mntss/clt-gemma-2-2b-2.5M",
        "description": "Gemma-2-2B + 2.5M cross-layer transcoder",
    },
}


def describe_forward() -> str:
    return (
        "CLT forward (per layer ℓ):\n"
        "  1. Encode residual x^ℓ → features a^ℓ via JumpReLU encoder.\n"
        "  2. Reconstruct MLP output ŷ^ℓ from all features in layers 1..ℓ.\n"
        "  3. In the replacement model, overwrite the MLP output with ŷ^ℓ.\n"
        "  4. Attribution edges use stop-grad Jacobians through frozen attn/LN.\n"
    )
