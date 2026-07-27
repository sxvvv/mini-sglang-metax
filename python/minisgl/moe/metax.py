"""MetaX pure-PyTorch MoE backend.

Replaces the NVIDIA sgl_kernel dependencies (topk_softmax,
moe_align_block_size) in FusedMoe with vanilla PyTorch ops so that the
MoE forward pass runs on MACA without any external CUDA kernels.

Weight layout (same as FusedMoe / SGLang convention):
    w1  [E, 2*N_int, K]   gate+up projection fused  (N_int = intermediate_per_rank)
    w2  [E, K,       N_int]  down projection

Forward data-flow:
    1. topk routing    : gating_output [T, E] -> topk_weights [T, k], topk_ids [T, k]
    2. for each expert e:
           gather tokens routed to e
           gate_up   = x @ w1[e].T               # [t, 2*N_int]
           gate, up  = gate_up.chunk(2, dim=-1)   # [t, N_int] each
           x_mid     = silu(gate) * up            # [t, N_int]
           x_out     = x_mid @ w2[e].T            # [t, K]
           scatter-add weighted x_out back to output buffer
    3. return output [T, K]

Performance note: this implementation loops over experts in Python; it is
intentionally a correctness prototype (M1).  A batched-einsum variant or
MACA custom kernel can be added later.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from minisgl.moe import BaseMoeBackend

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _pt_topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch replacement for sgl_kernel.topk_softmax.

    Args:
        gating_output: [T, E] raw router logits.
        topk:          number of experts per token.
        renormalize:   if True, re-normalize selected weights to sum=1.

    Returns:
        topk_weights [T, topk] float32
        topk_ids     [T, topk] int32
    """
    scores = torch.softmax(gating_output.float(), dim=-1)          # [T, E]
    topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)      # [T, k]
    if renormalize:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    return topk_weights, topk_ids.to(torch.int32)


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "silu":
        return F.silu(x)
    if activation == "gelu":
        return F.gelu(x)
    raise ValueError(f"Unsupported MoE activation: {activation!r}")


# ---------------------------------------------------------------------------
# MetaxMoe backend
# ---------------------------------------------------------------------------

class MetaxMoe(BaseMoeBackend):
    """MACA-compatible MoE backend using pure PyTorch grouped matmul.

    Drop-in replacement for FusedMoe on MetaX hardware where sgl_kernel
    NVIDIA binaries are unavailable.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:             [T, K]
            w1:                        [E, 2*N_int, K]
            w2:                        [E, K, N_int]
            gating_output:             [T, E]
            topk:                      experts per token
            renormalize:               re-normalize routing weights
            activation:                "silu" | "gelu"
            apply_router_weight_on_input:
                                       True  → weight hidden before expert compute
                                       False → weight expert output (standard)

        Returns:
            output: [T, K]  same shape / dtype as hidden_states
        """
        T, K = hidden_states.shape
        E = w1.shape[0]
        N2 = w1.shape[1]            # 2 * N_int
        N_int = N2 // 2             # intermediate_per_rank

        # ── 1. routing ───────────────────────────────────────────────────────
        topk_weights, topk_ids = _pt_topk_softmax(gating_output, topk, renormalize)
        # topk_weights: [T, k] float32  (always float32 for precision)
        # topk_ids:     [T, k] int32

        # ── 2. expert compute ─────────────────────────────────────────────────
        output = torch.zeros(T, K, dtype=hidden_states.dtype, device=hidden_states.device)

        for e in range(E):
            # Collect all (token, slot) pairs routed to expert e
            # slot_mask: [T, k] bool
            slot_mask = (topk_ids == e)           # [T, topk] bool
            token_mask = slot_mask.any(dim=-1)    # [T] bool
            if not token_mask.any():
                continue

            x = hidden_states[token_mask]         # [t, K]

            # Optional: scale input by routing weight before expert compute
            if apply_router_weight_on_input:
                # sum weights across slots for the same expert (usually 1 slot per token)
                w_in = (slot_mask[token_mask].float() * topk_weights[token_mask]).sum(dim=-1)
                x = x * w_in.to(x.dtype).unsqueeze(-1)

            # Gate + up projection  →  [t, 2*N_int]
            # w1[e]: [2*N_int, K],  x: [t, K]
            gate_up = x @ w1[e].T                 # [t, 2*N_int]
            gate, up = gate_up.chunk(2, dim=-1)   # [t, N_int] each

            # Activation (SwiGLU / GeGLU)
            x_mid = _apply_activation(gate, activation) * up  # [t, N_int]

            # Down projection  →  [t, K]
            # w2[e]: [K, N_int]
            x_out = x_mid @ w2[e].T               # [t, K]

            # Scale output by routing weight (standard path)
            if not apply_router_weight_on_input:
                # Each selected token may match expert e in multiple slots
                # (unusual with topk, but handle correctly)
                w_out = (slot_mask[token_mask].float() * topk_weights[token_mask]).sum(dim=-1)
                x_out = x_out * w_out.to(x_out.dtype).unsqueeze(-1)

            # Scatter-add back to output buffer
            output[token_mask] += x_out

        return output
