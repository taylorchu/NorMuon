# NorMuon

Official implementation of **NorMuon: Making Muon more efficient and scalable** ([arXiv:2510.05491](https://arxiv.org/abs/2510.05491)).

> 🎉 **Accepted as a Spotlight at ICML 2026.**

NorMuon augments [Muon](https://github.com/KellerJordan/Muon) with a per-row second-moment normalizer (similar in spirit to Adam's second moment) applied to the orthogonalized update. The normalizer is rescaled so that the overall update norm matches Muon's, giving better-conditioned per-neuron step sizes without changing the effective learning rate.

NorMuon is also used in [karpathy/nanochat](https://github.com/karpathy/nanochat). For a fully distributed FSDP-style implementation, see our PR to the [Dion](https://github.com/microsoft/dion/pull/19) codebase. For a `modded-nanogpt` integration, see [this PR](https://github.com/KellerJordan/modded-nanogpt/pull/141).

## Installation

```
pip install git+https://github.com/zichongli5/NorMuon.git
```

`normuon.py` has no dependencies beyond PyTorch, so you can also just drop it into your project.

## What's in `normuon.py`

- `NorMuon` / `SingleDeviceNorMuon` — distributed (DDP) and single-GPU optimizers.
- `NorMuonWithAuxAdam` / `SingleDeviceNorMuonWithAuxAdam` — bundle NorMuon for hidden weights with AdamW for the rest. **Recommended.**

## Usage

Like Muon, NorMuon is meant for the hidden 2D weight matrices of the network. Embeddings, the classifier head, and gains/biases should be optimized with AdamW. The `WithAuxAdam` variants take care of routing for you:

```python
from normuon import NorMuonWithAuxAdam

hidden_weights      = [p for p in model.body.parameters() if p.ndim >= 2]
hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
nonhidden_params    = [*model.head.parameters(), *model.embed.parameters()]

param_groups = [
    dict(params=hidden_weights, use_muon=True,
         lr=0.02, momentum=0.95, beta2=0.95, weight_decay=0.01),
    dict(params=hidden_gains_biases + nonhidden_params, use_muon=False,
         lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
]
optimizer = NorMuonWithAuxAdam(param_groups)
```

The defaults (`lr=0.02`, `momentum=0.95`, `beta2=0.95`) match Muon's; in our experiments only `lr` and `weight_decay` typically need tuning, and the same values that work for Muon are a good starting point.

## Per-head Muon

Applied to a whole attention projection, Newton-Schulz couples every head through a single polar
factor. Per-head Muon, from the [Kimi K3 technical report](https://arxiv.org/abs/2607.24653) (§2.5),
instead partitions the momentum along the head dimension and orthogonalizes each head's block
separately:

> instead of applying Newton–Schulz orthogonalization to the full Q, K, and V projection matrices, we
> partition their momentum matrices along the head dimension and orthogonalize each head's block
> separately \[…\] full-matrix orthogonalization treats all heads as a single coupled block, so heads
> with larger gradient or momentum scales dominate the shared update direction, while smaller-scale
> heads receive insufficiently normalized updates; per-head orthogonalization equalizes the update
> scale across heads.

[GLM-5](https://arxiv.org/abs/2602.15763) does the same thing under the name "Muon Split", splitting
the MLA up-projections `W^UQ`, `W^UK`, `W^UV` — the latent down-projections have no head structure and
must be left fused. It describes the effect as letting "projection weights for different attention
heads to update at different scales", where K3 says per-head orthogonalization "equalizes the update
scale across heads"; these are the same mechanism seen from either end, since dropping the shared
spectral normalization is what decouples each head's scale and what makes the resulting per-head norms
come out equal. GLM-5 also reports that Muon Split alone kept attention logits stable through
pre-training with no clipping strategy. In NorMuon the second-moment normalizer follows the same split
and becomes per-head too.

One caveat specific to NorMuon. The imbalance the report describes is an artifact of *approximate*
Newton-Schulz: the spectral pre-normalization is computed over the whole matrix, so a head whose
gradient is orders of magnitude smaller keeps singular values far from 1 and comes out
under-orthogonalized. With an exact polar factor the effect vanishes entirely. Giving eight heads
gradients spanning `1e-4` to `1e3`, plain Muon's per-head update norms spread by a factor of 1.6e5,
and per-head splitting flattens that to 1.008 — but NorMuon's per-row second-moment normalizer
already flattens it to 1.002 on its own, because per-row normalization subsumes per-head for the
`"row"` axis. So the report's headline motivation is largely already covered here. What per-head
splitting still changes is the direction of the update, since each head gets its own polar factor
rather than sharing one; expect it to matter less on top of NorMuon than on top of plain Muon.

**Single-device only for now** — see [below](#why-the-distributed-optimizers-do-not-support-this).

Set `num_heads` on a muon param group, plus `head_axis` to say which side of the matrix the heads
partition — `"row"` (the default) for the query/key/value projections, whose `out_features` is
`num_heads * head_dim`:

```python
from normuon import SingleDeviceNorMuonWithAuxAdam

num_heads = model.config.num_attention_heads
param_groups = [
    dict(params=[*q_proj_weights, *k_proj_weights, *v_proj_weights], use_muon=True,
         lr=0.02, num_heads=num_heads, head_axis="row"),
    dict(params=other_hidden_weights, use_muon=True, lr=0.02),
    dict(params=gains_biases + nonhidden_params, use_muon=False, lr=3e-4),
]
optimizer = SingleDeviceNorMuonWithAuxAdam(param_groups)
```

Group by head count, not just by role: under GQA the key/value projections have `num_key_value_heads`
heads and need their own group. Leave `num_heads` unset (the default `None`) for the MLP and any other
weight without a head structure. `SingleDeviceNorMuon` takes `num_heads` / `head_axis` as constructor
arguments, which apply to every parameter it is given.

Each block is rescaled by Muon's `max(1, rows/cols)**0.5` using the block's own shape, so a row split
leaves the total update norm at exactly `sqrt(out_features)` — the same as the fused version — and
`lr` carries over without retuning. This holds for the tall MLA up-projections too, where
`num_heads * head_dim` far exceeds the latent width. That identity assumes exact orthogonalization;
5-step Newton-Schulz only approximates it, and does so a little differently on a head block than on
the full matrix, so the measured norm runs a few percent under the fused one on square projections and
up to roughly ten percent under on very tall ones.

### Why the distributed optimizers do not support this

`NorMuon` and `NorMuonWithAuxAdam` are deliberately left untouched and reject `num_heads` — the
constructor with a `TypeError`, the param group with an `AssertionError`. Only the two
`SingleDevice*` classes take it.

Per-head splitting itself is device-agnostic: it lives entirely in `normuon_update`, and the update
body of each distributed class is identical to its single-device counterpart, so a verified
single-device implementation would carry over unchanged. What we could not verify is the surrounding
shard-and-gather code, and that code has failure modes a single-rank test cannot reach:

- `dist.all_gather` wants equally-sized tensors, but each call passes a window of `world_size`
  consecutive params and `params_pad` pads with `torch.empty_like(params[-1])`. Sorting by size
  clusters like shapes without aligning those clusters to window boundaries. At `world_size=1` every
  window holds one tensor, so the case never arises.
- Optimizer state is only created for the params a rank owns, so `state_dict()` from a single rank
  covers `1/world_size` of the momentum and second-moment buffers. At `world_size=1` one rank owns
  everything.
- When `len(params) % world_size != 0` a rank contributes an uninitialized pad tensor to the gather.

None of that is specific to per-head Muon, and none of it is a reason to trust an unrun code path.
Wiring the two distributed classes up is a mechanical change once there is a multi-rank environment
to check it in: thread `num_heads` / `head_axis` from the group into `second_momentum_buffer` and
`normuon_update`, exactly as the single-device classes do. Read `head_config` before you do — group
keys added this way are dropped by `load_state_dict` when resuming an older checkpoint.

### Splitting the output projection

`head_axis="col"` splits along `in_features` instead, for the output projection, whose columns are
grouped by head. This is an extension, not part of the report: K3 and GLM-5 both split Q/K/V only.
Because each head's block is then its own `head_dim`-wide layer, the RMS rescale gives a total update
norm `sqrt(num_heads)` times the fused one, so it needs a correspondingly lower `lr` for that group.

## Citation

```bibtex
@misc{li2025normuon,
  title         = {NorMuon: Making Muon more efficient and scalable},
  author        = {Li, Zichong and Liu, Liming and Liang, Chen and Chen, Weizhu and Zhao, Tuo},
  year          = {2025},
  eprint        = {2510.05491},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```

## Acknowledgements

Built directly on [Keller Jordan's Muon](https://github.com/KellerJordan/Muon); the Newton–Schulz iteration and the distributed update-sharding pattern are taken from there with minimal changes.
