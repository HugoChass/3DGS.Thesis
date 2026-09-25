# Semantic Supervision and Representation Design in 3D Gaussian Splatting for Urban Scene Understanding

Code for my MSc Robotics thesis at TU Delft (Cognitive Robotics, 2026).

**Thesis:** [TU Delft Repository](https://repository.tudelft.nl/record/uuid:29cbbd42-f383-4b67-893d-d6650e14b352) · [PDF](https://repository.tudelft.nl/file/File_e6fd087a-f8b8-495e-b4e3-885e1dcbb798?preview=1)
**Author:** Hugo E. Chassagnette
**Supervisor:** Dr. Holger Caesar · **Committee:** Michael Weinmann

---

## Overview

This repository is a modified clone of [DriveStudio](https://github.com/ziyc/drivestudio), using its Street Gaussians implementation within the OmniRe urban reconstruction pipeline. The original framework optimises 3D Gaussians for RGB reconstruction only. This work makes the Gaussians **semantically aware**: semantic labels from the nuScenes LiDAR data are attached to the Gaussian primitives, and the optimisation is extended so that semantic supervision takes part in the reconstruction itself.

The research question is:

> How do different semantic-aware training strategies and representation organisations influence semantic and RGB rendering quality in 3D Gaussian Splatting?

> **Note:** all modifications were made directly inside the cloned DriveStudio codebase, so there is no clean, isolated diff against upstream. Refer to the thesis (Section 3 and Appendix C) for the exact formulation of everything described below.

## What was changed

### 0. Semantic LiDAR labels

nuScenes only provides LiDAR semantic labels (lidarseg) on keyframes, at 2 Hz. To get labelled LiDAR for every frame, the keyframe labels are propagated to the intermediate sweeps with a KD-tree: each point in an unlabelled sweep takes the class of its nearest neighbour in the closest labelled keyframe. These labelled points provide the semantic ground truth used to initialise and supervise the semantic Gaussians.

### 1. Semantic Gaussian primitives (Thesis §3.2)

Each Gaussian is extended with per-class semantic logits `s ∈ R^K` alongside its usual mean, covariance, opacity and colour. The rasteriser renders both an RGB image and a semantic map, which is supervised against the ground-truth semantic labels.

Two ways of organising this are implemented:

| Representation | Description | Trade-off |
|---|---|---|
| **Joint Gaussians** (§3.2.1) | RGB and semantics live on the *same* Gaussians; both losses update geometry and appearance. | Compact, slightly better semantics, but RGB quality degrades due to task interference. |
| **Separated Gaussians** (§3.2.2) | Two independent Gaussian sets (`G_rgb`, `G_sem`), each optimised only by its own loss. | RGB fidelity is preserved; semantics slightly lower; slightly slower. |

### 2. Semantic-aware training objective (Thesis §3.3–3.4, Appendix C)

The training loss is extended to:

```
L_total = L_rgb + λ_main(t)·L_main^w + λ_clip·L_clip + λ_reg·L_reg + λ_depth·L_depth
```

Implemented and evaluated components:

| Component | Options | Thesis |
|---|---|---|
| Main semantic loss | Cross-entropy, Focal loss | §3.4.1, C.1 |
| Regularisation | Entropy, L2 on semantic logits | §3.4.1, C.2 |
| CLIP feature alignment | Cosine alignment with CLIP image embeddings | §3.4.1, C.3 |
| Depth supervision | LiDAR depth loss extended to the semantic Gaussian set (separated method) | §3.4.2, C.4 |
| Class weighting | Manual foreground/background weights, inverse class frequency | §3.4.3, C.5 |
| Boundary-aware weighting | Up-weights pixels near semantic class edges | §3.4.3, C.6 |
| Warm-up schedule | Linear ramp-up of the semantic loss weight (joint method) | §3.4.3, C.7 |

### 3. Semantic-aware densification (Thesis §3.4.4, C.8)

Adaptive density control is extended with a per-Gaussian **semantic importance score** combining predicted class, classification confidence and inverse class frequency. The score can drive culling, splitting and duplication of Gaussians, individually or combined.

### 4. Semantic Gaussian capacity (Thesis §4.4)

For the separated method, the number of initialised semantic Gaussians is configurable, to study how much capacity the (simpler) semantic map actually needs.

## Key findings

- **The choice of main semantic loss dominates.** Focal loss consistently beats cross-entropy and was used as the base for all later experiments.
- **Auxiliary strategies bring little.** Regularisation, CLIP alignment, class/boundary weighting and semantic densification gave negligible or negative effects overall.
- **Warm-up hurts semantics** in the joint method: RGB improves, but semantic supervision starts too late to recover.
- **Depth supervision matters for separated Gaussians**, since the semantic set lacks the structural guidance normally given by RGB gradients.
- **Semantic Gaussians can be much sparser:** initial count reduced from 800k to 100k without semantic degradation, with a small speed-up.
- **Joint vs separated is a real trade-off.** Joint gives sharper boundaries and marginally higher mIoU at a clear RGB cost; separated keeps RGB quality with near-equal semantics and is the more balanced option.

## Results

Averages over 10 nuScenes scenes (every 10th frame held out for testing), best configuration per method (Thesis Table 10):

| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ | Test mIoU ↑ | Novel-view mIoU ↑ | Train time (h) |
|---|---|---|---|---|---|---|
| Vanilla (StreetGS, RGB only) | 0.806 | 27.46 | 0.242 | – | – | 0.85 |
| Vanilla Semantics (logits, no supervision) | 0.797 | 26.72 | 0.240 | 0.320 | 0.250 | 1.13 |
| **Joint** (Focal + entropy reg.) | 0.773 | 25.78 | 0.266 | **0.614** | **0.351** | 1.27 |
| **Separated** (Focal + L2 reg. + depth) | **0.794** | **26.62** | **0.243** | 0.607 | 0.344 | 1.52 |

RGB metrics are on training views. Full per-component sweeps are in Appendix A of the thesis.

## Running

Installation, nuScenes preparation, training and evaluation work exactly as in upstream [DriveStudio](https://github.com/ziyc/drivestudio). The only difference is the config file you pass to training:

```bash
cd drivestudio
export PYTHONPATH=$(pwd)

python tools/train.py \
    --config_file <config.yaml> \
    --output_root $output_root \
    --project $project \
    --run_name $expname \
    dataset=nuscenes/6cams \
    data.scene_idx=$scene_idx \
    data.start_timestep=0 \
    data.end_timestep=-1
```

### Best configurations (Thesis Table 10)

| Method | Config |
|---|---|
| Joint Gaussians | [`configs/semantictest/entropy/streetgsSemantic_entropy_0.000008_Focal_0,0016.yaml`](drivestudio/configs/semantictest/entropy/streetgsSemantic_entropy_0.000008_Focal_0%2C0016.yaml) |
| Separated Gaussians | [`configs/semantictest/multidepth/MultigsSemantic_depth_0,008_l2_0.00004_Focal_0,0016.yaml`](drivestudio/configs/semantictest/multidepth/MultigsSemantic_depth_0%2C008_l2_0.00004_Focal_0%2C0016.yaml) |

Every other experiment from the thesis has its own config under `configs/semantictest/` (one folder per component; folders prefixed `multi` are the separated-Gaussian runs).

### What the semantic configs add to `streetgs.yaml`

| Key | Purpose |
|---|---|
| `nbr_pass` | `1` = joint Gaussians (RGB and semantics rendered in one pass from the same Gaussians), `2` = separated Gaussians (second render pass for the semantic set). |
| `semantics.use_ce` / `use_focal` + `loss_ce_w` / `loss_focal_w` | Main semantic loss and its weight. |
| `semantics.use_reg`, `reg_type` (`entropy` / `l2`), `loss_reg_w` | Semantic logit regularisation. |
| `semantics.use_contrastive`, `loss_contrastive_w` | CLIP feature alignment. |
| `semantics.use_depth`, `loss_depth_w` | LiDAR depth supervision on the semantic Gaussian set (separated method). |
| `semantics.class_weight_mode` (`manual` / `inv_freq` / `null`), `class_weights` | Class-based loss weighting (18 classes). |
| `semantics.boundary` | Boundary-aware weighting. |
| `semantics.warmup_start`, `full_weight_step` | Semantic loss warm-up schedule (`0`, `0` = off). |
| `use_semantic_cull` / `split` / `dup`, `semantic_*_importance_thresh`, `semantic_conf_power` | Semantic-importance densification. |
| `SemanticBackground`, `SemanticRigidNodes` | Separate Gaussian model for the semantic set (separated method only). |

## Citation

```bibtex
@mastersthesis{chassagnette2026semantic3dgs,
  title  = {Semantic Supervision and Representation Design in 3D Gaussian Splatting for Urban Scene Understanding},
  author = {Chassagnette, Hugo E.},
  school = {Delft University of Technology},
  year   = {2026},
  url    = {https://resolver.tudelft.nl/uuid:29cbbd42-f383-4b67-893d-d6650e14b352}
}
```

## Acknowledgements

Built on [DriveStudio](https://github.com/ziyc/drivestudio) / [OmniRe](https://arxiv.org/abs/2408.16760) and [Street Gaussians](https://arxiv.org/abs/2401.01339). Experiments use the [nuScenes](https://www.nuscenes.org/) dataset (CC BY-SA 4.0). Please also cite the upstream works and respect their licenses.
