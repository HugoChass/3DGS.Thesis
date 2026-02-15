from typing import Dict, List, Tuple
from omegaconf import OmegaConf
import os
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import kornia
from enum import IntEnum
import viser
import nerfview
from pytorch_msssim import SSIM
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from models.gaussians.basics import *

# Map indices 0..31 to distinct, readable colors (RGB in [0,1])
_PALETTE_255 = [
    (0, 0, 0),          # 0 void
    (102, 44, 22),      # 1 barrier
    (0, 191, 255),      # 2 bicycle
    (59, 59, 219),      # 3 bus
    (0, 0, 255),        # 4 vehicle
    (224, 117, 9),      # 5 construction vehicule
    (9, 224, 206),      # 6 motercycle
    (220, 20, 60),      # 7 human
    (255, 69, 0),       # 8 traffic cone
    (46, 82, 24),       # 9 trailer
    (81, 22, 102),      # 10 truck
    (50, 50, 50),       # 11 driveable_surface
    (205, 133, 63),     # 12 flat.other
    (244, 164, 96),     # 13 sidewalk
    (143, 188, 143),    # 14 terrain
    (105, 105, 105),    # 15 static_object
    (34, 139, 34),      # 16 vegetation
    (75, 117, 117),     # 17 UNLABELLED
    (0, 255, 0),        # 18 BACKGROUND
    ]
def get_semantic_palette(device=None, dtype=torch.float32):
    pal = torch.tensor(_PALETTE_255, device=device, dtype=dtype) / 255.0  # [32,3]
    return pal

logger = logging.getLogger()

class GSModelType(IntEnum):
    Background = 0
    RigidNodes = 1
    SMPLNodes = 2
    DeformableNodes = 3
    SemanticBackground = 4
    SemanticRigidNodes = 5

def lr_scheduler_fn(
    cfg: OmegaConf,
    lr_init: float
):
    if cfg.lr_final is None:
        lr_final = lr_init
    else:
        lr_final = cfg.lr_final

    def func(step):
        step = step - cfg.opt_after
        if step < 0:
            return 0.
        
        if step < cfg.warmup_steps:
            if cfg.ramp == "cosine":
                lr = cfg.lr_pre_warmup + (lr_init - cfg.lr_pre_warmup) * np.sin(
                    0.5 * np.pi * np.clip(step / cfg.warmup_steps, 0, 1)
                )
            else:
                lr = (
                    cfg.lr_pre_warmup
                    + (lr_init - cfg.lr_pre_warmup) * step / cfg.warmup_steps
                )
        else:
            t = np.clip(
                (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps), 0, 1
            )
            lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return lr  # divided by lr_init because the multiplier is with the initial learning rate

    return func

class BasicTrainer(nn.Module):
    def __init__(
        self,
        type: str = "basic",
        optim: OmegaConf = None,
        losses: OmegaConf = None,
        render: OmegaConf = None,
        res_schedule: OmegaConf = None,
        gaussian_optim_general_cfg: OmegaConf = None,
        gaussian_ctrl_general_cfg: OmegaConf = None,
        model_config: OmegaConf = None,
        num_train_images: int = 0,
        num_full_images: int = 0,
        test_set_indices: List[int] = None,
        scene_aabb: torch.Tensor = None,
        device=None,
    ):
        super().__init__()
        self._type = type
        self.optim_general = optim
        self.losses_dict = losses
        self.render_cfg = render
        self.res_schedule = res_schedule
        self.model_config = model_config
        self.num_iters = self.optim_general.get("num_iters", 30000)
        self.gaussian_optim_general_cfg = gaussian_optim_general_cfg
        self.gaussian_ctrl_general_cfg = gaussian_ctrl_general_cfg
        self.step = 0
        self.device = device
        
        self.num_semantic_classes = 18  # K
        self.semantic_feat_dim = 256        # D
        self.clip_feat_dim = 512

        # Projection head from logits (K) to semantic feature space (D)
        self.semantic_proj = nn.Linear(self.num_semantic_classes,
                                       self.semantic_feat_dim)
        

        # Teacher: CLIP feature space (D_clip) → same space as student (D_student)
        self.teacher_proj  = nn.Linear(self.clip_feat_dim,
                                       self.semantic_feat_dim)

        # dataset infos
        self.num_train_images = num_train_images
        self.num_full_images = num_full_images
        
        # init scene scale
        self._init_scene(scene_aabb=scene_aabb)
        
        # init models
        self.models = {}
        self.misc_classes_keys = [
            'Sky', 'Affine', 'CamPose', 'CamPosePerturb'
        ]
        self.gaussian_classes = {}
        self._init_models()
        self.pts_labels = None # will be overwritten in forward
        self.render_dynamic_mask = False
        
        # init losses fn
        self._init_losses()
        
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(self.device)
        self.step = 0

        # background color
        self.back_color = torch.zeros(3).to(self.device)
    
        # for evaluation
        self.cur_frame = torch.tensor(0, device=self.device)
        self.test_set_indices = test_set_indices # will be override
        
        # a simple viewer for background visualization
        self.viewer = None
    
    @property
    def in_test_set(self):
        return self.cur_frame.item() in self.test_set_indices
    
    def set_train(self):
        for model in self.models.values():
            model.train()
        self.train()
    
    def set_eval(self):
        for model in self.models.values():
            model.eval()
        self.eval()

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max((self.res_schedule.downscale_times - self.step // self.res_schedule.double_steps), 0)
        else:
            return 1
        
    def update_gaussian_cfg(self, model_cfg: OmegaConf) -> OmegaConf:
        class_optim_cfg = model_cfg.get('optim', None)
        class_ctrl_cfg = model_cfg.get('ctrl', None)
        new_optim_cfg = self.gaussian_optim_general_cfg.copy()
        new_ctrl_cfg = self.gaussian_ctrl_general_cfg.copy()
        if class_optim_cfg is not None:
            new_optim_cfg.update(class_optim_cfg)
        if class_ctrl_cfg is not None:
            new_ctrl_cfg.update(class_ctrl_cfg)
        model_cfg['optim'] = new_optim_cfg
        model_cfg['ctrl'] = new_ctrl_cfg

        return model_cfg
        
    def _init_scene(self, scene_aabb) -> None:
        self.aabb = scene_aabb.to(self.device)
        scene_origin = (self.aabb[0] + self.aabb[1]) / 2
        scene_radius = torch.max(self.aabb[1] - self.aabb[0]) / 2 * 1.1
        self.scene_radius = scene_radius.item()
        self.scene_origin = scene_origin
        logger.info(f"scene origin: {scene_origin}")
        logger.info(f"scene radius: {scene_radius}")
    
    def _init_models(self) -> None:
        raise NotImplementedError("Please implement the _init_models function")
    
    def initialize_optimizer(self) -> None:
        # get param groups first
        self.param_groups = {}
        for class_name, model in self.models.items():
            self.param_groups.update(model.get_param_groups())
                 
        groups = []
        lr_schedulers = {}
        for params_name, params in self.param_groups.items():
            class_name = params_name.split("#")[0]
            component_name = params_name.split("#")[1]
            class_cfg = self.model_config.get(class_name)
            class_optim_cfg = class_cfg["optim"]
            raw_optim_cfg = class_optim_cfg.get(component_name, None)
            lr_scale_factor = raw_optim_cfg.get("scale_factor", 1.0)
            if isinstance(lr_scale_factor, str) and lr_scale_factor == "scene_radius":
                # scale the spatial learning rate to scene scale
                lr_scale_factor = self.scene_radius

            optim_cfg = OmegaConf.create({
                "lr": raw_optim_cfg.get('lr', 0.0005),
                "eps": raw_optim_cfg.get('eps', 1.0e-15),
                "weight_decay": raw_optim_cfg.get('weight_decay', 0),
            })
            optim_cfg.lr = optim_cfg.lr * lr_scale_factor
            assert optim_cfg is not None, f"param group {params_name} not found in config"
            lr_init = optim_cfg.lr
            groups.append({
                'params': params,
                'name': params_name,
                'lr': optim_cfg.lr,
                'eps': optim_cfg.eps,
                'weight_decay': optim_cfg.weight_decay
            })
            
            if raw_optim_cfg.get("lr_final", None) is not None:
                sched_cfg = OmegaConf.create({
                    "opt_after": raw_optim_cfg.get('opt_after', 0),
                    "warmup_steps": raw_optim_cfg.get('warmup_steps', 0),
                    "max_steps": raw_optim_cfg.get('max_steps', self.num_iters),
                    "lr_pre_warmup": raw_optim_cfg.get('lr_pre_warmup', 1.0e-8),
                    "lr_final": raw_optim_cfg.get('lr_final', None),
                    "ramp": raw_optim_cfg.get('ramp', "cosine"),
                })
                # scale the learning rate according to the scene scale
                sched_cfg.lr_pre_warmup = sched_cfg.lr_pre_warmup * lr_scale_factor
                sched_cfg.lr_final = sched_cfg.lr_final * lr_scale_factor if sched_cfg.lr_final is not None else None
                # adjust max_steps to account for opt_after
                sched_cfg.max_steps = sched_cfg.max_steps - sched_cfg.opt_after
                lr_schedulers[params_name] = lr_scheduler_fn(sched_cfg, lr_init)

        self.optimizer = torch.optim.Adam(groups, lr=0.0, eps=1e-15)
        self.lr_schedulers = lr_schedulers
        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=self.optim_general.get("use_grad_scaler", False))
    
    def _init_losses(self) -> None:
        sky_opacity_loss_fn = None
        if "Sky" in self.models:
            if self.losses_dict.mask.opacity_loss_type == "bce":
                from models.losses import binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: binary_cross_entropy(pred, gt, reduction="mean")
            elif self.losses_dict.mask.opacity_loss_type == "safe_bce":
                from models.losses import safe_binary_cross_entropy
                sky_opacity_loss_fn = lambda pred, gt: safe_binary_cross_entropy(pred, gt, limit=0.1, reduction="mean")
        self.sky_opacity_loss_fn = sky_opacity_loss_fn
        
        depth_loss_fn = None
        depth_loss_cfg = self.losses_dict.get("depth", None)
        if depth_loss_cfg is not None:
            from models.losses import DepthLoss
            depth_loss_fn = DepthLoss(
                loss_type=depth_loss_cfg.loss_type,
                normalize=depth_loss_cfg.normalize,
                use_inverse_depth=depth_loss_cfg.inverse_depth,
            )
        self.depth_loss_fn = depth_loss_fn

        self.semantic_loss_cfg = self.losses_dict.get("semantics", None)
    
    def optimizer_zero_grad(self) -> None:
        self.optimizer.zero_grad()
    
    def optimizer_step(self) -> None:
        # for params_name, optimizer in self.optimizers.items():
        #     class_name = params_name.split("#")[0]
        #     component_name = params_name.split("#")[1]
        #     max_norm = self.model_config[class_name]["optim"][component_name].get("max_norm", None)
        #     if max_norm is not None:
        #         self.grad_scaler.unscale_(optimizer)
        #         torch.nn.utils.clip_grad_norm_(self.param_groups[params_name], max_norm)
        #     if any(any(p.grad is not None for p in g["params"]) for g in optimizer.param_groups):
        #         self.grad_scaler.step(optimizer)
        self.optimizer.step()

    def preprocess_per_train_step(self, step: int) -> None:
        self.step = step
        for class_name in self.gaussian_classes.keys():
            self.models[class_name].preprocess_per_train_step(step)

        # viewer
        if self.viewer is not None:
            while self.viewer.state.status == "paused":
                time.sleep(0.01)
            self.viewer.lock.acquire()
            self.tic = time.time()
        
    def postprocess_per_train_step(self, step: int) -> None:
        info_rgb = self.info["rgb"]
        info_sem = self.info["sem"]

        idx_rgb = self.info["idx_rgb"]  # LongTensor [N_rgb] global indices
        idx_sem = self.info["idx_sem"]  # LongTensor [N_sem] global indices

        # width/height might be stored in the top-level dict or inside info_rgb
        width  = self.info.get("width",  info_rgb.get("width"))
        height = self.info.get("height", info_rgb.get("height"))

        # --- grads (per-pass) ---
        if self.render_cfg.absgrad:
            grads_rgb = info_rgb["means2d"].absgrad.clone()
            if info_sem is not None:
                grads_sem = info_sem["means2d"].absgrad.clone()
        else:
            # Only valid if you called retain_grad() on the actual means2d tensors
            grads_rgb = info_rgb["means2d"].grad.clone()
            if info_sem is not None:
                grads_sem = info_sem["means2d"].grad.clone()

        # scale grads like before (keep shapes [1, N_pass, 2])
        grads_rgb[..., 0] *= width / 2.0 * self.render_cfg.batch_size
        grads_rgb[..., 1] *= height / 2.0 * self.render_cfg.batch_size

        radii_rgb = info_rgb["radii"]  # [1, N_rgb]

        if idx_rgb is not None:
            labels_rgb = self.pts_labels[idx_rgb]  # [N_rgb]
        else:
            labels_rgb = self.pts_labels
        
        if info_sem is not None:
            grads_sem[..., 0] *= width / 2.0 * self.render_cfg.batch_size
            grads_sem[..., 1] *= height / 2.0 * self.render_cfg.batch_size
            radii_sem = info_sem["radii"]  # [1, N_sem]
            labels_sem = self.pts_labels[idx_sem]  # [N_sem]

        last_size = max(width, height)

        for class_name, class_id in self.gaussian_classes.items():
            if "Semantic" not in class_name:
                # Non-semantic classes use RGB pass info
                local_mask = (labels_rgb == class_id)  # [N_rgb]
                self.models[class_name].postprocess_per_train_step(
                    step=step,
                    optimizer=self.optimizer,
                    radii=radii_rgb[0, local_mask],
                    xys_grad=grads_rgb[0, local_mask],
                    last_size=last_size,
                )
            else:
                # Semantic classes use SEM pass info
                local_mask = (labels_sem == class_id)  # [N_sem]
                self.models[class_name].postprocess_per_train_step(
                    step=step,
                    optimizer=self.optimizer,
                    radii=radii_sem[0, local_mask],
                    xys_grad=grads_sem[0, local_mask],
                    last_size=last_size,
                )

        # viewer (unchanged)
        if self.viewer is not None:
            num_train_rays_per_step = self.render_cfg.batch_size * width * height
            self.viewer.lock.release()
            num_train_steps_per_sec = 1.0 / (time.time() - self.tic)
            num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
            self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
            self.viewer.update(step, num_train_rays_per_step)

    
    def update_visibility_filter(self) -> None:
        # these must be stored from render_fn each step
        idx_rgb = self.info["idx_rgb"]   # LongTensor [N_rgb] global ids

        if idx_rgb is not None:
            labels_rgb = self.pts_labels[idx_rgb]  # [N_rgb]
        else:
            labels_rgb = self.pts_labels

        info_rgb = self.info["rgb"]

        for class_name in self.gaussian_classes.keys():
            class_id = self.gaussian_classes[class_name]

            if "Semantic" not in class_name:
                # local mask within rgb pass
                local_mask = (labels_rgb == class_id)   # [N_rgb]
                self.models[class_name].cur_radii = info_rgb["radii"][0, local_mask]

            else:
                idx_sem = self.info["idx_sem"]   # LongTensor [N_sem] global ids
                labels_sem = self.pts_labels[idx_sem]  # [N_sem]
                info_sem = self.info["sem"]
                # local mask within sem pass
                local_mask = (labels_sem == class_id)   # [N_sem]
                self.models[class_name].cur_radii = info_sem["radii"][0, local_mask]


    def process_camera(
        self,
        camera_infos: Dict[str, torch.Tensor],
        image_ids: torch.Tensor,
        novel_view: bool = False
    ) -> dataclass_camera:
        camtoworlds = camtoworlds_gt = camera_infos["camera_to_world"]
        
        if "CamPosePerturb" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPosePerturb"](camtoworlds, image_ids)

        if "CamPose" in self.models.keys() and not novel_view:
            camtoworlds = self.models["CamPose"](camtoworlds, image_ids)
        
        # collect camera information
        camera_dict = dataclass_camera(
            camtoworlds=camtoworlds,
            camtoworlds_gt=camtoworlds_gt,
            Ks=camera_infos["intrinsics"],
            H=camera_infos["height"],
            W=camera_infos["width"]
        )
        
        return camera_dict

    def collect_gaussians(
        self,
        cam: dataclass_camera,
        image_ids: torch.Tensor # leave it here for future use
    ) -> dataclass_gs:
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
            "class_labels": [],
            "_semantics": [],
            "_semantics_bool": [],
        }
        for class_name in self.gaussian_classes.keys():
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue
    
            # collect gaussians
            gs["class_labels"] = torch.full((gs["_means"].shape[0],), self.gaussian_classes[class_name], device=self.device)
            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)
            
        # get the class labels
        self.pts_labels = gs_dict.pop("class_labels")
        if self.render_dynamic_mask:
            self.dynamic_pts_mask = (self.pts_labels != 0).float()

        gaussians = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            _semantics=gs_dict["_semantics"],
            _semantics_bool=gs_dict["_semantics_bool"],
            detach_keys=[],    # if "means" in detach_keys, then the means will be detached
            extras=None        # to save some extra information (TODO) more flexible way
        )
        
        return gaussians
    
    def render_gaussians_old(
        self,
        gs: dataclass_gs,
        cam: dataclass_camera,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        
        def render_fn(opaticy_mask=None, return_info=False, override_colors=None):
            renders, alphas, info = rasterization(
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=gs.opacities.squeeze()*opaticy_mask if opaticy_mask is not None else gs.opacities.squeeze(),
                colors=override_colors if override_colors is not None else gs.rgbs,
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
                Ks=cam.Ks[None, ...],  # [C, 3, 3]
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )
            renders = renders[0]
            alphas = alphas[0].squeeze(-1)
            assert self.render_cfg.batch_size == 1, "batch size must be 1, will support batch size > 1 in the future"
            
            assert renders.shape[-1] == 4, f"Must render rgb, depth and alpha"
            rendered_rgb, rendered_depth = torch.split(renders, [3, 1], dim=-1)
            
            if not return_info:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None]
            else:
                return torch.clamp(rendered_rgb, max=1.0), rendered_depth, alphas[..., None], info
        
        # render rgb and opacity
        rgb, depth, opacity, self.info = render_fn(return_info=True)
        results = {
            "rgb_gaussians": rgb,
            "depth": depth, 
            "opacity": opacity
        }
        
        if self.training:
            self.info["means2d"].retain_grad()
        
        # render semantics
        device, dtype = gs.rgbs.device, gs.rgbs.dtype
        H, W, _ = rgb.shape
        num_classes = 17 # KNOWN CLASSES

        palette = get_semantic_palette(device=device, dtype=dtype)
        
        # total coverage/alpha for normalization (remove last dim)
        alpha_total = opacity.squeeze(-1)  # [H,W] 
        
        # class ids 0..K-1  (known classes)
        class_ids = list(range(num_classes))

        # temperature optional; you can anneal from ~1.5 → 1.0 → 0.7
        T = 1.0
        sem_probs = torch.softmax(gs.semantics / T, dim=-1)   # [N, K(+1)]

        # class masses for known classes
        class_masses = []
        for k in class_ids:
            #class_color = (gs.semantics == k).float()[:, None].expand(-1, 3)  # [N,3]
            class_color = sem_probs[:, k][:, None].expand(-1, 3)
            rgb_k, _, _ = render_fn(opaticy_mask=None, override_colors=class_color)
            class_masses.append(rgb_k[..., 0] )

        class_masses = torch.stack(class_masses, dim=-1)  # [H,W,K]

        # unknown gaussian mass (ℓ = 17)
        unknown_mask = sem_probs[:, 17]
        unk_color = unknown_mask[:, None].expand(-1, 3)
        rgb_unk, _, _ = render_fn(opaticy_mask=None, override_colors=unk_color)
        m_unknown = rgb_unk[..., 0][..., None]            # [H,W,1]

        alpha_total = opacity.squeeze(-1)                  # [H,W]

        # (optional) empty-space background
        m_bg_empty = (1.0 - alpha_total).clamp_min(0.0)[..., None]  # [H,W,1]

        # probs over K + 1 (+1) classes
        m_all = torch.cat([class_masses, m_bg_empty, m_unknown], dim=-1)  # [H,W,K+2]
        probs = m_all / (m_all.sum(dim=-1, keepdim=True) + 1e-6)          # closed simplex

        # For display-only labels (no grad):
        probs_vis = probs[..., :num_classes+2]                   
        labels = probs_vis.detach().argmax(dim=-1)   # [H,W]
        labels_safe = labels.clamp_min(0)            # map -1 to 0 for palette indexing
        labels_rgb = palette[labels_safe.view(-1)].view(H, W, 3).clone()

        # For display-only labels (no grad) no unlabelled:
        probs_vis = probs[..., :num_classes+1]                   
        labels = probs_vis.detach().argmax(dim=-1)   # [H,W]
        labels_safe = labels.clamp_min(0)            # map -1 to 0 for palette indexing
        labels_rgb_no_unlabelled = palette[labels_safe.view(-1)].view(H, W, 3).clone()

        # # Colorize the hard label map
        # safe = torch.clamp(labels, min=0)                               # map -1 -> 0 for indexing
        # palette = palette.to(device=device, dtype=dtype)
        # labels_rgb = palette[safe.view(-1)].view(H, W, 3).clone()       # [H,W,3]
        # unl_mask = (labels == -1)
        # if unl_mask.any():
        #     labels_rgb[unl_mask] = unlabeled_color
        
        results.update({
            "semantic_probs": probs,           # [H,W,K]
            "semantic_label": labels,         # [H,W] int64, -1 for background
            "semantic_rgb": labels_rgb, # [H,W,3] float in [0,1]
            "semantic_rgb_no_unlabelled": labels_rgb_no_unlabelled
        })

        return results, render_fn

    def render_gaussians(self, gs: dataclass_gs, cam: dataclass_camera, **kwargs):
        n = getattr(self.render_cfg, "nbr_pass", 1)

        # ----------------------------
        # Testing

        # def compare_arrays_percent(a, b, value_range=None, pixel_mode=False):
        #     a = np.asarray(a)
        #     b = np.asarray(b)

        #     if a.shape != b.shape:
        #         raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")

        #     # 1) exact difference %
        #     if pixel_mode and a.ndim == 3:
        #         # pixel differs if any channel differs
        #         exact_percent = 100.0 * np.mean(np.any(a != b, axis=-1))
        #     else:
        #         exact_percent = 100.0 * np.mean(a != b)

        #     # numeric difference
        #     af = a.astype(np.float32)
        #     bf = b.astype(np.float32)

        #     mae = np.mean(np.abs(af - bf))
        #     rmse = np.sqrt(np.mean((af - bf) ** 2))

        #     # pick range
        #     if value_range is None:
        #         # auto range based on data
        #         value_range = np.max([af.max(), bf.max()]) - np.min([af.min(), bf.min()])
        #         if value_range == 0:
        #             value_range = 1.0

        #     mae_percent = 100.0 * (mae / value_range)
        #     rmse_percent = 100.0 * (rmse / value_range)

        #     return {
        #         "exact_difference_%": exact_percent,
        #         "mae_%": mae_percent,
        #         "rmse_%": rmse_percent
        #     }

        # results_N1 = self.render_gaussians_one_pass(gs, cam, **kwargs)
        # results_N2 = self.render_gaussians_two_pass(gs, cam, **kwargs)

        # print(f"Compare RGB: {compare_arrays_percent(results_N1["rgb_gaussians"], results_N2["rgb_gaussians"])}") 
        # print(f"Compare Logits: {compare_arrays_percent(results_N1["semantic_logits"], results_N2["semantic_logits"])}") 
        # print(f"Compare Probs: {compare_arrays_percent(results_N1["semantic_probs"], results_N2["semantic_probs"])}") 
        # print(f"Compare Labels: {compare_arrays_percent(results_N1["semantic_label"], results_N2["semantic_label"])}") 
        # exit(0)

        if n == 1:
            return self.render_gaussians_one_pass(gs, cam, **kwargs)
        elif n == 2:
            return self.render_gaussians_two_pass(gs, cam, **kwargs)
        else:
            raise ValueError(
                f"render_cfg.nbr_pass must be 1 or 2, got {n}"
            )

    def render_gaussians_one_pass(self, gs: dataclass_gs, cam: dataclass_camera, **kwargs):

        device, dtype = gs.rgbs.device, gs.rgbs.dtype
        palette = get_semantic_palette(device=device, dtype=dtype)  # [C+1,3] incl. bg

        # Get semantic size
        assert hasattr(gs, "semantics"), "gs.semantics must exist"
        C = gs.semantics.shape[-1]  # num semantic classes (no bg)

        def render_fn(opacity_mask=None, return_info=False):
            opacities = gs.opacities.squeeze()
            if opacity_mask is not None:
                opacities = opacities * opacity_mask

            # Pack RGB + semantic logits
            sem_probs = torch.softmax(gs.semantics, dim=-1)  # [N,C]
            colors_all = torch.cat([gs.rgbs, sem_probs.to(gs.rgbs.dtype)], dim=-1)

            renders, alphas, info = rasterization(
                means=gs.means,
                quats=gs.quats,
                scales=gs.scales,
                opacities=opacities,
                colors=colors_all,  # [N, 3+C]
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],
                Ks=cam.Ks[None, ...],
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )

            img    = renders[0]               # [H, W, 3+C+1]
            alphas = alphas[0].squeeze(-1)    # [H, W]

            # Split channels: [RGB | C semantic | depth]
            rgb        = img[..., :3]
            sem_logits = img[..., 3:3 + C]        # [H, W, C]
            depth      = img[..., 3 + C:3 + C + 1]

            rgb = torch.clamp(rgb, max=1.0)

            if not return_info:
                return rgb, depth, alphas[..., None], sem_logits, None, None
            else:
                return rgb, depth, alphas[..., None], sem_logits, {"rgb": info, "sem": None, "idx_rgb": None, "idx_sem": None}

        # main call
        rgb, depth, opacity, sem_logits, self.info = render_fn(return_info=True)

        results = {
            "rgb_gaussians": rgb,
            "depth": depth,
            "opacity": opacity,
        }

        # ---- semantic post-processing ----
        # sem_logits: [H, W, C] (no bg)
        if sem_logits is not None:
            # foreground probs from logits
            fg_probs = sem_logits.float().softmax(-1)  # [H,W,C]

            # background mass ~ 1 - accumulated alpha
            alpha_total = opacity.squeeze(-1)          # [H,W]
            bg_prob = (1.0 - alpha_total).clamp_min(0.0)[..., None]  # [H,W,1]

            # combine foreground + background probs, normalize
            sem_all = torch.cat([fg_probs, bg_prob], dim=-1)  # [H,W,C+1]
            sem_all = sem_all / (sem_all.sum(dim=-1, keepdim=True) + 1e-6)

            # class indices including bg
            sem_labels = sem_all.argmax(-1, keepdim=True).to(torch.int32)  # [H,W,1]

            sem_logits = sem_all[:,:,:-1]

            results["semantic_logits"] = sem_logits          # blended logits (no bg)
            results["semantic_probs"]  = sem_all             # probs (C+1 incl. bg)
            results["semantic_label"]  = sem_labels          # [H,W,1]

            # pred_semantic_logits: [H, W, K]
            H, W, K = sem_logits.shape
            assert K == self.num_semantic_classes, "Logit dim must match num_semantic_classes"
            device = sem_logits.device

            # Make sure the projection layers are on the same device as the logits
            self.semantic_proj = self.semantic_proj.to(device)

            # Flatten spatial dims so we can apply Linear(K → D)
            logits_flat = sem_logits.view(-1, K)            # [H*W, K]

            # Project to D-dim semantic feature space (student features)
            feat_flat = self.semantic_proj(logits_flat)               # [H*W, D]

            # Reshape back to image grid
            pred_semantic_features = feat_flat.view(H, W, self.semantic_feat_dim)  # [H, W, D]
            results["semantic_features"] = pred_semantic_features     # [H, W, D]

            # ---- Palette visualization (with bg color = last palette entry) ----
            if palette is not None:
                assert palette.shape[0] == sem_all.shape[-1], \
                    f"palette rows {palette.shape[0]} must match C+1={sem_all.shape[-1]}"
                pal = palette.to(device=rgb.device, dtype=rgb.dtype)  # [C+1,3]

                idx_vis = sem_all.detach().argmax(dim=-1)   # [H,W]
                sem_vis = pal[idx_vis]                      # [H,W,3]
                results["semantic_rgb"] = sem_vis

                # ---- Visualization without 'unknown' labels ----
                # assume 'unknown' is second-to-last index, map it to background
                num_cls_plus_bg = sem_all.shape[-1]
                unknown_idx = num_cls_plus_bg - 2   # second-to-last
                bg_idx      = num_cls_plus_bg - 1   # last (background)

                idx_no_unknown = idx_vis.clone()
                idx_no_unknown[idx_no_unknown == unknown_idx] = bg_idx

                sem_vis_no_unl = pal[idx_no_unknown]        # [H,W,3]
                results["semantic_rgb_no_unlabelled"] = sem_vis_no_unl

        if self.training:
            self.info["rgb"]["means2d"].retain_grad()

        return results, render_fn

    def render_gaussians_two_pass(self, gs: dataclass_gs, cam: dataclass_camera, **kwargs):

        device, dtype = gs.rgbs.device, gs.rgbs.dtype
        palette = get_semantic_palette(device=device, dtype=dtype)  # [C+1,3] incl. bg

        # Get semantic size
        assert hasattr(gs, "semantics"), "gs.semantics must exist"
        C = gs.semantics.shape[-1]  # num semantic classes (no bg)
        def render_fn(opacity_mask=None, return_info=False):
            mask_rgb = (gs.semantics_bool == 0).all(dim=1)
            gs_rgb = gs.masked(mask_rgb)
            opacities = gs_rgb.opacities.squeeze()
            if opacity_mask is not None:
                opacity_mask = opacity_mask[mask_rgb]
                opacities = opacities * opacity_mask

            # -------------------------
            # Pass 1: RGB (+ depth)
            # -------------------------
            renders_rgb, alphas_rgb, info_rgb = rasterization(
                means=gs_rgb.means,
                quats=gs_rgb.quats,
                scales=gs_rgb.scales,
                opacities=opacities,
                colors=gs_rgb.rgbs,  # [N, 3]
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],
                Ks=cam.Ks[None, ...],
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )

            img_rgb   = renders_rgb[0]              # [H, W, 3+1] (RGB + depth)
            alpha_rgb = alphas_rgb[0].squeeze(-1)   # [H, W]

            rgb   = img_rgb[..., :3]
            depth = img_rgb[..., 3:4]               # [H, W, 1]

            rgb = torch.clamp(rgb, max=1.0)

            # -------------------------
            # Pass 2: Semantics (+ depth)
            # -------------------------
            # Render semantic logits as "colors" (features). Keep dtype aligned.

            mask_sem = (gs.semantics_bool == 1).all(dim=1)
            gs_sem = gs.masked(mask_sem)
            sem_colors = torch.softmax(gs_sem.semantics, dim=-1)  # [N,C]

            opacities = gs_sem.opacities.squeeze()

            renders_sem, alphas_sem, info_sem = rasterization(
                means=gs_sem.means,
                quats=gs_sem.quats,
                scales=gs_sem.scales,
                opacities=opacities,
                colors=sem_colors,  # [N, C]
                viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],
                Ks=cam.Ks[None, ...],
                width=cam.W,
                height=cam.H,
                packed=self.render_cfg.packed,
                absgrad=self.render_cfg.absgrad,
                sparse_grad=self.render_cfg.sparse_grad,
                rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
                **kwargs,
            )

            img_sem   = renders_sem[0]              # [H, W, C+1] (sem_logits + depth)
            alpha_sem = alphas_sem[0].squeeze(-1)   # [H, W]

            sem_logits = img_sem[..., :C]           # [H, W, C]
            sem_depth  = img_sem[..., C:C+1]      # [H, W, 1]  (optional, usually not needed)

            # For background prob, prefer RGB alpha (ties bg mass to the appearance compositing)
            opacity = alpha_rgb[..., None]          # [H, W, 1]
            sem_opacity = alpha_sem[..., None]

            idx_rgb = mask_rgb.nonzero(as_tuple=False).squeeze(1)  # [N_rgb]
            idx_sem = mask_sem.nonzero(as_tuple=False).squeeze(1)  # [N_sem]

            if return_info:
                return rgb, depth, opacity, sem_logits, sem_depth, sem_opacity, {"rgb": info_rgb, "sem": info_sem, "idx_rgb": idx_rgb, "idx_sem": idx_sem}
            else:
                return rgb, depth, opacity, sem_logits, sem_depth, sem_opacity

        # main call
        rgb, depth, opacity, sem_logits, sem_depth, sem_opacity, info = render_fn(return_info=True)
        # Keep self.info compatible with your existing code (expects "means2d" etc.)
        # Use RGB info as the primary (so gradients/retains behave as before).


        self.info = info

        results = {
            "rgb_gaussians": rgb,
            "depth": depth,
            "opacity": opacity,
        }

        # ---- semantic post-processing ----
        # sem_logits: [H, W, C] (no bg)
        if sem_logits is not None:
            # foreground probs from logits
            fg_probs = sem_logits.float().softmax(-1)  # [H,W,C]

            # background mass ~ 1 - accumulated alpha (from RGB pass)
            alpha_total = sem_opacity.squeeze(-1)          # [H,W]
            bg_prob = (1.0 - alpha_total).clamp_min(0.0)[..., None]  # [H,W,1]

            # combine foreground + background probs, normalize
            sem_all = torch.cat([fg_probs, bg_prob], dim=-1)  # [H,W,C+1]
            sem_all = sem_all / (sem_all.sum(dim=-1, keepdim=True) + 1e-6)

            # class indices including bg
            sem_labels = sem_all.argmax(-1, keepdim=True).to(torch.int32)  # [H,W,1]

            # testing
            sem_logits = sem_all[:,:,:-1]

            results["semantic_logits"] = sem_logits          # blended logits (no bg)
            results["semantic_probs"]  = sem_all             # probs (C+1 incl. bg)
            results["semantic_label"]  = sem_labels          # [H,W,1]
            results["semantic_depth"]  = sem_depth

            # pred_semantic_logits: [H, W, K]
            H, W, K = sem_logits.shape
            assert K == self.num_semantic_classes, "Logit dim must match num_semantic_classes"
            device = sem_logits.device

            # Make sure the projection layers are on the same device as the logits
            self.semantic_proj = self.semantic_proj.to(device)

            # Flatten spatial dims so we can apply Linear(K → D)
            logits_flat = sem_logits.view(-1, K)  # [H*W, K]

            # Project to D-dim semantic feature space (student features)
            feat_flat = self.semantic_proj(logits_flat)  # [H*W, D]

            # Reshape back to image grid
            pred_semantic_features = feat_flat.view(H, W, self.semantic_feat_dim)  # [H, W, D]
            results["semantic_features"] = pred_semantic_features

            # ---- Palette visualization (with bg color = last palette entry) ----
            if palette is not None:
                assert palette.shape[0] == sem_all.shape[-1], \
                    f"palette rows {palette.shape[0]} must match C+1={sem_all.shape[-1]}"
                pal = palette.to(device=rgb.device, dtype=rgb.dtype)  # [C+1,3]

                idx_vis = sem_all.detach().argmax(dim=-1)   # [H,W]
                sem_vis = pal[idx_vis]                      # [H,W,3]
                results["semantic_rgb"] = sem_vis

                # ---- Visualization without 'unknown' labels ----
                # assume 'unknown' is second-to-last index, map it to background
                num_cls_plus_bg = sem_all.shape[-1]
                unknown_idx = num_cls_plus_bg - 2   # second-to-last
                bg_idx      = num_cls_plus_bg - 1   # last (background)

                idx_no_unknown = idx_vis.clone()
                idx_no_unknown[idx_no_unknown == unknown_idx] = bg_idx

                sem_vis_no_unl = pal[idx_no_unknown]        # [H,W,3]
                results["semantic_rgb_no_unlabelled"] = sem_vis_no_unl

        if self.training:
            # Keep behavior same as before (info from RGB pass)
            self.info["rgb"]["means2d"].retain_grad()
            self.info["sem"]["means2d"].retain_grad()

        return results, render_fn


    def affine_transformation(
        self,
        rgb_blended: torch.Tensor,
        image_infos: Dict[str, torch.Tensor]
        ):
        if "Affine" in self.models:
            affine_trs = self.models['Affine'](image_infos)
            rgb_transformed = (affine_trs[..., :3, :3] @ rgb_blended[..., None] + affine_trs[..., :3, 3:])[..., 0]
            
            return rgb_transformed
        else:       
            return rgb_blended
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
            novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
        """

        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set
        
        # prapare data
        processed_cam = self.process_camera(
            camera_infos=camera_infos,
            image_ids=image_infos["img_idx"].flatten()[0],
            novel_view=novel_view
        )
        gs = self.collect_gaussians(
            cam=processed_cam,
            image_ids=image_infos["img_idx"].flatten()[0]
        )

        # render gaussians
        outputs, _ = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.)
        )
        
        # render GT semantic map
        device, dtype = gs.rgbs.device, gs.rgbs.dtype
        palette = get_semantic_palette(device=device, dtype=dtype)
        gt_semantic = image_infos["lidar_semantics_map"]
        H, W = gt_semantic.shape
        gt_semantic = torch.tensor(gt_semantic, device=device, dtype=torch.long)
        gt_semantic_map = palette[gt_semantic.view(-1)].view(H, W, 3).clone()
        outputs.update({"gt_semantic_map": gt_semantic_map})
        outputs["gt_semantics"] = gt_semantic

        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        # affine transformation
        outputs["rgb"] = self.affine_transformation(
            outputs["rgb_gaussians"] + outputs["rgb_sky"] * (1.0 - outputs["opacity"]), image_infos
        )
        return outputs
    
    def backward(self, loss_dict: Dict[str, torch.Tensor]) -> None:
        # ----------------- backward ----------------
        total_loss = sum(loss for loss in loss_dict.values())

        self._dump_autograd_graph(
            loss=total_loss,                 # IMPORTANT: unscaled loss is fine
            out_dir="/tudelft.net/staff-umbrella/hchassagnette/Workspace/output/graph",
            tag="train"
        )
        exit(0)

        self.grad_scaler.scale(total_loss).backward()
        self.optimizer_step()
        
        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            for group in self.optimizer.param_groups:
                if group["name"] in self.lr_schedulers:
                    new_lr = self.lr_schedulers[group["name"]](self.step)
                    group["lr"] = new_lr
    
    def _dump_autograd_graph(self, loss: torch.Tensor, out_dir: str, tag: str = ""):
        """
        Saves autograd graph for `loss` using torchviz.
        Writes: <out_dir>/autograd_<tag>_stepXXXXX.{dot,png}
        """
        try:
            from torchviz import make_dot
        except ImportError as e:
            logger.warning("torchviz not installed. `pip install torchviz` to enable graph dumps.")
            return

        os.makedirs(out_dir, exist_ok=True)
        step = getattr(self, "step", 0)

        # Param mapping is optional but makes the graph readable (names on leaves)
        params = {}
        try:
            params = dict(self.named_parameters())
        except Exception:
            pass

        dot = make_dot(loss, params=params)

        base = f"autograd{('_' + tag) if tag else ''}_step{step:05d}"
        dot_path = os.path.join(out_dir, base + ".dot")
        png_path = os.path.join(out_dir, base + ".png")

        # Save DOT (for later inspection / custom rendering)
        dot.save(dot_path)

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        # calculate loss
        loss_dict = {}
        
        if "egocar_masks" in image_infos:
            # in the case of egocar, we need to mask out the egocar region
            valid_loss_mask = (1.0 - image_infos["egocar_masks"]).float()
        else:
            valid_loss_mask = torch.ones_like(image_infos["sky_masks"])
        gt_rgb = image_infos["pixels"] * valid_loss_mask[..., None]
        predicted_rgb = outputs["rgb"] * valid_loss_mask[..., None]
        
        gt_occupied_mask = (1.0 - image_infos["sky_masks"]).float() * valid_loss_mask
        pred_occupied_mask = outputs["opacity"].squeeze() * valid_loss_mask
        
        # rgb loss
        Ll1 = torch.abs(gt_rgb - predicted_rgb).mean()
        simloss = 1 - self.ssim(gt_rgb.permute(2, 0, 1)[None, ...], predicted_rgb.permute(2, 0, 1)[None, ...])
        loss_dict.update({
            "rgb_loss": self.losses_dict.rgb.w * Ll1,
            "ssim_loss": self.losses_dict.ssim.w * simloss,
        })
        
        # mask loss
        if self.sky_opacity_loss_fn is not None:
            sky_loss_opacity = self.sky_opacity_loss_fn(pred_occupied_mask, gt_occupied_mask) * self.losses_dict.mask.w
            loss_dict.update({"sky_loss_opacity": sky_loss_opacity})
        
        # depth loss
        if self.depth_loss_fn is not None:
            gt_depth = image_infos["lidar_depth_map"] 
            lidar_hit_mask = (gt_depth > 0).float() * valid_loss_mask
            pred_depth = outputs["depth"]
            depth_loss = self.depth_loss_fn(pred_depth, gt_depth, lidar_hit_mask)
            
            lidar_w_decay = self.losses_dict.depth.get("lidar_w_decay", -1)
            if lidar_w_decay > 0:
                decay_weight = np.exp(-self.step / 8000 * lidar_w_decay)
            else:
                decay_weight = 1
            depth_loss = depth_loss * self.losses_dict.depth.w * decay_weight
            loss_dict.update({"depth_loss": depth_loss})
            
        # semantic loss
        if self.semantic_loss_cfg is not None:
            cfg = self.semantic_loss_cfg

            # Toggles
            use_ce          = cfg.get("use_ce", True)
            use_focal       = cfg.get("use_focal", False)
            use_contrastive = cfg.get("use_contrastive", False)
            use_reg         = cfg.get("use_reg", False)
            use_sem_depth   = cfg.get("use_depth", False)

            # Weights
            semce_w     = cfg.get("loss_ce_w", 0.1)
            semfocal_w  = cfg.get("loss_focal_w", 0.0)
            semcont_w   = cfg.get("loss_contrastive_w", 0.0)
            semreg_w    = cfg.get("loss_reg_w", 0.0)
            semdepth_w  = cfg.get("loss_depth_w", 0.0)

            # Focal parameters
            focal_alpha = cfg.get("focal_alpha", 0.25)
            focal_gamma = cfg.get("focal_gamma", 2.0)

            # Regularization parameters
            # reg_type: "entropy" (default) or "l2"
            reg_type = cfg.get("reg_type", "entropy")

            # Shared data
            pred_semantic_labels = outputs["semantic_label"]
            pred_semantic_logits = outputs["semantic_logits"]
            gt_semantics         = image_infos["lidar_semantics_map"]

            # mask out unlabeled classes (17, 18)
            labeled_mask = (gt_semantics != 17) & (gt_semantics != 18)
            total_labeled = labeled_mask.float().sum().clamp_min(1.0)

            sem_losses = {}

            # Precompute flattened tensors once (for all pixel-wise losses)
            H, W, K = pred_semantic_logits.shape  # [H, W, num_classes]

            logits_flat = pred_semantic_logits.view(-1, K)  # [HW, K]
            gt_flat     = gt_semantics.view(-1)             # [HW]
            mask_flat   = labeled_mask.view(-1)             # [HW]

            idx = mask_flat.nonzero(as_tuple=False).squeeze(1)  # [N] indices of labeled pixels
            has_labeled = idx.numel() > 0

            # Small helper for warmup on *all* semantic losses
            warmup_start = cfg.get("warmup_start", 5000)
            full_weight_step = cfg.get("full_weight_step", 15000)

            def apply_warmup(base_weight: float) -> float:
                if base_weight <= 0.0:
                    return 0.0
                if self.step < warmup_start:
                    return 0.0
                t = (self.step - warmup_start) / max(1, full_weight_step - warmup_start)
                return base_weight * t
            
                        # Small helper to get class-weights tensor on the right device
            class_weight_mode = cfg.get("class_weight_mode", None)  # "manual", "inv_freq", or None
            manual_class_weights = cfg.get("class_weights", None)
            inv_freq_eps = cfg.get("inv_freq_eps", 1e-6)
            normalize_class_weights = cfg.get("normalize_class_weights", True)

            class_weights_tensor = None  # default: no class weighting

            if class_weight_mode == "manual" and manual_class_weights is not None:
                # Manual weights from config
                assert len(manual_class_weights) == K, (
                    f"class_weights length {len(manual_class_weights)} "
                    f"must match num_classes {K}"
                )
                class_weights_tensor = logits_flat.new_tensor(
                    manual_class_weights, dtype=torch.float32
                )  # [K]

            elif class_weight_mode == "inv_freq" and has_labeled:
                # Inverse class frequency from current batch (only over labeled pixels)
                gt_lab_for_weights = gt_flat[idx].long()  # [N]

                # Count occurrences per class (including all K)
                counts = torch.bincount(gt_lab_for_weights, minlength=K).float()  # [K]

                # Avoid division by zero: for classes not present, we set weight 0
                weights = torch.zeros_like(counts)
                nonzero = counts > 0
                weights[nonzero] = 1.0 / (counts[nonzero] + inv_freq_eps)

                if normalize_class_weights and nonzero.any():
                    # Normalize so that mean(weight over seen classes) ~ 1
                    mean_w = weights[nonzero].mean()
                    weights[nonzero] = weights[nonzero] / mean_w

                class_weights_tensor = weights  # [K]
            # else: keep class_weights_tensor = None (no per-class weighting)

            # ------------------------------------------------------------------
            # 1) CE loss (already implemented, just slightly refactored)
            # ------------------------------------------------------------------
            if use_ce:
                if has_labeled:
                    logits_lab = logits_flat[idx]    # [N, K]
                    gt_lab     = gt_flat[idx].long() # [N]

                    # IMPORTANT: ensure gt_lab in [0, K-1]; 17 & 18 were masked out
                    loss_ce = F.cross_entropy(
                        logits_lab, gt_lab,
                        weight=class_weights_tensor,  # None or [K]
                        reduction="mean"
                    )
                else:
                    loss_ce = logits_flat.new_tensor(0.0)

                semce_weight = apply_warmup(semce_w)
                sem_losses["semantic_CE_loss"] = semce_weight * loss_ce

            # ------------------------------------------------------------------
            # 2) Focal loss (same logits / labels as CE, better for class imbalance)
            # ------------------------------------------------------------------
            if use_focal:
                if has_labeled:
                    logits_lab = logits_flat[idx]    # [N, K]
                    gt_lab     = gt_flat[idx].long() # [N]

                    # standard focal loss on logits_lab, gt_lab
                    # Compute softmax probabilities
                    log_probs = F.log_softmax(logits_lab, dim=-1)             # [N, K]
                    probs     = log_probs.exp()                               # [N, K]

                    # Gather probabilities of the true class
                    gt_lab_unsqueezed = gt_lab.unsqueeze(1)                  # [N, 1]
                    p_t = probs.gather(1, gt_lab_unsqueezed).clamp_min(1e-6) # [N, 1]

                    # Focal weight
                    focal_weight = (1.0 - p_t) ** focal_gamma                # [N, 1]

                    # Optional class-balancing alpha: assume scalar alpha for foreground,
                    # we apply it to all classes for simplicity; you can extend this to per-class alpha.
                    alpha_factor = focal_alpha

                    # CE term (negative log-likelihood)
                    ce_term = F.nll_loss(
                        log_probs, gt_lab, reduction="none"
                    ).unsqueeze(1)                                           # [N, 1]

                    if class_weights_tensor is not None:
                        per_class_w = class_weights_tensor[gt_lab].unsqueeze(1)  # [N, 1]
                    else:
                        per_class_w = 1.0

                    loss_focal = (per_class_w * alpha_factor * focal_weight * ce_term).mean()
                else:
                    loss_focal = logits_flat.new_tensor(0.0)

                semfocal_weight = apply_warmup(semfocal_w)
                sem_losses["semantic_focal_loss"] = semfocal_weight * loss_focal

            # ------------------------------------------------------------------
            # 3) Contrastive / feature-alignment loss (global, no broadcasting)
            # ------------------------------------------------------------------
            if use_contrastive:
                device = pred_semantic_logits.device
                pred_sem_feats = outputs.get("semantic_features", None).to(device)
                teacher_vec    = image_infos.get("semantic_teacher_features", None).to(device)

                if (pred_sem_feats is not None) and (teacher_vec is not None):
                    Hs, Ws, D_student = pred_sem_feats.shape

                    # Flatten spatial dims
                    student_flat = pred_sem_feats.view(-1, D_student)  # [H*W, D_student]

                    # Optionally restrict pooling to labeled pixels
                    if has_labeled:
                        student_lab = student_flat[idx]              # [N, D_student]
                        if student_lab.numel() == 0:
                            # fallback: use all pixels if no labeled ones
                            student_global = student_flat.mean(dim=0, keepdim=True)  # [1, D_student]
                        else:
                            student_global = student_lab.mean(dim=0, keepdim=True)   # [1, D_student]
                    else:
                        student_global = student_flat.mean(dim=0, keepdim=True)       # [1, D_student]

                    # Teacher is a single vector [D_clip] for the whole image
                    teacher_vec = teacher_vec.to(student_global.device)               # [D_clip]
                    teacher_vec = teacher_vec.view(1, -1)                             # [1, D_clip]

                    # Project teacher to student space: nn.Linear(D_clip, D_student)
                    self.teacher_proj = self.teacher_proj.to(device)
                    teacher_proj = self.teacher_proj(teacher_vec)                     # [1, D_student]

                    # Normalize for cosine similarity
                    student_norm = F.normalize(student_global, p=2, dim=-1)          # [1, D_student]
                    teacher_norm = F.normalize(teacher_proj,  p=2, dim=-1)           # [1, D_student]

                    cos_sim = (student_norm * teacher_norm).sum(dim=-1)              # [1]
                    loss_contrastive = (1.0 - cos_sim).mean()
                else:
                    loss_contrastive = pred_semantic_logits.new_tensor(0.0)

                semcont_weight = apply_warmup(semcont_w)
                sem_losses["semantic_contrastive_loss"] = semcont_weight * loss_contrastive



            # ------------------------------------------------------------------
            # 4) Regularization loss on semantic logits (entropy or L2)
            # ------------------------------------------------------------------
            if use_reg:
                if has_labeled:
                    logits_lab = logits_flat[idx]    # [N, K]
                else:
                    logits_lab = logits_flat         # fall back to all pixels [HW, K]

                if reg_type == "entropy":
                    # Encourage low-entropy (confident) predictions
                    probs = F.softmax(logits_lab, dim=-1)           # [N, K]
                    log_probs = torch.log(probs.clamp_min(1e-6))    # [N, K]
                    entropy = -(probs * log_probs).sum(dim=-1)      # [N]
                    loss_reg = entropy.mean()
                elif reg_type == "l2":
                    # Simple L2 penalty on logits magnitude
                    loss_reg = (logits_lab ** 2).mean()
                else:
                    # Unknown reg type -> no-op
                    loss_reg = logits_flat.new_tensor(0.0)

                semreg_weight = apply_warmup(semreg_w)
                sem_losses["semantic_reg_loss"] = semreg_weight * loss_reg
            
            # ------------------------------------------------------------------
            # 5) Depth loss on semantic Gaussians
            # ------------------------------------------------------------------
            
            if use_sem_depth:
                pred_semantic_depth  = outputs["semantic_depth"]
                gt_depth = image_infos["lidar_depth_map"] 
                lidar_hit_mask = (gt_depth > 0).float() * valid_loss_mask
                sem_depth_loss = self.depth_loss_fn(pred_semantic_depth, gt_depth, lidar_hit_mask)
                
                sem_losses["sem_depth_loss"] = sem_depth_loss * semdepth_w

            # ------------------------------------------------------------------
            # Add semantic losses to main loss dict
            # ------------------------------------------------------------------
            if len(sem_losses) > 0:
                loss_dict.update(sem_losses)


        # ----- reg loss -----
        opacity_entropy_reg = self.losses_dict.get("opacity_entropy", None)
        if opacity_entropy_reg is not None:
            pred_opacity = torch.clamp(outputs["opacity"].squeeze(), 1e-6, 1 - 1e-6)
            loss_dict.update({
                "opacity_entropy_loss": opacity_entropy_reg.w * (-pred_opacity * torch.log(pred_opacity)).mean()
            })
            
        # from pvg: https://github.com/fudan-zvg/PVG/blob/b4162a9135282e0f3c929054f16be1b3fbacd77a/train.py#L161
        inverse_depth_smoothness_reg = self.losses_dict.get("inverse_depth_smoothness", None)
        if inverse_depth_smoothness_reg is not None:
            inverse_depth = 1 / (outputs["depth"] + 1e-5)
            loss_inv_depth = kornia.losses.inverse_depth_smoothness_loss(
                inverse_depth[None].repeat(1, 1, 1, 3).permute(0, 3, 1, 2),
                image_infos["pixels"][None].permute(0, 3, 1, 2)
            )
            loss_dict.update({
                "inverse_depth_smoothness_loss": inverse_depth_smoothness_reg.w * loss_inv_depth
            })
            
        # affine reg loss
        affine_reg = self.losses_dict.get("affine", None)
        if affine_reg is not None and "Affine" in self.models:
            affine_trs = self.models['Affine']({"img_idx": image_infos["img_idx"].flatten()[0]})
            reg_mat = torch.eye(3, device=self.device)
            reg_shift = torch.zeros(3, device=self.device)
            loss_affine = torch.abs(affine_trs[..., :3, :3] - reg_mat).mean() + torch.abs(affine_trs[..., :3, 3:] - reg_shift).mean()
            loss_dict.update({
                "affine_loss": affine_reg.w * loss_affine
            })

        # dynamic region loss
        dynamic_region_weighted_losses = self.losses_dict.get("dynamic_region", None)
        if dynamic_region_weighted_losses is not None:
            weight_factor = dynamic_region_weighted_losses.get("w", 1.0)
            start_from = dynamic_region_weighted_losses.get("start_from", 0)
            if self.step == start_from:
                self.render_dynamic_mask = True
            if self.step > start_from and "Dynamic_opacity" in outputs:
                dynamic_pred_mask = (outputs["Dynamic_opacity"].data > 0.2).squeeze()
                dynamic_pred_mask = dynamic_pred_mask & valid_loss_mask.bool()
                
                if dynamic_pred_mask.sum() > 0:
                    Ll1 = torch.abs(gt_rgb[dynamic_pred_mask] - predicted_rgb[dynamic_pred_mask]).mean()
                    loss_dict.update({
                        "vehicle_region_rgb_loss": weight_factor * Ll1,
                    })
            
        # compute gaussian reg loss
        for class_name in self.gaussian_classes.keys():
            class_reg_loss = self.models[class_name].compute_reg_loss()
            for k, v in class_reg_loss.items():
                loss_dict[f"{class_name}_{k}"] = v
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        metric_dict = {}
        psnr = self.psnr(outputs["rgb"], image_infos["pixels"])
        metric_dict.update({"psnr": psnr})
        return metric_dict
    
    def get_gaussian_count(self):
        num_dict = {}
        for class_name in self.gaussian_classes.keys():
            num_dict[class_name] = self.models[class_name].num_points
        return num_dict
    
    def state_dict(self, only_model: bool = True):
        state_dict = super().state_dict()
        state_dict.update({
            "models": {k: v.state_dict() for k, v in self.models.items()},
            "step": self.step,
        })
        if not only_model:
            state_dict.update({
                "optimizer": {k: v.state_dict() for k, v in self.optimizer.items()},
                # "lr_schedulers": {k: v.state_dict() for k, v in self.lr_schedulers.items()},
                # "grad_scaler": self.grad_scaler.state_dict(),
            })
        return state_dict

    def load_state_dict(self, state_dict: dict, load_only_model: bool =True, strict: bool = True):
        step = state_dict.pop("step")
        self.step = step
        logger.info(f"Loading checkpoint at step {step}")

        # load optimizer and schedulers
        if "optimizer" in state_dict:
            loaded_state_optimizers = state_dict.pop("optimizer")
        # if "schedulers" in state_dict:
        #     loaded_state_schedulers = state_dict.pop("schedulers")
        # if "grad_scaler" in state_dict:
        #     loaded_grad_scaler = state_dict.pop("grad_scaler")
        if not load_only_model:
            raise NotImplementedError("Now only support loading model, \
                it seems there is no need to load optimizer and schedulers")
            for k, v in loaded_state_optimizers.items():
                self.optimizer[k].load_state_dict(v)
            for k, v in loaded_state_schedulers.items():
                self.schedulers[k].load_state_dict(v)
            self.grad_scaler.load_state_dict(loaded_grad_scaler)
        
        # load model
        model_state_dict = state_dict.pop("models")
        for class_name in self.models.keys():
            model = self.models[class_name]
            model.step = step
            if class_name not in model_state_dict:
                if class_name in self.gaussian_classes:
                    self.gaussian_classes.pop(class_name)
                logger.warning(f"Cannot find {class_name} in the checkpoint")
                continue
            msg = model.load_state_dict(model_state_dict[class_name], strict=strict)
            logger.info(f"{class_name}: {msg}")
        msg = super().load_state_dict(state_dict, strict)
        logger.info(f"BasicTrainer: {msg}")
        
    def resume_from_checkpoint(
        self,
        ckpt_path: str,
        load_only_model: bool=True
    ) -> None:
        """
        Load model from checkpoint.
        """
        logger.info(f"Loading checkpoint from {ckpt_path}")
        state_dict = torch.load(ckpt_path)
        self.load_state_dict(state_dict, load_only_model=load_only_model, strict=True)
        
    def save_checkpoint(
        self,
        log_dir: str,
        save_only_model: bool=True,
        is_final: bool=False
    ) -> None:
        """
        Save model to checkpoint.
        """
        if is_final:
            ckpt_path = os.path.join(log_dir, f"checkpoint_final.pth")
        else:
            ckpt_path = os.path.join(log_dir, f"checkpoint_{self.step:05d}.pth")
        torch.save(self.state_dict(only_model=save_only_model), ckpt_path)
        logger.info(f"Saved a checkpoint to {ckpt_path}")
        
    def init_viewer(self, port: int = 8080):
        # a simple viewer for background ONLY visualization
        self.server = viser.ViserServer(port=port, verbose=False)
        self.viewer = nerfview.Viewer(
            server=self.server,
            render_fn=self._viewer_render_fn,
            mode="training",
        )

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)
        
        cam = dataclass_camera(
            camtoworlds=c2w,
            camtoworlds_gt=c2w,
            Ks=K,
            H=H,
            W=W
        )
        
        gs_dict = {
            "_means": [],
            "_scales": [],
            "_quats": [],
            "_rgbs": [],
            "_opacities": [],
        }
        for class_name in ["Background"]:
            gs = self.models[class_name].get_gaussians(cam)
            if gs is None:
                continue

            for k, _ in gs.items():
                gs_dict[k].append(gs[k])
        
        for k, v in gs_dict.items():
            gs_dict[k] = torch.cat(v, dim=0)

        gs = dataclass_gs(
            _means=gs_dict["_means"],
            _scales=gs_dict["_scales"],
            _quats=gs_dict["_quats"],
            _rgbs=gs_dict["_rgbs"],
            _opacities=gs_dict["_opacities"],
            detach_keys=[],
            extras=None
        )
        
        render_colors, _, _ = rasterization(
            means=gs.means,
            quats=gs.quats,
            scales=gs.scales,
            opacities=gs.opacities.squeeze(),
            colors=gs.rgbs,
            viewmats=torch.linalg.inv(cam.camtoworlds)[None, ...],  # [C, 4, 4]
            Ks=cam.Ks[None, ...],  # [C, 3, 3]
            width=cam.W,
            height=cam.H,
            packed=self.render_cfg.packed,
            absgrad=self.render_cfg.absgrad,
            sparse_grad=self.render_cfg.sparse_grad,
            rasterize_mode="antialiased" if self.render_cfg.antialiased else "classic",
            radius_clip=4.0,  # skip GSs that have small image radius (in pixels)
        )
        return render_colors[0].cpu().numpy()