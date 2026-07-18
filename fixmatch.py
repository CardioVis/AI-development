"""
FixMatch for semi-supervised medical image segmentation.

Supported models (via --model):
  deeplabv3plus_resnet101  (Models.DeepLabV3Plus)
  unet_resnet34            (segmentation_models_pytorch)
  segformer_b2             (segmentation_models_pytorch)

Example:
  python fixmatch.py --config_yml Configs/cardio.yml --model unet_resnet34 --exp dk_unet
"""

import json
import os
import sys
import time
import argparse

import yaml
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from itertools import cycle

from Datasets.create_dataset import Dataset, StrongWeakAugment, get_dataset_without_full_label
from Utils.utils import DotDict, fix_all_seed

from Models.DeepLabV3Plus import deeplabv3plus_resnet101
from monai.losses import GeneralizedDiceFocalLoss
from monai.metrics import DiceMetric, MeanIoU

SUPPORTED_MODELS = (
    "deeplabv3plus_resnet101",
    "unet_resnet34",
    "segformer_b2",
)

CE_LOSS_WEIGHT = 0.5


def _exclude_background(config) -> bool:
    return bool(getattr(config, "exclude_background", True))


def _use_class_weights(config) -> bool:
    return bool(getattr(config, "use_class_weights", False))


def compute_fg_class_weights(labels_dir: str, num_classes: int) -> torch.Tensor:
    """Pixel histogram over FG classes → weight_c = 1/sqrt(freq); bg weight = 0."""
    counts = np.zeros(num_classes, dtype=np.float64)
    label_files = sorted(
        f for f in os.listdir(labels_dir) if f.endswith(".npy")
    )
    for name in label_files:
        mask = np.load(os.path.join(labels_dir, name))
        if mask.ndim > 2:
            mask = np.argmax(mask, axis=0) if mask.shape[0] == num_classes else mask[..., 0]
        flat = mask.astype(np.int64).ravel()
        bc = np.bincount(flat, minlength=num_classes)
        counts += bc[:num_classes]
    weights = np.zeros(num_classes, dtype=np.float64)
    for c in range(1, num_classes):
        freq = max(counts[c], 1.0)
        weights[c] = 1.0 / np.sqrt(freq)
    fg_sum = weights[1:].sum()
    if fg_sum > 0:
        weights[1:] = weights[1:] / fg_sum * (num_classes - 1)
    weights[0] = 0.0
    return torch.tensor(weights, dtype=torch.float32)


def load_or_compute_class_weights(config) -> torch.Tensor:
    num_classes = int(getattr(config.data, "num_classes", 3))
    train_folder = config.data.train_folder
    weights_path = os.path.join(train_folder, "class_weights.json")
    if os.path.isfile(weights_path):
        data = json.loads(open(weights_path, encoding="utf-8").read())
        w = data.get("weights")
        if isinstance(w, list) and len(w) == num_classes:
            print(f"Loaded class weights from {weights_path}")
            return torch.tensor(w, dtype=torch.float32)
    labels_dir = os.path.join(train_folder, "labels")
    weights = compute_fg_class_weights(labels_dir, num_classes)
    payload = {
        "weights": weights.tolist(),
        "formula": "1/sqrt(freq_fg); bg=0; renormalized over FG",
        "num_classes": num_classes,
    }
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote class weights to {weights_path}")
    return weights


def labeled_frame_sample_weights(lb_dataset, class_weights: torch.Tensor) -> list[float]:
    """Boost frames that contain rare FG classes (inverse of class weight rarity)."""
    # Higher class_weight ⇒ rarer ⇒ higher boost
    boost = class_weights.detach().cpu().numpy().copy()
    boost[0] = 0.0
    if boost[1:].max() > 0:
        boost[1:] = boost[1:] / boost[1:].max() * 4.0  # scale boost to [0, 4]
    weights: list[float] = []
    labels_dir = os.path.join(lb_dataset.root_dir, "labels")
    for name in lb_dataset.dataset:
        mask = np.load(os.path.join(labels_dir, name))
        if mask.ndim > 2:
            mask = np.argmax(mask, axis=0)
        present = set(int(x) for x in np.unique(mask) if int(x) > 0)
        w = 1.0 + float(sum(boost[c] for c in present if c < len(boost)))
        weights.append(w)
    return weights


def segmentation_ce_loss(
    logits: torch.Tensor,
    target_onehot_or_idx: torch.Tensor,
    class_weights: torch.Tensor | None,
    ignore_bg: bool = True,
    conf_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pixel CE; ignore background (and optional low-confidence FixMatch mask)."""
    if target_onehot_or_idx.dim() == 4 and target_onehot_or_idx.size(1) > 1:
        target = target_onehot_or_idx.argmax(dim=1)
    else:
        target = target_onehot_or_idx.long()
        if target.dim() == 4:
            target = target.squeeze(1)

    ignore_index = -100
    target_ce = target.clone()
    if ignore_bg:
        target_ce[target_ce == 0] = ignore_index
    if conf_mask is not None:
        target_ce[conf_mask < 0.5] = ignore_index

    weight = class_weights.to(logits.device) if class_weights is not None else None
    return F.cross_entropy(logits, target_ce, weight=weight, ignore_index=ignore_index)


def supervised_loss(
    criterion_gdl,
    logits: torch.Tensor,
    label: torch.Tensor,
    config,
    class_weights: torch.Tensor | None = None,
    conf_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = criterion_gdl(logits, label)
    if _use_class_weights(config) and class_weights is not None:
        loss = loss + CE_LOSS_WEIGHT * segmentation_ce_loss(
            logits,
            label,
            class_weights,
            ignore_bg=_exclude_background(config),
            conf_mask=conf_mask,
        )
    return loss


def load_class_names(config) -> list[str]:
    """Resolve class names from class_map.json next to train/test folders."""
    import json

    num_classes = int(getattr(config.data, "num_classes", 3))
    data = None
    for key in ("train_folder", "test_folder", "val_folder"):
        folder = getattr(config.data, key, None)
        if not folder:
            continue
        path = os.path.join(folder, "class_map.json")
        if os.path.isfile(path):
            data = json.loads(open(path, encoding="utf-8").read())
            break
    if not isinstance(data, dict):
        return [f"class_{i}" for i in range(num_classes)]

    id_to_name: dict[int, str] = {0: "background"}
    for k, v in data.items():
        if k.startswith("_"):
            continue
        if k == "background":
            id_to_name[0] = "background"
        elif isinstance(v, int):
            id_to_name[int(v)] = str(k)
    return [id_to_name.get(i, f"class_{i}") for i in range(num_classes)]


def hd95_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    """Hausdorff-95 between two binary 2D masks (CPU, SciPy). nan if undefined."""
    from scipy.ndimage import distance_transform_edt

    pred = pred.astype(bool)
    gt = gt.astype(bool)
    if not pred.any() and not gt.any():
        return float("nan")
    if not pred.any() or not gt.any():
        return float("nan")
    dt_pred = distance_transform_edt(~pred)
    dt_gt = distance_transform_edt(~gt)
    d_gt = dt_pred[gt]
    d_pred = dt_gt[pred]
    return float(np.percentile(np.concatenate([d_gt, d_pred]), 95))


def dice_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return float("nan")
    return float(2.0 * inter / denom)


def iou_binary(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    union = pred.sum() + gt.sum() - inter
    if union == 0:
        return float("nan")
    return float(inter / union)


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def build_model(model_name: str, num_classes: int):
    name = (model_name or "deeplabv3plus_resnet101").lower()
    if name == "deeplabv3plus_resnet101":
        return deeplabv3plus_resnet101(
            num_classes=num_classes, output_stride=8, pretrained_backbone=True
        )
    if name == "unet_resnet34":
        import segmentation_models_pytorch as smp

        return smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    if name == "segformer_b2":
        import segmentation_models_pytorch as smp

        return smp.Segformer(
            encoder_name="mit_b2",
            encoder_weights="imagenet",
            in_channels=3,
            classes=num_classes,
        )
    raise ValueError(f"Unknown model {model_name!r}. Choose from {SUPPORTED_MODELS}")


def load_name_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_external_test_loader(config):
    """Patient-held-out test set from data.test_folder/test.txt when present."""
    test_folder = getattr(config.data, "test_folder", None)
    if not test_folder:
        return None
    list_path = os.path.join(test_folder, "test.txt")
    if not os.path.isfile(list_path):
        return None
    names = load_name_list(list_path)
    if not names:
        return None
    num_classes = getattr(config.data, "num_classes", 3)
    ds = Dataset(
        dataset=names,
        img_size=config.data.img_size,
        use_aug=False,
        data_path=test_folder,
        num_classes=num_classes,
    )
    return DataLoader(
        ds,
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True,
    )


def main(config):
    dataset = get_dataset_without_full_label(
        config,
        img_size=config.data.img_size,
        train_aug=config.data.train_aug,
        k=config.fold,
        lb_dataset=Dataset,
        ulb_dataset=StrongWeakAugment,
    )

    class_weights = None
    sampler = None
    shuffle = True
    if _use_class_weights(config):
        class_weights = load_or_compute_class_weights(config)
        frame_w = labeled_frame_sample_weights(dataset["lb_dataset"], class_weights)
        sampler = WeightedRandomSampler(
            weights=frame_w,
            num_samples=len(frame_w),
            replacement=True,
        )
        shuffle = False
        print(
            f"Class-weight sampling ON (CE λ={CE_LOSS_WEIGHT}); "
            f"frame weight range [{min(frame_w):.2f}, {max(frame_w):.2f}]"
        )

    l_train_loader = DataLoader(
        dataset["lb_dataset"],
        batch_size=config.train.l_batchsize,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,  # BatchNorm (DeepLab ASPP) needs batch>1 in train mode
    )
    u_train_loader = DataLoader(
        dataset["ulb_dataset"],
        batch_size=config.train.u_batchsize,
        shuffle=True,
        num_workers=config.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if len(l_train_loader) == 0 or len(u_train_loader) == 0:
        raise RuntimeError(
            f"Empty train loader after drop_last "
            f"(labeled_batches={len(l_train_loader)}, unlabeled_batches={len(u_train_loader)}). "
            f"Reduce batch size or add more data."
        )
    val_loader = DataLoader(
        dataset["val_dataset"],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True,
    )

    external_test = build_external_test_loader(config)
    test_loader = external_test or DataLoader(
        dataset["val_dataset"],
        batch_size=config.test.batch_size,
        shuffle=False,
        num_workers=config.test.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if external_test is not None:
        print(f"Using patient-held-out test_folder: {config.data.test_folder}")

    train_loader = {
        "l_loader": l_train_loader,
        "u_loader": u_train_loader,
        "class_weights": class_weights,
    }
    print(
        f"Unlabeled folder: {config.data.unlabeled_folder} | "
        f"Unlabeled batches: {len(u_train_loader)}, Labeled batches: {len(l_train_loader)}"
    )
    print(
        f"exclude_background={_exclude_background(config)} "
        f"use_class_weights={_use_class_weights(config)}"
    )

    num_classes = getattr(config.data, "num_classes", 3)
    model_name = getattr(config, "model_name", "deeplabv3plus_resnet101")
    model = build_model(model_name, num_classes=num_classes)
    print(f"Model: {model_name}")

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_params / 1e6:.2f}M total parameters")
    print(f"{total_trainable_params / 1e6:.2f}M trainable parameters")

    device = config.device
    model = model.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    criterion = [
        GeneralizedDiceFocalLoss(
            include_background=not _exclude_background(config),
            to_onehot_y=False,
            softmax=True,
        ).to(device)
    ]

    if config.test.only_test:
        test(config, model, best_model_dir, test_loader, criterion)
    else:
        train_val(config, model, train_loader, val_loader, criterion)
        test(config, model, best_model_dir, test_loader, criterion)


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))


def get_current_consistency_weight(epoch):
    return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)


def train_val(config, model, train_loader, val_loader, criterion):
    num_classes = getattr(config.data, "num_classes", 3)
    num_epochs = int(config.train.num_epochs)
    fold = int(getattr(config, "fold", 1))
    class_weights = train_loader.get("class_weights")
    exclude_bg = _exclude_background(config)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config.train.optimizer.adamw.lr),
        weight_decay=float(config.train.optimizer.adamw.weight_decay),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    train_dice = DiceMetric(
        include_background=not exclude_bg, num_classes=num_classes, reduction="mean"
    )
    train_iou = MeanIoU(include_background=not exclude_bg, reduction="mean")

    max_dice_score = -float("inf")
    best_epoch = 0
    warmup_epochs = getattr(config.train, "warmup_epochs", 15)
    n_u = len(train_loader["u_loader"])

    torch.save(model.state_dict(), best_model_dir)
    print(
        f"[fold {fold}] Start training: {num_epochs} epochs, "
        f"warmup={warmup_epochs}, unlabeled_batches/epoch={n_u}, "
        f"exclude_bg={exclude_bg}, class_weights={class_weights is not None}",
        flush=True,
    )

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        train_metrics = {"dice": 0, "iou": 0, "loss": 0}
        num_train = 0
        phase = "warmup" if epoch < warmup_epochs else "fixmatch"

        source_dataset = zip(cycle(train_loader["l_loader"]), train_loader["u_loader"])
        train_loop = tqdm(
            source_dataset,
            total=n_u,
            desc=f"Fold{fold} Ep {epoch + 1}/{num_epochs} [{phase}]",
            leave=True,
            file=sys.stderr,
            dynamic_ncols=True,
            mininterval=1.0,
        )

        for idx, (batch, batch_w_s) in enumerate(train_loop):
            device = config.device
            img = batch["image"].to(device).float()
            label = batch["label"].to(device).float()
            weak_batch = batch_w_s["img_w"].to(device).float()
            strong_batch = batch_w_s["img_s"].to(device).float()

            sup_batch_len = img.shape[0]
            unsup_batch_len = weak_batch.shape[0]

            output = model(img)

            if epoch < warmup_epochs:
                loss = supervised_loss(
                    criterion[0], output, label, config, class_weights=class_weights
                )
            else:
                with torch.no_grad():
                    outputs_weak = model(weak_batch)
                    max_probs, pseudo_labels = torch.max(outputs_weak.softmax(dim=1), dim=1)
                    mask = max_probs.ge(config.semi.conf_thresh).float()
                    num_pred_classes = outputs_weak.size(1)
                    pseudo_labels_one_hot = (
                        F.one_hot(pseudo_labels, num_classes=num_pred_classes)
                        .permute(0, 3, 1, 2)
                        .float()
                    )
                    pseudo_labels_masked = pseudo_labels_one_hot * mask.unsqueeze(1)

                outputs_strong = model(strong_batch)
                sup_loss = supervised_loss(
                    criterion[0], output, label, config, class_weights=class_weights
                )
                unsup_loss = supervised_loss(
                    criterion[0],
                    outputs_strong,
                    pseudo_labels_masked,
                    config,
                    class_weights=class_weights,
                    conf_mask=mask,
                )
                consistency_weight = get_current_consistency_weight(epoch)
                loss = (
                    sup_loss
                    + unsup_loss * consistency_weight * (sup_batch_len / unsup_batch_len)
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                output_onehot = torch.zeros_like(output)
                output_onehot.scatter_(1, output.argmax(dim=1, keepdim=True), 1)
                train_dice(y_pred=output_onehot, y=label)
                train_iou(y_pred=output_onehot, y=label)
                train_metrics["loss"] = (
                    train_metrics["loss"] * num_train + loss.item() * sup_batch_len
                ) / (num_train + sup_batch_len)
                num_train += sup_batch_len

            train_loop.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{train_dice.aggregate().item():.4f}",
                refresh=False,
            )

            if config.debug:
                break

        train_metrics["dice"] = train_dice.aggregate().item()
        train_metrics["iou"] = train_iou.aggregate().item()
        train_dice.reset()
        train_iou.reset()

        val_metrics = validate_model(model, val_loader, criterion)
        time_elapsed = time.time() - start
        is_best = val_metrics["dice"] > max_dice_score
        if is_best:
            max_dice_score = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_dir)

        log_message = (
            f"[fold {fold}] Epoch {epoch + 1}/{num_epochs} [{phase}] "
            f"steps={idx + 1} | "
            f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} | "
            f"val dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} | "
            f"best={max_dice_score:.4f}@ep{best_epoch + 1} | "
            f"{time_elapsed // 60:.0f}m{time_elapsed % 60:.0f}s"
            + (" *BEST*" if is_best else "")
        )
        print(log_message, flush=True)
        file_log.write(log_message + "\n")
        file_log.flush()

        scheduler.step()
        if config.debug:
            break

    print(
        f"[fold {fold}] Training completed. Best epoch: {best_epoch + 1} "
        f"(val dice={max_dice_score:.4f})",
        flush=True,
    )

def validate_model(model, val_loader, criterion):
    """Fast val for model selection: overall Dice/IoU (MONAI mean). No HD."""
    model.eval()
    metrics = {"dice": 0, "iou": 0, "hd": float("nan"), "loss": 0}
    num_val = 0
    num_classes = getattr(config.data, "num_classes", 3)
    exclude_bg = _exclude_background(config)
    dice_metric = DiceMetric(
        include_background=not exclude_bg, num_classes=num_classes, reduction="mean"
    )
    iou_metric = MeanIoU(include_background=not exclude_bg, reduction="mean")

    val_loop = tqdm(val_loader, desc="Validation", leave=False, file=sys.stderr, mininterval=1.0)
    for batch in val_loop:
        device = getattr(config, "device", torch.device("cuda"))
        img = batch["image"].to(device).float()
        label = batch["label"].to(device).float()
        batch_len = img.shape[0]

        with torch.no_grad():
            output = model(img)
            loss = criterion[0](output, label)
            preds = torch.argmax(output, dim=1, keepdim=True)
            preds_onehot = torch.zeros_like(output)
            preds_onehot.scatter_(1, preds, 1)
            if len(label.shape) == 4:
                labels_onehot = label
            else:
                labels_onehot = torch.zeros_like(output)
                labels_onehot.scatter_(1, label.unsqueeze(1), 1)
            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            iou_metric(y_pred=preds_onehot, y=labels_onehot)
            metrics["loss"] = (metrics["loss"] * num_val + loss.item() * batch_len) / (
                num_val + batch_len
            )
            num_val += batch_len
            val_loop.set_postfix(
                loss=f"{loss.item():.4f}",
                dice=f"{dice_metric.aggregate().item():.4f}",
                refresh=False,
            )

    metrics["dice"] = dice_metric.aggregate().item()
    metrics["iou"] = iou_metric.aggregate().item()
    dice_metric.reset()
    iou_metric.reset()
    return metrics


def evaluate_detailed(model, loader, criterion, class_names: list[str], compute_hd: bool = True):
    """Full test metrics: overall + per-class Dice/IoU/(optional HD95 via SciPy CPU)."""
    model.eval()
    num_classes = len(class_names)
    exclude_bg = _exclude_background(config)
    start_c = 1 if exclude_bg else 0
    per_dice = [[] for _ in range(num_classes)]
    per_iou = [[] for _ in range(num_classes)]
    per_hd = [[] for _ in range(num_classes)]
    loss_avg = 0.0
    num_val = 0

    loop = tqdm(loader, desc="Test+metrics", leave=True, file=sys.stderr, mininterval=1.0)
    for batch in loop:
        device = getattr(config, "device", torch.device("cuda"))
        img = batch["image"].to(device).float()
        label = batch["label"].to(device).float()
        batch_len = img.shape[0]

        with torch.no_grad():
            output = model(img)
            loss = criterion[0](output, label)
            pred_idx = torch.argmax(output, dim=1).detach().cpu().numpy()  # B,H,W
            # Dataset labels are one-hot (B,C,H,W)
            if label.dim() == 4 and label.size(1) > 1:
                gt_idx = torch.argmax(label, dim=1).detach().cpu().numpy()
            else:
                gt_idx = label.detach().long().cpu().numpy()
                if gt_idx.ndim == 4:
                    gt_idx = gt_idx.squeeze(1)

            for b in range(batch_len):
                pred_b = pred_idx[b]
                gt_b = gt_idx[b]
                for c in range(start_c, num_classes):
                    p = pred_b == c
                    g = gt_b == c
                    per_dice[c].append(dice_binary(p, g))
                    per_iou[c].append(iou_binary(p, g))
                    if compute_hd:
                        per_hd[c].append(hd95_binary(p, g))

            loss_avg = (loss_avg * num_val + loss.item() * batch_len) / (num_val + batch_len)
            num_val += batch_len

    class_rows = []
    for c in range(start_c, num_classes):
        name = class_names[c] if c < len(class_names) else f"class_{c}"
        d = _nanmean(per_dice[c])
        i = _nanmean(per_iou[c])
        h = _nanmean(per_hd[c]) if compute_hd else float("nan")
        class_rows.append({"id": c, "name": name, "dice": d, "iou": i, "hd95": h})

    # Macro over reported classes (FG-only when exclude_bg)
    overall = {
        "loss": loss_avg,
        "dice": _nanmean([r["dice"] for r in class_rows]),
        "iou": _nanmean([r["iou"] for r in class_rows]),
        "hd95": _nanmean([r["hd95"] for r in class_rows]) if compute_hd else float("nan"),
        "per_class": class_rows,
        "exclude_background": exclude_bg,
    }
    return overall


def format_detailed_results(metrics: dict) -> str:
    def fmt(x):
        return f"{x:.4f}" if x == x else "nan"

    scope = (
        "macro over foreground classes (background excluded)"
        if metrics.get("exclude_background", True)
        else "macro over all classes"
    )
    lines = [
        f"Test Results ({scope}):",
        f"Loss: {metrics['loss']:.4f}",
        f"Dice: {fmt(metrics['dice'])}",
        f"IoU:  {fmt(metrics['iou'])}",
        f"HD95: {fmt(metrics['hd95'])}  (SciPy CPU; nan = class absent / undefined)",
        "",
        f"{'id':>3}  {'class':<32}  {'Dice':>8}  {'IoU':>8}  {'HD95':>8}",
        "-" * 70,
    ]
    for r in metrics["per_class"]:
        lines.append(
            f"{r['id']:3d}  {r['name']:<32}  {fmt(r['dice']):>8}  {fmt(r['iou']):>8}  {fmt(r['hd95']):>8}"
        )
    return "\n".join(lines)


def test(config, model, model_dir, test_loader, criterion):
    device = getattr(
        config, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.load_state_dict(torch.load(model_dir, map_location=device))
    class_names = load_class_names(config)
    print(f"Evaluating {len(class_names)} classes with per-organ Dice/IoU/HD95...", flush=True)
    metrics = evaluate_detailed(
        model, test_loader, criterion, class_names=class_names, compute_hd=True
    )
    results_str = format_detailed_results(metrics)

    with open(test_results_dir, "w", encoding="utf-8") as f:
        f.write(results_str + "\n")

    print("=" * 80)
    print(results_str)
    print("=" * 80)

    file_log.write("\n" + "=" * 80 + "\n")
    file_log.write(results_str + "\n")
    file_log.write("=" * 80 + "\n")
    file_log.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with FixMatch")
    parser.add_argument("--exp", type=str, default="tmp")
    parser.add_argument("--config_yml", type=str, default="Configs/cardio.yml")
    parser.add_argument(
        "--model",
        type=str,
        default="deeplabv3plus_resnet101",
        choices=list(SUPPORTED_MODELS),
    )
    parser.add_argument("--adapt_method", type=str, default=False)
    parser.add_argument("--num_domains", type=str, default=False)
    parser.add_argument("--dataset", type=str, nargs="+", default="chase_db1")
    parser.add_argument("--k_fold", type=str, default="No")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Single fold (legacy). Prefer --folds for multi-fold.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=None,
        help="Folds to train, e.g. --folds 1 2 3 4 5",
    )
    parser.add_argument("--consistency", type=float, default=0.5)
    parser.add_argument("--consistency_rampup", type=float, default=75.0)
    # Notebook-friendly overrides
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--l_batchsize", type=int, default=None)
    parser.add_argument("--u_batchsize", type=int, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--conf_thresh", type=float, default=None)
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="If set, rewrite train/val/test/unlabeled folders under this dat_kim_goc root",
    )
    parser.add_argument(
        "--unlabeled_folder",
        type=str,
        default=None,
        help="Override unlabeled folder (e.g. .../unlabeled_data_dense for ablation M1)",
    )
    parser.add_argument(
        "--exclude_background",
        type=int,
        default=1,
        choices=[0, 1],
        help="Exclude background from loss/metrics (1=yes, default)",
    )
    parser.add_argument(
        "--use_class_weights",
        type=int,
        default=0,
        choices=[0, 1],
        help="Ablation M2: weighted CE + rare-frame sampler (1=on)",
    )

    args = parser.parse_args()

    config = yaml.load(open(args.config_yml, encoding="utf-8"), Loader=yaml.FullLoader)
    config["model_adapt"]["adapt_method"] = args.adapt_method
    config["model_adapt"]["num_domains"] = args.num_domains
    config["data"]["k_fold"] = args.k_fold
    config["seed"] = args.seed
    config["model_name"] = args.model
    config["exclude_background"] = bool(args.exclude_background)
    config["use_class_weights"] = bool(args.use_class_weights)

    if args.data_root:
        root = args.data_root.rstrip("/")
        config["data"]["train_folder"] = f"{root}/labeled_data"
        config["data"]["val_folder"] = f"{root}/labeled_data"
        config["data"]["test_folder"] = f"{root}/test_data"
        config["data"]["unlabeled_folder"] = f"{root}/unlabeled_data"

    if args.unlabeled_folder:
        config["data"]["unlabeled_folder"] = args.unlabeled_folder.rstrip("/")

    if args.num_epochs is not None:
        config["train"]["num_epochs"] = args.num_epochs
    if args.warmup_epochs is not None:
        config["train"]["warmup_epochs"] = args.warmup_epochs
    if args.l_batchsize is not None:
        config["train"]["l_batchsize"] = args.l_batchsize
    if args.u_batchsize is not None:
        config["train"]["u_batchsize"] = args.u_batchsize
    if args.img_size is not None:
        config["data"]["img_size"] = args.img_size
    if args.conf_thresh is not None:
        config["semi"]["conf_thresh"] = args.conf_thresh

    # Resolve folds: --folds > --fold > yml train.folds > [1]
    if args.folds:
        folds_cli = list(args.folds)
    elif args.fold is not None:
        folds_cli = [args.fold]
    else:
        folds_cli = list(config.get("train", {}).get("folds") or [1])
    config["train"]["folds"] = folds_cli
    config["fold"] = folds_cli[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config["device"] = device
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    else:
        print("(Running on CPU)")

    fix_all_seed(config["seed"])
    print(yaml.dump(config, default_flow_style=False))
    for arg in vars(args):
        print(f"{arg:<20}: {getattr(args, arg)}")

    store_config = config
    config = DotDict(config)

    folds_to_run = getattr(config.train, "folds", [1])
    for fold in folds_to_run:
        print(f"\n=== Training Fold {fold} ===")
        config["fold"] = fold

        exp_dir = f"{config.data.save_folder}/{args.exp}/fold{fold}"
        os.makedirs(exp_dir, exist_ok=True)
        best_model_dir = f"{exp_dir}/best.pth"
        test_results_dir = f"{exp_dir}/test_results.txt"

        if not config.debug:
            yaml.dump(store_config, open(f"{exp_dir}/exp_config.yml", "w"))

        with open(f"{exp_dir}/log.txt", "w") as file_log:
            main(config)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
