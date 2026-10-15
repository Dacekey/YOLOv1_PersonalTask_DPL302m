"""Single-GPU/DDP YOLOv1 training with resumable epoch checkpoints and ZIP export."""
import argparse
import csv
import json
import logging
import signal
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import distributed_utils as distributed

from dataset import VOC_CLASSES, VOCDataset, collate, encode_target, find_voc_root
from evaluate import evaluate_model, save_metrics
from export_results import export_results
from loss import YOLOv1Loss
from predict import save_dataset_examples
from pretrained import load_pretrained
from run_utils import (atomic_save, load_detector, restore_rng, rng_state, runtime_info,
                       seed_everything, seed_worker, write_json)
from yolov1 import YOLOv1


def build_model(config, device, pretrained_path=None):
    model = YOLOv1(dropout=config.get("dropout", 0.5))
    metadata = load_pretrained(model.backbone, pretrained_path) if pretrained_path else None
    return model.to(device), metadata


def write_history(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def train_run(config, data, output, pretrained_path=None, resume=None, device="cuda",
              max_hours=None, zip_path=None):
    config = dict(config)
    rank, world = distributed.rank(), distributed.world_size()
    primary = rank == 0
    if config.get("expected_world_size", world) != world:
        raise ValueError(f"Config expects {config['expected_world_size']} processes; got {world}. Use torchrun.")
    required = ("epochs", "batch_size", "accumulation", "learning_rate", "seed")
    for key in required:
        if key not in config:
            raise ValueError(f"Missing config key {key}")
    if min(config["epochs"], config["batch_size"], config["accumulation"]) < 1:
        raise ValueError("epochs, batch_size and accumulation must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select a Kaggle GPU or use --device cpu for checks")
    if max_hours is not None and max_hours <= 0:
        raise ValueError("max-hours must be positive")
    device = torch.device(device)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    if (output / "checkpoints/last.pt").exists() and not resume:
        raise FileExistsError("Run already exists. Choose a new output directory or pass --resume.")
    seed_everything(config["seed"])
    torch.set_num_threads(config.get("cpu_threads", 4))
    logger = logging.getLogger("yolov1")
    logger.setLevel(logging.INFO); logger.propagate = False
    for handler in list(logger.handlers):
        handler.close(); logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handlers = (logging.StreamHandler(), logging.FileHandler(output / "training.log")) if primary else (logging.NullHandler(),)
    for handler in handlers:
        handler.setFormatter(formatter); logger.addHandler(handler)
    started = time.monotonic()
    root = find_voc_root(data)
    train_set = VOCDataset(root, "train", augment=config.get("augment", True),
                           limit=config.get("train_limit"))
    if config.get("overfit", False):
        val_set = VOCDataset(root, "train", ids=train_set.ids)
    else:
        val_set = VOCDataset(root, "val", limit=config.get("val_limit"))
        if set(train_set.ids) & set(val_set.ids):
            raise ValueError("Training and validation IDs overlap")
    generator = torch.Generator().manual_seed(config["seed"] + rank)
    workers = config.get("num_workers", 2)
    worker_options = {"multiprocessing_context": "spawn"} if workers > 0 and world > 1 else {}
    sampler = DistributedSampler(train_set, num_replicas=world, rank=rank, seed=config["seed"],
                                 shuffle=True, drop_last=False) if world > 1 else None
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=sampler is None, sampler=sampler,
        collate_fn=collate, num_workers=workers, worker_init_fn=seed_worker, generator=generator,
        pin_memory=device.type == "cuda", **worker_options)
    val_loader = DataLoader(val_set, batch_size=config.get("eval_batch_size", 1), shuffle=False,
        collate_fn=collate, num_workers=workers, pin_memory=device.type == "cuda", **worker_options)
    checkpoint = None
    if resume:
        model, checkpoint = load_detector(resume, device)
        if checkpoint["config"] != config:
            raise ValueError("Resume config differs. Use the exact saved config, including epochs.")
        if checkpoint["split_ids"] != {"train": train_set.ids, "val": val_set.ids}:
            raise ValueError("Resume dataset IDs differ from the checkpoint")
        pretrained_metadata = checkpoint.get("pretrained")
    else:
        if not pretrained_path and config.get("require_pretrained", True):
            raise ValueError("Provide --pretrained; --random-init explicitly allows training from scratch")
        model, pretrained_metadata = build_model(config, device, pretrained_path)
    pretrained_names = {f"backbone.conv{i:02d}.{field}" for i in range(1, 17)
                        for field in ("weight", "bias")} if pretrained_metadata else set()
    old_params, new_params = [], []
    for name, parameter in model.named_parameters():
        (old_params if name in pretrained_names else new_params).append(parameter)
    groups = [{"params": new_params, "lr": config["learning_rate"]}]
    if old_params:
        groups.append({"params": old_params,
                       "lr": config["learning_rate"] * config.get("backbone_lr_factor", 0.1)})
    optimizer = torch.optim.SGD(groups, momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 0.0005))
    warmup = config.get("warmup_epochs", 3)
    milestones = config.get("lr_milestones", [30, 45])

    def lr_factor(epoch):
        return min(1.0, (epoch + 1) / max(warmup, 1)) * 0.1 ** sum(epoch >= m for m in milestones)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = bool(config.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = YOLOv1Loss(config.get("lambda_coord", 5), config.get("lambda_noobj", 0.5))
    history, best_map, global_step, start_epoch = [], -1.0, 0, 0
    if checkpoint:
        for key in ("optimizer", "scheduler", "scaler", "rng", "loader_rng"):
            if key not in checkpoint:
                raise ValueError("Use last.pt for resume; best.pt is inference-only")
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        if checkpoint.get("world_size", 1) != world:
            raise ValueError("Resume requires the same world size as the saved run")
        state = checkpoint.get("rank_states", [{"rng": checkpoint["rng"], "loader_rng": checkpoint["loader_rng"]}])[rank]
        resume_rng = state["rng"]
        generator.set_state(state["loader_rng"].cpu())
        history, best_map = checkpoint["history"], checkpoint["best_map"]
        global_step, start_epoch = checkpoint["global_step"], checkpoint["epoch"] + 1
        # Resume needs just last.pt; its best-state pointer is reconstructed if supplied alongside it.
        sibling_best = Path(resume).parent / "best.pt"
        if primary and not (output / "checkpoints/best.pt").exists() and sibling_best.is_file():
            import shutil
            (output / "checkpoints").mkdir(exist_ok=True)
            shutil.copyfile(sibling_best, output / "checkpoints/best.pt")
        if primary and Path(resume).resolve() != (output / "checkpoints/last.pt").resolve():
            import shutil
            (output / "checkpoints").mkdir(exist_ok=True)
            shutil.copyfile(resume, output / "checkpoints/last.pt")
        del checkpoint
    bare_model = model
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None,
                                        broadcast_buffers=False, gradient_as_bucket_view=True)
    if resume:
        restore_rng(resume_rng)
    else:
        seed_everything(config["seed"] + rank)
    manifest = {**runtime_info(), "config": config, "classes": list(VOC_CLASSES),
        "dataset_root": str(root), "dataset_source": "stpeteishii/pascal-voc-2007-dataset",
        "split_ids": {"train": train_set.ids, "val": val_set.ids},
        "evaluation_split": "train (overfit diagnostic)" if config.get("overfit") else "val",
        "pretrained": pretrained_metadata, "resume": str(resume) if resume else None,
        "parameters": sum(p.numel() for p in model.parameters()), "amp": amp,
        "world_size": world, "distributed_backend": dist.get_backend() if world > 1 else None,
        "per_gpu_batch_size": config["batch_size"],
        "effective_batch_size": config["batch_size"] * config["accumulation"] * world,
        "train_samples_per_rank": len(sampler) if sampler else len(train_set),
        "train_padding_duplicates_per_epoch": (len(sampler) * world - len(train_set)) if sampler else 0,
        "preprocessing": "RGB [0,1], direct bilinear resize 448x448; no ImageNet mean/std",
        "box_parameterization": "cell-relative xy; raw sqrt(w/h); linear class/confidence",
        "noobj_policy": "All non-responsible boxes, except cells with difficult-only targets",
        "collision_policy": "Largest non-difficult object per cell; full GT retained for evaluation",
        "train_collisions": sum(encode_target(a)[2] for a in train_set.annotations.values()),
        "status": "running"}
    if primary:
        write_json(output / "config.json", config); write_json(output / "run_manifest.json", manifest)
        # Snapshot code used by this run, independently of the remote branch moving later.
        import shutil
        source_dir = output / "source"; source_dir.mkdir(exist_ok=True)
        for file in Path(__file__).parent.glob("*.py"):
            shutil.copyfile(file, source_dir / file.name)
        write_history(output / "history.csv", history)
    stopped = [False]
    def request_stop(signum, frame):
        stopped[0] = True
        logger.warning("Stop requested; saving after the current epoch")
    previous_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGINT)}
    logger.info("Train images=%d; validation images=%d; params=%d; AMP=%s", len(train_set),
                len(val_set), manifest["parameters"], amp)
    if world > 1:
        dist.barrier()
    logger.info("DDP world_size=%d; per-GPU batch=%d; accumulation=%d; effective batch=%d",
                world, config["batch_size"], config["accumulation"], manifest["effective_batch_size"])
    reason = "completed"
    try:
        for epoch in range(start_epoch, config["epochs"]):
            elapsed = time.monotonic() - started
            estimate = history[-1]["seconds"] if history else 0
            reserve = config.get("export_reserve_seconds", 600)
            if distributed.any_rank(stopped[0] or (max_hours and elapsed + estimate + reserve >= max_hours * 3600), device):
                reason = "time_budget_or_signal"; break
            epoch_started = time.monotonic()
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train(); optimizer.zero_grad(set_to_none=True)
            totals = {k: 0.0 for k in ("loss", "xy", "wh", "object", "no_object", "class")}
            seen = 0
            epoch_lr = optimizer.param_groups[0]["lr"]
            for batch_index, (images, targets) in enumerate(train_loader):
                images = images.to(device, non_blocking=True)
                grids = torch.stack([t["grid"] for t in targets]).to(device)
                ignored = torch.stack([t["ignore"] for t in targets]).to(device)
                group_start = (batch_index // config["accumulation"]) * config["accumulation"]
                group_samples = min(config["accumulation"] * config["batch_size"],
                                    len(train_loader.sampler) - group_start * config["batch_size"])
                boundary = ((batch_index + 1) % config["accumulation"] == 0 or
                            batch_index + 1 == len(train_loader))
                sync_context = model.no_sync() if world > 1 and not boundary else nullcontext()
                with sync_context:
                    with torch.autocast(device_type=device.type, enabled=amp):
                        prediction = model(images)
                        loss, terms = criterion(prediction, grids, ignored)
                    if distributed.any_rank(not bool(torch.isfinite(loss)), device):
                        raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
                    scaler.scale(loss * images.shape[0] / group_samples).backward()
                if boundary:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("grad_clip", 10),
                                                   error_if_nonfinite=True)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                n = images.shape[0]; seen += n
                for key, value in {"loss": float(loss.detach()), **terms}.items():
                    totals[key] += value * n
                if batch_index % config.get("log_every", 50) == 0:
                    logger.info("epoch=%d batch=%d/%d loss=%.5f", epoch + 1, batch_index + 1,
                                len(train_loader), float(loss.detach()))
            # Rank zero evaluates the full validation set with the unwrapped module.
            # No DistributedSampler padding can duplicate validation images.
            metrics = evaluate_model(bare_model, val_loader, device, config)[0] if primary else None
            metrics = distributed.broadcast(metrics)
            packed = torch.tensor([totals[k] for k in totals] + [seen], dtype=torch.float64, device=device)
            if world > 1:
                dist.all_reduce(packed)
            totals = dict(zip(totals, packed[:-1].tolist()))
            seen = int(packed[-1].item())
            scheduler.step()
            row = {"epoch": epoch + 1, "train_loss": totals["loss"] / seen,
                   "val_loss": metrics["loss"], "map50": metrics["map50"], "lr": epoch_lr,
                   "seconds": time.monotonic() - epoch_started,
                   **{k: totals[k] / seen for k in totals if k != "loss"}}
            history.append(row)
            if primary:
                write_history(output / "history.csv", history)
            improved = metrics["map50"] > best_map
            best_map = max(best_map, metrics["map50"])
            base = {"format": "personal-yolov1-v1", "model": bare_model.state_dict(), "config": config,
                "epoch": epoch, "global_step": global_step, "best_map": best_map,
                "pretrained": pretrained_metadata, "split_ids": manifest["split_ids"]}
            states = distributed.gather_states({"rng": rng_state(), "loader_rng": generator.get_state()})
            if primary:
                if improved:
                    atomic_save(base, output / "checkpoints/best.pt")
                atomic_save({**base, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                             "scaler": scaler.state_dict(), "rng": states[0]["rng"],
                             "loader_rng": states[0]["loader_rng"], "rank_states": states, "world_size": world,
                             "history": history}, output / "checkpoints/last.pt")
            logger.info("Epoch %d: train=%.5f val=%.5f mAP50=%.5f seconds=%.1f", epoch + 1,
                        row["train_loss"], row["val_loss"], row["map50"], row["seconds"])
            if distributed.any_rank(stopped[0], device):
                reason = "signal"; break
        # Release training allocations before loading a potentially different best checkpoint.
        if history and len(history) > start_epoch:
            del base, prediction, loss
        del optimizer, scheduler, scaler, model, bare_model, groups, old_params, new_params, parameter
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if primary:
            best_path = output / "checkpoints/best.pt"
            if best_path.is_file():
                best_model, best_checkpoint = load_detector(best_path, device)
                metrics, predictions = evaluate_model(best_model, val_loader, device, config)
                metrics.update({"split": manifest["evaluation_split"],
                                "checkpoint_epoch": best_checkpoint["epoch"] + 1})
                save_metrics(metrics, predictions, output)
                save_dataset_examples(val_set, predictions, output / "examples",
                                      config.get("example_count", 10), config.get("demo_threshold", 0.25))
                del best_model, best_checkpoint
            elif history:
                # If resuming last.pt alone with no new improvement, evaluate last as an explicit fallback.
                last_model, last_checkpoint = load_detector(output / "checkpoints/last.pt" if
                    (output / "checkpoints/last.pt").exists() else resume, device)
                metrics, predictions = evaluate_model(last_model, val_loader, device, config)
                metrics.update({"split": manifest["evaluation_split"], "checkpoint": "last (best unavailable)",
                                "checkpoint_epoch": last_checkpoint["epoch"] + 1})
                save_metrics(metrics, predictions, output)
            manifest.update({"status": reason, "completed_epochs": history[-1]["epoch"] if history else 0,
                             "best_map50": best_map if history else None,
                             "session_seconds": time.monotonic() - started,
                             "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
                             if device.type == "cuda" else None})
            write_json(output / "run_manifest.json", manifest)
    except BaseException as error:
        manifest.update({"status": "failed", "error": f"{type(error).__name__}: {error}"})
        if primary:
            write_json(output / "run_manifest.json", manifest)
        logger.exception("Training failed; last.pt contains the last completed epoch, if any")
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for handler in list(logger.handlers):
            handler.flush(); handler.close(); logger.removeHandler(handler)
        if zip_path and primary:
            export_results(output, zip_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pretrained")
    parser.add_argument("--resume")
    parser.add_argument("--random-init", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-hours", type=float)
    parser.add_argument("--zip")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.random_init:
        config["require_pretrained"] = False
    device = distributed.initialize(args.device)
    try:
        train_run(config, args.data, args.output, args.pretrained, args.resume, device,
                  args.max_hours, args.zip)
        if distributed.world_size() > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
