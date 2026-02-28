"""
=============================================================================
YOLO Custom Object Detection — Complete Training Pipeline
=============================================================================
Based on Ultralytics YOLOv8/v11
Reference: https://docs.ultralytics.com

This script covers:
  - Dataset structure setup
  - Dataset YAML configuration
  - Full hyperparameter configuration (all categories)
  - Model training
  - Validation and evaluation
  - Inference on images / video / webcam
  - Model export for deployment

Requirements:
    pip install ultralytics
=============================================================================
"""

# ─── IMPORTS ──────────────────────────────────────────────────────────────────

import os
import yaml
from pathlib import Path
from ultralytics import YOLO


# =============================================================================
# STEP 1 — DATASET DIRECTORY STRUCTURE
# =============================================================================
#
# Your dataset folder must follow this structure:
#
#   dataset/
#   ├── images/
#   │   ├── train/        ← training images (.jpg / .png)
#   │   ├── val/          ← validation images
#   │   └── test/         ← (optional) test images
#   └── labels/
#       ├── train/        ← YOLO-format label .txt files
#       ├── val/
#       └── test/
#
# Each label .txt file contains one row per object:
#   <class_id> <x_center> <y_center> <width> <height>
#   All values are normalized between 0.0 and 1.0
#
# Example label line:
#   0 0.512 0.483 0.230 0.415
#
# =============================================================================


def create_dataset_yaml(
    dataset_path: str,
    class_names: list,
    yaml_save_path: str = "dataset.yaml"
) -> str:
    """
    Creates the dataset YAML configuration file required by YOLO.

    Args:
        dataset_path  : Absolute path to the root dataset directory.
        class_names   : List of class name strings in order of their IDs.
                        Example: ['cat', 'dog', 'car']
        yaml_save_path: Where to save the generated YAML file.

    Returns:
        Path to the saved YAML file.
    """

    dataset_config = {
        "path"  : str(Path(dataset_path).resolve()),  # root dataset directory
        "train" : "images/train",                      # relative path to train images
        "val"   : "images/val",                        # relative path to val images
        "test"  : "images/test",                       # relative path to test images (optional)
        "nc"    : len(class_names),                    # number of classes
        "names" : class_names                          # list of class name strings
    }

    with open(yaml_save_path, "w") as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)

    print(f"[INFO] Dataset YAML saved to: {yaml_save_path}")
    print(f"[INFO] Classes ({len(class_names)}): {class_names}")
    return yaml_save_path


# =============================================================================
# STEP 2 — HYPERPARAMETER CONFIGURATION
# =============================================================================
#
# All hyperparameters are documented below with their default values,
# what they control, and the recommended best values for production training.
#
# Categories:
#   A. Training Core
#   B. Learning Rate
#   C. Loss Function
#   D. Augmentation
#   E. Validation
#
# =============================================================================


TRAIN_HYPERPARAMS = {

    # ── A. TRAINING CORE HYPERPARAMETERS ────────────────────────────────────

    # epochs (Default: 100)
    # Total complete passes through the dataset. More epochs = more learning
    # but risk of overfitting. Always pair with patience for early stopping.
    # Best Value: 100–150 for standard datasets.
    "epochs": 150,

    # patience (Default: 50)
    # Stops training if validation mAP does not improve for N consecutive epochs.
    # Acts as an automatic brake to save compute and prevent overfitting.
    # Best Value: 50 as a safe default. Use 30 for faster experimentation.
    "patience": 50,

    # batch (Default: 16)
    # Number of images per training step. Larger = more stable gradients
    # but requires more GPU memory. Use -1 to let YOLO pick automatically.
    # Best Value: 32 or 64 depending on GPU VRAM. Use -1 for AutoBatch.
    "batch": 32,

    # imgsz (Default: 640)
    # All images are resized to this resolution before training.
    # Larger = better small object detection but much slower training.
    # Best Value: 640 for general use. 1280 only for tiny object datasets.
    "imgsz": 640,

    # save (Default: True)
    # Saves best.pt and last.pt checkpoints during training.
    # Always keep True to prevent losing training progress.
    # Best Value: Always True.
    "save": True,

    # save_period (Default: -1)
    # Saves an extra checkpoint every N epochs as a backup recovery point.
    # -1 saves only best.pt and last.pt. Use 10 for long training runs.
    # Best Value: 10 for runs over 100 epochs.
    "save_period": 10,

    # cache (Default: False)
    # Preloads images into RAM or disk to eliminate per-epoch disk reads.
    # Options: False | "ram" | "disk"
    # Best Value: "ram" if memory is sufficient (~1 GB per 1000 images at 640px).
    "cache": "disk",

    # device (Default: None)
    # Hardware to train on. None = auto-detect GPU.
    # Options: None | 0 | "0,1" | "cpu" | "mps"
    # Best Value: 0 for single GPU. "0,1" for multi-GPU.
    "device": 0,

    # workers (Default: 8)
    # CPU threads for parallel image loading. More = GPU stays fed with data.
    # Best Value: 8 on Linux/Mac. 4 on Windows. 0 to debug dataloader errors.
    "workers": 8,

    # pretrained (Default: True)
    # Start from COCO-pretrained weights instead of random initialization.
    # Dramatically reduces training time and improves final accuracy.
    # Best Value: Always True. Transfer learning almost always wins.
    "pretrained": True,

    # optimizer (Default: "auto")
    # Weight update algorithm. SGD generalizes well. AdamW better for fine-tuning.
    # Options: "auto" | "SGD" | "Adam" | "AdamW" | "RMSProp"
    # Best Value: "SGD" or "auto" for scratch training. "AdamW" for fine-tuning.
    "optimizer": "SGD",

    # seed (Default: 0)
    # Fixes all random operations for fully reproducible training runs.
    # Best Value: 0 for reproducibility. Change to verify variance across runs.
    "seed": 0,

    # deterministic (Default: True)
    # Forces identical GPU results every run when combined with fixed seed.
    # Small speed cost. Use False in production for marginal speed gain.
    # Best Value: True during research. False in final production runs.
    "deterministic": True,

    # single_cls (Default: False)
    # Treats all classes as one. Only use when class identity does not matter.
    # Best Value: False for multi-class detection tasks.
    "single_cls": False,

    # rect (Default: False)
    # Batches images by aspect ratio to reduce padding waste.
    # Incompatible with mosaic augmentation.
    # Best Value: False for standard training. True for non-square image datasets.
    "rect": False,

    # cos_lr (Default: False)
    # Applies cosine annealing LR schedule instead of linear decay.
    # Produces smoother convergence and better final accuracy.
    # Best Value: True for runs of 100+ epochs.
    "cos_lr": True,

    # close_mosaic (Default: 10)
    # Disables mosaic augmentation for the final N epochs so the model
    # finalizes on clean, realistic images before training ends.
    # Best Value: 10–15. Never set to 0.
    "close_mosaic": 10,

    # resume (Default: False)
    # Resumes from last.pt if training was previously interrupted.
    # Best Value: True only when recovering from a crashed training session.
    "resume": False,

    # amp (Default: True)
    # Automatic Mixed Precision — uses FP16 to speed up training ~2x.
    # Requires no accuracy trade-off. Disable only if NaN losses appear.
    # Best Value: Always True.
    "amp": True,

    # fraction (Default: 1.0)
    # Fraction of training dataset to use. 1.0 = use everything.
    # Best Value: 1.0 for production. Lower only for quick experiments.
    "fraction": 1.0,

    # freeze (Default: None)
    # Freezes first N backbone layers so their weights don't update.
    # Use for transfer learning on small custom datasets.
    # Best Value: None to train all layers with a full dataset.
    "freeze": None,

    # dropout (Default: 0.0)
    # Randomly deactivates neurons to prevent overfitting.
    # Best Value: 0.0 for most tasks. 0.1–0.2 only if clearly overfitting.
    "dropout": 0.0,

    # multi_scale (Default: False)
    # Randomly resizes images between 0.5x–1.5x each batch.
    # Improves scale robustness at ~30% extra training time cost.
    # Best Value: True if objects vary greatly in size across the dataset.
    "multi_scale": False,

    # ── B. LEARNING RATE HYPERPARAMETERS ────────────────────────────────────

    # lr0 (Default: 0.01)
    # Initial learning rate — size of first weight update steps.
    # Most important hyperparameter. Tune this first above all others.
    # Best Value: 0.01 for SGD from scratch. 0.001 for fine-tuning.
    "lr0": 0.01,

    # lrf (Default: 0.01)
    # Final LR as a fraction of lr0. Actual final LR = lr0 × lrf.
    # Best Value: 0.01 gives a 100x reduction by the final epoch.
    "lrf": 0.01,

    # momentum (Default: 0.937)
    # Carries gradient direction history forward to smooth updates.
    # Best Value: 0.937 is optimized for YOLO. Rarely needs changing.
    "momentum": 0.937,

    # weight_decay (Default: 0.0005)
    # Penalizes large weights to prevent overfitting.
    # Best Value: 0.0005 standard. Increase to 0.001 if overfitting.
    "weight_decay": 0.0005,

    # warmup_epochs (Default: 3.0)
    # Ramps LR from near zero to lr0 over the first N epochs.
    # Prevents destructive large updates at the start of training.
    # Best Value: 3.0 default. Increase to 5.0 for large or complex models.
    "warmup_epochs": 3.0,

    # warmup_momentum (Default: 0.8)
    # Initial momentum during warmup — builds up gradually to full momentum.
    # Best Value: 0.8 is standard and rarely needs adjustment.
    "warmup_momentum": 0.8,

    # warmup_bias_lr (Default: 0.1)
    # Higher dedicated LR for bias parameters during warmup phase only.
    # Best Value: 0.1 standard. Leave unchanged in most configurations.
    "warmup_bias_lr": 0.1,

    # nbs (Default: 64)
    # Nominal batch size reference for automatic LR scaling.
    # LR is scaled proportionally when actual batch != 64.
    # Best Value: 64 standard. Change only for custom LR scaling scenarios.
    "nbs": 64,

    # ── C. LOSS FUNCTION HYPERPARAMETERS ────────────────────────────────────

    # box (Default: 7.5)
    # Weight on bounding box regression loss.
    # Higher = model prioritizes spatial accuracy of predicted boxes.
    # Best Value: 7.5 default. Increase to 10–12 for precision-critical apps.
    "box": 7.5,

    # cls (Default: 0.5)
    # Weight on classification loss.
    # Higher = model prioritizes getting class labels correct.
    # Best Value: 0.5 standard. Increase to 1.0 if classification is primary.
    "cls": 0.5,

    # dfl (Default: 1.5)
    # Weight on Distribution Focal Loss for box edge probability distributions.
    # Only applies to YOLOv8 and later (anchor-free models).
    # Best Value: 1.5 default. Slightly increase for precise edge localization.
    "dfl": 1.5,

    # label_smoothing (Default: 0.0)
    # Softens hard 0/1 labels to prevent overconfident predictions.
    # Best Value: 0.0 for most tasks. Try 0.01–0.05 if model is overconfident.
    "label_smoothing": 0.0,

    # ── D. AUGMENTATION HYPERPARAMETERS ─────────────────────────────────────

    # hsv_h (Default: 0.015)
    # Random hue shift fraction. Teaches color tone invariance.
    # Best Value: 0.015 standard. Increase to 0.03 for varied lighting datasets.
    "hsv_h": 0.015,

    # hsv_s (Default: 0.7)
    # Random saturation change. Simulates different lighting intensities.
    # Best Value: 0.7 standard. Reduce to 0.3–0.5 for controlled environments.
    "hsv_s": 0.7,

    # hsv_v (Default: 0.4)
    # Random brightness change. Simulates day/night and indoor/outdoor lighting.
    # Best Value: 0.4 standard. Increase to 0.6 for variable-light deployments.
    "hsv_v": 0.4,

    # degrees (Default: 0.0)
    # Random image rotation range in degrees.
    # Best Value: 0.0 for upright imagery. 10–45 for aerial or drone datasets.
    "degrees": 0.0,

    # translate (Default: 0.1)
    # Random horizontal and vertical image shift as fraction of image size.
    # Best Value: 0.1 standard. Increase to 0.2 for edge-of-frame objects.
    "translate": 0.1,

    # scale (Default: 0.5)
    # Random zoom in/out. Teaches the model scale invariance.
    # Best Value: 0.5 provides ±50% zoom. Reduce for fixed-size object datasets.
    "scale": 0.5,

    # shear (Default: 0.0)
    # Random slanting distortion in degrees. Simulates tilted cameras.
    # Best Value: 0.0 for frontal imagery. 2–5 for angled camera datasets.
    "shear": 0.0,

    # perspective (Default: 0.0)
    # Random 3D perspective warp. Simulates different camera viewpoints.
    # Best Value: 0.0 standard. Use 0.0001–0.0005 for vehicle detection tasks.
    "perspective": 0.0,

    # flipud (Default: 0.0)
    # Probability of vertical (upside-down) flip.
    # Best Value: 0.0 for ground-level imagery. 0.5 for aerial/overhead datasets.
    "flipud": 0.0,

    # fliplr (Default: 0.5)
    # Probability of horizontal mirror flip. Effectively doubles dataset diversity.
    # Best Value: 0.5 for most datasets. 0.0 for direction-sensitive tasks.
    "fliplr": 0.5,

    # mosaic (Default: 1.0)
    # Combines 4 images into a 2x2 grid. Most impactful YOLO augmentation.
    # Teaches multi-scale and multi-context detection simultaneously.
    # Best Value: 1.0 always. close_mosaic handles end-of-training disabling.
    "mosaic": 1.0,

    # mixup (Default: 0.0)
    # Blends two images together. Teaches calibrated uncertain predictions.
    # Combined with mosaic = most powerful augmentation combo in YOLO.
    # Best Value: 0.0–0.1. Add 0.1 if model is overfitting training data.
    "mixup": 0.1,

    # copy_paste (Default: 0.0)
    # Cuts objects from one image and pastes into another.
    # Best Value: 0.0 standard. Use 0.1–0.2 for rare/underrepresented classes.
    "copy_paste": 0.0,

    # erasing (Default: 0.4)
    # Randomly blacks out patches to simulate occlusion.
    # Best Value: 0.4 standard. Reduce to 0.2 for clean, uncluttered scenes.
    "erasing": 0.4,

    # crop_fraction (Default: 1.0)
    # Fraction of image to crop during classification augmentation.
    # Best Value: 1.0 for detection tasks. 0.8–0.9 for classification tasks.
    "crop_fraction": 1.0,

    # ── E. VALIDATION HYPERPARAMETERS ───────────────────────────────────────

    # val (Default: True)
    # Run validation after every epoch. Required for early stopping and
    # best checkpoint saving. Always keep True.
    "val": True,

    # plots (Default: True)
    # Auto-generates training curves, confusion matrix, and PR curves.
    # Essential for diagnosing training behavior. Always keep True.
    "plots": True,

    # save_json (Default: False)
    # Saves predictions in COCO JSON format for official benchmark evaluation.
    # Best Value: False during training. True for benchmark submissions only.
    "save_json": False,

    # project and name — output directory configuration
    "project": "runs/detect",
    "name"   : "custom_training",

    # exist_ok (Default: False)
    # If True, overwrites existing run directory instead of creating run2, run3...
    "exist_ok": False,

    # verbose (Default: True)
    # Print detailed training logs to console each epoch.
    "verbose": True,
}


# =============================================================================
# STEP 3 — MODEL SELECTION
# =============================================================================
#
# Choose the model size based on your accuracy vs speed requirements:
#
#   Model          Parameters   Speed    Accuracy   Use Case
#   ─────────────────────────────────────────────────────────
#   yolov8n.pt       3.2M       ████     ██         Edge devices, real-time
#   yolov8s.pt       11.2M      ███      ███        Balanced speed/accuracy
#   yolov8m.pt       25.9M      ██       ████       Standard GPU training
#   yolov8l.pt       43.7M      █        █████      High-accuracy applications
#   yolov8x.pt       68.2M      ░        ██████     Maximum accuracy
#
# =============================================================================

MODEL_SIZES = {
    "nano"   : "yolov8n.pt",   # fastest, lowest accuracy
    "small"  : "yolov8s.pt",   # good balance for simple tasks
    "medium" : "yolov8m.pt",   # recommended for most custom datasets
    "large"  : "yolov8l.pt",   # high accuracy, needs strong GPU
    "xlarge" : "yolov8x.pt",   # maximum accuracy
}


# =============================================================================
# STEP 4 — TRAINING FUNCTION
# =============================================================================

def train_yolo(
    dataset_yaml : str,
    model_size   : str = "medium",
    output_dir   : str = "runs/detect",
    run_name     : str = "custom_training"
) -> YOLO:
    """
    Trains a YOLO model on a custom dataset using all configured hyperparameters.

    Args:
        dataset_yaml : Path to the dataset YAML configuration file.
        model_size   : One of 'nano', 'small', 'medium', 'large', 'xlarge'.
        output_dir   : Root directory where training results are saved.
        run_name     : Name of this specific training run folder.

    Returns:
        Trained YOLO model object.
    """

    model_path = MODEL_SIZES.get(model_size, "yolov8m.pt")
    print(f"\n{'='*60}")
    print(f"  YOLO Custom Object Detection — Training")
    print(f"{'='*60}")
    print(f"  Model    : {model_path}")
    print(f"  Dataset  : {dataset_yaml}")
    print(f"  Output   : {output_dir}/{run_name}")
    print(f"{'='*60}\n")

    # Load the pretrained model
    model = YOLO(model_path)

    # Update run metadata in hyperparameters
    TRAIN_HYPERPARAMS["project"] = output_dir
    TRAIN_HYPERPARAMS["name"]    = run_name

    # Start training with all hyperparameters
    results = model.train(
        data=dataset_yaml,
        **TRAIN_HYPERPARAMS
    )

    print(f"\n[INFO] Training complete.")
    print(f"[INFO] Best model saved to: {output_dir}/{run_name}/weights/best.pt")
    print(f"[INFO] Last model saved to: {output_dir}/{run_name}/weights/last.pt")

    return model


# =============================================================================
# STEP 5 — VALIDATION / EVALUATION
# =============================================================================

def validate_model(
    model_path   : str,
    dataset_yaml : str,
    split        : str = "val",
    conf         : float = 0.001,
    iou          : float = 0.6,
    max_det      : int = 300,
    save_json    : bool = False,
    plots        : bool = True
):
    """
    Evaluates a trained model on the validation or test split.

    Args:
        model_path   : Path to trained weights file (best.pt or last.pt).
        dataset_yaml : Path to the dataset YAML file.
        split        : Dataset split to evaluate on: 'val' or 'test'.
        conf         : Confidence threshold for validation detections.
        iou          : IoU threshold for counting a detection as correct.
        max_det      : Maximum detections per image during validation.
        save_json    : Save predictions in COCO JSON format.
        plots        : Generate confusion matrix and PR curve plots.

    Returns:
        Validation metrics object.
    """

    print(f"\n[INFO] Running validation on split='{split}' ...")
    model = YOLO(model_path)

    metrics = model.val(
        data     = dataset_yaml,
        split    = split,      # 'val' during training, 'test' for final evaluation
        conf     = conf,       # low conf for complete mAP calculation
        iou      = iou,        # IoU threshold to count detection as correct
        max_det  = max_det,    # max detections per image
        save_json= save_json,  # True for COCO benchmark submissions
        plots    = plots,      # generate confusion matrix and PR curve
        verbose  = True
    )

    print(f"\n[RESULTS] mAP@0.5     : {metrics.box.map50:.4f}")
    print(f"[RESULTS] mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"[RESULTS] Precision   : {metrics.box.mp:.4f}")
    print(f"[RESULTS] Recall      : {metrics.box.mr:.4f}")

    return metrics


# =============================================================================
# STEP 6 — INFERENCE
# =============================================================================

def run_inference(
    model_path  : str,
    source      : str,
    conf        : float = 0.25,
    iou         : float = 0.7,
    max_det     : int = 300,
    imgsz       : int = 640,
    half        : bool = False,
    agnostic_nms: bool = False,
    augment     : bool = False,
    save        : bool = True,
    save_txt    : bool = False,
    save_conf   : bool = False,
    show        : bool = False,
    classes     : list = None,
    project     : str = "runs/detect",
    name        : str = "predict"
):
    """
    Runs inference on images, video, folder, or webcam.

    Args:
        model_path   : Path to trained weights (best.pt).
        source       : Input source. Options:
                         - Path to image     : "image.jpg"
                         - Path to folder    : "images/"
                         - Path to video     : "video.mp4"
                         - Webcam            : 0
                         - YouTube URL       : "https://youtube.com/..."
                         - Screen capture    : "screen"
        conf         : Minimum confidence to report a detection (0.1–0.9).
        iou          : NMS IoU threshold to remove duplicate boxes (0.3–0.9).
        max_det      : Maximum detections per image.
        imgsz        : Inference image size in pixels.
        half         : Use FP16 for 2x faster inference on NVIDIA GPUs.
        agnostic_nms : Apply NMS across all classes together.
        augment      : Test-time augmentation — slower but more accurate.
        save         : Save annotated output images/video to disk.
        save_txt     : Save detection labels as .txt files.
        save_conf    : Include confidence scores in saved label files.
        show         : Display predictions in a window (requires display).
        classes      : Filter to specific class IDs. None = all classes.
        project      : Root directory for saving results.
        name         : Subfolder name for this inference run.

    Returns:
        List of Results objects with detections.
    """

    print(f"\n[INFO] Running inference on: {source}")
    model = YOLO(model_path)

    results = model.predict(
        source      = source,
        conf        = conf,          # filter out low-confidence detections
        iou         = iou,           # NMS overlap threshold
        max_det     = max_det,       # cap total detections per image
        imgsz       = imgsz,         # resize input to this resolution
        half        = half,          # FP16 for faster GPU inference
        agnostic_nms= agnostic_nms,  # class-agnostic NMS
        augment     = augment,       # TTA for maximum accuracy
        save        = save,          # save annotated output to disk
        save_txt    = save_txt,      # save labels as .txt files
        save_conf   = save_conf,     # include confidence in labels
        show        = show,          # display live in a window
        classes     = classes,       # filter specific class IDs
        project     = project,
        name        = name,
        verbose     = True
    )

    # Print detection summary for each image
    for i, result in enumerate(results):
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            print(f"\n[Image {i+1}] Detections: {len(boxes)}")
            for box in boxes:
                cls_id  = int(box.cls[0])
                conf_val= float(box.conf[0])
                xyxy    = box.xyxy[0].tolist()
                class_name = result.names[cls_id]
                print(f"  - {class_name:20s} | conf: {conf_val:.3f} | box: {[round(v,1) for v in xyxy]}")
        else:
            print(f"\n[Image {i+1}] No detections above conf={conf}")

    print(f"\n[INFO] Results saved to: {project}/{name}/")
    return results


# =============================================================================
# STEP 7 — MODEL EXPORT FOR DEPLOYMENT
# =============================================================================

def export_model(
    model_path : str,
    format     : str  = "onnx",
    imgsz      : int  = 640,
    half       : bool = False,
    int8       : bool = False,
    dynamic    : bool = False,
    simplify   : bool = True,
    opset      : int  = None,
    nms        : bool = False,
    batch      : int  = 1
):
    """
    Exports the trained model to a deployment format.

    Args:
        model_path : Path to trained weights (best.pt).
        format     : Export format. Options:
                       'torchscript' — PyTorch CPU/GPU deployment
                       'onnx'        — Cross-platform universal (recommended)
                       'openvino'    — Intel CPU/VPU hardware
                       'tflite'      — Android mobile devices
                       'coreml'      — iPhone/iPad/Mac (Apple)
                       'engine'      — NVIDIA TensorRT (fastest GPU inference)
                       'paddle'      — PaddlePaddle ecosystem
        imgsz      : Fixed input image size baked into the exported model.
        half       : Export with FP16 weights for faster GPU inference.
        int8       : Quantize to INT8 for ~4x smaller, faster edge models.
        dynamic    : Allow variable batch sizes and image dimensions in ONNX.
        simplify   : Remove redundant nodes from ONNX graph (recommended).
        opset      : ONNX opset version. None = auto-select.
        nms        : Bake NMS post-processing into the exported model graph.
        batch      : Static batch size for non-dynamic exports.

    Returns:
        Path to the exported model file.
    """

    print(f"\n[INFO] Exporting model to format='{format}' ...")
    model = YOLO(model_path)

    export_path = model.export(
        format  = format,    # deployment target format
        imgsz   = imgsz,     # fixed input resolution
        half    = half,      # FP16 for GPU deployment
        int8    = int8,      # INT8 for edge/mobile deployment
        dynamic = dynamic,   # variable input shapes in ONNX
        simplify= simplify,  # clean up ONNX graph
        opset   = opset,     # ONNX opset version
        nms     = nms,       # embed NMS into model
        batch   = batch      # static batch size
    )

    print(f"[INFO] Model exported to: {export_path}")
    return export_path


# =============================================================================
# STEP 8 — RESUME INTERRUPTED TRAINING
# =============================================================================

def resume_training(last_checkpoint: str):
    """
    Resumes a previously interrupted training run from the last checkpoint.

    Args:
        last_checkpoint : Path to last.pt from the interrupted training run.
                          Example: "runs/detect/custom_training/weights/last.pt"
    """

    print(f"\n[INFO] Resuming training from: {last_checkpoint}")
    model = YOLO(last_checkpoint)

    results = model.train(resume=True)

    print("[INFO] Training resumed and completed.")
    return model


# =============================================================================
# STEP 9 — MAIN PIPELINE
# =============================================================================

def main():
    """
    Complete end-to-end YOLO custom object detection pipeline.
    Edit the configuration below before running.
    """

    # ── USER CONFIGURATION ───────────────────────────────────────────────────

    # Path to your dataset root directory
    DATASET_PATH = "/path/to/your/dataset"

    # Your class names in order of their integer IDs (starting from 0)
    CLASS_NAMES = [
        "class_0",
        "class_1",
        "class_2",
        # Add all your classes here
    ]

    # Path where the dataset YAML will be saved
    YAML_PATH = "dataset.yaml"

    # Model size: 'nano' | 'small' | 'medium' | 'large' | 'xlarge'
    MODEL_SIZE = "medium"

    # Training output directory and run name
    OUTPUT_DIR = "runs/detect"
    RUN_NAME   = "custom_training_v1"

    # Source for inference after training (image, folder, video, or 0 for webcam)
    INFERENCE_SOURCE = "test_images/"

    # Export format after training
    EXPORT_FORMAT = "onnx"

    # ── PIPELINE EXECUTION ───────────────────────────────────────────────────

    # 1. Create dataset YAML configuration
    print("\n[STEP 1] Creating dataset configuration...")
    yaml_path = create_dataset_yaml(
        dataset_path  = DATASET_PATH,
        class_names   = CLASS_NAMES,
        yaml_save_path= YAML_PATH
    )

    # 2. Train the model
    print("\n[STEP 2] Starting training...")
    model = train_yolo(
        dataset_yaml = yaml_path,
        model_size   = MODEL_SIZE,
        output_dir   = OUTPUT_DIR,
        run_name     = RUN_NAME
    )

    # Path to the best trained model weights
    best_weights = f"{OUTPUT_DIR}/{RUN_NAME}/weights/best.pt"

    # 3. Validate on validation split
    print("\n[STEP 3] Validating on validation split...")
    val_metrics = validate_model(
        model_path   = best_weights,
        dataset_yaml = yaml_path,
        split        = "val"
    )

    # 4. Final evaluation on test split (holdout — run only once)
    print("\n[STEP 4] Final evaluation on test split...")
    test_metrics = validate_model(
        model_path   = best_weights,
        dataset_yaml = yaml_path,
        split        = "test"
    )

    # 5. Run inference on new images
    print("\n[STEP 5] Running inference...")
    results = run_inference(
        model_path = best_weights,
        source     = INFERENCE_SOURCE,
        conf       = 0.25,
        iou        = 0.7,
        save       = True
    )

    # 6. Export model for deployment
    print("\n[STEP 6] Exporting model...")
    export_path = export_model(
        model_path = best_weights,
        format     = EXPORT_FORMAT,
        imgsz      = 640,
        simplify   = True
    )

    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Best Weights : {best_weights}")
    print(f"  Exported To  : {export_path}")
    print(f"  Val mAP@0.5  : {val_metrics.box.map50:.4f}")
    print(f"  Test mAP@0.5 : {test_metrics.box.map50:.4f}")
    print(f"{'='*60}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()


# =============================================================================
# QUICK USAGE EXAMPLES
# =============================================================================
#
# ── Train from scratch ───────────────────────────────────────────────────────
#
#   model = train_yolo(
#       dataset_yaml = "dataset.yaml",
#       model_size   = "medium",
#       run_name     = "experiment_1"
#   )
#
# ── Resume interrupted training ──────────────────────────────────────────────
#
#   resume_training("runs/detect/experiment_1/weights/last.pt")
#
# ── Validate trained model ───────────────────────────────────────────────────
#
#   validate_model(
#       model_path   = "runs/detect/experiment_1/weights/best.pt",
#       dataset_yaml = "dataset.yaml",
#       split        = "val"
#   )
#
# ── Run inference on images ───────────────────────────────────────────────────
#
#   run_inference(
#       model_path = "best.pt",
#       source     = "test_images/",
#       conf       = 0.25,
#       save       = True
#   )
#
# ── Run inference on webcam ───────────────────────────────────────────────────
#
#   run_inference(
#       model_path = "best.pt",
#       source     = 0,        # 0 = default webcam
#       conf       = 0.4,
#       show       = True
#   )
#
# ── Export to ONNX ───────────────────────────────────────────────────────────
#
#   export_model("best.pt", format="onnx", simplify=True)
#
# ── Export to TensorRT (fastest GPU inference) ───────────────────────────────
#
#   export_model("best.pt", format="engine", half=True)
#
# ── Export to TFLite (Android mobile) ────────────────────────────────────────
#
#   export_model("best.pt", format="tflite", int8=True)
#
# =============================================================================