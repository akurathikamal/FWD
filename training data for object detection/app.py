


# ------------------------roboflow like dataset convertion---------------
import os
import shutil
import random
import yaml  # for creating data.yaml

# -----------------------
# Paths
# -----------------------
original_path = r"/media/fwd/UBUNTU 22_0/DJI_DRONE_FEB21_COLLECTION/Full_overall_dataset"   # your original dataset with "images" and "labels"
new_dataset_path = r"/media/fwd/UBUNTU 22_0/DJI_DRONE_FEB21_COLLECTION/roboflowdataset"      # new dataset split will be saved here

# -----------------------
# Split ratios
# -----------------------
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1  # optional, can set to 0 if not needed

# -----------------------
# Make new dataset structure
# -----------------------
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(new_dataset_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(new_dataset_path, split, "labels"), exist_ok=True)

# -----------------------
# Collect images
# -----------------------
images_path = os.path.join(original_path, "images")
labels_path = os.path.join(original_path, "labels")
all_images = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

random.shuffle(all_images)

# -----------------------
# Split sizes
# -----------------------
total = len(all_images)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

train_files = all_images[:train_end]
val_files = all_images[train_end:val_end]
test_files = all_images[val_end:]

# -----------------------
# Copy helper
# -----------------------
def move_files(files, split):
    for img in files:
        img_src = os.path.join(images_path, img)
        label_src = os.path.join(labels_path, os.path.splitext(img)[0] + ".txt")

        img_dst = os.path.join(new_dataset_path, split, "images", img)
        label_dst = os.path.join(new_dataset_path, split, "labels", os.path.splitext(img)[0] + ".txt")

        shutil.copy(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.copy(label_src, label_dst)



# -----------------------
# Copy all files
# -----------------------
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print(f"✅ Split completed! New dataset saved in: {new_dataset_path}")
print(f"Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# -----------------------
# Create data.yaml
# -----------------------
# ⚠ Update class names here:
class_names = ["drone"]  # example, replace with your actual classes

data_yaml = {
    'train': os.path.join(new_dataset_path, "train", "images").replace("\\", "/"),
    'val': os.path.join(new_dataset_path, "val", "images").replace("\\", "/"),
    'test': os.path.join(new_dataset_path, "test", "images").replace("\\", "/"),
    'nc': len(class_names),
    'names': class_names
}

yaml_path = os.path.join(new_dataset_path, "data.yaml")
with open(yaml_path, 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print(f"📂 data.yaml created at: {yaml_path}")













"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║        AIRBORNE DRONE DETECTION  —  Full Training Pipeline                       ║
║  ─────────────────────────────────────────────────────────────────────────────   ║
║  Task    : Detect moving drones from a moving drone (drone-vs-drone)             ║
║  Camera  : 1280×720 @ 120 fps                                                    ║
║  Model   : YOLOv8s  (best accuracy/speed tradeoff on Jetson AGX/Orin)            ║
║  Train HW: RTX 5070 Ti  16 GB VRAM  |  25-core CPU                               ║
║  Deploy  : Jetson → TensorRT FP16 engine                                         ║
║  Tracking: MLflow                                                                ║
║  Augment : Albumentations  (night, motion-blur, noise, atmospheric)              ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

# ── stdlib ─────────────────────────────────────────────────────────────────────
import os, sys, shutil, random, logging, warnings, argparse, time
from pathlib import Path
from datetime import datetime

# ── 3rd-party ──────────────────────────────────────────────────────────────────
import cv2, yaml, numpy as np
import torch

import albumentations as A
from albumentations.pytorch import ToTensorV2

import mlflow, mlflow.pytorch
from ultralytics import YOLO

warnings.filterwarnings("ignore")
logging.basicConfig(
    level   = logging.INFO,
    format  = "[%(asctime)s | %(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("drone_train")


# ══════════════════════════════════════════════════════════════════════════════
#  ①  HARDWARE-AWARE CONFIGURATION
#     RTX 5070 Ti  →  16 384 MB VRAM, CUDA 13.1, driver 590
#     25 CPU cores  →  workers=16 (leave cores for OS / dataloader overhead)
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # ── Camera / Input ────────────────────────────────────────────────────────
    cam_w           = 1280,
    cam_h           = 720,
    imgsz           = 640  ,    # keep full res – 16 GB can handle it
                                 # set to 640 if you want faster iteration

    # ── Model ─────────────────────────────────────────────────────────────────
    # yolov8s  — best balance for Jetson deployment
    # switch to yolov8m for more accuracy if Jetson is AGX Orin
    model           = "yolo26n.pt",
    nc              = 1,
    class_names     = ["drone"],

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset_root    = "/media/fwd/UBUNTU 22_0/DJI_DRONE_FEB21_COLLECTION/roboflowdataset",
    data_yaml       = "/media/fwd/UBUNTU 22_0/DJI_DRONE_FEB21_COLLECTION/roboflowdataset/data.yaml",


    # ── Training — tuned for RTX 5070 Ti ─────────────────────────────────────
    epochs          = 1000,
    batch           = 16,        # 32 fits in 16 GB at imgsz=1280 with AMP
    workers         = 16,        # 25 cores → 16 safe for dataloader
    device          = "0",       # single RTX 5070 Ti
    optimizer       = "AdamW",
    lr0             = 0.0001,
    lrf             = 0.005,     # final lr = lr0 × lrf
    momentum        = 0.937,
    weight_decay    = 5e-4,
    warmup_epochs   = 5,
    warmup_bias_lr  = 0.1,
    cos_lr          = True,
    amp             = True,      # automatic mixed precision (FP16) — RTX 5070 Ti
    cache           = "ram",     # cache dataset in RAM (25-core system has plenty)
    patience        = 60,
    save_period     = 25,

    # ── Loss weights (small fast objects need strong box loss) ────────────────
    box             = 9.5,       # ↑ for tiny drones
    cls             = 0.3,
    dfl             = 1.5,

    # ── Small-object tricks ───────────────────────────────────────────────────
    multi_scale     = False,      # random scale 0.5×–1.5× per batch
    overlap_mask    = False,

    # ── Output ────────────────────────────────────────────────────────────────
    project         = "/home/fwd/Documents/FEB_21_notincludefirstdatasetimg_DJIDRONESTRAINED",
    name            = f"exp_{datetime.now().strftime('%Y%m%d_%H%M')}",

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow_uri = "mlruns",
    mlflow_exp      = "Drone_vs_Drone_Detection",

    # ── Jetson Export ─────────────────────────────────────────────────────────
    jetson_format   = "engine",  # TensorRT
    jetson_fp16     = True,
    jetson_imgsz    = 640,       # smaller for Jetson inference speed
    jetson_workspace= 4,         # GB TRT builder memory
)


# ══════════════════════════════════════════════════════════════════════════════
#  ②  ALBUMENTATIONS AUGMENTATION PIPELINES
#
#  Design rationale for drone-vs-drone @ night:
#    A) MOTION BLUR       — both platforms moving, fast shutter/ISO tradeoffs
#    B) LOW-LIGHT / NIGHT — stars, IR illuminators, searchlights
#    C) SENSOR NOISE      — high-ISO, thermal-like rolling noise
#    D) ATMOSPHERIC       — haze, fog, rain streaks at altitude
#    E) GEOMETRIC         — rolling/banking platform, vibration, wind
#    F) JPEG/STREAM       — video encoding artefacts over telemetry link
# ══════════════════════════════════════════════════════════════════════════════

def build_train_transform() -> A.Compose:
    bp = A.BboxParams(
        format         = "yolo",           # normalised [cx, cy, w, h]
        label_fields   = ["class_labels"],
        min_area       = 9,                # drop boxes < 3×3 px after crop
        min_visibility = 0.20,
    )
    return A.Compose([

        # ── A. MOTION BLUR ─────────────────────────────────────────────────
        # Drones move 5–30 m/s  →  strong directional blur at 120 fps
        A.OneOf([
            A.MotionBlur(blur_limit=(7, 25), p=1.0),          # linear motion
            A.ZoomBlur(max_factor=(1.0, 1.10), p=1.0),        # approach/recede
            A.Blur(blur_limit=(3, 7), p=1.0),                 # mild general
        ], p=0.65),

        # ── B. NIGHT / LOW-LIGHT ───────────────────────────────────────────
        A.OneOf([
            # Deep night: very dark frame + boosted contrast
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.65, -0.25),
                    contrast_limit=(0.1, 0.5), p=1.0),
                A.RandomGamma(gamma_limit=(20, 70), p=1.0),
            ]),
            # Dusk / twilight
            A.RandomBrightnessContrast(
                brightness_limit=(-0.35, -0.05),
                contrast_limit=(-0.1, 0.3), p=1.0),
            # Night-vision / thermal-like (desaturate + equalize)
            A.Compose([
                A.ToGray(p=1.0),
                A.CLAHE(clip_limit=6.0, tile_grid_size=(4, 4), p=1.0),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.4, 0.0),
                    contrast_limit=(0.2, 0.5), p=1.0),
            ]),
            # IR illuminator glow (slightly blue/green tint + bright centre)
            A.Compose([
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.5, -0.1),
                    contrast_limit=(0.0, 0.4), p=1.0),
                A.HueSaturationValue(
                    hue_shift_limit=15,
                    sat_shift_limit=(-40, -10),
                    val_shift_limit=(-20, 10), p=1.0),
            ]),
        ], p=0.55),

        # ── C. SENSOR NOISE ────────────────────────────────────────────────
        # High-ISO rolling noise + fixed pattern + hot pixels
        A.OneOf([
            A.GaussNoise(var_limit=(30.0, 150.0), mean=0, p=1.0),
            A.ISONoise(color_shift=(0.01, 0.06), intensity=(0.15, 0.55), p=1.0),
            A.MultiplicativeNoise(
                multiplier=(0.80, 1.20), per_channel=True,
                elementwise=True, p=1.0),
        ], p=0.55),

        # ── D. ATMOSPHERIC (altitude weather) ─────────────────────────────
        A.OneOf([
            A.RandomFog(
                fog_coef_lower=0.04, fog_coef_upper=0.25,
                alpha_coef=0.10, p=1.0),
            # A.RandomRain(
            #     slant_lower=-15, slant_upper=15,
            #     drop_length=15, drop_width=1,
            #     drop_color=(180, 180, 180),
            #     blur_value=3, brightness_coefficient=0.85,
            #     rain_type=None, p=1.0),
            A.RandomRain(
                slant_lower=-15, slant_upper=15,
                drop_length=15, drop_width=1,
                drop_color=(180, 180, 180),
                blur_value=3, brightness_coefficient=0.85,
                rain_type="drizzle", p=1.0),   # ← fixed
            A.RandomSunFlare(
                flare_roi=(0, 0, 1, 0.4),
                angle_lower=0.0, angle_upper=1.0,
                num_flare_circles_lower=2, num_flare_circles_upper=5,
                src_radius=100, src_color=(255, 220, 160), p=1.0),
            A.RandomSnow(
                snow_point_lower=0.02, snow_point_upper=0.12,
                brightness_coeff=1.5, p=1.0),
        ], p=0.25),

        # ── E. GEOMETRIC (moving platform) ────────────────────────────────
        # Camera roll / bank angle
        A.Rotate(limit=20, border_mode=cv2.BORDER_CONSTANT,
                 value=114, p=0.45),
        # Platform vibration / wind
        A.Affine(
            translate_percent={"x": (-0.06, 0.06), "y": (-0.06, 0.06)},
            shear=(-6, 6), p=0.35),
        # Perspective (3D parallax as both drones manoeuvre)
        A.Perspective(scale=(0.02, 0.07), p=0.30),
        # Flip – drone can appear from any side
        A.HorizontalFlip(p=0.50),
        A.VerticalFlip(p=0.10),

        # ── F. STREAM / ENCODING ARTEFACTS ────────────────────────────────
        A.ImageCompression(quality_lower=40, quality_upper=90, p=0.30),
        A.Defocus(radius=(1, 4), alias_blur=(0.1, 0.5), p=0.15),

        # ── G. OCCLUSION (partial cloud, bird, rotor wash) ─────────────────
        A.CoarseDropout(
            max_holes=8, max_height=40, max_width=40,
            min_holes=1, min_height=8, min_width=8,
            fill_value=114, p=0.20),

        # ── H. COLOUR JITTER ───────────────────────────────────────────────
        A.HueSaturationValue(
            hue_shift_limit=12, sat_shift_limit=35, val_shift_limit=25,
            p=0.35),
        A.RGBShift(r_shift_limit=15, g_shift_limit=10,
                   b_shift_limit=15, p=0.25),

    ], bbox_params=bp)


def build_val_transform() -> A.Compose:
    bp = A.BboxParams(
        format       = "yolo",
        label_fields = ["class_labels"],
        min_area     = 4,
        min_visibility = 0.1,
    )
    # Val: only resize/pad — no augmentation
    return A.Compose([
        A.LongestMaxSize(max_size=CFG["imgsz"]),
        A.PadIfNeeded(
            min_height=CFG["imgsz"], min_width=CFG["imgsz"],
            border_mode=cv2.BORDER_CONSTANT, value=114),
    ], bbox_params=bp)


# ══════════════════════════════════════════════════════════════════════════════
#  ③  AUGMENTED DATASET WRITER
#     Reads original images+labels, applies Albumentations N times,
#     writes augmented copies ready for YOLO training.
# ══════════════════════════════════════════════════════════════════════════════

def write_augmented_split(
    src_img_dir : str,
    src_lbl_dir : str,
    dst_img_dir : str,
    dst_lbl_dir : str,
    multiplier  : int = 4,
    mode        : str = "train",
) -> int:
    transform = build_train_transform() if mode == "train" else build_val_transform()
    imgs = sorted(Path(src_img_dir).glob("*.*"))
    Path(dst_img_dir).mkdir(parents=True, exist_ok=True)
    Path(dst_lbl_dir).mkdir(parents=True, exist_ok=True)

    written = 0
    for img_path in imgs:
        lbl_path = Path(src_lbl_dir) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # parse YOLO labels
        bboxes, classes = [], []
        for line in lbl_path.read_text().strip().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            classes.append(int(p[0]))
            bboxes.append([min(max(float(x), 0.0), 1.0) for x in p[1:5]])

        # always copy original
        versions = [("orig", image, bboxes, classes)]
        for i in range(multiplier):
            try:
                aug = transform(image=image, bboxes=bboxes, class_labels=classes)
                versions.append((f"a{i:02d}", aug["image"],
                                 aug["bboxes"], aug["class_labels"]))
            except Exception as e:
                log.debug(f"Aug skip {img_path.name}:{i} — {e}")

        for tag, aug_img, aug_boxes, aug_cls in versions:
            stem   = f"{img_path.stem}_{tag}"
            out_im = Path(dst_img_dir) / f"{stem}.jpg"
            out_lb = Path(dst_lbl_dir) / f"{stem}.txt"
            cv2.imwrite(str(out_im),
                        cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(out_lb, "w") as f:
                for cls_id, box in zip(aug_cls, aug_boxes):
                    f.write(f"{cls_id} {box[0]:.6f} {box[1]:.6f} "
                            f"{box[2]:.6f} {box[3]:.6f}\n")
            written += 1

    log.info(f"[augment] {mode}: {len(imgs)} → {written} samples  ({dst_img_dir})")
    return written


def build_augmented_dataset(cfg: dict, multiplier: int = 4) -> str:
    root     = Path(cfg["dataset_root"])
    aug_root = root / "augmented"

    # train: augment
    write_augmented_split(
        src_img_dir = str(root / "images" / "train"),
        src_lbl_dir = str(root / "labels" / "train"),
        dst_img_dir = str(aug_root / "images" / "train"),
        dst_lbl_dir = str(aug_root / "labels" / "train"),
        multiplier  = multiplier,
        mode        = "train",
    )
    # val/test: copy only
    for split in ["val", "test"]:
        for sub in ["images", "labels"]:
            src = root / sub / split
            dst = aug_root / sub / split
            if src.exists():
                shutil.copytree(str(src), str(dst), dirs_exist_ok=True)

    # write new data.yaml
    new_yaml = aug_root / "drone_data.yaml"
    with open(cfg["data_yaml"]) as f:
        orig = yaml.safe_load(f)
    orig["path"] = str(aug_root.resolve())
    with open(new_yaml, "w") as f:
        yaml.dump(orig, f, default_flow_style=False)

    log.info(f"Augmented dataset YAML → {new_yaml}")
    return str(new_yaml)


# ══════════════════════════════════════════════════════════════════════════════
#  ④  DATASET YAML SCAFFOLDING
# ══════════════════════════════════════════════════════════════════════════════

def create_dataset_yaml(cfg: dict) -> str:
    root = Path(cfg["dataset_root"]).resolve()
    for split in ["train", "val", "test"]:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    data = {
        "path"  : str(root),
        "train" : "images/train",
        "val"   : "images/val",
        "test"  : "images/test",
        "nc"    : cfg["nc"],
        "names" : cfg["class_names"],
    }
    out = root / "drone_data.yaml"
    with open(out, "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    log.info(f"Dataset YAML created → {out}")
    log.info("  ► Place images in:  dataset/images/{train,val,test}/")
    log.info("  ► Place labels in:  dataset/labels/{train,val,test}/")
    log.info("  ► Label format   :  YOLO  [class cx cy w h]  (normalised)")
    return str(out)


# ══════════════════════════════════════════════════════════════════════════════
#  ⑤  MLflow CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

class DroneMLflowCallback:
    def __init__(self, run, cfg: dict):
        self.run = run
        self.cfg = cfg
        self.best_map50 = 0.0

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch
        # losses
        for k, v in trainer.label_loss_items(trainer.tloss, prefix="train").items():
            try: mlflow.log_metric(f"loss/{k}", float(v), step=epoch)
            except: pass
        # val metrics
        for k, v in trainer.metrics.items():
            try:
                key = k.replace("(B)", "").strip()
                mlflow.log_metric(f"val/{key}", float(v), step=epoch)
                if "mAP50" in key and float(v) > self.best_map50:
                    self.best_map50 = float(v)
                    mlflow.log_metric("best_mAP50", self.best_map50, step=epoch)
            except: pass
        # learning rate
        for i, lr in enumerate(trainer.scheduler.get_last_lr()):
            mlflow.log_metric(f"lr/pg{i}", lr, step=epoch)

    def on_train_end(self, trainer):
        save_dir = Path(trainer.save_dir)
        # weights
        for w in ["best.pt", "last.pt"]:
            p = save_dir / "weights" / w
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="weights")
        # plots
        for f in save_dir.glob("*.png"):
            mlflow.log_artifact(str(f), artifact_path="plots")
        for f in save_dir.glob("*.csv"):
            mlflow.log_artifact(str(f), artifact_path="results")
        # final metrics
        for k, v in trainer.metrics.items():
            try: mlflow.log_metric(f"final/{k.replace('(B)','').strip()}", float(v))
            except: pass

        log.info("═" * 70)
        log.info(f"  MLflow Run ID  : {self.run.info.run_id}")
        log.info(f"  Experiment     : {self.cfg['mlflow_exp']}")
        log.info(f"  Best mAP50     : {self.best_map50:.4f}")
        log.info(f"  Weights saved  : {save_dir / 'weights'}")
        log.info(f"  Dashboard      : {self.cfg['mlflow_uri']}")
        log.info("═" * 70)


# ══════════════════════════════════════════════════════════════════════════════
#  ⑥  MAIN TRAINING FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def train(cfg: dict = CFG, use_albumentations: bool = True, aug_mult: int = 4):
    log.info("RTX 5070 Ti detected — using AMP=True, batch=32, imgsz=1280")
    log.info(f"CUDA available: {torch.cuda.is_available()} | "
             f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    # ── optional: pre-generate augmented dataset on disk ─────────────────────
    active_yaml = cfg["data_yaml"]
    if use_albumentations:
        log.info(f"Building Albumentations dataset (×{aug_mult})…")
        active_yaml = build_augmented_dataset(cfg, multiplier=aug_mult)
    else:
        log.info("Skipping Albumentations pre-augmentation (using YOLO built-in)")

    # ── MLflow ────────────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg["mlflow_uri"])
    mlflow.set_experiment(cfg["mlflow_exp"])

    with mlflow.start_run(run_name=cfg["name"]) as run:
        # log full config
        loggable = {k: str(v) for k, v in cfg.items()
                    if not isinstance(v, (list, dict))}
        loggable["aug_multiplier"]   = str(aug_mult)
        loggable["use_albumentations"] = str(use_albumentations)
        loggable["cuda_device"]      = torch.cuda.get_device_name(0) \
                                       if torch.cuda.is_available() else "cpu"
        loggable["torch_version"]    = torch.__version__
        mlflow.log_params(loggable)
        mlflow.log_artifact(active_yaml, artifact_path="dataset")

        # ── load model ────────────────────────────────────────────────────────
        model = YOLO(cfg["model"])

        # ── attach MLflow callback ────────────────────────────────────────────
        cb = DroneMLflowCallback(run, cfg)
        model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)
        model.add_callback("on_train_end",       cb.on_train_end)

        # ── YOLO train ────────────────────────────────────────────────────────
        # Note: YOLO built-in augmentations COMPLEMENT Albumentations
        # They run at batch-load time; Albumentations runs as pre-processing
        results = model.train(
            data            = active_yaml,
            epochs          = cfg["epochs"],
            imgsz           = cfg["imgsz"],
            batch           = cfg["batch"],
            device          = cfg["device"],
            optimizer       = cfg["optimizer"],
            lr0             = cfg["lr0"],
            lrf             = cfg["lrf"],
            momentum        = cfg["momentum"],
            weight_decay    = cfg["weight_decay"],
            warmup_epochs   = cfg["warmup_epochs"],
            warmup_bias_lr  = cfg["warmup_bias_lr"],
            cos_lr          = cfg["cos_lr"],
            amp             = cfg["amp"],
            cache           = cfg["cache"],
            workers         = cfg["workers"],
            box             = cfg["box"],
            cls             = cfg["cls"],
            dfl             = cfg["dfl"],
            multi_scale     = cfg["multi_scale"],
            patience        = cfg["patience"],
            save_period     = cfg["save_period"],
            project         = cfg["project"],
            name            = cfg["name"],
            exist_ok        = True,
            verbose         = True,
            # ── YOLO built-in augmentation (adds diversity on top of Albumentations)
            mosaic          = 1.0,      # 4-image mosaic — key for small targets
            mixup           = 0.15,     # blend 2 images
            copy_paste      = 0.10,     # paste drone instances onto new backgrounds
            hsv_h           = 0.015,
            hsv_s           = 0.4,
            hsv_v           = 0.3,
            degrees         = 10.0,
            translate       = 0.1,
            scale           = 0.5,
            flipud          = 0.1,
            fliplr          = 0.5,
            perspective     = 0.0005,
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  ⑦  VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate(model_path: str, cfg: dict = CFG) -> dict:
    model = YOLO(model_path)
    metrics = model.val(
        data   = cfg["data_yaml"],
        imgsz  = cfg["imgsz"],
        device = cfg["device"],
        conf   = 0.25,
        iou    = 0.5,
        split  = "test",
        verbose= True,
    )
    summary = {
        "mAP50"     : metrics.box.map50,
        "mAP50_95"  : metrics.box.map,
        "precision" : metrics.box.mp,
        "recall"    : metrics.box.mr,
    }
    log.info("\n[VALIDATION]")
    for k, v in summary.items():
        log.info(f"  {k:<12}: {v:.4f}")
    return summary


# ══════════════════════════════════════════════════════════════════════════════
#  ⑧  JETSON EXPORT  (TensorRT FP16)
#     ► Run this command ON THE JETSON (or cross-compile with trtexec)
#     ► Alternatively run on your RTX 5070 Ti host and copy .engine to Jetson
# ══════════════════════════════════════════════════════════════════════════════

def export_for_jetson(model_path: str, cfg: dict = CFG) -> str:
    """
    Export best.pt → TensorRT .engine optimised for Jetson inference.

    Steps (also see jetson_deploy.sh):
      1. Load PyTorch weights
      2. Export to ONNX  (intermediate)
      3. Build TensorRT engine (FP16)

    The resulting .engine file should be copied to Jetson and loaded with:
        model = YOLO("drone_best.engine")
        results = model.predict(source=frame, imgsz=640, conf=0.3)
    """
    log.info(f"Exporting {model_path} → TensorRT FP16 engine…")
    model = YOLO(model_path)
    engine_path = model.export(
        format    = cfg["jetson_format"],   # "engine"
        imgsz     = cfg["jetson_imgsz"],    # 640 for Jetson speed
        half      = cfg["jetson_fp16"],     # FP16
        device    = cfg["device"],
        workspace = cfg["jetson_workspace"],# GB
        simplify  = True,
        dynamic   = False,                  # static shapes = faster on Jetson
    )
    log.info(f"Engine saved: {engine_path}")
    log.info("Copy drone_best.engine to Jetson and use:")
    log.info("  from ultralytics import YOLO")
    log.info("  model = YOLO('drone_best.engine')")
    log.info("  model.predict(source=frame, imgsz=640, conf=0.3, iou=0.5)")
    return engine_path


# ══════════════════════════════════════════════════════════════════════════════
#  ⑨  JETSON INFERENCE SCRIPT  (written out as a standalone file)
# ══════════════════════════════════════════════════════════════════════════════

JETSON_INFERENCE_CODE = '''"""
Jetson Real-Time Drone Detection — runs on Jetson AGX / Orin / NX
Camera: CSI or USB  1280×720 @ 120 fps
Model : TensorRT FP16 engine (drone_best.engine)
"""
import cv2, time
from ultralytics import YOLO

ENGINE  = "drone_best.engine"   # path to TensorRT engine on Jetson
CONF    = 0.30
IOU     = 0.50
IMGSZ   = 640
SRC     = 0                     # camera index; or RTSP URL

model = YOLO(ENGINE, task="detect")
cap   = cv2.VideoCapture(SRC)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS,         120)

prev_t = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source     = frame,
        imgsz      = IMGSZ,
        conf       = CONF,
        iou        = IOU,
        device     = 0,     # Jetson GPU
        verbose    = False,
        half       = True,  # FP16
    )

    # Draw detections
    annotated = results[0].plot()

    # FPS overlay
    now   = time.time()
    fps   = 1.0 / (now - prev_t + 1e-9)
    prev_t = now
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Drone count
    n_drones = len(results[0].boxes)
    cv2.putText(annotated, f"Drones: {n_drones}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Drone Detector — Jetson", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
'''

JETSON_DEPLOY_SH = '''#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Jetson Setup Script  —  Install Ultralytics + TensorRT environment
# Tested on JetPack 5.x / 6.x  (Jetson AGX Orin / Orin NX / Xavier)
# ─────────────────────────────────────────────────────────────────────────────
set -e

echo "► Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y python3-pip libopencv-dev python3-opencv \\
    liblapack-dev libblas-dev gfortran libfreetype6-dev \\
    libopenblas-dev cmake

echo "► Installing Python packages..."
pip3 install --upgrade pip
pip3 install ultralytics==8.3.*          # pin for stability
pip3 install onnx onnxruntime

# TensorRT Python bindings are pre-installed on JetPack
# Verify:
python3 -c "import tensorrt; print('TensorRT:', tensorrt.__version__)"

echo "► Copying engine to Jetson..."
# scp user@host:/path/to/drone_best.engine ./drone_best.engine

echo "► Running inference..."
python3 jetson_inference.py

echo "Done."
'''


def write_jetson_files(out_dir: str = ".") -> None:
    out = Path(out_dir)
    (out / "jetson_inference.py").write_text(JETSON_INFERENCE_CODE)
    (out / "jetson_deploy.sh").write_text(JETSON_DEPLOY_SH)
    os.chmod(out / "jetson_deploy.sh", 0o755)
    log.info(f"Jetson files written → {out}/jetson_inference.py + jetson_deploy.sh")


# ══════════════════════════════════════════════════════════════════════════════
#  ⑩  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Drone-vs-Drone Detection | RTX 5070 Ti → Jetson")
    p.add_argument("--mode",
        choices=["train", "validate", "export", "scaffold", "jetson-files"],
        default="train")
    p.add_argument("--weights",  default="",      help="Path to .pt for val/export")
    p.add_argument("--epochs",   type=int,  default=CFG["epochs"])
    p.add_argument("--batch",    type=int,  default=CFG["batch"])
    p.add_argument("--imgsz",    type=int,  default=CFG["imgsz"])
    p.add_argument("--model",    default=CFG["model"])
    p.add_argument("--data",     default=CFG["data_yaml"])
    p.add_argument("--device",   default=CFG["device"])
    p.add_argument("--aug-mult", type=int,  default=4,
                   help="Albumentations copies per image (default 4)")
    p.add_argument("--no-albumentations", action="store_true",
                   help="Skip Albumentations, use YOLO built-in aug only")
    p.add_argument("--mlflow-uri",  default=CFG["mlflow_uri"])
    p.add_argument("--mlflow-exp",  default=CFG["mlflow_exp"])
    p.add_argument("--name",     default=CFG["name"])
    return p.parse_args()


def main():
    args  = parse_args()
    cfg   = {**CFG,
             "epochs"    : args.epochs,
             "batch"     : args.batch,
             "imgsz"     : args.imgsz,
             "model"     : args.model,
             "data_yaml" : args.data,
             "device"    : args.device,
             "mlflow_uri": args.mlflow_uri,
             "mlflow_exp": args.mlflow_exp,
             "name"      : args.name}

    if args.mode == "scaffold":
        create_dataset_yaml(cfg)

    elif args.mode == "train":
        train(cfg,
              use_albumentations = not args.no_albumentations,
              aug_mult           = args.aug_mult)

    elif args.mode == "validate":
        assert args.weights, "--weights required for validate mode"
        validate(args.weights, cfg)

    elif args.mode == "export":
        assert args.weights, "--weights required for export mode"
        export_for_jetson(args.weights, cfg)

    elif args.mode == "jetson-files":
        write_jetson_files(".")

    else:
        log.error(f"Unknown mode: {args.mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()