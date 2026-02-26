# YOLO Hyperparameters — Complete Reference

> One-line definition for every hyperparameter across all 8 categories.

---

## 1. Training Core Hyperparameters

| Hyperparameter | Default | One Line |
|---|---|---|
| `epochs` | 100 | How many times the model sees the entire dataset |
| `patience` | 50 | Stop training if no improvement for N epochs |
| `batch` | 16 | How many images processed together before updating weights |
| `imgsz` | 640 | Resolution every image is resized to before training |
| `save` | True | Whether to save model checkpoints during training |
| `save_period` | -1 | Save a checkpoint every N epochs |
| `cache` | False | Preload all images into RAM to speed up training |
| `device` | None | Which hardware to train on — GPU, CPU, or Apple Silicon |
| `workers` | 8 | CPU threads that prepare data while GPU is computing |
| `pretrained` | True | Start from weights already trained on millions of images |
| `optimizer` | auto | Algorithm that updates model weights after each error |
| `seed` | 0 | Starting number that makes all random operations reproducible |
| `deterministic` | True | Forces identical results every single run |
| `single_cls` | False | Treat all object classes as one unified "object" class |
| `rect` | False | Batch images by similar aspect ratio to reduce padding waste |
| `cos_lr` | False | Decay learning rate along a smooth cosine curve |
| `close_mosaic` | 10 | Disable mosaic augmentation for the final N epochs |
| `resume` | False | Continue training from a saved checkpoint |
| `amp` | True | Use 16-bit numbers where safe to speed up training |
| `fraction` | 1.0 | Use only a fraction of the dataset for training |
| `freeze` | None | Lock first N layers so their weights don't change |
| `dropout` | 0.0 | Randomly deactivate neurons to prevent overfitting |
| `multi_scale` | False | Randomly resize images during training for scale robustness |
| `overlap_mask` | True | Allow segmentation masks to overlap each other |
| `mask_ratio` | 4 | Downsample segmentation masks by this factor to save memory |

---

## 2. Learning Rate Hyperparameters

| Hyperparameter | Default | One Line |
|---|---|---|
| `lr0` | 0.01 | Starting learning rate — how big the first correction steps are |
| `lrf` | 0.01 | Final learning rate fraction — how small steps get at the end |
| `momentum` | 0.937 | Carry forward previous gradient direction to smooth updates |
| `weight_decay` | 0.0005 | Gently shrink large weights to prevent overfitting |
| `warmup_epochs` | 3.0 | Gradually ramp up learning rate from zero at training start |
| `warmup_momentum` | 0.8 | Start momentum low during warmup and build it gradually |
| `warmup_bias_lr` | 0.1 | Higher learning rate for bias parameters during warmup |
| `nbs` | 64 | Reference batch size used to automatically scale learning rate |

---

## 3. Loss Function Hyperparameters

| Hyperparameter | Default | One Line |
|---|---|---|
| `box` | 7.5 | How heavily wrong bounding box coordinates are penalized |
| `cls` | 0.5 | How heavily predicting the wrong class is penalized |
| `dfl` | 1.5 | Penalizes wrong probability distribution over box edge positions |
| `pose` | 12.0 | How heavily wrong keypoint positions are penalized |
| `kobj` | 2.0 | How heavily wrong keypoint visibility predictions are penalized |
| `label_smoothing` | 0.0 | Soften hard 0/1 labels to prevent overconfident predictions |
| `nbs` | 64 | Reference batch size for consistent loss scaling |
| `obj` | 1.0 | How heavily wrong objectness scores are penalized (v5/v7) |
| `obj_pw` | 1.0 | Upweight positive object examples in objectness loss (v5) |
| `cls_pw` | 1.0 | Upweight positive class examples in classification loss (v5) |
| `anchor_t` | 4.0 | Maximum size ratio between ground truth and anchor for assignment |
| `fl_gamma` | 0.0 | Focus training on hard misclassified examples by down-weighting easy ones |
| `iou_t` | 0.20 | Minimum box overlap to count as a positive training example |

---

## 4. Augmentation Hyperparameters

| Hyperparameter | Default | One Line |
|---|---|---|
| `hsv_h` | 0.015 | Randomly shift image hue to teach color invariance |
| `hsv_s` | 0.7 | Randomly change color saturation to simulate different lighting |
| `hsv_v` | 0.4 | Randomly change brightness to simulate different lighting conditions |
| `degrees` | 0.0 | Randomly rotate images to teach orientation invariance |
| `translate` | 0.1 | Randomly shift images so objects aren't always centered |
| `scale` | 0.5 | Randomly zoom in or out to teach scale invariance |
| `shear` | 0.0 | Apply slanting distortion to simulate angled viewpoints |
| `perspective` | 0.0 | Apply 3D warp to simulate different camera angles |
| `flipud` | 0.0 | Randomly flip images vertically for aerial or overhead datasets |
| `fliplr` | 0.5 | Randomly mirror images horizontally to double effective dataset size |
| `bgr` | 0.0 | Randomly swap RGB to BGR to handle different camera conventions |
| `mosaic` | 1.0 | Combine 4 images into one grid to force multi-scale detection |
| `mixup` | 0.0 | Blend two images together to teach calibrated uncertain predictions |
| `copy_paste` | 0.0 | Cut objects from one image and paste into another |
| `copy_paste_mode` | flip | Whether to flip or blend the pasted object |
| `auto_augment` | randaugment | Apply a pre-designed automatic augmentation policy |
| `erasing` | 0.4 | Randomly black out image patches to simulate occlusion |
| `crop_fraction` | 1.0 | Randomly crop images to teach partial object recognition |

---

## 5. Inference / Prediction Hyperparameters

| Hyperparameter | Default | One Line |
|---|---|---|
| `conf` | 0.25 | Only report detections above this confidence score |
| `iou` | 0.7 | Remove duplicate boxes that overlap more than this threshold |
| `max_det` | 300 | Maximum number of detections reported per image |
| `half` | False | Use 16-bit precision at inference for faster speed |
| `agnostic_nms` | False | Apply duplicate removal across all classes together |
| `retina_masks` | False | Generate full-resolution segmentation masks |
| `classes` | None | Only return detections for specific class IDs |
| `vid_stride` | 1 | Process only every Nth frame of a video |
| `stream_buffer` | False | Process every frame in order vs only the latest frame |
| `visualize` | False | Save intermediate feature maps to see what the model sees |
| `augment` | False | Run inference on multiple augmented versions and merge results |

---

## 6. Anchor Hyperparameters *(YOLOv5/v7 only)*

| Hyperparameter | Default | One Line |
|---|---|---|
| `anchors` | 3 | Number of reference box templates per detection head |
| `anchor_t` | 4.0 | Maximum size ratio allowed between ground truth and anchor |

> **Note:** YOLOv8 and later are anchor-free — these parameters no longer apply.

---

## 7. Validation Hyperparameters

| Hyperparameter | Default | One Line |
|---|---|---|
| `val` | True | Whether to evaluate on validation data after each epoch |
| `split` | val | Which dataset split to use for validation |
| `save_json` | False | Save predictions in COCO JSON format for official evaluation |
| `save_hybrid` | False | Save labels combining ground truth and model predictions |
| `conf` | 0.001 | Confidence threshold used during validation evaluation |
| `iou` | 0.6 | Overlap threshold for counting a detection as correct |
| `max_det` | 300 | Maximum detections per image during validation |
| `plots` | True | Auto-generate training curves and confusion matrices |

---

## 8. Export Hyperparameters

| Hyperparameter | Default | One Line |
|---|---|---|
| `format` | torchscript | Target deployment format — ONNX, TensorRT, TFLite, CoreML |
| `keras` | False | Use Keras API when exporting to TensorFlow format |
| `optimize` | False | Apply TorchScript graph optimizations for CPU deployment |
| `int8` | False | Convert weights to 8-bit integers for 4× smaller faster model |
| `half` | False | Convert to 16-bit floats for faster NVIDIA GPU inference |
| `dynamic` | False | Allow variable batch sizes and image dimensions in ONNX |
| `simplify` | True | Remove redundant nodes to clean up the ONNX graph |
| `opset` | None | ONNX operator set version controlling available operations |
| `workspace` | 4 | GPU memory TensorRT can use during engine optimization |
| `nms` | False | Bake duplicate removal directly into the exported model |

---

## Quick Reference — Most Impactful Parameters

| Priority | Parameter | Why It Matters |
|---|---|---|
| 1 | `lr0` + `lrf` | Controls the entire learning speed and convergence |
| 2 | `mosaic` + `mixup` | Most powerful augmentation combination |
| 3 | `imgsz` | Bigger = detects smaller objects |
| 4 | `batch` | Larger = more stable training gradients |
| 5 | `box` + `cls` + `dfl` | Balances what the model prioritizes |
| 6 | `close_mosaic` | Stabilizes final convergence |
| 7 | `warmup_epochs` | Prevents early training chaos |
| 8 | `weight_decay` + `dropout` | Controls overfitting |

---

## Format Options Reference

| Format | Best For |
|---|---|
| `torchscript` | PyTorch CPU/GPU deployment |
| `onnx` | Cross-platform universal deployment |
| `openvino` | Intel CPU/VPU hardware |
| `tflite` | Android mobile devices |
| `coreml` | iPhone / iPad / Mac (Apple) |
| `engine` | NVIDIA TensorRT — fastest GPU inference |
| `paddle` | PaddlePaddle ecosystem |

---

*Based on YOLOv8/YOLOv11 — the most complete YOLO implementations.*