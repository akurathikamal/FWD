import os
import hashlib
from PIL import Image
import yaml

# -------------------------------
# 1. Image Quality & Format
# -------------------------------
def check_images(images_folder, extensions=[".jpg", ".png"]):
    bad_files = []
    resolutions = set()
    for f in os.listdir(images_folder):
        if any(f.lower().endswith(ext) for ext in extensions):
            path = os.path.join(images_folder, f)
            try:
                with Image.open(path) as img:
                    img.verify()
                    resolutions.add(img.size)
            except Exception as e:
                bad_files.append((f, str(e)))
    return bad_files, resolutions

# -------------------------------
# 2. Label File Integrity
# -------------------------------
def check_labels(labels_folder, num_classes, extension=".txt"):
    bad_files = []
    empty_files = []
    for f in os.listdir(labels_folder):
        if f.endswith(extension):
            path = os.path.join(labels_folder, f)
            with open(path, "r") as file:
                lines = file.readlines()
                if not lines:
                    empty_files.append(f)
                else:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            bad_files.append((f, "Invalid format"))
                            break
                        class_id, x, y, w, h = parts
                        if not class_id.isdigit() or int(class_id) >= num_classes:
                            bad_files.append((f, f"Invalid class ID {class_id}"))
                        try:
                            x, y, w, h = map(float, (x, y, w, h))
                            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                                bad_files.append((f, "Coordinates not normalized"))
                        except ValueError:
                            bad_files.append((f, "Non-numeric values"))
    return bad_files, empty_files

# -------------------------------
# 3. Pairing Check
# -------------------------------
def check_pairs(images_folder, labels_folder, img_exts=[".jpg", ".png"], lbl_ext=".txt"):
    img_names = {os.path.splitext(f)[0] for f in os.listdir(images_folder) if any(f.endswith(ext) for ext in img_exts)}
    lbl_names = {os.path.splitext(f)[0] for f in os.listdir(labels_folder) if f.endswith(lbl_ext)}

    unpaired_images = img_names - lbl_names
    unpaired_labels = lbl_names - img_names

    return unpaired_images, unpaired_labels

# -------------------------------
# 4. Duplicates
# -------------------------------
def file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def find_duplicates(folder):
    seen = {}
    duplicates = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path):
            h = file_hash(path)
            if h in seen:
                duplicates.append((seen[h], f))
            else:
                seen[h] = f
    return duplicates

# -------------------------------
# 5. Bounding Box Validity
# -------------------------------
def check_bboxes(labels_folder, extension=".txt"):
    bad_boxes = []
    for f in os.listdir(labels_folder):
        if f.endswith(extension):
            path = os.path.join(labels_folder, f)
            with open(path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, x, y, w, h = parts
                        try:
                            x, y, w, h = map(float, (x, y, w, h))
                            if w <= 0 or h <= 0:
                                bad_boxes.append((f, "Zero/negative width/height"))
                        except ValueError:
                            bad_boxes.append((f, "Invalid bbox values"))
    return bad_boxes

# -------------------------------
# 6. Dataset Split Check
# -------------------------------
def check_split(train_folder, val_folder, test_folder):
    train_files = set(os.listdir(train_folder))
    val_files = set(os.listdir(val_folder))
    test_files = set(os.listdir(test_folder))

    leakage = (train_files & val_files) | (train_files & test_files) | (val_files & test_files)
    return leakage

# -------------------------------
# 7. Configuration File Check
# -------------------------------
def check_yaml(yaml_file, num_classes):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    errors = []
    if "nc" not in data or data["nc"] != num_classes:
        errors.append("Number of classes mismatch")
    if "names" not in data or len(data["names"]) != num_classes:
        errors.append("Class names mismatch")
    return errors


# 🐍 Code for Class Consistency
# This will:
# - Verify all class IDs are within the defined range (0–nc-1).
# - Check that no label file contains undefined class IDs.
# - Ensure class names in data.yaml match the IDs used in labels.
import os
import yaml

def check_class_consistency(labels_folder, yaml_file, extension=".txt"):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    num_classes = data["nc"]
    class_names = data["names"]

    bad_files = []
    used_classes = set()

    for f in os.listdir(labels_folder):
        if f.endswith(extension):
            path = os.path.join(labels_folder, f)
            with open(path, "r") as file:
                for line in file:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = parts[0]
                        if not class_id.isdigit():
                            bad_files.append((f, "Non-numeric class ID"))
                        else:
                            cid = int(class_id)
                            if cid < 0 or cid >= num_classes:
                                bad_files.append((f, f"Invalid class ID {cid}"))
                            else:
                                used_classes.add(cid)

    missing_classes = set(range(num_classes)) - used_classes
    return bad_files, missing_classes, class_names

if __name__ == "__main__":
    labels_folder = "dataset/labels"
    yaml_file = "dataset/data.yaml"

    bad_files, missing_classes, class_names = check_class_consistency(labels_folder, yaml_file)

    print("Bad label files:", bad_files)
    print("Classes not used in dataset:", missing_classes)
    print("Class names from YAML:", class_names)

# 

# 🐍 Code for Annotation Quality (Spot-Check)
# Since annotation quality is partly visual, we can:
# - Randomly sample a few label files.
# - Print their bounding boxes and class IDs.
# - Optionally overlay boxes on images for manual inspection.
import os
import random
import cv2

def spot_check_annotations(images_folder, labels_folder, sample_size=5, img_ext=".jpg"):
    files = [f for f in os.listdir(images_folder) if f.endswith(img_ext)]
    sample = random.sample(files, min(sample_size, len(files)))

    for f in sample:
        img_path = os.path.join(images_folder, f)
        lbl_path = os.path.join(labels_folder, os.path.splitext(f)[0] + ".txt")

        if not os.path.exists(lbl_path):
            print(f"No label for {f}")
            continue

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        with open(lbl_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 5:
                    cid, x, y, bw, bh = parts
                    x, y, bw, bh = map(float, (x, y, bw, bh))
                    # Convert normalized coords to pixel coords
                    x1 = int((x - bw/2) * w)
                    y1 = int((y - bh/2) * h)
                    x2 = int((x + bw/2) * w)
                    y2 = int((y + bh/2) * h)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(img, cid, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        cv2.imshow("Annotation Check", img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    images_folder = "dataset/images"
    labels_folder = "dataset/labels"

    spot_check_annotations(images_folder, labels_folder, sample_size=5)

# 

# 🔍 What This Gives You
# - Class Consistency: Ensures IDs are valid and match data.yaml.
# - Annotation Quality: Lets you visually inspect a random subset of images with bounding boxes drawn.


# -------------------------------
# Run All Checks
# -------------------------------
if __name__ == "__main__":
    images_folder = "dataset/images"
    labels_folder = "dataset/labels"
    train_folder = "dataset/train/images"
    val_folder = "dataset/val/images"
    test_folder = "dataset/test/images"
    yaml_file = "dataset/data.yaml"
    num_classes = 5  # <-- set your number of classes here

    # 1. Image check
    bad_images, resolutions = check_images(images_folder)
    print("Bad images:", bad_images)
    print("Image resolutions found:", resolutions)

    # 2. Label check
    bad_labels, empty_labels = check_labels(labels_folder, num_classes)
    print("Bad labels:", bad_labels)
    print("Empty labels:", empty_labels)

    # 3. Pairing check
    unpaired_images, unpaired_labels = check_pairs(images_folder, labels_folder)
    print("Unpaired images:", unpaired_images)
    print("Unpaired labels:", unpaired_labels)

    # 4. Duplicates
    dup_images = find_duplicates(images_folder)
    dup_labels = find_duplicates(labels_folder)
    print("Duplicate images:", dup_images)
    print("Duplicate labels:", dup_labels)

    # 5. Bounding box validity
    bad_boxes = check_bboxes(labels_folder)
    print("Invalid bounding boxes:", bad_boxes)

    # 6. Dataset split leakage
    leakage = check_split(train_folder, val_folder, test_folder)
    print("Data leakage between splits:", leakage)

    # 7. YAML config check
    yaml_errors = check_yaml(yaml_file, num_classes)
    print("YAML errors:", yaml_errors)

    print("✅ Dataset validation complete.")