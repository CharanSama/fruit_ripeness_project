import os
import cv2
import numpy as np

# Use a non-GUI backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity
from sklearn.metrics import confusion_matrix
import seaborn as sns



# -----------------------------------------------------------
# 1. PREPROCESSING
# -----------------------------------------------------------
def preprocess_image(bgr_img, save_debug=False, label="image"):
    """Resize → Gaussian blur → HSV → CLAHE."""
    os.makedirs("outputs", exist_ok=True)

    # Resize
    bgr_resized = cv2.resize(bgr_img, (256, 256))

    # Gaussian smoothing
    bgr_gaussian = cv2.GaussianBlur(bgr_resized, (5, 5), 0)

    # HSV conversion
    hsv_before = cv2.cvtColor(bgr_gaussian, cv2.COLOR_BGR2HSV)

    # CLAHE on V channel
    h, s, v = cv2.split(hsv_before)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    hsv_after = cv2.merge([h, s, v_eq])
    bgr_preprocessed = cv2.cvtColor(hsv_after, cv2.COLOR_HSV2BGR)

    return bgr_resized, hsv_before, hsv_after, bgr_preprocessed



# -----------------------------------------------------------
# 2. GAUSSIAN PYRAMID + TEXTURE FEATURES
# -----------------------------------------------------------
def gaussian_pyramid(img, levels=4):
    pyr = [img]
    temp = img.copy()
    for _ in range(1, levels):
        temp = cv2.pyrDown(temp)
        pyr.append(temp)
    return pyr


def combine_pyramid_images(pyr, out_path):
    """Combine 4 levels horizontally into one panel."""
    resized = [cv2.resize(level, (256, 256)) for level in pyr]
    panel = cv2.hconcat(resized)
    cv2.imwrite(out_path, panel)
    print(f"Saved combined pyramid → {out_path}")


def laplacian_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()



# -----------------------------------------------------------
# 3. FEATURE EXTRACTION
# -----------------------------------------------------------
def extract_hsv_features(hsv_img):
    h, s, v = cv2.split(hsv_img)
    return np.array([
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v),
    ], dtype=np.float32)


def extract_texture_features(pyr):
    sharp = [laplacian_sharpness(level) for level in pyr]
    decay = sharp[0] - sharp[-1]  # how fast texture disappears
    return np.array([sharp[0], sharp[1], sharp[2], sharp[3], decay], dtype=np.float32)



# -----------------------------------------------------------
# 4. CLASSIFIER
# -----------------------------------------------------------
def classify_nearest(feat, all_features, labels):
    d = np.linalg.norm(all_features - feat, axis=1)
    return labels[np.argmin(d)]



# -----------------------------------------------------------
# 5. CONFUSION MATRIX PLOTTING
# -----------------------------------------------------------
def save_confusion_matrix(y_true, y_pred, class_names, out_path="outputs/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=class_names)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Ripeness Classification)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved confusion matrix → {out_path}")



# -----------------------------------------------------------
# 6. MAIN PROGRAM — WITH TRAIN/TEST SPLIT
# -----------------------------------------------------------
def main():
    train_root = "dataset"
    test_root = "testset"

    # ------------ TRAINING DATA -------------------
    train_features = []
    train_labels = []
    pyramid_saved = {}

    class_names = sorted(os.listdir(train_root))

    print("Training on classes:", class_names)

    for label in class_names:
        class_dir = os.path.join(train_root, label)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, fname)
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue

            # Preprocess
            _, _, hsv_after, pre = preprocess_image(bgr)

            # Gaussian pyramid
            pyr = gaussian_pyramid(pre)

            # Save 1 combined pyramid panel PER class
            if label not in pyramid_saved:
                combine_pyramid_images(pyr, f"outputs/{label}_pyramid_panel.png")
                pyramid_saved[label] = True

            # Extract features
            color = extract_hsv_features(hsv_after)
            texture = extract_texture_features(pyr)
            full_feat = np.concatenate([color, texture])

            train_features.append(full_feat)
            train_labels.append(label)

    train_features = np.vstack(train_features)
    train_labels = np.array(train_labels)

    print("Training samples loaded:", len(train_labels))


    # ------------ TESTING DATA -------------------
    y_true = []
    y_pred = []

    print("\nTesting classifier...")

    for label in class_names:
        class_dir = os.path.join(test_root, label)
        if not os.path.isdir(class_dir):
            continue

        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(class_dir, fname)
            bgr = cv2.imread(img_path)
            if bgr is None:
                continue

            y_true.append(label)

            _, _, hsv_after, pre = preprocess_image(bgr)
            pyr = gaussian_pyramid(pre)

            color = extract_hsv_features(hsv_after)
            texture = extract_texture_features(pyr)
            feat = np.concatenate([color, texture])

            pred = classify_nearest(feat, train_features, train_labels)
            y_pred.append(pred)

            print(f"{fname}: true={label}, predicted={pred}")


    # ------------ METRICS -------------------
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall_acc = np.mean(y_true == y_pred)
    print("\n==============================")
    print("     RIPENESS ACCURACY REPORT")
    print("==============================")
    print(f"Overall Accuracy: {overall_acc * 100:.2f}%\n")

    # Per-class accuracy
    for cls in class_names:
        idx = (y_true == cls)
        cls_acc = np.mean(y_pred[idx] == cls)
        print(f"{cls} accuracy: {cls_acc * 100:.2f}%")

    # Confusion matrix
    save_confusion_matrix(y_true, y_pred, class_names)



if __name__ == "__main__":
    main()
