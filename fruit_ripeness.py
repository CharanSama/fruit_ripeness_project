import os
import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns


#processes the images correctly
def preprocess_image(bgr_img, save_debug=False, label="image"):
    """Resize → Gaussian blur → HSV → CLAHE."""
    os.makedirs("outputs", exist_ok=True)

    #resize image
    bgr_resized = cv2.resize(bgr_img, (256, 256))

    #gaussian smoothing
    bgr_gaussian = cv2.GaussianBlur(bgr_resized, (5, 5), 0)

    #HSV conversion
    hsv_before = cv2.cvtColor(bgr_gaussian, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_before)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    hsv_after = cv2.merge([h, s, v_eq])
    bgr_preprocessed = cv2.cvtColor(hsv_after, cv2.COLOR_HSV2BGR)

    return bgr_resized, hsv_before, hsv_after, bgr_preprocessed


#Gaussian Pyramid
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


#Extracts the hsv features from the fruit
def extract_hsv_features(hsv_img):
    """
    Color features:
      - mean / std of H, S, V
      - percentage of pixels in 3 hue ranges:
        * greenish
        * yellowish
        * brownish
    Uses a mask to ignore low-saturation / low-value background.
    """
    h, s, v = cv2.split(hsv_img)

    #focus the mask of the fruit on the image
    mask = (s > 40) & (v > 40)
    if np.sum(mask) < 50:
        h_valid = h.flatten()
        s_valid = s.flatten()
        v_valid = v.flatten()
    else:
        h_valid = h[mask]
        s_valid = s[mask]
        v_valid = v[mask]

    mean_h = np.mean(h_valid)
    std_h  = np.std(h_valid)
    mean_s = np.mean(s_valid)
    std_s  = np.std(s_valid)
    mean_v = np.mean(v_valid)
    std_v  = np.std(v_valid)

    #Hue percentages displayed
    total_pix = len(h_valid) + 1e-8

    #ranges for the fruit used (banana) set at these values, dpending on the hue and fruit
    yellow_mask = (h_valid >= 15) & (h_valid <= 35)
    green_mask  = (h_valid >= 35) & (h_valid <= 85)
    brown_mask  = (h_valid < 15) | (h_valid > 85)

    frac_yellow = np.sum(yellow_mask) / total_pix
    frac_green  = np.sum(green_mask) / total_pix
    frac_brown  = np.sum(brown_mask) / total_pix

    return np.array([
        mean_h, std_h,
        mean_s, std_s,
        mean_v, std_v,
        frac_green, frac_yellow, frac_brown
    ], dtype=np.float32)


def extract_texture_features(pyr):
    sharp = [laplacian_sharpness(level) for level in pyr]
    decay = sharp[0] - sharp[-1]
    while len(sharp) < 4:
        sharp.append(0.0)
    return np.array([sharp[0], sharp[1], sharp[2], sharp[3], decay], dtype=np.float32)


#Confusion Matrix displaying correct information
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


#main function that runs the program
def main():
    train_root = "dataset"
    test_root = "testset"

    #training data
    train_features = []
    train_labels = []
    pyramid_saved = {}

    class_names = sorted([
        d for d in os.listdir(train_root)
        if os.path.isdir(os.path.join(train_root, d))
    ])

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

            #Preprocess image
            _, _, hsv_after, pre = preprocess_image(bgr)

            #Gaussian pyramid
            pyr = gaussian_pyramid(pre)
            if label not in pyramid_saved:
                combine_pyramid_images(pyr, f"outputs/{label}_pyramid_panel.png")
                pyramid_saved[label] = True

            #Extract features from the images like color and texture
            color = extract_hsv_features(hsv_after)
            texture = extract_texture_features(pyr)
            full_feat = np.concatenate([color, texture])

            train_features.append(full_feat)
            train_labels.append(label)

    train_features = np.vstack(train_features)
    train_labels = np.array(train_labels)

    print("Training samples loaded:", len(train_labels))

    #scaling the data and fitting the model
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_features_scaled, train_labels)

    #testing data for making predictions
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
            feat_scaled = scaler.transform(feat.reshape(1, -1))
            pred = clf.predict(feat_scaled)[0]
            y_pred.append(pred)

            print(f"{fname}: true={label}, predicted={pred}")

    #set for the accuracy report, what would show in the terminal
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    overall_acc = np.mean(y_true == y_pred)
    print("\n========RIPENESS ACCURACY REPORT========")
    print(f"    ====Overall Accuracy: {overall_acc * 100:.2f}%====\n")

    #accuracy set for the classes
    for cls in class_names:
        idx = (y_true == cls)
        if np.sum(idx) == 0:
            continue
        cls_acc = np.mean(y_pred[idx] == cls)
        print(f"{cls} accuracy: {cls_acc * 100:.2f}%")

    #saves the confusion matrix in our outputs folder
    save_confusion_matrix(y_true, y_pred, class_names)


if __name__ == "__main__":
    main()

