import os
import cv2
import numpy as np

# Use a non-GUI backend so we don't need Tk / windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity


# -------------------------
# 1. Preprocessing
# -------------------------
def preprocess_image(bgr_img, save_debug=False, label="image"):
    """
    Resize, blur, convert to HSV, apply CLAHE.
    Also saves intermediate images if save_debug=True.
    """

    # Make sure output folder exists
    os.makedirs("outputs", exist_ok=True)

    # --- Step 1: Resize ---
    bgr_resized = cv2.resize(bgr_img, (256, 256), interpolation=cv2.INTER_AREA)
    if save_debug:
        cv2.imwrite(f"outputs/{label}_step_original.png", bgr_resized)

    # --- Step 2: Gaussian smoothing ---
    bgr_gaussian = cv2.GaussianBlur(bgr_resized, (5, 5), 0)
    if save_debug:
        cv2.imwrite(f"outputs/{label}_step_gaussian.png", bgr_gaussian)

    # --- Step 3: Convert to HSV ---
    hsv_before = cv2.cvtColor(bgr_gaussian, cv2.COLOR_BGR2HSV)

    # To *visualize HSV* we convert back to RGB after scaling channels
    hsv_visual = cv2.cvtColor(hsv_before, cv2.COLOR_HSV2BGR)
    if save_debug:
        cv2.imwrite(f"outputs/{label}_step_hsv_visualized.png", hsv_visual)

    # --- Step 4: CLAHE on V channel ---
    h, s, v = cv2.split(hsv_before)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    hsv_after = cv2.merge([h, s, v_eq])
    bgr_preprocessed = cv2.cvtColor(hsv_after, cv2.COLOR_HSV2BGR)

    if save_debug:
        cv2.imwrite(f"outputs/{label}_step_preprocessed.png", bgr_preprocessed)

    return bgr_resized, hsv_before, hsv_after, bgr_preprocessed



# -------------------------
# 2. Feature extraction
# -------------------------
def extract_features(hsv_img):
    h, s, v = cv2.split(hsv_img)
    feats = [
        np.mean(h), np.std(h),
        np.mean(s), np.std(s),
        np.mean(v), np.std(v),
    ]
    return np.array(feats, dtype=np.float32)


def compute_histograms(hsv_img, bins=32):
    h, s, _ = cv2.split(hsv_img)

    hist_h = cv2.calcHist([h], [0], None, [bins], [0, 180])
    hist_s = cv2.calcHist([s], [0], None, [bins], [0, 256])

    hist_h = hist_h / (hist_h.sum() + 1e-8)
    hist_s = hist_s / (hist_s.sum() + 1e-8)

    return hist_h, hist_s


# -------------------------
# 3. Simple PSNR
# -------------------------
def compute_psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")

    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


# -------------------------
# 4. Nearest-neighbor "classifier"
# -------------------------
def classify_nearest(feature, all_features, all_labels):
    diffs = all_features - feature
    dists = np.linalg.norm(diffs, axis=1)
    idx = np.argmin(dists)
    return all_labels[idx]


# -------------------------
# 5. Visualization helpers
# -------------------------
def ensure_outputs_dir():
    os.makedirs("outputs", exist_ok=True)


def show_images(original_bgr, preprocessed_bgr, true_label, predicted_label):
    """
    Save original vs preprocessed image comparison to outputs/example_images.png
    """
    ensure_outputs_dir()

    rgb_orig = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    rgb_prep = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(rgb_orig)
    plt.axis("off")
    plt.title(f"Original\nLabel: {true_label}")

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_prep)
    plt.axis("off")
    plt.title(f"Preprocessed\nPredicted: {predicted_label}")

    plt.tight_layout()
    out_path = os.path.join("outputs", "example_images.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved image comparison to {out_path}")


def show_histograms(hsv_before, hsv_after, title_prefix="Example"):
    """
    Save H and S histograms before and after equalization
    to outputs/example_histograms.png
    """
    ensure_outputs_dir()

    hist_h_before, hist_s_before = compute_histograms(hsv_before)
    hist_h_after, hist_s_after = compute_histograms(hsv_after)

    bins_h = np.arange(len(hist_h_before))
    bins_s = np.arange(len(hist_s_before))

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(bins_h, hist_h_before, label="Before")
    plt.plot(bins_h, hist_h_after, label="After")
    plt.title(f"{title_prefix} Hue Histogram")
    plt.xlabel("Hue bin")
    plt.ylabel("Normalized count")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(bins_s, hist_s_before, label="Before")
    plt.plot(bins_s, hist_s_after, label="After")
    plt.title(f"{title_prefix} Saturation Histogram")
    plt.xlabel("Saturation bin")
    plt.ylabel("Normalized count")
    plt.legend()

    plt.tight_layout()
    out_path = os.path.join("outputs", "example_histograms.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved histograms to {out_path}")

def plot_feature_bars(all_data):
    """
    Creates a bar chart comparing mean H, S, V for each ripeness class.
    Saves to outputs/feature_bars.png.
    """
    ensure_outputs_dir()

    # Collect features per class
    per_class = {}
    for sample in all_data:
        label = sample["label"]
        feats = sample["features"]  # [meanH, stdH, meanS, stdS, meanV, stdV]
        per_class.setdefault(label, []).append(feats)

    # Compute average feature vector for each class
    class_names = sorted(per_class.keys())
    avg_feats = {}
    for label in class_names:
        arr = np.vstack(per_class[label])
        avg_feats[label] = np.mean(arr, axis=0)

    # We only care about meanH (0), meanS (2), meanV (4)
    metrics_idx = [0, 2, 4]
    metrics_names = ["Hue", "Saturation", "Value"]

    x = np.arange(len(metrics_idx))  # 0,1,2
    width = 0.25                     # bar width

    plt.figure(figsize=(8, 4))

    for i, label in enumerate(class_names):
        means = [avg_feats[label][j] for j in metrics_idx]
        plt.bar(x + i * width, means, width, label=label)

    plt.xticks(x + width, metrics_names)
    plt.ylabel("Mean channel value")
    plt.title("Average HSV Features by Ripeness Class")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join("outputs", "feature_bars.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved feature bar chart to {out_path}")


def plot_feature_scatter(all_data):
    """
    Scatter plot of mean Hue vs mean Saturation for each sample.
    Saves to outputs/feature_scatter.png.
    """
    ensure_outputs_dir()

    colors = {
        "unripe": "g",
        "ripe": "y",
        "overripe": "r"
    }

    plt.figure(figsize=(5, 5))

    for sample in all_data:
        label = sample["label"]
        feats = sample["features"]
        mean_h = feats[0]
        mean_s = feats[2]
        c = colors.get(label, "b")
        plt.scatter(mean_h, mean_s, c=c, label=label)

        # annotate with filename (optional)
        name = os.path.basename(sample["path"])
        plt.annotate(name, (mean_h, mean_s), fontsize=8)

    # Avoid duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.xlabel("Mean Hue")
    plt.ylabel("Mean Saturation")
    plt.title("Feature Space: Mean Hue vs Mean Saturation")
    plt.tight_layout()

    out_path = os.path.join("outputs", "feature_scatter.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved feature scatter plot to {out_path}")



# -------------------------
# 6. Main
# -------------------------
def main():
    dataset_root = "dataset"

    all_features = []
    all_labels = []
    all_data = []

    class_folders = [d for d in os.listdir(dataset_root)
                     if os.path.isdir(os.path.join(dataset_root, d))]
    class_folders.sort()
    print("Classes found:", class_folders)

    for label in class_folders:
        folder_path = os.path.join(dataset_root, label)

        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(folder_path, fname)
            bgr = cv2.imread(img_path)
            if bgr is None:
                print("Could not read:", img_path)
                continue

            save_steps = (len(all_data) == 0)

            bgr_resized, hsv_before, hsv_after, bgr_preprocessed = preprocess_image(
                bgr,
                save_debug=save_steps,
                label=label  # example: "ripe" or "unripe"
            )
            feats = extract_features(hsv_after)

            all_features.append(feats)
            all_labels.append(label)
            all_data.append({
                "path": img_path,
                "label": label,
                "bgr_orig": bgr_resized,
                "bgr_prep": bgr_preprocessed,
                "hsv_before": hsv_before,
                "hsv_after": hsv_after,
                "features": feats
            })

    all_features = np.vstack(all_features)
    all_labels = np.array(all_labels)

    print("Feature matrix shape:", all_features.shape)

    print("\n=== Classification using 1-NN on HSV features ===")
    for sample in all_data:
        feat = sample["features"]
        true_label = sample["label"]
        predicted = classify_nearest(feat, all_features, all_labels)
        print(f"{os.path.basename(sample['path'])}: true = {true_label}, predicted = {predicted}")

    # Use the first image for PSNR/SSIM and plots
    example = all_data[0]
    orig_gray = cv2.cvtColor(example["bgr_orig"], cv2.COLOR_BGR2GRAY)
    prep_gray = cv2.cvtColor(example["bgr_prep"], cv2.COLOR_BGR2GRAY)

    psnr_value = compute_psnr(orig_gray, prep_gray)
    ssim_value = structural_similarity(orig_gray, prep_gray, data_range=255)

    print("\n=== Quality metrics (original vs preprocessed) ===")
    print(f"PSNR: {psnr_value:.3f} dB")
    print(f"SSIM: {ssim_value:.3f}")

    predicted_label = classify_nearest(example["features"], all_features, all_labels)
    show_images(example["bgr_orig"], example["bgr_prep"],
                true_label=example["label"],
                predicted_label=predicted_label)

    show_histograms(example["hsv_before"], example["hsv_after"],
                    title_prefix=example["label"].capitalize())
    
    plot_feature_bars(all_data)
    plot_feature_scatter(all_data)




if __name__ == "__main__":
    main()
