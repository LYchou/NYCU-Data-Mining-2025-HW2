import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from skimage.feature import hog, local_binary_pattern


def list_image_paths(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Directory not found: {directory}")
    image_paths = []
    for name in os.listdir(directory):
        if name.lower().endswith(".png"):
            image_paths.append(os.path.join(directory, name))
    if not image_paths:
        raise RuntimeError(f"No PNG images found in: {directory}")
    return image_paths


def numeric_stem(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        return sys.maxsize


def load_gray(image_path: str, size: Tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as img:
        img = img.convert("L").resize(size, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


def extract_hog(image: np.ndarray, orientations: int, pixels_per_cell: Tuple[int, int], cells_per_block: Tuple[int, int]) -> np.ndarray:
    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    ).astype(np.float32)


def extract_lbp_hist(image: np.ndarray, radius: int, n_points: int) -> np.ndarray:
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    n_bins = n_points + 2  # per uniform LBP definition
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist.astype(np.float32)


def extract_features(paths: List[str], size: Tuple[int, int], hog_params: dict, lbp_params: dict) -> np.ndarray:
    features: List[np.ndarray] = []
    for idx, p in enumerate(paths):
        img = load_gray(p, size)
        f_hog = extract_hog(img, **hog_params)
        f_lbp = extract_lbp_hist(img, **lbp_params)
        feat = np.concatenate([f_hog, f_lbp], axis=0)
        features.append(feat)
        if (idx + 1) % 500 == 0:
            print(f"Extracted features for {idx + 1}/{len(paths)} images", file=sys.stderr)
    return np.stack(features, axis=0)


def _nearest_center_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x_norm2 = np.sum(X * X, axis=1, keepdims=True)
    c_norm2 = np.sum(centers * centers, axis=1)
    cross = X @ centers.T
    d2 = x_norm2 + c_norm2[None, :] - 2.0 * cross
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(np.min(d2, axis=1))


def minmax_by_quantile(values: np.ndarray, q_low: float = 0.05, q_high: float = 0.95) -> Tuple[np.ndarray, float, float]:
    lo = float(np.quantile(values, q_low))
    hi = float(np.quantile(values, q_high))
    if hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values) + 1e-6)
    scaled = (values - lo) / max(hi - lo, 1e-6)
    return np.clip(scaled, 0.0, 1.0), lo, hi


def ensure_v2_prefix(path: str) -> str:
    directory = os.path.dirname(path)
    base = os.path.basename(path)
    if not base.startswith("v2_"):
        base = "v2_" + base
    return os.path.join(directory, base) if directory else base


def write_submission(rows: List[Tuple[int, int]], output_csv: str) -> None:
    dirpath = os.path.dirname(output_csv)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("id,prediction\n")
        for idx, pred in rows:
            f.write(f"{idx},{pred}\n")
    print(f"Wrote submission to {output_csv}")


def write_scores(rows: List[Tuple[int, float]], output_csv: str) -> None:
    dirpath = os.path.dirname(output_csv)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("id,prediction\n")
        for idx, score in rows:
            f.write(f"{idx},{score}\n")
    print(f"Wrote scores to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v2 HOG+LBP features with KMeans+OCSVM fusion for anomaly detection")
    parser.add_argument("--train_dir", type=str, default=os.path.abspath("Dataset/train"))
    parser.add_argument("--test_dir", type=str, default=os.path.abspath("Dataset/test"))
    parser.add_argument("--output_csv", type=str, default=os.path.abspath("v2_submission.csv"))
    parser.add_argument("--image_size", type=int, nargs=2, default=(128, 128), metavar=("W", "H"))
    # HOG
    parser.add_argument("--hog_orientations", type=int, default=9)
    parser.add_argument("--hog_pixels_per_cell", type=int, nargs=2, default=(8, 8))
    parser.add_argument("--hog_cells_per_block", type=int, nargs=2, default=(2, 2))
    # LBP
    parser.add_argument("--lbp_radius", type=int, default=1)
    parser.add_argument("--lbp_points", type=int, default=8)
    # PCA / KMeans
    parser.add_argument("--pca_components", type=int, default=128)
    parser.add_argument("--kmeans_k", type=int, default=30)
    # OCSVM
    parser.add_argument("--ocsvm_nu", type=float, default=0.05)
    parser.add_argument("--ocsvm_gamma", type=str, default="scale")
    # Fusion & output
    parser.add_argument("--weight_kmeans", type=float, default=0.5)
    parser.add_argument("--weight_ocsvm", type=float, default=0.5)
    parser.add_argument("--output_mode", type=str, choices=["binary", "score"], default="binary")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # Prepare paths
    print("Listing training images...", file=sys.stderr)
    train_paths = list_image_paths(args.train_dir)
    train_paths.sort(key=numeric_stem)

    print("Listing test images...", file=sys.stderr)
    test_paths = list_image_paths(args.test_dir)
    test_paths.sort(key=numeric_stem)

    hog_params = {
        "orientations": args.hog_orientations,
        "pixels_per_cell": tuple(args.hog_pixels_per_cell),
        "cells_per_block": tuple(args.hog_cells_per_block),
    }
    lbp_params = {
        "radius": args.lbp_radius,
        "n_points": args.lbp_points,
    }

    # Feature extraction
    print("Extracting train features (HOG+LBP)...", file=sys.stderr)
    X_train = extract_features(train_paths, tuple(args.image_size), hog_params, lbp_params)

    print("Standardizing and PCA on train...", file=sys.stderr)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_std = scaler.fit_transform(X_train)
    pca = PCA(n_components=args.pca_components, random_state=args.random_state)
    X_train_pca = pca.fit_transform(X_train_std)

    # KMeans
    print(f"Fitting KMeans (k={args.kmeans_k})...", file=sys.stderr)
    kmeans = KMeans(n_clusters=args.kmeans_k, n_init=10, random_state=args.random_state)
    kmeans.fit(X_train_pca)

    # Calibrate KMeans distances on train
    train_km_dists = _nearest_center_distances(X_train_pca, kmeans.cluster_centers_)
    _, km_lo, km_hi = minmax_by_quantile(train_km_dists)

    # OCSVM
    print("Fitting One-Class SVM...", file=sys.stderr)
    ocsvm = OneClassSVM(kernel="rbf", nu=args.ocsvm_nu, gamma=args.ocsvm_gamma)
    ocsvm.fit(X_train_pca)

    # Calibrate OCSVM anomaly score on train (-decision_function)
    train_oc_scores = -ocsvm.decision_function(X_train_pca).reshape(-1)
    _, oc_lo, oc_hi = minmax_by_quantile(train_oc_scores)

    # Test features
    print("Extracting test features (HOG+LBP)...", file=sys.stderr)
    X_test = extract_features(test_paths, tuple(args.image_size), hog_params, lbp_params)
    print("Transforming test (scaler+PCA)...", file=sys.stderr)
    X_test_std = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_std)

    # KMeans distances -> [0,1]
    print("Computing KMeans distances...", file=sys.stderr)
    km_dists = _nearest_center_distances(X_test_pca, kmeans.cluster_centers_)
    km_scores = np.clip((km_dists - km_lo) / max(km_hi - km_lo, 1e-6), 0.0, 1.0)

    # OCSVM scores -> [0,1]
    print("Computing OCSVM scores...", file=sys.stderr)
    oc_scores = -ocsvm.decision_function(X_test_pca).reshape(-1)
    oc_scores = np.clip((oc_scores - oc_lo) / max(oc_hi - oc_lo, 1e-6), 0.0, 1.0)

    # Fusion
    w_km = max(args.weight_kmeans, 0.0)
    w_oc = max(args.weight_ocsvm, 0.0)
    if w_km + w_oc <= 0:
        w_km, w_oc = 0.5, 0.5
    w_sum = w_km + w_oc
    fused = (w_km * km_scores + w_oc * oc_scores) / w_sum

    # Output
    out_path = ensure_v2_prefix(args.output_csv)
    if args.output_mode == "score":
        rows = [(i, float(s)) for i, s in enumerate(fused)]
        write_scores(rows, out_path)
    else:
        preds = (fused >= args.threshold).astype(int)
        rows = [(i, int(v)) for i, v in enumerate(preds)]
        write_submission(rows, out_path)


if __name__ == "__main__":
    main()



