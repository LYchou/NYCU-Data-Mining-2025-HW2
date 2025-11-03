"""
v1 Baseline KMeans Anomaly Detection for MVTec AD Dataset
=========================================================

程式脈絡與想法：
---------------

1. 任務背景：
   - 工業異常偵測：訓練集全為正常圖像，測試集包含正常與異常圖像
   - 目標：對測試集每張圖像輸出異常分數（0-1，越高越異常）
   - 評分指標：AUC（Area Under Curve）
   - 限制：不可使用預訓練模型、不可使用外部資料

2. 核心思路 - KMeans 距離法：
   - 假設：正常圖像在特徵空間中會形成若干個緊密的群集
   - 異常圖像會偏離這些正常群集，距離群集中心較遠
   - 策略：用 KMeans 找到正常圖像的群集中心，計算測試圖像到最近中心的距離作為異常分數

3. 技術流程：
   a) 資料前處理：
      - 轉灰階、縮放到固定尺寸（64x64，平衡效果與速度）
      - 標準化（StandardScaler）避免數值範圍差異
   
   b) 降維：
      - PCA 降維到 64 維，減少計算量並去除冗餘資訊
      - 避免維度詛咒，提升 KMeans 效果
   
   c) 群集學習：
      - KMeans 將正常圖像分為 30 個群集
      - 計算每個測試圖像到最近群集中心的歐氏距離
   
   d) 分數校準：
      - 用訓練集距離的分位數（5%, 95%）做 MinMax 正規化
      - 將距離映射到 [0,1] 區間，距離越大分數越高

4. 優化細節：
   - 高效距離計算：使用 ||x-c||² = ||x||² + ||c||² - 2x·c 避免重複計算
   - 確定性排序：按檔名數字排序確保結果可重現
   - 記憶體友善：分批載入圖像，避免一次載入全部
   - 自動前綴：輸出檔案自動加上 v1_ 前綴

5. 參數選擇考量：
   - 影像尺寸 64x64：足夠保留主要特徵，計算快速
   - PCA 64 維：保留主要變異，避免過度降維
   - KMeans k=30：假設正常圖像有約 30 種主要模式
   - 分位數校準：對抗極值影響，提升分數穩定性

6. 預期效果：
   - 此方法對紋理異常、局部缺陷較敏感
   - 作為強基線，後續可與 OCSVM、Autoencoder 等方法融合
   - 連續分數輸出適合 AUC 評分，無需閾值選擇

7. 使用方式：
   python v1_baseline_kmeans.py [--image_size 64 64] [--pca_components 64] [--kmeans_k 30] [--threshold 0.5]
   
   輸出：v1_submission.csv（id, prediction 格式，prediction 為 0 或 1 的整數值）
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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


def load_image_as_vector(image_path: str, size: Tuple[int, int]) -> np.ndarray:
    # Convert image to grayscale and resize, then flatten to 1D vector
    with Image.open(image_path) as img:
        img = img.convert("L")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr.reshape(-1)


def load_images_matrix(image_paths: List[str], size: Tuple[int, int]) -> np.ndarray:
    features = np.empty((len(image_paths), size[0] * size[1]), dtype=np.float32)
    for idx, path in enumerate(image_paths):
        features[idx] = load_image_as_vector(path, size)
        if (idx + 1) % 500 == 0:
            print(f"Loaded {idx + 1}/{len(image_paths)} images", file=sys.stderr)
    return features


def numeric_stem(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return int(stem)
    except ValueError:
        # Fallback: use lexicographic order if not purely numeric
        # Note: this will still be deterministic
        return sys.maxsize


def fit_kmeans_on_train(train_dir: str,
                        image_size: Tuple[int, int] = (64, 64),
                        pca_components: int = 64,
                        kmeans_k: int = 30,
                        random_state: int = 42):
    print("Listing training images...", file=sys.stderr)
    train_paths = list_image_paths(train_dir)
    # Sort for determinism
    train_paths.sort(key=numeric_stem)

    print("Loading training images...", file=sys.stderr)
    X = load_images_matrix(train_paths, image_size)

    print("Standardizing features...", file=sys.stderr)
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)

    print(f"Fitting PCA to {pca_components} components...", file=sys.stderr)
    pca = PCA(n_components=pca_components, svd_solver="auto", random_state=random_state)
    X_pca = pca.fit_transform(X_std)

    print(f"Fitting KMeans with k={kmeans_k}...", file=sys.stderr)
    kmeans = KMeans(n_clusters=kmeans_k, n_init=10, random_state=random_state)
    kmeans.fit(X_pca)

    # Compute training distances to nearest centroid for calibration
    print("Calibrating distance distribution on training data...", file=sys.stderr)
    train_dists = _nearest_center_distances(X_pca, kmeans.cluster_centers_)
    q_low = np.quantile(train_dists, 0.05)
    q_high = np.quantile(train_dists, 0.95)
    if q_high <= q_low:
        # Fallback to min/max if quantiles collapse
        q_low = float(np.min(train_dists))
        q_high = float(np.max(train_dists) + 1e-6)

    calibration = {
        "q_low": float(q_low),
        "q_high": float(q_high),
    }

    return scaler, pca, kmeans, calibration


def _nearest_center_distances(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    # Efficient nearest-center distance computation
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    x_norm2 = np.sum(X * X, axis=1, keepdims=True)  # (n, 1)
    c_norm2 = np.sum(centers * centers, axis=1)     # (k,)
    cross = X @ centers.T                            # (n, k)
    d2 = x_norm2 + c_norm2[None, :] - 2.0 * cross   # (n, k)
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(np.min(d2, axis=1))
    return d


def distances_to_scores(distances: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    # Map distances to [0, 1] where higher means more anomalous
    scores = (distances - q_low) / max(q_high - q_low, 1e-6)
    return np.clip(scores, 0.0, 1.0)


def scores_to_binary(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    # Convert continuous scores to binary 0/1 predictions
    return (scores >= threshold).astype(int)


def ensure_v1_prefix(path: str) -> str:
    directory = os.path.dirname(path)
    base = os.path.basename(path)
    if not base.startswith("v1_"):
        base = "v1_" + base
    return os.path.join(directory, base) if directory else base


def predict_test_scores(test_dir: str,
                        scaler: StandardScaler,
                        pca: PCA,
                        kmeans: KMeans,
                        calibration: dict,
                        image_size: Tuple[int, int] = (64, 64),
                        threshold: float = 0.5) -> List[Tuple[int, int]]:
    print("Listing test images...", file=sys.stderr)
    test_paths = list_image_paths(test_dir)
    # Sort by numeric stem for determinism and to align with common assignment conventions
    test_paths = sorted(test_paths, key=numeric_stem)

    print("Loading test images...", file=sys.stderr)
    X_test = load_images_matrix(test_paths, image_size)

    print("Transforming test features (scaler + PCA)...", file=sys.stderr)
    X_test_std = scaler.transform(X_test)
    X_test_pca = pca.transform(X_test_std)

    print("Computing distances to nearest centroid...", file=sys.stderr)
    dists = _nearest_center_distances(X_test_pca, kmeans.cluster_centers_)
    scores = distances_to_scores(dists, calibration["q_low"], calibration["q_high"])
    
    print(f"Converting scores to binary predictions (threshold={threshold})...", file=sys.stderr)
    binary_predictions = scores_to_binary(scores, threshold)

    # Map to rows with sequential ids 0..N-1 as per assignment's sample_submission
    rows: List[Tuple[int, int]] = []
    for idx, prediction in enumerate(binary_predictions):
        rows.append((idx, int(prediction)))
    return rows


def write_submission(rows: List[Tuple[int, int]], output_csv: str) -> None:
    dirpath = os.path.dirname(output_csv)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("id,prediction\n")
        for idx, score in rows:
            f.write(f"{idx},{score}\n")
    print(f"Wrote submission to {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v1 Baseline KMeans anomaly detection (no pretrained models)")
    parser.add_argument("--train_dir", type=str, default=os.path.abspath("Dataset/train"), help="Path to training images (normal only)")
    parser.add_argument("--test_dir", type=str, default=os.path.abspath("Dataset/test"), help="Path to test images")
    parser.add_argument("--output_csv", type=str, default=os.path.abspath("v1_submission.csv"), help="Output CSV path (will be prefixed with v1_ if not)")
    parser.add_argument("--image_size", type=int, nargs=2, default=(64, 64), metavar=("W", "H"), help="Resize images to W H")
    parser.add_argument("--pca_components", type=int, default=64, help="Number of PCA components")
    parser.add_argument("--kmeans_k", type=int, default=30, help="Number of KMeans clusters")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for converting scores to binary 0/1")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    scaler, pca, kmeans, calibration = fit_kmeans_on_train(
        train_dir=args.train_dir,
        image_size=tuple(args.image_size),
        pca_components=args.pca_components,
        kmeans_k=args.kmeans_k,
        random_state=args.random_state,
    )

    rows = predict_test_scores(
        test_dir=args.test_dir,
        scaler=scaler,
        pca=pca,
        kmeans=kmeans,
        calibration=calibration,
        image_size=tuple(args.image_size),
        threshold=args.threshold,
    )

    output_csv = ensure_v1_prefix(args.output_csv)
    write_submission(rows, output_csv)


if __name__ == "__main__":
    main()


