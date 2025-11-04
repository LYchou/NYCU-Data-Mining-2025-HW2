"""
v4 Cluster-wise One-Class Anomaly Detection

- Step 1: Global embedding (HOG + HSV hist) and auto-select k via silhouette
- Step 2: For each cluster, train scaler->PCA->KMeans/OCSVM and per-cluster calibrators
- Step 3: Inference: assign image to a cluster via embedding → cluster models on multi-scale patches → top-k aggregation

Artifacts are saved under v4_artifacts/ with v4_ prefix. Submission: v4_submission.csv
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
from PIL import Image
from tqdm import tqdm

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import hog, local_binary_pattern
from skimage.filters import sobel, laplace, gaussian


# -------------------------------
# Config
# -------------------------------

CONFIG = {
    "paths": {
        "train_dir": "Dataset/train",
        "test_dir": "Dataset/test",
        "workdir": "v4_artifacts",
        "output_csv": "v4_submission.csv",
    },
    "embed": {  # global embedding for clustering
        "resize": (128, 128),
        "cand_k_min": 8,
        "cand_k_max": 50,
        "cand_k_step": 2,
        "subsample": 1,  # 1 means use all, 2 means take every 2nd image
    },
    "per_cluster": {
        "pca_keep_var": 0.95,
        "pca_max_dim": 64,
        "kmeans_k": 32,
        "kmeans_n_init": 10,
        "ocsvm_nu": 0.05,
        "ocsvm_gamma_mode": "scale",  # or "1_over_d"
        "ocsvm_subsample": 16000,
        "q_low": 0.05,
        "q_high": 0.995,
    },
    "patch": {"sizes": [32, 64, 128], "stride_ratio": 0.5},
    "aggregate": {"topk_ratio": 0.01, "threshold": 0.8, "blend_kmeans": 0.7, "blend_ocsvm": 0.3},
    "seed": 42,
}

FILE_PREFIX = "v4_"


# -------------------------------
# Utils
# -------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)


def list_png_numeric(dir_path: str) -> List[str]:
    p = Path(dir_path)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    paths = [str(x) for x in p.glob("*.png")]
    if not paths:
        raise RuntimeError(f"No PNG images in {dir_path}")

    def key_fn(s: str) -> Tuple[int, str]:
        stem = Path(s).stem
        try:
            return (int(stem), "")
        except ValueError:
            return (sys.maxsize, stem)

    paths.sort(key=key_fn)
    return paths


def load_image_rgb(path: str, size_hw: Tuple[int, int]) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize((size_hw[1], size_hw[0]), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def wfile(args: argparse.Namespace, name: str) -> Path:
    return Path(args.workdir) / f"{FILE_PREFIX}{name}"


# -------------------------------
# Global embedding (HOG + HSV hist)
# -------------------------------

def embed_feature(img: np.ndarray) -> np.ndarray:
    # HOG (gray)
    g = rgb2gray(img).astype(np.float32)
    h = hog(g, pixels_per_cell=(16, 16), cells_per_block=(2, 2), feature_vector=True)
    # HSV 3D hist (8x8x8)
    hsv = rgb2hsv(img)
    hist, _ = np.histogramdd(
        hsv.reshape(-1, 3), bins=(8, 8, 8), range=((0, 1), (0, 1), (0, 1)), density=True
    )
    return np.concatenate([h.astype(np.float32), hist.flatten().astype(np.float32)])


def build_embeddings(image_paths: Sequence[str], size_hw: Tuple[int, int], step: int = 1) -> np.ndarray:
    feats: List[np.ndarray] = []
    for p in tqdm(image_paths[::step], desc="Build embeddings"):
        img = load_image_rgb(p, size_hw)
        feats.append(embed_feature(img))
    return np.stack(feats, axis=0).astype(np.float32)


def auto_select_k(X: np.ndarray, k_min: int, k_max: int, k_step: int) -> int:
    best_k, best_s = None, -1.0
    for k in range(k_min, k_max + 1, k_step):
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=CONFIG["seed"]).fit(X)
        try:
            s = silhouette_score(X, km.labels_, metric="euclidean")
        except Exception:
            s = -1.0
        if s > best_s:
            best_s, best_k = s, k
    if best_k is None:
        best_k = max(k_min, 2)
    return best_k


# -------------------------------
# Patch features (LBP uint8 + edges + DoG + FFT)
# -------------------------------

@dataclass
class FeatureConfig:
    use_lbp: bool = True
    use_edges: bool = True
    use_dog: bool = True
    use_fft: bool = True
    lbp_P: int = 8
    lbp_R: float = 1.0
    edge_threshold: float = 0.2
    dog_sigma1: float = 1.0
    dog_sigma2: float = 2.5
    fft_bins: Tuple[float, float, float, float] = (0.0, 0.2, 0.5, 1.0)


def sliding_windows(img: np.ndarray, patch: int, stride: int) -> Iterable[np.ndarray]:
    H, W, _ = img.shape
    if H < patch or W < patch:
        return []
    for y in range(0, H - patch + 1, stride):
        for x in range(0, W - patch + 1, stride):
            yield img[y : y + patch, x : x + patch, :]


def make_patches(img: np.ndarray, patch_sizes: Sequence[int], stride_ratio: float) -> List[np.ndarray]:
    patches: List[np.ndarray] = []
    for p in patch_sizes:
        stride = max(1, int(p * stride_ratio))
        patches.extend(list(sliding_windows(img, p, stride)))
    return patches


def feats_from_patch(rgb_patch: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    g_f = rgb2gray(rgb_patch).astype(np.float32)
    g_u8 = (np.clip(np.round(g_f * 255.0), 0, 255)).astype(np.uint8)
    vec: List[np.ndarray] = []

    if cfg.use_lbp:
        lbp = local_binary_pattern(g_u8, P=cfg.lbp_P, R=cfg.lbp_R, method="uniform")
        bins = np.arange(0, cfg.lbp_P + 3)
        hist, _ = np.histogram(lbp, bins=bins, range=(0, cfg.lbp_P + 2), density=True)
        vec.append(hist.astype(np.float32))

    if cfg.use_edges:
        ed1, ed2 = sobel(g_f), laplace(g_f)
        thr = cfg.edge_threshold
        vec.append(
            np.array(
                [ed1.mean(), ed1.std(), (ed1 > thr).mean(), ed2.mean(), ed2.std(), (ed2 > thr).mean()],
                dtype=np.float32,
            )
        )

    if cfg.use_dog:
        dog = np.abs(gaussian(g_f, cfg.dog_sigma1) - gaussian(g_f, cfg.dog_sigma2))
        vec.append(np.array([dog.mean(), dog.std(), (dog > 0.1).mean()], dtype=np.float32))

    if cfg.use_fft:
        F = np.abs(np.fft.rfft2(g_f))
        F = F / (F.sum() + 1e-8)
        Hf, Wf = F.shape
        yy, xx = np.mgrid[0:Hf, 0:Wf]
        r = np.sqrt((yy / max(Hf - 1, 1)) ** 2 + (xx / max(Wf - 1, 1)) ** 2)
        bins = list(cfg.fft_bins)
        bucket: List[float] = []
        for i in range(len(bins) - 1):
            m = (r >= bins[i]) & (r < bins[i + 1])
            bucket.append(float(F[m].sum()))
        vec.append(np.asarray(bucket, dtype=np.float32))

    return np.concatenate(vec, axis=0)


def extract_features_from_patches(patches: Sequence[np.ndarray], cfg: FeatureConfig) -> np.ndarray:
    feats = [feats_from_patch(p, cfg) for p in patches]
    return np.stack(feats, axis=0).astype(np.float32) if feats else np.empty((0, 0), dtype=np.float32)


# -------------------------------
# Calibration & aggregation
# -------------------------------

def fit_quantile_calibrator(scores: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float]:
    lo = float(np.quantile(scores, q_low))
    hi = float(np.quantile(scores, q_high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def apply_calibrator(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    z = (scores - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0)


def kmeans_distance_scores(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1)
    cross = X @ centers.T
    d2 = x2 + c2[None, :] - 2.0 * cross
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(np.min(d2, axis=1))
    return d.astype(np.float32)


def topk_mean(values: np.ndarray, ratio: float) -> float:
    if values.size == 0:
        return 0.0
    n = max(1, int(values.size * ratio))
    return float(np.sort(values)[-n:].mean())


# -------------------------------
# Train / Infer
# -------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(CONFIG["seed"])
    work = Path(args.workdir); work.mkdir(parents=True, exist_ok=True)

    # 1) Build embeddings on train and choose k
    train_paths = list_png_numeric(args.train_dir)
    Xtr_embed = build_embeddings(train_paths, CONFIG["embed"]["resize"], step=CONFIG["embed"]["subsample"])
    k = auto_select_k(Xtr_embed, CONFIG["embed"]["cand_k_min"], CONFIG["embed"]["cand_k_max"], CONFIG["embed"]["cand_k_step"])
    embed_km = KMeans(n_clusters=k, n_init=10, random_state=CONFIG["seed"]).fit(Xtr_embed)
    joblib.dump(embed_km, wfile(args, "embed_kmeans.joblib"))
    with open(wfile(args, "embed_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"k": int(k), "resize": CONFIG["embed"]["resize"]}, f)

    # 2) For each cluster, gather patch features from member images and train cluster models
    # Assign cluster labels for all train images (no subsample)
    # Recompute embedding for exact alignment
    Xtr_all = build_embeddings(train_paths, CONFIG["embed"]["resize"], step=1)
    labels = embed_km.predict(Xtr_all)

    # compute global mean/std for RGB normalization for patch features
    # reuse 256x256 like v3 pipeline
    def online_mean_std(paths: Sequence[str], size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        count = 0
        s = np.zeros(3, dtype=np.float64)
        s2 = np.zeros(3, dtype=np.float64)
        for pth in tqdm(paths, desc="RGB mean/std"):
            arr = load_image_rgb(pth, size)
            pix = arr.reshape(-1, 3).astype(np.float64)
            count += pix.shape[0]
            s += pix.sum(axis=0)
            s2 += (pix * pix).sum(axis=0)
        m = s / max(count, 1)
        v = s2 / max(count, 1) - m * m
        v = np.maximum(v, 1e-12)
        return m.astype(np.float32), (np.sqrt(v) + 1e-6).astype(np.float32)

    mean, std = online_mean_std(train_paths, (256, 256))
    with open(wfile(args, "rgb_norm.json"), "w", encoding="utf-8") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, f)

    feat_cfg = FeatureConfig(edge_threshold=0.2, dog_sigma1=1.0, dog_sigma2=2.5)

    def standardize_rgb(arr: np.ndarray) -> np.ndarray:
        return (arr - mean[None, None, :]) / std[None, None, :]

    for cid in range(k):
        member_idx = np.where(labels == cid)[0]
        if member_idx.size == 0:
            continue
        X_list: List[np.ndarray] = []
        for idx in tqdm(member_idx.tolist(), desc=f"Cluster {cid} features"):
            img = load_image_rgb(train_paths[idx], (256, 256))
            img = standardize_rgb(img)
            patches = make_patches(img, CONFIG["patch"]["sizes"], CONFIG["patch"]["stride_ratio"])
            Xp = extract_features_from_patches(patches, feat_cfg)
            if Xp.size > 0:
                X_list.append(Xp)
        if not X_list:
            continue
        X = np.concatenate(X_list, axis=0)

        scaler = StandardScaler(with_mean=True, with_std=True)
        X_std = scaler.fit_transform(X)
        pca = PCA(n_components=min(CONFIG["per_cluster"]["pca_max_dim"], X_std.shape[1]), random_state=CONFIG["seed"])
        pca.fit(X_std)
        # keep_var trimming
        cum = np.cumsum(pca.explained_variance_ratio_)
        d = int(np.searchsorted(cum, CONFIG["per_cluster"]["pca_keep_var"]) + 1)
        if d < pca.n_components_:
            pca = PCA(n_components=d, random_state=CONFIG["seed"]).fit(X_std)
        X_pca = pca.transform(X_std)

        km = KMeans(n_clusters=CONFIG["per_cluster"]["kmeans_k"], n_init=CONFIG["per_cluster"]["kmeans_n_init"], random_state=CONFIG["seed"]).fit(X_pca)
        km_scores = kmeans_distance_scores(X_pca, km.cluster_centers_)
        km_lo, km_hi = fit_quantile_calibrator(km_scores, CONFIG["per_cluster"]["q_low"], CONFIG["per_cluster"]["q_high"])

        # OCSVM subsample
        sub_n = min(CONFIG["per_cluster"]["ocsvm_subsample"], X_pca.shape[0])
        sub_idx = np.random.choice(X_pca.shape[0], size=sub_n, replace=False)
        gamma = (1.0 / max(1, X_pca.shape[1]) if CONFIG["per_cluster"]["ocsvm_gamma_mode"] == "1_over_d" else "scale")
        oc = OneClassSVM(kernel="rbf", nu=CONFIG["per_cluster"]["ocsvm_nu"], gamma=gamma).fit(X_pca[sub_idx])
        oc_scores = (-oc.decision_function(X_pca).reshape(-1)).astype(np.float32)
        oc_lo, oc_hi = fit_quantile_calibrator(oc_scores, CONFIG["per_cluster"]["q_low"], CONFIG["per_cluster"]["q_high"]) 

        # save cluster models
        joblib.dump(scaler, wfile(args, f"group_{cid}_scaler.joblib"))
        joblib.dump(pca,    wfile(args, f"group_{cid}_pca.joblib"))
        joblib.dump(km,     wfile(args, f"group_{cid}_kmeans.joblib"))
        joblib.dump(oc,     wfile(args, f"group_{cid}_ocsvm.joblib"))
        with open(wfile(args, f"group_{cid}_calib.json"), "w", encoding="utf-8") as f:
            json.dump({"km": {"lo": km_lo, "hi": km_hi}, "oc": {"lo": oc_lo, "hi": oc_hi}}, f)

    # Optional: estimate image-level threshold from train images (subset)
    blend = CONFIG["aggregate"]

    def image_score_from_models(img_arr: np.ndarray, cfg: FeatureConfig,
                                sizes: Sequence[int], stride_ratio: float,
                                scaler: StandardScaler, pca: PCA,
                                km: KMeans, oc: OneClassSVM,
                                calib: Dict, blend_cfg: Dict) -> float:
        patches = make_patches(img_arr, sizes, stride_ratio)
        X = extract_features_from_patches(patches, cfg)
        if X.size == 0:
            return 0.0
        X_std = scaler.transform(X)
        X_pca = pca.transform(X_std)
        km_scores = kmeans_distance_scores(X_pca, km.cluster_centers_)
        km_z = apply_calibrator(km_scores, float(calib["km"]["lo"]), float(calib["km"]["hi"]))
        oc_scores = (-oc.decision_function(X_pca).reshape(-1)).astype(np.float32)
        oc_z = apply_calibrator(oc_scores, float(calib["oc"]["lo"]), float(calib["oc"]["hi"]))
        return (
            blend_cfg["blend_kmeans"] * topk_mean(km_z, blend_cfg["topk_ratio"]) +
            blend_cfg["blend_ocsvm"]  * topk_mean(oc_z, blend_cfg["topk_ratio"]) 
        ) / max(blend_cfg["blend_kmeans"] + blend_cfg["blend_ocsvm"], 1e-8)

    scores_train_imgs: List[float] = []
    max_eval = min(500, len(train_paths))
    for pth in tqdm(train_paths[:max_eval], desc="Train image-level scores"):
        # cluster assignment
        img_embed = load_image_rgb(pth, CONFIG["embed"]["resize"])
        z = embed_feature(img_embed)
        cid = int(embed_km.predict(z.reshape(1, -1))[0])
        # load cluster models
        scaler: StandardScaler = joblib.load(wfile(args, f"group_{cid}_scaler.joblib"))
        pca: PCA = joblib.load(wfile(args, f"group_{cid}_pca.joblib"))
        km: KMeans = joblib.load(wfile(args, f"group_{cid}_kmeans.joblib"))
        oc: OneClassSVM = joblib.load(wfile(args, f"group_{cid}_ocsvm.joblib"))
        with open(wfile(args, f"group_{cid}_calib.json"), "r", encoding="utf-8") as f:
            calib = json.load(f)
        # full image normalized
        img_full = load_image_rgb(pth, (256, 256))
        img_full = (img_full - mean[None, None, :]) / std[None, None, :]
        s = image_score_from_models(img_full, feat_cfg, CONFIG["patch"]["sizes"], CONFIG["patch"]["stride_ratio"],
                                    scaler, pca, km, oc, calib, blend)
        scores_train_imgs.append(float(s))

    if scores_train_imgs:
        thr = float(np.quantile(np.asarray(scores_train_imgs), 0.99))
        with open(wfile(args, "auto_threshold.json"), "w", encoding="utf-8") as f:
            json.dump({"threshold": thr}, f)
        print(f"[v4] suggested threshold = {thr:.4f} (saved to auto_threshold.json)")

    print(f"v4 training finished. Artifacts at {work}")


def infer(args: argparse.Namespace) -> None:
    set_seed(CONFIG["seed"])
    work = Path(args.workdir)

    # load embedding KMeans
    embed_km: KMeans = joblib.load(wfile(args, "embed_kmeans.joblib"))
    with open(wfile(args, "embed_meta.json"), "r", encoding="utf-8") as f:
        embed_meta = json.load(f)
    resize = tuple(embed_meta["resize"])  # type: ignore

    # rgb normalization
    with open(wfile(args, "rgb_norm.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    mean = np.array(d["mean"], dtype=np.float32)
    std = np.array(d["std"], dtype=np.float32)

    def standardize_rgb(arr: np.ndarray) -> np.ndarray:
        return (arr - mean[None, None, :]) / std[None, None, :]

    feat_cfg = FeatureConfig(edge_threshold=0.2, dog_sigma1=1.0, dog_sigma2=2.5)

    # load auto threshold if available
    thr_path = wfile(args, "auto_threshold.json")
    if thr_path.exists():
        with open(thr_path, "r", encoding="utf-8") as f:
            t = json.load(f).get("threshold", CONFIG["aggregate"]["threshold"])
        image_threshold = float(t)
    else:
        image_threshold = CONFIG["aggregate"]["threshold"]

    test_paths = list_png_numeric(args.test_dir)
    scores_img: List[float] = []
    for p in tqdm(test_paths, desc="v4 infer"):
        # cluster assignment by embedding
        img_embed = load_image_rgb(p, resize)
        z = embed_feature(img_embed)
        cid = int(embed_km.predict(z.reshape(1, -1))[0])

        # load cluster models
        scaler: StandardScaler = joblib.load(wfile(args, f"group_{cid}_scaler.joblib"))
        pca: PCA = joblib.load(wfile(args, f"group_{cid}_pca.joblib"))
        km: KMeans = joblib.load(wfile(args, f"group_{cid}_kmeans.joblib"))
        oc: OneClassSVM = joblib.load(wfile(args, f"group_{cid}_ocsvm.joblib"))
        with open(wfile(args, f"group_{cid}_calib.json"), "r", encoding="utf-8") as f:
            calib = json.load(f)

        # patch features and scores
        img_full = load_image_rgb(p, (256, 256))
        img_full = standardize_rgb(img_full)
        patches = make_patches(img_full, CONFIG["patch"]["sizes"], CONFIG["patch"]["stride_ratio"])
        X = extract_features_from_patches(patches, feat_cfg)
        if X.size == 0:
            scores_img.append(0.0)
            continue
        X_std = scaler.transform(X)
        X_pca = pca.transform(X_std)
        km_scores = kmeans_distance_scores(X_pca, km.cluster_centers_)
        km_z = apply_calibrator(km_scores, float(calib["km"]["lo"]), float(calib["km"]["hi"]))
        oc_scores = (-oc.decision_function(X_pca).reshape(-1)).astype(np.float32)
        oc_z = apply_calibrator(oc_scores, float(calib["oc"]["lo"]), float(calib["oc"]["hi"]))

        blend = CONFIG["aggregate"]
        img_score = (
            blend["blend_kmeans"] * topk_mean(km_z, blend["topk_ratio"]) +
            blend["blend_ocsvm"]  * topk_mean(oc_z, blend["topk_ratio"]) 
        ) / max(blend["blend_kmeans"] + blend["blend_ocsvm"], 1e-8)
        scores_img.append(float(img_score))

    # to binary
    preds = (np.asarray(scores_img) >= image_threshold).astype(int)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("id,prediction\n")
        for i, y in enumerate(preds.tolist()):
            f.write(f"{i},{y}\n")
    print(f"Wrote submission to {out_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v4 Cluster-wise OC")
    p.add_argument("--mode", type=str, default="all", choices=["train", "infer", "all"])
    p.add_argument("--train_dir", type=str, default=str(Path(CONFIG["paths"]["train_dir"]).resolve()))
    p.add_argument("--test_dir", type=str, default=str(Path(CONFIG["paths"]["test_dir"]).resolve()))
    p.add_argument("--workdir", type=str, default=str(Path(CONFIG["paths"]["workdir"]).resolve()))
    p.add_argument("--output_csv", type=str, default=str(Path(CONFIG["paths"]["output_csv"]).resolve()))
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode in ("train", "all"):
        train(args)
    if args.mode in ("infer", "all"):
        infer(args)


if __name__ == "__main__":
    main()


