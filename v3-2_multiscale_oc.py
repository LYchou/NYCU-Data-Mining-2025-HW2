"""
v3-2 Multiscale One-Class Anomaly Detection (prioritized improvements)

- Train only on normal images in Dataset/train
- Infer on Dataset/test and output v3-2_submission.csv with id,prediction
  where id is the numeric filename order (0.png -> id 0)

Improvements over v3:
- Standardize features with StandardScaler BEFORE PCA/KMeans/OCSVM
- Add 32x32 patches and reduce top-k ratio for finer defects
- LBP computed on uint8 grayscale to avoid float artifacts
- All artifacts/use files use prefix v3-2_* under v3-2_artifacts/
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

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern
from skimage.filters import sobel, laplace, gaussian


# -------------------------------
# Global configuration (centralized defaults)
# -------------------------------

CONFIG = {
    "paths": {
        "train_dir": "Dataset/train",
        "test_dir": "Dataset/test",
        "workdir": "v3-2_artifacts",
        "output_csv": "v3-2_submission.csv",
    },
    "image": {"input_w": 256, "input_h": 256},
    "patch": {"sizes": [32, 64, 128], "stride_ratio": 0.5},
    "feature": {"edge_threshold": 0.2, "dog_sigma1": 1.0, "dog_sigma2": 2.5},
    "pca": {"keep_var": 0.95, "max_dim": 64},
    "kmeans": {"k": 32, "n_init": 10},
    "ocsvm": {"nu": 0.1, "gamma_mode": "1_over_d", "subsample": 16000},
    "calibrate": {"q_low": 0.05, "q_high": 0.95},
    "aggregate": {
        "topk_ratio": 0.02,
        "blend_kmeans": 0.5,
        "blend_ocsvm": 0.5,
        "blend_ae": 0.4,
        "threshold": 0.5,
    },
    "ae": {
        "enabled": False,
        "latent_dim": 48,
        "epochs": 20,
        "batch_size": 256,
        "lr": 1.5e-3,
        "device": "cpu",
    },
    "seed": 42,
}

FILE_PREFIX = "v3-2_"


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


def online_mean_std(image_paths: Sequence[str], size_hw: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    count = 0
    sum_c = np.zeros(3, dtype=np.float64)
    sumsq_c = np.zeros(3, dtype=np.float64)
    for p in tqdm(image_paths, desc="Compute train mean/std"):
        img = load_image_rgb(p, size_hw)
        pix = img.reshape(-1, 3).astype(np.float64)
        count += pix.shape[0]
        sum_c += pix.sum(axis=0)
        sumsq_c += (pix * pix).sum(axis=0)
    mean = sum_c / max(count, 1)
    var = sumsq_c / max(count, 1) - mean * mean
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var)
    return mean.astype(np.float32), (std + 1e-6).astype(np.float32)


def standardize_rgb(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arr - mean[None, None, :]) / std[None, None, :]


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


def feats_from_patch(rgb_patch: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    # float32 for continuous ops, uint8 for LBP stability
    g_f = rgb2gray(rgb_patch).astype(np.float32)
    g_u8 = (np.clip(np.round(g_f * 255.0), 0, 255)).astype(np.uint8)
    vec_parts: List[np.ndarray] = []

    if cfg.use_lbp:
        lbp = local_binary_pattern(g_u8, P=cfg.lbp_P, R=cfg.lbp_R, method="uniform")
        bins = np.arange(0, cfg.lbp_P + 3)
        hist, _ = np.histogram(lbp, bins=bins, range=(0, cfg.lbp_P + 2), density=True)
        vec_parts.append(hist.astype(np.float32))

    if cfg.use_edges:
        ed1 = sobel(g_f)
        ed2 = laplace(g_f)
        thr = cfg.edge_threshold
        edge_stats = np.array([
            float(ed1.mean()), float(ed1.std()), float((ed1 > thr).mean()),
            float(ed2.mean()), float(ed2.std()), float((ed2 > thr).mean())
        ], dtype=np.float32)
        vec_parts.append(edge_stats)

    if cfg.use_dog:
        g1 = gaussian(g_f, sigma=cfg.dog_sigma1)
        g2 = gaussian(g_f, sigma=cfg.dog_sigma2)
        dog = np.abs(g1 - g2)
        dog_stats = np.array([float(dog.mean()), float(dog.std()), float((dog > 0.1).mean())], dtype=np.float32)
        vec_parts.append(dog_stats)

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
        vec_parts.append(np.asarray(bucket, dtype=np.float32))

    return np.concatenate(vec_parts, axis=0)


def extract_features_from_patches(patches: Sequence[np.ndarray], cfg: FeatureConfig) -> np.ndarray:
    if len(patches) == 0:
        return np.empty((0, 0), dtype=np.float32)
    feats: List[np.ndarray] = []
    for p in patches:
        feats.append(feats_from_patch(p, cfg))
    X = np.stack(feats, axis=0).astype(np.float32)
    return X


def fit_pca_dynamic(X: np.ndarray, keep_var: float, max_dim: int, random_state: int) -> PCA:
    max_dim_eff = min(max_dim, X.shape[1])
    pca = PCA(n_components=max_dim_eff, random_state=random_state)
    pca.fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    idx = int(np.searchsorted(cum, keep_var) + 1)
    n_components = min(max_dim_eff, max(1, idx))
    if n_components != pca.n_components_:
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(X)
    return pca


def kmeans_distance_scores(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
    x2 = np.sum(X * X, axis=1, keepdims=True)
    c2 = np.sum(centers * centers, axis=1)
    cross = X @ centers.T
    d2 = x2 + c2[None, :] - 2.0 * cross
    d2 = np.maximum(d2, 0.0)
    d = np.sqrt(np.min(d2, axis=1))
    return d.astype(np.float32)


def ocsvm_scores(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    return (-model.decision_function(X).reshape(-1)).astype(np.float32)


def fit_quantile_calibrator(scores: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float]:
    lo = float(np.quantile(scores, q_low))
    hi = float(np.quantile(scores, q_high))
    if not math.isfinite(lo):
        lo = float(np.min(scores))
    if not math.isfinite(hi):
        hi = float(np.max(scores) + 1e-6)
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


def apply_calibrator(scores: np.ndarray, lo: float, hi: float) -> np.ndarray:
    z = (scores - lo) / (hi - lo)
    return np.clip(z, 0.0, 1.0)


def topk_mean(values: np.ndarray, ratio: float) -> float:
    if values.size == 0:
        return 0.0
    n = max(1, int(values.size * ratio))
    return float(np.sort(values)[-n:].mean())


def aggregate_image_scores(per_patch_scores: Dict[str, Optional[np.ndarray]], topk_ratio: float, blend: Dict[str, float]) -> float:
    pooled: Dict[str, float] = {}
    for name, arr in per_patch_scores.items():
        if arr is None or arr.size == 0:
            continue
        pooled[name] = topk_mean(arr, topk_ratio)
    num = 0.0
    den = 0.0
    for name, w in blend.items():
        if name in pooled:
            num += w * pooled[name]
            den += w
    return num / max(den, 1e-8)


class TinyAE:
    def __init__(self, latent_dim: int = 48, device: str = "cpu") -> None:
        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self, z: int):
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Conv2d(1, 16, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(16, 32, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                )
                self.enc_fc = nn.Linear(64 * 8 * 8, z)
                self.dec_fc = nn.Linear(z, 64 * 8 * 8)
                self.dec = nn.Sequential(
                    nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
                    nn.Sigmoid(),
                )

            def forward(self, x):
                h = self.enc(x)
                h = h.flatten(1)
                z = self.enc_fc(h)
                h2 = self.dec_fc(z).view(x.size(0), 64, 8, 8)
                out = self.dec(h2)
                return out

        self.device = device
        self.net = Net(latent_dim).to(self.device)

    def train_fit(self, patches64_gray: np.ndarray, epochs: int = 20, batch_size: int = 256, lr: float = 1.5e-3) -> None:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        x = torch.from_numpy(patches64_gray).float().unsqueeze(1)
        ds = TensorDataset(x, x)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        crit = nn.MSELoss(reduction="mean")
        self.net.train()
        for _ in tqdm(range(epochs), desc="Train AE"):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                opt.zero_grad(set_to_none=True)
                out = self.net(xb)
                loss = crit(out, yb)
                loss.backward()
                opt.step()

    def score(self, patches64_gray: np.ndarray) -> np.ndarray:
        import torch
        import torch.nn.functional as F
        self.net.eval()
        with torch.no_grad():
            x = torch.from_numpy(patches64_gray).float().unsqueeze(1).to(self.device)
            out = self.net(x)
            mse = F.mse_loss(out, x, reduction="none")
            mse = mse.view(mse.size(0), -1).mean(dim=1).cpu().numpy().astype(np.float32)
        return mse

    def save(self, path: str) -> None:
        import torch
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        import torch
        self.net.load_state_dict(torch.load(path, map_location=self.device))


def wfile(args: argparse.Namespace, name: str) -> Path:
    return Path(args.workdir) / f"{FILE_PREFIX}{name}"


def train_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    train_paths = list_png_numeric(args.train_dir)
    ih, iw = CONFIG["image"]["input_h"], CONFIG["image"]["input_w"]
    mean, std = online_mean_std(train_paths, (ih, iw))

    feat_cfg = FeatureConfig(
        use_lbp=True,
        use_edges=True,
        use_dog=True,
        use_fft=True,
        lbp_P=8,
        lbp_R=1.0,
        edge_threshold=CONFIG["feature"]["edge_threshold"],
        dog_sigma1=CONFIG["feature"]["dog_sigma1"],
        dog_sigma2=CONFIG["feature"]["dog_sigma2"],
        fft_bins=(0.0, 0.2, 0.5, 1.0),
    )

    X_list: List[np.ndarray] = []
    patches64_for_ae: List[np.ndarray] = []
    for p in tqdm(train_paths, desc="Extract train features"):
        img = load_image_rgb(p, (ih, iw))
        img = standardize_rgb(img, mean, std)
        patches = make_patches(img, CONFIG["patch"]["sizes"], CONFIG["patch"]["stride_ratio"])
        Xp = extract_features_from_patches(patches, feat_cfg)
        X_list.append(Xp)
        if args.ae_enabled:
            for patch_sz in CONFIG["patch"]["sizes"]:
                if patch_sz == 64:
                    stride = max(1, int(patch_sz * CONFIG["patch"]["stride_ratio"]))
                    for win in sliding_windows(img, patch_sz, stride):
                        g = rgb2gray(win).astype(np.float32)
                        if g.shape[0] == 64 and g.shape[1] == 64:
                            patches64_for_ae.append(g)

    X = np.concatenate(X_list, axis=0) if len(X_list) > 0 else np.empty((0, 0), dtype=np.float32)

    # Standardize -> PCA -> Models
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)
    pca = fit_pca_dynamic(X_std, CONFIG["pca"]["keep_var"], CONFIG["pca"]["max_dim"], args.seed)
    X_pca = pca.transform(X_std)

    kmeans = KMeans(n_clusters=CONFIG["kmeans"]["k"], n_init=CONFIG["kmeans"]["n_init"], random_state=args.seed)
    kmeans.fit(X_pca)
    km_scores_train = kmeans_distance_scores(X_pca, kmeans.cluster_centers_)
    km_lo, km_hi = fit_quantile_calibrator(km_scores_train, CONFIG["calibrate"]["q_low"], CONFIG["calibrate"]["q_high"])

    ocsvm_sub_n = min(CONFIG["ocsvm"]["subsample"], X_pca.shape[0])
    if ocsvm_sub_n <= 0:
        raise RuntimeError("Invalid OCSVM subsample size")
    idx = np.random.choice(X_pca.shape[0], size=ocsvm_sub_n, replace=False)
    X_sub = X_pca[idx]
    gamma = 1.0 / max(1, X_pca.shape[1]) if CONFIG["ocsvm"]["gamma_mode"] == "1_over_d" else "scale"
    ocsvm = OneClassSVM(kernel="rbf", nu=CONFIG["ocsvm"]["nu"], gamma=gamma)
    ocsvm.fit(X_sub)
    svm_scores_train = ocsvm_scores(ocsvm, X_pca)
    svm_lo, svm_hi = fit_quantile_calibrator(svm_scores_train, CONFIG["calibrate"]["q_low"], CONFIG["calibrate"]["q_high"])

    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(wfile(args, "mean_std.npz"), mean=mean, std=std)
    joblib.dump(scaler, wfile(args, "scaler.joblib"))
    joblib.dump(pca, wfile(args, "pca.joblib"))
    joblib.dump(kmeans, wfile(args, "kmeans.joblib"))
    joblib.dump(ocsvm, wfile(args, "ocsvm.joblib"))
    with open(wfile(args, "calib_kmeans.json"), "w", encoding="utf-8") as f:
        json.dump({"lo": km_lo, "hi": km_hi}, f)
    with open(wfile(args, "calib_ocsvm.json"), "w", encoding="utf-8") as f:
        json.dump({"lo": svm_lo, "hi": svm_hi}, f)

    if args.ae_enabled and len(patches64_for_ae) > 0:
        try:
            import torch  # noqa: F401
            ae = TinyAE(latent_dim=CONFIG["ae"]["latent_dim"], device=args.ae_device)
            patches64 = np.stack(patches64_for_ae, axis=0)
            ae.train_fit(patches64, epochs=CONFIG["ae"]["epochs"], batch_size=CONFIG["ae"]["batch_size"], lr=CONFIG["ae"]["lr"])
            ae_scores = ae.score(patches64)
            ae_lo, ae_hi = fit_quantile_calibrator(ae_scores, CONFIG["calibrate"]["q_low"], CONFIG["calibrate"]["q_high"])
            ae.save(str(wfile(args, "ae.pt")))
            with open(wfile(args, "calib_ae.json"), "w", encoding="utf-8") as f:
                json.dump({"lo": float(ae_lo), "hi": float(ae_hi)}, f)
        except Exception as e:
            print(f"[AE] failed to train/save AE: {e}")

    print(f"Artifacts saved to {work} with prefix {FILE_PREFIX}")


def ensure_artifacts_exist(args: argparse.Namespace, need_ae: bool) -> None:
    required = ["mean_std.npz", "scaler.joblib", "pca.joblib", "kmeans.joblib", "ocsvm.joblib", "calib_kmeans.json", "calib_ocsvm.json"]
    miss = [n for n in required if not wfile(args, n).exists()]
    if miss:
        raise RuntimeError(f"Artifacts missing {[FILE_PREFIX + n for n in miss]}. 請先執行 --mode train")
    if need_ae and not wfile(args, "ae.pt").exists():
        raise RuntimeError("缺少 v3-2_ae.pt。請先以 --ae_enabled true 訓練，或推論時關閉 AE。")


def infer_pipeline(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    ensure_artifacts_exist(args, need_ae=args.ae_enabled)

    mean_std = np.load(wfile(args, "mean_std.npz"))
    mean = mean_std["mean"].astype(np.float32)
    std = mean_std["std"].astype(np.float32)
    scaler: StandardScaler = joblib.load(wfile(args, "scaler.joblib"))
    pca: PCA = joblib.load(wfile(args, "pca.joblib"))
    kmeans: KMeans = joblib.load(wfile(args, "kmeans.joblib"))
    ocsvm: OneClassSVM = joblib.load(wfile(args, "ocsvm.joblib"))
    with open(wfile(args, "calib_kmeans.json"), "r", encoding="utf-8") as f:
        km_calib = json.load(f)
    with open(wfile(args, "calib_ocsvm.json"), "r", encoding="utf-8") as f:
        svm_calib = json.load(f)

    ae = None
    ae_calib = None
    if args.ae_enabled and wfile(args, "ae.pt").exists() and wfile(args, "calib_ae.json").exists():
        try:
            import torch  # noqa: F401
            ae = TinyAE(latent_dim=CONFIG["ae"]["latent_dim"], device=args.ae_device)
            ae.load(str(wfile(args, "ae.pt")))
            with open(wfile(args, "calib_ae.json"), "r", encoding="utf-8") as f:
                ae_calib = json.load(f)
        except Exception as e:
            print(f"[AE] Unable to load AE ({e}); AE disabled.")
            ae = None
            ae_calib = None

    test_paths = list_png_numeric(args.test_dir)
    feat_cfg = FeatureConfig(
        use_lbp=True,
        use_edges=True,
        use_dog=True,
        use_fft=True,
        lbp_P=8,
        lbp_R=1.0,
        edge_threshold=CONFIG["feature"]["edge_threshold"],
        dog_sigma1=CONFIG["feature"]["dog_sigma1"],
        dog_sigma2=CONFIG["feature"]["dog_sigma2"],
        fft_bins=(0.0, 0.2, 0.5, 1.0),
    )

    image_scores: List[float] = []
    ih, iw = CONFIG["image"]["input_h"], CONFIG["image"]["input_w"]
    for p in tqdm(test_paths, desc="Infer test"):
        img = load_image_rgb(p, (ih, iw))
        img = standardize_rgb(img, mean, std)
        patches = make_patches(img, CONFIG["patch"]["sizes"], CONFIG["patch"]["stride_ratio"])
        X = extract_features_from_patches(patches, feat_cfg)
        X_std = scaler.transform(X)
        X_pca = pca.transform(X_std)
        km_scores = kmeans_distance_scores(X_pca, kmeans.cluster_centers_)
        km_z = apply_calibrator(km_scores, float(km_calib["lo"]), float(km_calib["hi"]))
        svm_scores = ocsvm_scores(ocsvm, X_pca)
        svm_z = apply_calibrator(svm_scores, float(svm_calib["lo"]), float(svm_calib["hi"]))

        ae_z = None
        if ae is not None and ae_calib is not None:
            patches64: List[np.ndarray] = []
            for patch_sz in CONFIG["patch"]["sizes"]:
                if patch_sz == 64:
                    stride = max(1, int(patch_sz * CONFIG["patch"]["stride_ratio"]))
                    for win in sliding_windows(img, patch_sz, stride):
                        g = rgb2gray(win).astype(np.float32)
                        if g.shape[0] == 64 and g.shape[1] == 64:
                            patches64.append(g)
            if len(patches64) > 0:
                arr = np.stack(patches64, axis=0)
                ae_scores = ae.score(arr)
                ae_z = apply_calibrator(ae_scores, float(ae_calib["lo"]), float(ae_calib["hi"]))

        if args.ae_enabled and (ae is not None) and (ae_z is not None):
            blend = {"kmeans": CONFIG["aggregate"]["blend_kmeans"], "ocsvm": CONFIG["aggregate"]["blend_ocsvm"], "ae": CONFIG["aggregate"]["blend_ae"]}
        else:
            blend = {"kmeans": CONFIG["aggregate"]["blend_kmeans"], "ocsvm": CONFIG["aggregate"]["blend_ocsvm"]}

        img_score = aggregate_image_scores({"kmeans": km_z, "ocsvm": svm_z, "ae": ae_z}, CONFIG["aggregate"]["topk_ratio"], blend)
        image_scores.append(img_score)

    preds = (np.asarray(image_scores) >= CONFIG["aggregate"]["threshold"]).astype(int)
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("id,prediction\n")
        for idx, y in enumerate(preds.tolist()):
            f.write(f"{idx},{y}\n")
    print(f"Wrote submission to {out_csv}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v3-2 Multiscale OC Anomaly Detection")
    p.add_argument("--mode", type=str, default="all", choices=["train", "infer", "all"])
    p.add_argument("--ae_enabled", type=lambda x: str(x).lower() in {"1", "true", "yes"}, default=CONFIG["ae"]["enabled"])
    p.add_argument("--ae_device", type=str, choices=["cpu", "cuda"], default=CONFIG["ae"]["device"])  # manual select
    p.add_argument("--output_csv", type=str, default=str(Path(CONFIG["paths"]["output_csv"]).resolve()))
    p.add_argument("--train_dir", type=str, default=str(Path(CONFIG["paths"]["train_dir"]).resolve()))
    p.add_argument("--test_dir", type=str, default=str(Path(CONFIG["paths"]["test_dir"]).resolve()))
    p.add_argument("--workdir", type=str, default=str(Path(CONFIG["paths"]["workdir"]).resolve()))
    p.add_argument("--seed", type=int, default=CONFIG["seed"])
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode in ("train", "all"):
        train_pipeline(args)
    if args.mode in ("infer", "all"):
        infer_pipeline(args)


if __name__ == "__main__":
    main()


