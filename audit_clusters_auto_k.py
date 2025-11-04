# audit_clusters_auto_k.py
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances, silhouette_score
from skimage.color import rgb2hsv
from skimage.feature import hog

def list_png(dirp):
    p = Path(dirp)
    return sorted([str(x) for x in p.glob("*.png")], key=lambda s: int(Path(s).stem))

def load_rgb(path, size=(256,256)):
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)/255.0

def feat(img):
    # HOG(灰階) + HSV 3D 直方圖(8x8x8)
    g = np.dot(img, [0.299,0.587,0.114]).astype(np.float32)
    h = hog(g, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
    hsv = rgb2hsv(img)
    hist, _ = np.histogramdd(hsv.reshape(-1,3), bins=(8,8,8), range=((0,1),(0,1),(0,1)), density=True)
    return np.concatenate([h, hist.flatten().astype(np.float32)])

def embed(paths, step=1):
    X=[]
    for p in paths[::step]:
        X.append(feat(load_rgb(p)))
    return np.stack(X,0)

def kl_div(p, q):
    p = p + 1e-12; q = q + 1e-12
    return float(np.sum(p*np.log(p/q)))

if __name__ == "__main__":
    train_dir="Dataset/train"; test_dir="Dataset/test"
    train_paths = list_png(train_dir); test_paths = list_png(test_dir)

    # 可調: 子抽樣步長（加快速度）
    step_train = 1
    step_test  = 1

    Xtr = embed(train_paths, step=step_train)
    Xte = embed(test_paths,  step=step_test)

    # 候選 k；也可擴到 {6,8,10,12,14,16,18}
    cand_k = [8,10,12,14,16,20,25,30,40,50,70]

    # 先用訓練特徵選 k（silhouette 越大越好）
    best_k, best_score = None, -1
    for k in cand_k:
        km_tmp = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xtr)
        labels = km_tmp.labels_
        try:
            s = silhouette_score(Xtr, labels, metric="euclidean")
        except Exception:
            s = -1
        print(f"k={k}, silhouette={s:.4f}")
        if s > best_score:
            best_score, best_k = s, k
    print(f"best_k={best_k} (silhouette={best_score:.4f})")

    # 用 best_k 重新擬合並審核分佈
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(Xtr)
    ctr = km.cluster_centers_
    tr_lab = km.predict(Xtr); te_lab = km.predict(Xte)

    # 群分佈對比（KL）
    k = best_k
    tr_hist = np.bincount(tr_lab, minlength=k); te_hist = np.bincount(te_lab, minlength=k)
    tr_prob = tr_hist/np.sum(tr_hist); te_prob = te_hist/np.sum(te_hist)
    print("train分佈:", np.round(tr_prob,3))
    print("test 分佈:", np.round(te_prob,3))
    print("KL(test||train):", kl_div(te_prob, tr_prob))

    # 最近中心距離覆蓋率（是否大量超過 train 的 99% 距離）
    tr_d = pairwise_distances(Xtr, ctr, metric="euclidean")
    te_d = pairwise_distances(Xte, ctr, metric="euclidean")
    tr_near = np.min(tr_d, axis=1); te_near = np.min(te_d, axis=1)

    # 也做「同群內」門檻：對每個群算各自 99% 分位
    global_q99 = np.quantile(tr_near, 0.99)
    print("global q99:", float(global_q99))
    print("test 超過 global q99 的比例:", float((te_near>global_q99).mean()))

    # 同群內覆蓋率
    over_rates = []
    for c in range(k):
        tr_c = tr_near[tr_lab==c]
        te_c = te_near[te_lab==c]
        if tr_c.size<10 or te_c.size==0:
            continue
        q99 = np.quantile(tr_c, 0.99)
        over = float((te_c>q99).mean())
        over_rates.append(over)
    if over_rates:
        print("test 同群內超過各群 q99 的比例均值:", float(np.mean(over_rates)))