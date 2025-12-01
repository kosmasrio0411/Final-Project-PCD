import numpy as np
import cv2
import scipy.ndimage as ndi
from skimage.feature import graycomatrix, graycoprops

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def preprocess_image(img):
    # Baca grayscale

    # Resize
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    # CLAHE (adaptHistEq versi OpenCV)
    img = clahe.apply(img)

    # Gaussian filter
    img = cv2.GaussianBlur(img, (5, 5), 0)

    return img


def get_body_mask(img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    h, w = img.shape

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, body = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k, iterations=4)

    body[int(0.95 * h):, :] = 0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(body)
    mask = np.zeros_like(img, np.uint8)
    if num_labels > 1:
        main = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == main).astype(np.uint8)

    mask = ndi.binary_fill_holes(mask > 0).astype(np.uint8)
    return mask

def segment_lungs_final(img_pre, debug=False):
    # 1. pastikan uint8
    img = np.clip(img_pre, 0, 255).astype(np.uint8)
    h, w = img.shape

    # 2. body mask
    body_mask = get_body_mask(img).astype(np.uint8)
    body_area = body_mask.sum()
    if body_area == 0:
        lung_mask = np.zeros_like(img, np.uint8)
        return img, lung_mask

    # 3. nol-kan di luar badan
    img_body = img.copy()
    img_body[body_mask == 0] = 0

    # 4. Otsu di area badan
    blur = cv2.GaussianBlur(img_body, (5, 5), 0)
    torso_pixels = blur[body_mask == 1].reshape(-1, 1)

    thresh_val, _ = cv2.threshold(
        torso_pixels, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    # 5. apply ambang sekali saja dengan NumPy (paru lebih gelap)
    lung_bin = np.zeros_like(img, np.uint8)
    lung_bin[(blur < thresh_val) & (body_mask == 1)] = 255

    # 6. OPEN kecil + fill holes awal
    k_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lung_bin = cv2.morphologyEx(lung_bin, cv2.MORPH_OPEN, k_small, iterations=2)
    lung_mask_tmp = ndi.binary_fill_holes(lung_bin > 0).astype(np.uint8)

    # 7. CC: pilih 1–2 blob terbesar
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(lung_mask_tmp)
    lung_mask = np.zeros_like(img, np.uint8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        sorted_idx_desc = np.argsort(areas)[::-1]          # indeks di stats[1:]
        top_2_idx = sorted_idx_desc[:2]         # label asli
        chosen_labels = (top_2_idx + 1).tolist()
        lung_mask = np.isin(labels, chosen_labels).astype(np.uint8)

    # 8. batasi ke body, fill holes, smooth akhir
    lung_mask[body_mask == 0] = 0
    lung_mask = ndi.binary_fill_holes(lung_mask > 0).astype(np.uint8)

    r = max(int(0.01 * min(h, w)), 2)
    k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    lung_mask = cv2.morphologyEx(lung_mask, cv2.MORPH_CLOSE, k_smooth, iterations=3)
    lung_mask = ndi.binary_fill_holes(lung_mask > 0).astype(np.uint8)

    if debug:
        print("lung area ratio:", lung_mask.sum() / (h * w + 1e-8))

    segmented = img * lung_mask
    return segmented, lung_mask


def glcm_features(img,
                  distances=(1, 2, 4),
                  angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                  levels=16):
    """
    Ekstraksi fitur GLCM:
    - quantisasi ke 'levels' gray level
    - untuk setiap prop GLCM, simpan SEMUA nilai untuk setiap (distance, angle)
      => 6 * len(distances) * len(angles) fitur
    """
    # pastikan uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    # Quantize ke 0..levels-1
    img_q = (img / (256 // levels)).astype(np.uint8)

    glcm = graycomatrix(
        img_q,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    props = ["contrast", "dissimilarity", "homogeneity",
             "energy", "correlation", "ASM"]

    feat = []
    for prop in props:
        values = graycoprops(glcm, prop)   # shape: [len(distances), len(angles)]
        feat.extend(values.flatten())      # simpan SEMUA kombinasi jarak×sudut

    return np.array(feat, dtype=np.float32)


def glcm_stat_features(img,
                       distances=(1, 2, 4),
                       angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
                       levels=16):
    """
    Ekstraksi fitur:
    - 6 fitur GLCM: contrast, dissimilarity, homogeneity, energy, correlation, ASM
      -> disimpan SEMUA nilai untuk setiap (distance, angle)
         => 6 * len(distances) * len(angles) fitur
    - 8 fitur statistik intensitas: mean, std, entropy, RMS, variance, smoothness, kurtosis, skewness

    img: hasil segmentasi paru (seg_img), uint8 (0..255)
    return: vektor fitur 1D (float32)
    """

    # pastikan uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    h, w = img.shape

    # ==========================
    # 1) STATISTICAL FEATURES (8)
    # ==========================
    roi_vals = img[img > 0].astype(np.float32)  # hanya piksel paru (mask != 0)
    if roi_vals.size == 0:
        roi_vals = img.reshape(-1).astype(np.float32)

    mean_val = roi_vals.mean()
    std_val  = roi_vals.std()

    # entropy dari histogram
    hist, _ = np.histogram(roi_vals, bins=levels, range=(0, 255), density=True)
    hist = hist + 1e-12
    entropy = -np.sum(hist * np.log2(hist))

    # RMS & variance
    rms = np.sqrt(np.mean((roi_vals - mean_val) ** 2))
    variance = std_val ** 2

    # smoothness: 1 - 1/(1 + σ²)
    smoothness = 1.0 - 1.0 / (1.0 + variance)

    # kurtosis & skewness
    if std_val > 1e-6:
        z = (roi_vals - mean_val) / std_val
        kurtosis = np.mean(z**4) - 3.0
        skewness = np.mean(z**3)
    else:
        kurtosis = 0.0
        skewness = 0.0

    stat_feats = [
        mean_val, std_val, entropy, rms,
        variance, smoothness, kurtosis, skewness
    ]

    # ==========================
    # 2) GLCM FEATURES (6 × D × A)
    # ==========================
    # quantisasi
    step = max(256 // levels, 1)
    img_q = (img // step).astype(np.uint8)

    # bisa pakai bounding box paru biar GLCM tidak didominasi background
    ys, xs = np.where(img > 0)
    if len(xs) > 0:
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        img_q_sub = img_q[y_min:y_max+1, x_min:x_max+1]
    else:
        img_q_sub = img_q

    glcm = graycomatrix(
        img_q_sub,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    props = ["contrast", "correlation", "homogeneity",
             "energy", "entropy", "variance"]

    glcm_feats = []
    for p in props:
        vals = graycoprops(glcm, p)   # shape: [len(distances), len(angles)]
        glcm_feats.extend(vals.flatten())  # SIMPAN semua jarak × sudut

    # gabungkan: [GLCM per d×θ] + [8 statistik]
    feats = np.array(glcm_feats + stat_feats, dtype=np.float32)
    return feats