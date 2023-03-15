import cv2
import numpy as np
from matplotlib import pyplot as plt


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray):
    kernel = np.ones(shape=(7, 7), dtype=np.uint8) * 255
    while True:
        expanded = cv2.dilate(src=marker, kernel=kernel)
        expanded = cv2.bitwise_and(src1=expanded, src2=mask)

        if (marker == expanded).all():
            return expanded
        marker = expanded


def kmeans_extract_color_palette(image: np.ndarray, palette_size: int):
    data = cv2.resize(image, (100, 100)).reshape(-1, 3)
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _compactness, labels, centers = cv2.kmeans(data.astype(
        np.float32), palette_size, None, criteria, attempts=10, flags=flags)

    cluster_sizes = np.bincount(labels.flatten())

    palette = []
    palette_rgb = np.empty((palette_size, 3), dtype=np.uint8)

    i = 0
    for cluster_idx in np.argsort(-cluster_sizes):
        palette_rgb[i] = np.array(centers[cluster_idx].astype(int))
        i += 1

        palette.append(np.full(
            (image.shape[0], image.shape[1], 3), fill_value=centers[cluster_idx].astype(int), dtype=np.uint8))

    return palette, palette_rgb


coins_bgr = cv2.imread('coins.png', cv2.IMREAD_UNCHANGED)
coins_gray = cv2.cvtColor(coins_bgr, cv2.COLOR_BGR2GRAY)

_t, mask_all_coins = cv2.threshold(
    coins_gray, 150, 255, cv2.THRESH_BINARY_INV)

imgEdge = cv2.Canny(coins_gray, 145, 553, apertureSize=3, L2gradient=True)

mask_all_coins = cv2.morphologyEx(
    mask_all_coins, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (15, 15)))

mask_all_coins = cv2.erode(
    mask_all_coins, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

palette, palette_rgb = kmeans_extract_color_palette(coins_bgr, palette_size=12)
palette_rgb = palette_rgb[-2:]

avg = np.average(palette_rgb, axis=0)

palette_range_low, palette_range_high = palette_rgb[0], palette_rgb[-1]

error_arr = np.full((3,), 20)

marker_copper_coin = cv2.inRange(
    coins_bgr, palette_range_low, palette_range_high)


marker = cv2.bitwise_and(marker_copper_coin, mask_all_coins)

mask_reconstructed = morphological_reconstruction(
    marker, mask_all_coins)

coins_masked = cv2.bitwise_and(coins_bgr, coins_bgr, mask=mask_reconstructed)

palette = np.hstack(palette)

sf = coins_bgr.shape[1] / palette.shape[1]
out = np.vstack([coins_bgr, cv2.resize(palette, (0, 0), fx=sf, fy=sf)])

cv2.imshow('canny', imgEdge)
cv2.imshow('mask_all_coins', mask_all_coins)
cv2.imshow('marker_copper_coin', marker_copper_coin)
cv2.imshow('result', coins_masked)
cv2.imshow('kmeans color analysis', out)

cv2.imwrite('coin_mask.png', mask_reconstructed)
cv2.waitKey()
