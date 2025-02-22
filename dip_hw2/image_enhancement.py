import cv2
import numpy as np
import os


def gamma_correction(img, gamma):
    img_float = img.astype(np.float32) / 255.0
    corrected = np.power(img_float, 1 / gamma)
    brightness = 1.5
    corrected = corrected * brightness
    corrected = (corrected * 255).clip(0, 255).astype(np.uint8)
    return corrected


def histogram_equalization(img):
    hist = np.zeros(256)
    rows, cols = img.shape[:2]

    for i in range(rows):
        for j in range(cols):
            hist[img[i, j]] += 1

    clip_limit = (rows * cols) * 0.01
    hist = np.minimum(hist, clip_limit)

    cdf = np.cumsum(hist)
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    return cdf_normalized[img].astype(np.uint8)


def other_enhancement_algorithm(img):
    """using Unsharp Masking"""
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)

    # Calculate the unsharp mask
    unsharp_mask = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

    return unsharp_mask



def main():
    output_dir = "output/image_enhancement"
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread("data/image_enhancement/input.bmp")

    gamma_list = [0.5, 1.0, 2.0]  # parameter setting
    for gamma in gamma_list:
        gamma_correction_img = gamma_correction(img, gamma)

        output_path = os.path.join(output_dir, f"gamma_correction_{gamma}.png")
        cv2.imwrite(output_path, gamma_correction_img)

        comparison = np.vstack([img, gamma_correction_img])
        cv2.imshow(f"Gamma correction | Gamma = {gamma}", comparison)
        cv2.waitKey(0)

    hist_eq_img = histogram_equalization(img)
    output_path = os.path.join(output_dir, "histogram_equalization.png")
    cv2.imwrite(output_path, hist_eq_img)

    comparison = np.vstack([img, hist_eq_img])
    cv2.imshow("Histogram equalization", comparison)
    cv2.waitKey(0)

    enhanced_img = other_enhancement_algorithm(img)
    output_path = os.path.join(output_dir, "enhanced.png")
    cv2.imwrite(output_path, enhanced_img)

    comparison = np.vstack([img, enhanced_img])
    cv2.imshow("Enhanced image", comparison)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()