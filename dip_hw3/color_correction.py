import cv2
import numpy as np
import os


def white_patch_algorithm(img):

    # Split image into RGB
    b, g, r = cv2.split(img)

    max_b = np.max(b)
    max_g = np.max(g)
    max_r = np.max(r)

    scale_b = 255.0 / max_b if max_b > 0 else 1.0
    scale_g = 255.0 / max_g if max_g > 0 else 1.0
    scale_r = 255.0 / max_r if max_r > 0 else 1.0

    b = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r = np.clip(r * scale_r, 0, 255).astype(np.uint8)

    corrected_img = cv2.merge([b, g, r])

    return corrected_img


def gray_world_algorithm(img):

    b, g, r = cv2.split(img)

    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)

    gray_avg = (avg_b + avg_g + avg_r) / 3.0

    scale_b = gray_avg / avg_b if avg_b > 0 else 1.0
    scale_g = gray_avg / avg_g if avg_g > 0 else 1.0
    scale_r = gray_avg / avg_r if avg_r > 0 else 1.0

    b = np.clip(b * scale_b, 0, 255).astype(np.uint8)
    g = np.clip(g * scale_g, 0, 255).astype(np.uint8)
    r = np.clip(r * scale_r, 0, 255).astype(np.uint8)

    corrected_img = cv2.merge([b, g, r])

    return corrected_img



def main():

    os.makedirs("result/color_correction", exist_ok=True)
    for i in range(2):
        img = cv2.imread("data/color_correction/input{}.bmp".format(i + 1))

        # Apply white-balance algorithms
        white_patch_img = white_patch_algorithm(img)
        gray_world_img = gray_world_algorithm(img)

        cv2.imwrite("result/color_correction/white_patch_input{}.bmp".format(i + 1), white_patch_img)
        cv2.imwrite("result/color_correction/gray_world_input{}.bmp".format(i + 1), gray_world_img)


if __name__ == "__main__":
    main()