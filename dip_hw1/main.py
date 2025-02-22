import numpy as np
import cv2
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaussian', action='store_true')
    parser.add_argument('--median', action='store_true')
    parser.add_argument('--laplacian', action='store_true')
    args = parser.parse_args()
    return args


def padding(input_img, kernel_size):
    """對輸入的影像進行padding，以便在之後進行convolution時可以處理照片的邊緣"""
    if isinstance(kernel_size, int):
        pad_h = pad_w = kernel_size // 2
    else:
        pad_h = kernel_size[0] // 2
        pad_w = kernel_size[1] // 2

    if len(input_img.shape) == 3:
        padded_img = np.pad(input_img,
                            ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                            mode='edge')
    else:
        padded_img = np.pad(input_img,
                            ((pad_h, pad_h), (pad_w, pad_w)),
                            mode='edge')
    return padded_img


def convolution(input_img, kernel):
    """對輸入的照片進行2D-convolution"""
    #h為高度，w為寬度，c為chonnel數
    if len(input_img.shape) == 3:
        h, w, c = input_img.shape
    else:
        h, w = input_img.shape
        c = 1
        input_img = input_img.reshape(h, w, 1)

    kernel = np.flipud(np.fliplr(kernel))
    kernel_h, kernel_w = kernel.shape
    padded_img = padding(input_img, (kernel_h, kernel_w))
    output_img = np.zeros_like(input_img)

    for i in range(h):
        for j in range(w):
            for k in range(c):
                output_img[i, j, k] = np.sum(
                    padded_img[i:i + kernel_h, j:j + kernel_w, k] * kernel
                )

    return output_img


def gaussian_filter(input_img, sigma=1.0, kernel_size=3):
    """對輸入的照片使用 gaussian filter來減少noise"""
    # 建立一個Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2

    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = ((1 / (2 * np.pi * sigma ** 2)) *
                            np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))

    # Normalize kernel來避免改變整體照片的總量度
    kernel = kernel / np.sum(kernel)

    return convolution(input_img, kernel)


def median_filter(input_img, kernel_size=3):
    """對輸入的照片使用 median filter來減少noise"""
    if len(input_img.shape) == 3:
        h, w, c = input_img.shape
    else:
        h, w = input_img.shape
        c = 1
        input_img = input_img.reshape(h, w, 1)

    padded_img = padding(input_img, (kernel_size, kernel_size))
    output_img = np.zeros_like(input_img)
    pad = kernel_size // 2

    for i in range(h):
        for j in range(w):
            for k in range(c):
                window = padded_img[i:i + kernel_size, j:j + kernel_size, k]
                output_img[i, j, k] = np.median(window)

    return output_img


def laplacian_sharpening(input_img, kernel_type=1):
    """對輸入照片使用 laplacian sharpening，根據需求不同可以選擇type 1 or 2的kernel"""
    if kernel_type == 1:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ])
    else:
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])

    sharpened = convolution(input_img, kernel)

    # 為了避免 overflow ，使用clip
    sharpened = np.clip(sharpened, 0, 255)

    return sharpened.astype(np.uint8)


if __name__ == "__main__":
    args = parse_args()

    if args.gaussian:
        input_img = cv2.imread("input_part1.jpg")
        output_img = gaussian_filter(input_img, sigma=2.0, kernel_size=5)
    elif args.median:
        input_img = cv2.imread("input_part1.jpg")
        output_img = median_filter(input_img, kernel_size=7)
    elif args.laplacian:
        input_img = cv2.imread("input_part2.jpg")
        output_img = laplacian_sharpening(input_img, kernel_type=2)

    cv2.imwrite("output.jpg", output_img)