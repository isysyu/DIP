import cv2
import numpy as np


def generate_motion_blur_psf(size=15, angle=30, length=9, method='point'):
    """ 產生motion blur PSF, 依據testcase的不同可以調整想要的生成方法"""
    angle_rad = np.deg2rad(angle)
    center = size // 2
    psf = np.zeros((size, size))

    if method == 'point':
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        for i in range(length):
            offset = i - length // 2
            x = center + round(offset * cos_angle)
            y = center + round(offset * sin_angle)

            if 0 <= x < size and 0 <= y < size:
                psf[y, x] = 1

    elif method == 'sine':
        s0 = 1.0 / length
        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)

        rotated_X = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        psf = np.cos(2 * np.pi * s0 * rotated_X)

    elif method == 'line':
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)

        line_distance = X * sin_angle - Y * cos_angle
        psf = np.exp(-(line_distance ** 2) / (2 * (length / 6) ** 2))

    elif method == 'edge':
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        x = np.arange(size) - center
        y = np.arange(size) - center
        X, Y = np.meshgrid(x, y)

        edge_distance = X * cos_angle + Y * sin_angle
        psf = np.gradient(psf)[0]

    return psf / np.sum(psf)

def wiener_filtering(image, psf, K=0.01):
    """實作wiener filter的部分，實現image restoration"""
    result = np.zeros_like(image)
    for channel in range(3):
        channel_float = image[:, :, channel].astype(float)
        channel_fft = np.fft.fft2(channel_float)
        psf_fft = np.fft.fft2(psf, s=channel_float.shape)
        psf_fft_conj = np.conj(psf_fft)
        H = psf_fft_conj / (np.abs(psf_fft) ** 2 + K)
        result[:, :, channel] = np.abs(np.fft.ifft2(channel_fft * H))

    return np.clip(result, 0, 255).astype(np.uint8)


def constrained_least_square_filtering(image, psf, gamma=0.01):
    """實作csl filter的部分，實現image restoration"""
    result = np.zeros_like(image)
    laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

    for channel in range(3):
        channel_float = image[:, :, channel].astype(float)
        channel_fft = np.fft.fft2(channel_float)
        psf_fft = np.fft.fft2(psf, s=channel_float.shape)
        psf_fft_conj = np.conj(psf_fft)
        p_fft = np.fft.fft2(laplacian, s=channel_float.shape)
        H = psf_fft_conj / (np.abs(psf_fft) ** 2 + gamma * np.abs(p_fft) ** 2)
        result[:, :, channel] = np.abs(np.fft.ifft2(channel_fft * H))

    return np.clip(result, 0, 255).astype(np.uint8)


def compute_PSNR(image_original, image_restored):
    """教授提供的PSNR計算方法"""
    # PSNR = 10 * log10(max_pixel^2 / MSE)
    psnr = 10 * np.log10(255 ** 2 / np.mean((image_original.astype(np.float64) - image_restored.astype(np.float64)) ** 2))

    return psnr


def main():
    # Parameters for testcase1
    tc1_params = {
        'wiener': {
            'size': 31,
            'angle': -45,
            'length': 13,
            'method': 'line',
            'K': 0.01
        },
        'cls': {
            'size': 31,
            'angle': -45,
            'length': 13,
            'method': 'line',
            'gamma': 1.4
        }
    }

    # Parameters for testcase2
    tc2_params = {
        'wiener': {
            'size': 41,
            'angle': -45,
            'length': 41,
            'method': 'point',
            'K': 0.05
        },
        'cls': {
            'size': 27,
            'angle': -45,
            'length': 13,
            'method': 'line',
            'gamma': 7
        }
    }

    for i in range(2):
        img_original = cv2.imread(f"data/image_restoration/testcase{i + 1}/input_original.png")
        img_blurred = cv2.imread(f"data/image_restoration/testcase{i + 1}/input_blurred.png")

        params = tc1_params if i == 0 else tc2_params

        psf_wiener = generate_motion_blur_psf(
            size=params['wiener']['size'],
            angle=params['wiener']['angle'],
            length=params['wiener']['length'],
            method=params['wiener']['method']
        )
        wiener_img = wiener_filtering(img_blurred, psf_wiener, K=params['wiener']['K'])

        psf_cls = generate_motion_blur_psf(
            size=params['cls']['size'],
            angle=params['cls']['angle'],
            length=params['cls']['length'],
            method=params['cls']['method']
        )
        constrained_least_square_img = constrained_least_square_filtering(
            img_blurred, psf_cls, gamma=params['cls']['gamma']
        )

        print(f"\n---------- Testcase {i + 1} ----------")
        print("Wiener Parameters:", params['wiener'])
        print("CLS Parameters:", params['cls'])
        print("\nMethod: Wiener filtering")
        print(f"PSNR = {compute_PSNR(img_original, wiener_img)}\n")
        print("Method: Constrained least squares filtering")
        print(f"PSNR = {compute_PSNR(img_original, constrained_least_square_img)}\n")

        cv2.imshow("Results", np.hstack([img_blurred, wiener_img, constrained_least_square_img]))
        cv2.waitKey(0)

if __name__ == "__main__":
    main()