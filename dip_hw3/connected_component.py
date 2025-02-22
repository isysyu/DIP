import cv2
import numpy as np
import os
import colorsys
from collections import deque


def to_binary(img):

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    binary_fill = binary.copy()
    h, w = binary.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(binary_fill, mask, (0, 0), 255)
    binary_fill = cv2.bitwise_not(binary_fill)
    binary = binary | binary_fill

    return binary // 255


def find_root(parent, label):
    if parent[label] != label:
        parent[label] = find_root(parent, parent[label])
    return parent[label]


def union(parent, label1, label2):
    root1 = find_root(parent, label1)
    root2 = find_root(parent, label2)
    if root1 != root2:
        parent[root2] = root1


def two_pass(binary_img, connectivity):

    height, width = binary_img.shape
    # First pass: assign initial labels
    labels = np.zeros((height, width), dtype=np.int32)
    current_label = 1
    parent = {}

    for i in range(height):
        for j in range(width):
            if binary_img[i, j] == 0:
                continue

            neighbors = []
            if connectivity == 4:
                if i > 0: neighbors.append(labels[i - 1, j])  # N
                if j > 0: neighbors.append(labels[i, j - 1])  # W
            else:  # 8-connectivity
                if i > 0 and j > 0: neighbors.append(labels[i - 1, j - 1])  # NW
                if i > 0: neighbors.append(labels[i - 1, j])  # N
                if i > 0 and j < width - 1: neighbors.append(labels[i - 1, j + 1])  # NE
                if j > 0: neighbors.append(labels[i, j - 1])  # W

            neighbors = [n for n in neighbors if n > 0]

            if not neighbors:  # No neighbors with labels
                labels[i, j] = current_label
                parent[current_label] = current_label
                current_label += 1
            else:  # Has labeled neighbors
                min_label = min(neighbors)
                labels[i, j] = min_label
                for n in neighbors:
                    union(parent, min_label, n)

    # Second pass: resolve equivalences
    final_labels = {}
    next_label = 1

    for i in range(height):
        for j in range(width):
            if labels[i, j] > 0:
                root = find_root(parent, labels[i, j])
                if root not in final_labels:
                    final_labels[root] = next_label
                    next_label += 1
                labels[i, j] = final_labels[root]

    return labels


def seed_filling(binary_img, connectivity):

    height, width = binary_img.shape
    labels = np.zeros((height, width), dtype=np.int32)
    current_label = 1

    def get_neighbors(y, x):
        if connectivity == 4:
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        else:  # 8-connectivity
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0),
                          (1, 1), (1, -1), (-1, 1), (-1, -1)]

        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                yield (ny, nx)

    # Process each pxl
    for i in range(height):
        for j in range(width):
            if binary_img[i, j] == 1 and labels[i, j] == 0:
                # S new region
                queue = deque([(i, j)])
                labels[i, j] = current_label

                while queue:
                    y, x = queue.popleft()
                    for ny, nx in get_neighbors(y, x):
                        if binary_img[ny, nx] == 1 and labels[ny, nx] == 0:
                            labels[ny, nx] = current_label
                            queue.append((ny, nx))

                current_label += 1

    return labels


def color_mapping(label_img):

    n_labels = np.max(label_img)
    if n_labels == 0:
        return np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)


    np.random.seed(42)  # For reproduct
    colors = []
    hues = np.linspace(0, 1, n_labels + 1)[:-1]

    for hue in hues:
        rgb = tuple(int(i * 255) for i in colorsys.hsv_to_rgb(hue, 1.0, 1.0))
        colors.append(rgb)

    colors = np.array([[0, 0, 0]] + colors, dtype=np.uint8)

    height, width = label_img.shape
    colored = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(n_labels + 1):
        colored[label_img == i] = colors[i]

    return colored


def main():
    os.makedirs("result/connected_component/two_pass", exist_ok=True)
    os.makedirs("result/connected_component/seed_filling", exist_ok=True)
    connectivity_type = [4, 8]

    for i in range(2):
        img = cv2.imread(f"data/connected_component/input{i + 1}.png")
        if img is None:
            print(f"Error: Could not read input{i + 1}.png")
            continue

        for connectivity in connectivity_type:
            binary_img = to_binary(img)

            cv2.imwrite(f"result/connected_component/binary_input{i + 1}.png",
                        binary_img * 255)

            two_pass_label = two_pass(binary_img, connectivity)
            seed_filling_label = seed_filling(binary_img, connectivity)

            two_pass_color = color_mapping(two_pass_label)
            seed_filling_color = color_mapping(seed_filling_label)

            cv2.imwrite("result/connected_component/two_pass/input{}_c{}.png".format(i + 1, connectivity),
                        two_pass_color)
            cv2.imwrite("result/connected_component/seed_filling/input{}_c{}.png".format(i + 1, connectivity),
                        seed_filling_color)


if __name__ == "__main__":
    main()