import cv2
import numpy as np
import matplotlib.pyplot as plt

def binarize_image(image, threshold=128):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def connected_components_labeling(binary_image):
    label_image = np.zeros(binary_image.shape, dtype=np.int32)
    label = 1
    height, width = binary_image.shape
    
    def dfs(x, y):
        stack = [(x, y)]
        while stack:
            cx, cy = stack.pop()
            if label_image[cx, cy] == 0 and binary_image[cx, cy] == 255:
                label_image[cx, cy] = label
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < height and 0 <= ny < width and label_image[nx, ny] == 0:
                        stack.append((nx, ny))

    for i in range(height):
        for j in range(width):
            if binary_image[i, j] == 255 and label_image[i, j] == 0:
                dfs(i, j)
                label += 1
    
    return label_image, label - 1

def label_to_color_image(label_image, num_labels):
    color_image = np.zeros((*label_image.shape, 3), dtype=np.uint8)
    np.random.seed(0)
    colors = [tuple(np.random.choice(range(256), size=3)) for _ in range(num_labels + 1)]
    for i in range(label_image.shape[0]):
        for j in range(label_image.shape[1]):
            if label_image[i, j] > 0:
                color_image[i, j] = colors[label_image[i, j]]
    return color_image

image = cv2.imread('pinos-madeira.png')

if image is None:
    print("Erro ao carregar a imagem. Verifique o caminho e o nome da imagem.")
else:
    binary_image = binarize_image(image)

    label_image, num_labels = connected_components_labeling(binary_image)

    color_image = label_to_color_image(label_image, num_labels)

    plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Imagem original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title('Grupos')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(color_image)
    plt.title('Componentes rotulados')
    plt.axis('off')

    plt.show()
