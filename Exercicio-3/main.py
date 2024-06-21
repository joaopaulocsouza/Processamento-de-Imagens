import cv2
import numpy as np
import matplotlib.pyplot as plt

def quantize_image(image, num_levels):
    interval = 256 // num_levels
    
    quantized_image = (image // interval) * interval + interval // 2
    quantized_image = np.clip(quantized_image, 0, 255)
    
    return quantized_image.astype(np.uint8)

def display_images(images, titles):
    n = len(images)
    plt.figure(figsize=(20, 10))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis('off')
    plt.show()

image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)


quantized_images = []
titles = []

for levels in [256, 64, 16, 4, 2]:
    quantized_images.append(quantize_image(image, levels))
    titles.append(f'Quantização;: {levels}')

quantized_images.insert(0, image)
titles.insert(0, 'Imagem original')

display_images(quantized_images, titles)
