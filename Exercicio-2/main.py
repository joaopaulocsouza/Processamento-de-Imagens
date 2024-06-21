import cv2
import numpy as np
import matplotlib.pyplot as plt

def reduce_resolution(image, sampling_interval):
    height, width = image.shape[:2]
    
    new_height = height // sampling_interval
    new_width = width // sampling_interval
    
    downsampled_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    for i in range(new_height):
        for j in range(new_width):
            downsampled_image[i, j] = image[i * sampling_interval, j * sampling_interval]
    
    return downsampled_image

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

downsampled_images = []
titles = []

for interval in [2, 4, 8, 16]:
    downsampled_images.append(reduce_resolution(image, interval))
    titles.append(f'Intervalo: {interval}')

downsampled_images.insert(0, image)
titles.insert(0, 'Imagem original')

display_images(downsampled_images, titles)
