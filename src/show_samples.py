"""
Показывает сетку картинок из CIFAR-100 и сохраняет её в outputs/.
"""

import os
import matplotlib.pyplot as plt

from src.dataset import get_dataloaders, CIFAR100_MEAN, CIFAR100_STD


def denormalize(image):
    """
    Возвращает нормализованную картинку к обычному виду,
    чтобы её можно было нарисовать.
    (Обратная операция: умножаем на разброс, прибавляем среднее.)
    """
    image = image.clone()
    for channel, (mean, std) in enumerate(zip(CIFAR100_MEAN, CIFAR100_STD)):
        image[channel] = image[channel] * std + mean
    # imshow ждёт порядок (высота, ширина, каналы), а у нас (каналы, высота, ширина)
    return image.permute(1, 2, 0).clamp(0, 1)


def show_samples(num_images=16):
    train_loader, _, class_names = get_dataloaders(batch_size=num_images)
    images, labels = next(iter(train_loader))

    plt.figure(figsize=(8, 8))
    for i in range(num_images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(denormalize(images[i]))
        plt.title(class_names[labels[i]], fontsize=9)
        plt.axis("off")

    plt.suptitle("Примеры из CIFAR-100 (с аугментациями)")
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/samples.png", dpi=150)
    print("Картинка сохранена: outputs/samples.png")
    plt.show()


if __name__ == "__main__":
    show_samples()