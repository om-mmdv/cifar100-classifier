"""
Загрузка и подготовка данных CIFAR-100.

Главная функция здесь - get_dataloaders(). Она:
  1. скачивает картинки (если их ещё нет),
  2. применяет нормализацию и аугментации,
  3. возвращает готовые "порции" данных для обучения и теста.
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Среднее и разброс по цветовым каналам (красный, зелёный, синий).
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


def get_transforms():
    """Возвращает два набора преобразований: для обучения и для теста."""
    # Для ОБУЧЕНИЯ: добавляем случайные изменения (аугментации)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # Для ТЕСТА: никаких случайностей, только подготовка
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    return train_transform, test_transform


def get_dataloaders(data_dir="data", batch_size=128, num_workers=0):
    """
    Скачивает CIFAR-100 и возвращает загрузчики данных.

    Возвращает: train_loader, test_loader, class_names
    """
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    class_names = train_dataset.classes

    return train_loader, test_loader, class_names


if __name__ == "__main__":
    train_loader, test_loader, class_names = get_dataloaders()

    print(f"Картинок для обучения: {len(train_loader.dataset)}")
    print(f"Картинок для теста:    {len(test_loader.dataset)}")
    print(f"Количество классов:    {len(class_names)}")
    print(f"Первые 5 классов:      {class_names[:5]}")

    images, labels = next(iter(train_loader))
    print(f"Размер одной порции картинок: {tuple(images.shape)}")
    print(f"Размер одной порции ответов:  {tuple(labels.shape)}")