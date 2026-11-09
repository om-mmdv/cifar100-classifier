"""
Модели нейросетей для CIFAR-100.

Здесь две модели:
  - SimpleCNN:   простая свёрточная сеть (для понимания принципа)
  - SmallResNet: сеть с "обходными путями" (точнее, но медленнее)

Функция get_model() выбирает нужную модель по имени.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------
# Модель 1: простая свёрточная сеть
# ---------------------------------------------------------------
def conv_block(in_channels, out_channels):
    """
    Один "блок": свёртка -> нормализация -> ReLU.
    Мы будем строить сеть из таких одинаковых блоков.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class SimpleCNN(nn.Module):
    """
    Простая CNN. Путь картинки через сеть:

      вход 3x32x32
        -> 2 блока (64 фильтра) -> уменьшаем до 16x16
        -> 2 блока (128 фильтров) -> уменьшаем до 8x8
        -> 2 блока (256 фильтров) -> уменьшаем до 4x4
        -> "растягиваем" в один длинный ряд чисел
        -> финальный слой: 100 чисел (по одному на класс)
    """

    def __init__(self, num_classes=100):
        super().__init__()

        # Часть 1: извлекаем признаки из картинки
        self.features = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2),            # 32x32 -> 16x16

            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2),            # 16x16 -> 8x8

            conv_block(128, 256),
            conv_block(256, 256),
            nn.MaxPool2d(2),            # 8x8 -> 4x4
        )

        # Часть 2: принимаем решение, какой это класс
        self.classifier = nn.Sequential(
            nn.Flatten(),               # 256x4x4 -> ряд из 4096 чисел
            nn.Dropout(0.5),            # выключаем половину нейронов (против зубрёжки)
            nn.Linear(256 * 4 * 4, num_classes),
        )

    def forward(self, x):
        """Прогон картинок через сеть."""
        x = self.features(x)
        return self.classifier(x)


# ---------------------------------------------------------------
# Модель 2: мини-ResNet
# ---------------------------------------------------------------
class ResidualBlock(nn.Module):
    """
    Блок с "обходным путём".

    Вход проходит через два слоя свёртки, а потом к результату
    ДОБАВЛЯЕТСЯ исходный вход. Если слои сделали что-то не то,
    исходная информация всё равно не потеряется.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Основной путь: две свёртки
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Обходной путь. Если размеры изменились, подгоняем вход
        # к новому размеру с помощью свёртки 1x1.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()   # ничего не меняем

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main_path(x) + self.shortcut(x)   # складываем два пути
        return self.relu(out)


class SmallResNet(nn.Module):
    """
    Мини-ResNet: стартовая свёртка, три группы блоков, финальный слой.
    """

    def __init__(self, num_classes=100):
        super().__init__()

        # Стартовая свёртка
        self.start = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Три группы по два блока. В каждой следующей картинка
        # уменьшается вдвое, а фильтров становится вдвое больше.
        self.layers = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),

            ResidualBlock(64, 128, stride=2),    # 32x32 -> 16x16
            ResidualBlock(128, 128),

            ResidualBlock(128, 256, stride=2),   # 16x16 -> 8x8
            ResidualBlock(256, 256),
        )

        # Усредняем каждую карту признаков до одного числа
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.start(x)
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# ---------------------------------------------------------------
# Выбор модели по имени
# ---------------------------------------------------------------
def get_model(name="simple", num_classes=100):
    """Возвращает модель по имени: 'simple' или 'resnet'."""
    if name == "simple":
        return SimpleCNN(num_classes)
    if name == "resnet":
        return SmallResNet(num_classes)
    raise ValueError(f"Неизвестная модель: {name}. Доступно: 'simple', 'resnet'")


if __name__ == "__main__":
    # Быстрая проверка: прогоняем "пустую" порцию через обе модели
    fake_images = torch.randn(8, 3, 32, 32)   # 8 случайных картинок

    for model_name in ["simple", "resnet"]:
        model = get_model(model_name)
        output = model(fake_images)
        params = sum(p.numel() for p in model.parameters())
        print(f"{model_name:7s} | выход: {tuple(output.shape)} | параметров: {params:,}")