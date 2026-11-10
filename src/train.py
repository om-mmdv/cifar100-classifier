"""
Обучение модели на CIFAR-100.

Запуск (из папки проекта):
    python -m src.train --model simple --epochs 20
    python -m src.train --model resnet --epochs 30
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from src.dataset import get_dataloaders
from src.model import get_model


def get_device():
    """Берём видеокарту, если она есть, иначе процессор."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Одна эпоха обучения. Возвращает (средний loss, accuracy)."""
    model.train()  # включаем режим обучения (работают Dropout и BatchNorm)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  обучение", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()              # 1. сбрасываем старые подсказки
        outputs = model(images)            # 2. сеть угадывает
        loss = loss_fn(outputs, labels)    # 3. считаем ошибку
        loss.backward()                    # 4. ищем, какие веса виноваты
        optimizer.step()                   # 5. чуть-чуть исправляем веса

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()  # при проверке веса не меняем, считать "подсказки" не нужно
def evaluate(model, loader, loss_fn, device):
    """Проверка на тестовых картинках. Возвращает (loss, accuracy)."""
    model.eval()  # режим проверки (Dropout выключен)
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  проверка", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def plot_history(history, model_name):
    """Рисует графики обучения и сохраняет в outputs/."""
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(11, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="обучение")
    plt.plot(epochs, history["test_loss"], label="тест")
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Ошибка (чем меньше, тем лучше)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in history["train_acc"]], label="обучение")
    plt.plot(epochs, [a * 100 for a in history["test_acc"]], label="тест")
    plt.xlabel("Эпоха")
    plt.ylabel("Accuracy, %")
    plt.title("Точность (чем больше, тем лучше)")
    plt.legend()

    plt.tight_layout()
    path = f"outputs/curves_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Графики сохранены: {path}")


def main():
    # Настройки, которые можно менять из командной строки
    parser = argparse.ArgumentParser(description="Обучение на CIFAR-100")
    parser.add_argument("--model", default="simple", choices=["simple", "resnet"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05, help="размер шага обучения")
    args = parser.parse_args()

    os.makedirs("outputs", exist_ok=True)
    device = get_device()
    print(f"Модель: {args.model} | Эпох: {args.epochs} | Устройство: {device}")

    # Данные, модель, функция потерь, оптимизатор
    train_loader, test_loader, _ = get_dataloaders(batch_size=args.batch_size)
    model = get_model(args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=5e-4, nesterov=True,
    )
    # Шаг обучения плавно уменьшается от lr почти до нуля
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_acc = 0.0
    weights_path = f"outputs/best_{args.model}.pth"

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        # Сохраняем модель, только если она стала лучше
        mark = ""
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), weights_path)
            mark = "  <- лучшая, сохранена"

        print(
            f"Эпоха {epoch:2d}/{args.epochs} | "
            f"train acc {train_acc*100:5.1f}% | test acc {test_acc*100:5.1f}% | "
            f"{time.time() - start:.0f} сек{mark}"
        )

    # Сохраняем историю и рисуем графики
    with open(f"outputs/history_{args.model}.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_history(history, args.model)

    print(f"\nГотово. Лучшая точность на тесте: {best_acc*100:.2f}%")
    print(f"Веса модели: {weights_path}")


if __name__ == "__main__":
    main()