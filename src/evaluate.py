"""
Оценка обученной модели на тестовых картинках CIFAR-100.

Запуск (из папки проекта):
    python -m src.evaluate --model simple

Что делает:
  1. считает top-1 и top-5 точность,
  2. находит лучшие и худшие классы,
  3. находит самые частые путаницы между классами,
  4. сохраняет графики и примеры ошибок в outputs/.
"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from src.dataset import get_dataloaders, CIFAR100_MEAN, CIFAR100_STD
from src.model import get_model
from src.train import get_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    """
    Прогоняет все тестовые картинки через модель.

    Возвращает:
        labels:     настоящие классы, массив (10000,)
        top5:       5 лучших догадок для каждой картинки, массив (10000, 5)
        images:     сами картинки (чтобы потом показать ошибки)
    """
    model.eval()
    all_labels, all_top5, all_images = [], [], []

    for images, labels in tqdm(loader, desc="Оценка"):
        outputs = model(images.to(device))
        # topk(5): пять классов с самыми большими оценками
        top5 = outputs.topk(5, dim=1).indices.cpu()
        all_labels.append(labels)
        all_top5.append(top5)
        all_images.append(images)

    return (
        torch.cat(all_labels).numpy(),
        torch.cat(all_top5).numpy(),
        torch.cat(all_images),
    )


def per_class_accuracy(labels, predictions, num_classes=100):
    """Точность отдельно для каждого класса."""
    accuracies = np.zeros(num_classes)
    for c in range(num_classes):
        mask = labels == c                       # все картинки этого класса
        accuracies[c] = (predictions[mask] == c).mean()
    return accuracies


def most_confused_pairs(labels, predictions, class_names, top_n=10):
    """
    Считает, как часто один класс принимают за другой.
    Возвращает список: (настоящий класс, ошибочный ответ, сколько раз).
    """
    counts = {}
    for true, pred in zip(labels, predictions):
        if true != pred:
            counts[(true, pred)] = counts.get((true, pred), 0) + 1

    pairs = sorted(counts.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [(class_names[t], class_names[p], n) for (t, p), n in pairs]


def plot_class_accuracy(accuracies, class_names, model_name):
    """Столбчатый график: 10 лучших и 10 худших классов."""
    order = np.argsort(accuracies)
    worst, best = order[:10], order[-10:]
    chosen = np.concatenate([worst, best])

    colors = ["#d9534f"] * 10 + ["#5cb85c"] * 10   # красный - плохие, зелёный - хорошие

    plt.figure(figsize=(9, 7))
    plt.barh([class_names[i] for i in chosen], accuracies[chosen] * 100, color=colors)
    plt.xlabel("Accuracy, %")
    plt.title("10 худших и 10 лучших классов")
    plt.tight_layout()
    path = f"outputs/class_accuracy_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"График по классам сохранён: {path}")


def denormalize(image):
    """Возвращает картинку к обычным цветам, чтобы её можно было показать."""
    image = image.clone()
    for ch, (mean, std) in enumerate(zip(CIFAR100_MEAN, CIFAR100_STD)):
        image[ch] = image[ch] * std + mean
    return image.permute(1, 2, 0).clamp(0, 1)


def plot_mistakes(images, labels, predictions, class_names, model_name, num=16):
    """Сетка из картинок, на которых модель ошиблась."""
    wrong = np.where(labels != predictions)[0][:num]

    plt.figure(figsize=(9, 9))
    for i, idx in enumerate(wrong):
        plt.subplot(4, 4, i + 1)
        plt.imshow(denormalize(images[idx]))
        plt.title(
            f"было: {class_names[labels[idx]]}\nмодель: {class_names[predictions[idx]]}",
            fontsize=8,
        )
        plt.axis("off")
    plt.suptitle("Примеры ошибок модели")
    plt.tight_layout()
    path = f"outputs/mistakes_{model_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Примеры ошибок сохранены: {path}")


def main():
    parser = argparse.ArgumentParser(description="Оценка модели на CIFAR-100")
    parser.add_argument("--model", default="simple", choices=["simple", "resnet"])
    args = parser.parse_args()

    device = get_device()
    _, test_loader, class_names = get_dataloaders()

    # Создаём модель и загружаем в неё сохранённые веса
    model = get_model(args.model).to(device)
    model.load_state_dict(torch.load(f"outputs/best_{args.model}.pth", map_location=device))

    labels, top5, images = collect_predictions(model, test_loader, device)
    predictions = top5[:, 0]   # первая догадка = ответ модели

    # --- Общие метрики ---
    top1_acc = (predictions == labels).mean()
    top5_acc = np.mean([label in row for label, row in zip(labels, top5)])
    print(f"\nTop-1 accuracy: {top1_acc*100:.2f}%")
    print(f"Top-5 accuracy: {top5_acc*100:.2f}%")

    # --- По классам ---
    accuracies = per_class_accuracy(labels, predictions)
    order = np.argsort(accuracies)
    print("\nСамые трудные классы:")
    for i in order[:5]:
        print(f"  {class_names[i]:15s} {accuracies[i]*100:5.1f}%")
    print("Самые лёгкие классы:")
    for i in order[-5:][::-1]:
        print(f"  {class_names[i]:15s} {accuracies[i]*100:5.1f}%")

    # --- Путаницы ---
    print("\nЧаще всего путает (настоящий -> ответ модели, раз):")
    for true, pred, n in most_confused_pairs(labels, predictions, class_names):
        print(f"  {true} -> {pred}: {n}")

    # --- Графики ---
    plot_class_accuracy(accuracies, class_names, args.model)
    plot_mistakes(images, labels, predictions, class_names, args.model)

    # --- Сохраняем итоги в файл (пригодится для README) ---
    results = {
        "model": args.model,
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
    }
    with open(f"outputs/results_{args.model}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nИтоги сохранены: outputs/results_{args.model}.json")


if __name__ == "__main__":
    main()