"""
Предсказание для любой вашей картинки.

Запуск (из папки проекта):
    python -m src.predict путь\к\картинке.jpg --model simple
"""

import argparse

import torch
from PIL import Image
from torchvision import datasets, transforms

from src.dataset import CIFAR100_MEAN, CIFAR100_STD
from src.model import get_model
from src.train import get_device


def load_image(path):
    """
    Открывает картинку и готовит её для сети:
      1. вырезаем квадрат из центра (чтобы не исказить пропорции),
      2. уменьшаем до 32x32 (как в CIFAR-100),
      3. нормализуем так же, как при обучении.
    """
    image = Image.open(path).convert("RGB")

    prepare = transforms.Compose([
        transforms.CenterCrop(min(image.size)),   # квадрат из центра
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    # unsqueeze(0) добавляет "порцию из одной картинки": (3,32,32) -> (1,3,32,32)
    return prepare(image).unsqueeze(0)


@torch.no_grad()
def predict(image_path, model_name="simple", top_k=5):
    """Возвращает список из top_k пар: (название класса, уверенность в %)."""
    device = get_device()

    # Названия классов берём из уже скачанного датасета
    class_names = datasets.CIFAR100(root="data", train=False, download=False).classes

    model = get_model(model_name).to(device)
    model.load_state_dict(torch.load(f"outputs/best_{model_name}.pth", map_location=device))
    model.eval()

    outputs = model(load_image(image_path).to(device))
    # softmax превращает оценки в проценты, которые в сумме дают 100%
    probabilities = torch.softmax(outputs, dim=1)[0]
    top_probs, top_ids = probabilities.topk(top_k)

    return [(class_names[i], p.item() * 100) for i, p in zip(top_ids, top_probs)]


def main():
    parser = argparse.ArgumentParser(description="Что на картинке? (CIFAR-100)")
    parser.add_argument("image", help="путь к картинке (jpg или png)")
    parser.add_argument("--model", default="simple", choices=["simple", "resnet"])
    args = parser.parse_args()

    print(f"\nКартинка: {args.image}\n")
    for rank, (name, prob) in enumerate(predict(args.image, args.model), start=1):
        print(f"  {rank}. {name:15s} {prob:5.1f}%")


if __name__ == "__main__":
    main()