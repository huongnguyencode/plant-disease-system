#!/usr/bin/env python3
"""Fine-tune EfficientNetV2-S on cropped real-world leaf images.

Kaggle example:
    python finetune_real_world_effnetv2s.py \
        --data /kaggle/input/plant-real-data/cropped \
        --class-names /kaggle/input/plant-models/class_names.json \
        --checkpoint /kaggle/input/plant-models/best_effnetv2s.pth \
        --output-dir /kaggle/working/models

To split a flat ``cropped/<class_name>`` dataset first:
    python finetune_real_world_effnetv2s.py \
        --split-source real_data/cropped \
        --data real_data/cropped \
        --valid-ratio 0.2
"""

import argparse
import json
import random
import shutil
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune EfficientNetV2-S trên ảnh lá thực tế đã crop."
    )
    parser.add_argument("--data", default="real_data/cropped")
    parser.add_argument("--class-names", default="models/class_names.json")
    parser.add_argument("--checkpoint", default="models/best_effnetv2s.pth")
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--resume", help="Checkpoint fine-tune để tiếp tục huấn luyện")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-source",
        help="Chia thư mục <class_name> phẳng thành data/train và data/valid",
    )
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    return parser.parse_args()


def load_class_names(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data[str(index)] for index in range(len(data))]
    raise ValueError("class_names.json phải là list hoặc object có key là chỉ số lớp.")


def split_flat_dataset(source_dir, dataset_dir, valid_ratio=0.2, seed=42):
    """Copy a flat class-folder dataset into train/valid class folders."""
    source_dir = Path(source_dir).resolve()
    dataset_dir = Path(dataset_dir).resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu cần chia: {source_dir}")
    if not 0.0 < valid_ratio < 1.0:
        raise ValueError("valid_ratio phải nằm trong khoảng (0, 1).")

    class_dirs = [
        path
        for path in sorted(source_dir.iterdir())
        if path.is_dir() and path.name not in {"train", "valid"}
    ]
    if not class_dirs:
        raise ValueError("Không tìm thấy thư mục lớp để chia train/valid.")

    rng = random.Random(seed)
    copied_train = 0
    copied_valid = 0
    for class_dir in class_dirs:
        images = [
            path
            for path in sorted(class_dir.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not images:
            print(f"Cảnh báo: lớp {class_dir.name} không có ảnh, bỏ qua.")
            continue

        rng.shuffle(images)
        valid_count = max(1, round(len(images) * valid_ratio)) if len(images) > 1 else 0
        valid_images = set(images[:valid_count])
        for image_path in images:
            split_name = "valid" if image_path in valid_images else "train"
            destination = dataset_dir / split_name / class_dir.name / image_path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination)
            if split_name == "valid":
                copied_valid += 1
            else:
                copied_train += 1

    print(f"Đã chia dữ liệu: train={copied_train}, valid={copied_valid}")


def build_transforms():
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(
                brightness=0.35,
                contrast=0.35,
                saturation=0.25,
                hue=0.03,
            ),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    valid_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, valid_transform


def image_folder_with_json_order(root, transform, class_names):
    """Create ImageFolder and remap its alphabetical targets to JSON indices."""
    dataset = datasets.ImageFolder(root=root, transform=transform)
    json_class_to_idx = {name: index for index, name in enumerate(class_names)}
    unknown_classes = sorted(set(dataset.classes) - set(json_class_to_idx))
    if unknown_classes:
        raise ValueError(
            "Các lớp không có trong class_names.json: " + ", ".join(unknown_classes)
        )

    original_classes = list(dataset.classes)
    remapped_samples = [
        (path, json_class_to_idx[original_classes[target]])
        for path, target in dataset.samples
    ]
    dataset.samples = remapped_samples
    dataset.imgs = remapped_samples
    dataset.targets = [target for _, target in remapped_samples]
    dataset.classes = list(class_names)
    dataset.class_to_idx = json_class_to_idx
    return dataset


def build_model(num_classes):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError("Định dạng checkpoint không được hỗ trợ.")


def load_model_weights(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    cleaned = {
        key.replace("module.", "", 1): value for key, value in state_dict.items()
    }
    model.load_state_dict(cleaned, strict=True)
    return checkpoint


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_training_checkpoint(path, model, optimizer, epoch, best_accuracy, history):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "history": history,
        },
        path,
    )


def main():
    args = parse_args()
    if not 10 <= args.epochs <= 20:
        raise ValueError("--epochs phải nằm trong khoảng từ 10 đến 20.")
    if args.lr not in {1e-5, 3e-5}:
        raise ValueError("--lr phải là 1e-5 hoặc 3e-5.")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_dir = Path(args.data)
    if args.split_source:
        split_flat_dataset(args.split_source, data_dir, args.valid_ratio, args.seed)

    train_dir = data_dir / "train"
    valid_dir = data_dir / "valid"
    if not train_dir.is_dir() or not valid_dir.is_dir():
        raise FileNotFoundError(
            "Dữ liệu phải có hai thư mục train/ và valid/. "
            "Dùng --split-source nếu dữ liệu hiện ở dạng <class_name> phẳng."
        )

    class_names = load_class_names(args.class_names)
    train_transform, valid_transform = build_transforms()
    train_dataset = image_folder_with_json_order(
        train_dir, train_transform, class_names
    )
    valid_dataset = image_folder_with_json_order(
        valid_dir, valid_transform, class_names
    )
    if not train_dataset.samples or not valid_dataset.samples:
        raise ValueError("Tập train và valid đều phải có ít nhất một ảnh.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    model = build_model(len(class_names)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    start_epoch = 0
    best_accuracy = -1.0
    history = []

    if args.resume:
        resume_checkpoint = load_model_weights(model, args.resume, device)
        if isinstance(resume_checkpoint, dict):
            if "optimizer_state_dict" in resume_checkpoint:
                optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1
            best_accuracy = float(resume_checkpoint.get("best_accuracy", 0.0))
            history = list(resume_checkpoint.get("history", []))
        print(f"Tiếp tục từ checkpoint: {args.resume}, epoch {start_epoch + 1}")
    else:
        load_model_weights(model, args.checkpoint, device)
        print(f"Đã tải trọng số ban đầu: {args.checkpoint}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = output_dir / "best_effnetv2s_real_finetuned.pth"
    last_checkpoint_path = output_dir / "last_effnetv2s_real_finetuned.pth"
    history_path = output_dir / "history_real_finetune.json"

    print(f"Thiết bị: {device}")
    print(f"Số ảnh train: {len(train_dataset)}, valid: {len(valid_dataset)}")
    print(f"Số lớp đầu ra: {len(class_names)}")

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        with torch.no_grad():
            valid_loss, valid_accuracy = run_epoch(
                model, valid_loader, criterion, device
            )

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "valid_loss": valid_loss,
            "valid_accuracy": valid_accuracy,
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_accuracy:.4%} | "
            f"valid_loss={valid_loss:.4f} | valid_acc={valid_accuracy:.4%}"
        )

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            save_training_checkpoint(
                best_model_path,
                model,
                optimizer,
                epoch,
                best_accuracy,
                history,
            )
            print(f"Đã lưu model tốt nhất: {best_model_path}")

        save_training_checkpoint(
            last_checkpoint_path,
            model,
            optimizer,
            epoch,
            best_accuracy,
            history,
        )
        with history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, ensure_ascii=False, indent=2)

    print(f"Hoàn tất. Validation accuracy tốt nhất: {best_accuracy:.4%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
