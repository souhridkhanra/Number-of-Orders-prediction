import random
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


DATA_GLOB = "images_*"
TRAIN_CSV_PATH = Path("train.csv")
MODEL_PATH = Path("landmark_model.pth")
OUTPUT_DIR = Path("outputs")

SEED = 42
MAX_IMAGES = None
EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 1e-4
MIN_SAMPLES_PER_CLASS = 3
MAX_CLASSES = 0
USE_FULL_DATASET_EACH_EPOCH = False


def set_seed(seed: int = SEED) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def collect_image_paths(data_roots: list[Path]) -> list[Path]:
	image_paths: list[Path] = []
	for data_dir in data_roots:
		image_paths.extend(sorted(data_dir.rglob("*.jpg")))
	if not image_paths:
		raise FileNotFoundError("No .jpg images found under images_* directories")
	return image_paths


def plot_dataset_samples(image_paths: list[Path], out_path: Path, n: int = 12) -> None:
	sample_paths = random.sample(image_paths, k=min(n, len(image_paths)))
	cols = 4
	rows = int(np.ceil(len(sample_paths) / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
	axes = np.array(axes).reshape(-1)

	for ax in axes:
		ax.axis("off")

	for ax, p in zip(axes, sample_paths):
		img = Image.open(p).convert("RGB")
		ax.imshow(img)
		ax.set_title(p.stem[:8], fontsize=9)
		ax.axis("off")

	fig.suptitle("Random Dataset Samples", fontsize=14)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def load_supervised_labels(
	image_paths: list[Path],
	train_csv_path: Path,
) -> tuple[list[Path], np.ndarray, dict[int, int]]:
	if not train_csv_path.exists():
		raise FileNotFoundError(f"Missing required labels file: {train_csv_path.resolve()}")

	requested_ids = {p.stem for p in image_paths}
	id_to_landmark: dict[str, int] = {}

	with train_csv_path.open("r", newline="") as f:
		reader = csv.DictReader(f)
		if "id" not in reader.fieldnames or "landmark_id" not in reader.fieldnames:
			raise ValueError("train.csv must have columns: id, landmark_id")
		for row in reader:
			img_id = row["id"]
			if img_id in requested_ids:
				id_to_landmark[img_id] = int(row["landmark_id"])

	labeled_paths: list[Path] = []
	raw_landmark_ids: list[int] = []
	for p in image_paths:
		img_id = p.stem
		if img_id in id_to_landmark:
			labeled_paths.append(p)
			raw_landmark_ids.append(id_to_landmark[img_id])

	if not labeled_paths:
		raise RuntimeError("No matching labels found between local images and train.csv")

	unique_landmarks = sorted(set(raw_landmark_ids))
	landmark_to_class = {landmark_id: i for i, landmark_id in enumerate(unique_landmarks)}
	class_to_landmark = {i: landmark_id for landmark_id, i in landmark_to_class.items()}
	labels = np.array([landmark_to_class[l] for l in raw_landmark_ids], dtype=np.int64)
	return labeled_paths, labels, class_to_landmark


def filter_and_select_classes(
	image_paths: list[Path],
	labels: np.ndarray,
	class_to_landmark: dict[int, int],
	min_count: int = MIN_SAMPLES_PER_CLASS,
	max_classes: int = MAX_CLASSES,
) -> tuple[list[Path], np.ndarray, dict[int, int]]:
	unique_labels, counts = np.unique(labels, return_counts=True)
	label_count = {int(label): int(count) for label, count in zip(unique_labels.tolist(), counts.tolist())}
	eligible = [label for label, count in label_count.items() if count >= min_count]

	if not eligible:
		raise RuntimeError(
			"No classes meet the minimum samples per class requirement. "
			"Increase MAX_IMAGES or lower MIN_SAMPLES_PER_CLASS."
		)

	# Keep the most frequent classes to reduce class fragmentation in small local shards.
	eligible_sorted = sorted(eligible, key=lambda label: label_count[label], reverse=True)
	selected = eligible_sorted[:max_classes] if max_classes > 0 else eligible_sorted
	keep_labels = set(selected)
	keep_mask = np.array([label in keep_labels for label in labels])

	filtered_paths = [p for i, p in enumerate(image_paths) if keep_mask[i]]
	filtered_labels = labels[keep_mask]

	# Reindex labels after filtering so classes are contiguous.
	new_unique = sorted(set(filtered_labels.tolist()))
	old_to_new = {old: new for new, old in enumerate(new_unique)}
	reindexed_labels = np.array([old_to_new[int(x)] for x in filtered_labels], dtype=np.int64)
	reindexed_class_to_landmark = {
		new_idx: class_to_landmark[old_idx] for old_idx, new_idx in old_to_new.items()
	}
	return filtered_paths, reindexed_labels, reindexed_class_to_landmark


class LandmarkDataset(Dataset):
	def __init__(self, image_paths: list[Path], labels: np.ndarray, transform=None):
		self.image_paths = image_paths
		self.labels = labels.astype(np.int64)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.image_paths)

	def __getitem__(self, idx: int):
		image = Image.open(self.image_paths[idx]).convert("RGB")
		if self.transform:
			image = self.transform(image)
		label = int(self.labels[idx])
		return image, label, str(self.image_paths[idx])


class SmallCNN(nn.Module):
	def __init__(self, num_classes: int):
		super().__init__()
		try:
			weights = models.ResNet18_Weights.DEFAULT
		except Exception:
			weights = None
		self.backbone = models.resnet18(weights=weights)

		# Fine-tune only higher-level features to reduce overfitting on small local shard.
		for p in self.backbone.parameters():
			p.requires_grad = False
		for p in self.backbone.layer4.parameters():
			p.requires_grad = True
		in_features = self.backbone.fc.in_features
		self.backbone.fc = nn.Sequential(
			nn.Dropout(p=0.2),
			nn.Linear(in_features, num_classes),
		)
		for p in self.backbone.fc.parameters():
			p.requires_grad = True

	def forward(self, x):
		return self.backbone(x)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
	model.eval()
	correct, total = 0, 0
	with torch.no_grad():
		for images, labels, _ in loader:
			images = images.to(device)
			labels = labels.to(device)
			logits = model(images)
			preds = logits.argmax(dim=1)
			correct += (preds == labels).sum().item()
			total += labels.numel()
	return correct / max(total, 1)


def evaluate_loss_and_accuracy(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
) -> tuple[float, float]:
	model.eval()
	running_loss = 0.0
	correct = 0
	total = 0

	with torch.no_grad():
		for images, labels, _ in loader:
			images = images.to(device)
			labels = labels.to(device)

			logits = model(images)
			loss = criterion(logits, labels)
			preds = logits.argmax(dim=1)

			running_loss += loss.item() * images.size(0)
			correct += (preds == labels).sum().item()
			total += labels.numel()

	avg_loss = running_loss / max(total, 1)
	acc = correct / max(total, 1)
	return avg_loss, acc


def train_model(
	model: nn.Module,
	train_loader: DataLoader,
	val_loader: DataLoader | None,
	device: torch.device,
) -> tuple[nn.Module, dict[str, list[float]]]:
	criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
	optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
	scheduler = optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="min",
		factor=0.5,
		patience=2,
		min_lr=LEARNING_RATE * 0.05,
	)

	best_acc = 0.0
	best_val_loss = float("inf")
	epochs_without_improvement = 0
	best_state = None
	history: dict[str, list[float]] = {"train_acc": [], "val_acc": []}

	for epoch in range(1, EPOCHS + 1):
		model.train()
		running_loss = 0.0
		train_correct = 0
		train_total = 0

		for images, labels, _ in train_loader:
			images = images.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			logits = model(images)
			loss = criterion(logits, labels)
			loss.backward()
			optimizer.step()

			preds = logits.argmax(dim=1)
			running_loss += loss.item() * images.size(0)
			train_correct += (preds == labels).sum().item()
			train_total += labels.numel()

		train_loss = running_loss / max(train_total, 1)
		train_acc = train_correct / max(train_total, 1)
		if val_loader is not None:
			val_loss, val_acc = evaluate_loss_and_accuracy(model, val_loader, criterion, device)
		else:
			val_loss, val_acc = train_loss, train_acc

		print(
			f"Epoch {epoch}/{EPOCHS} | "
			f"train_acc: {train_acc * 100:.2f}% | "
			f"val_acc: {val_acc * 100:.2f}%"
		)
		history["train_acc"].append(train_acc)
		history["val_acc"].append(val_acc)

		scheduler.step(val_loss)

		if val_loss < (best_val_loss - EARLY_STOPPING_MIN_DELTA):
			best_val_loss = val_loss
			best_acc = val_acc
			epochs_without_improvement = 0
			best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
		else:
			epochs_without_improvement += 1

		if val_loader is not None and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
			print(
				"Early stopping triggered: "
				f"no val_loss improvement for {EARLY_STOPPING_PATIENCE} consecutive epochs."
			)
			break

	if best_state is not None:
		model.load_state_dict(best_state)

	print(f"Best validation loss: {best_val_loss:.4f}")
	print(f"Best validation accuracy: {best_acc * 100:.2f}%")
	return model, history


def plot_accuracy_curves(history: dict[str, list[float]], out_path: Path) -> None:
	train_acc = history.get("train_acc", [])
	val_acc = history.get("val_acc", [])
	if not train_acc or not val_acc:
		return

	epochs = np.arange(1, len(train_acc) + 1)
	fig, ax = plt.subplots(figsize=(8, 5))
	ax.plot(epochs, np.array(train_acc) * 100.0, label="Train Accuracy", linewidth=2)
	ax.plot(epochs, np.array(val_acc) * 100.0, label="Validation Accuracy", linewidth=2)
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Accuracy (%)")
	ax.set_title("Training vs Validation Accuracy")
	ax.grid(alpha=0.3)
	ax.legend()
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def plot_predictions(
	model: nn.Module,
	val_dataset: LandmarkDataset,
	device: torch.device,
	out_path: Path,
	n: int = 12,
) -> None:
	model.eval()
	idxs = random.sample(range(len(val_dataset)), k=min(n, len(val_dataset)))

	cols = 4
	rows = int(np.ceil(len(idxs) / cols))
	fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
	axes = np.array(axes).reshape(-1)

	for ax in axes:
		ax.axis("off")

	with torch.no_grad():
		for ax, i in zip(axes, idxs):
			img_tensor, label, img_path = val_dataset[i]
			logits = model(img_tensor.unsqueeze(0).to(device))
			probs = torch.softmax(logits, dim=1)
			pred = int(torch.argmax(probs, dim=1).item())
			conf = float(probs[0, pred].item())

			img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
			img_np = np.clip(img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)

			ax.imshow(img_np)
			color = "green" if pred == label else "red"
			ax.set_title(
				f"true={label} pred={pred}\nconf={conf:.2f}\n{Path(img_path).name[:10]}",
				color=color,
				fontsize=8,
			)
			ax.axis("off")

	fig.suptitle("Validation Predictions (Supervised Landmark Classes)", fontsize=14)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def main() -> None:
	set_seed(SEED)
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	image_dirs = sorted([p for p in Path(".").glob(DATA_GLOB) if p.is_dir()])
	if not image_dirs:
		raise FileNotFoundError(f"No directories matching {DATA_GLOB!r} found")

	image_paths = collect_image_paths(image_dirs)
	if MAX_IMAGES is not None and len(image_paths) > MAX_IMAGES:
		image_paths = random.sample(image_paths, k=MAX_IMAGES)

	print(f"Using {len(image_paths)} images from {len(image_dirs)} shard(s): {[d.name for d in image_dirs]}")
	print(f"Loading supervised labels from: {TRAIN_CSV_PATH.resolve()}")

	plot_dataset_samples(image_paths, OUTPUT_DIR / "dataset_samples.png")

	labeled_paths, labels, class_to_landmark = load_supervised_labels(image_paths, TRAIN_CSV_PATH)
	print(f"Matched labels for {len(labeled_paths)} images from train.csv")

	applied_min_samples = None
	base_paths = labeled_paths
	base_labels = labels
	base_class_to_landmark = class_to_landmark
	for candidate_min in [MIN_SAMPLES_PER_CLASS, 4, 3, 2]:
		try:
			candidate_paths, candidate_labels, candidate_class_to_landmark = filter_and_select_classes(
				base_paths,
				base_labels,
				base_class_to_landmark,
				min_count=candidate_min,
				max_classes=MAX_CLASSES,
			)
			if len(set(candidate_labels.tolist())) >= 2:
				labeled_paths = candidate_paths
				labels = candidate_labels
				class_to_landmark = candidate_class_to_landmark
				applied_min_samples = candidate_min
				break
		except RuntimeError:
			continue

	if applied_min_samples is None or len(set(labels.tolist())) < 2:
		raise RuntimeError(
			"Not enough classes after adaptive filtering. Increase MAX_IMAGES substantially."
		)
	print(
		f"Using {len(labeled_paths)} labeled images across {len(set(labels.tolist()))} classes "
		f"(min_samples_per_class={applied_min_samples}, max_classes={MAX_CLASSES})."
	)

	val_ratio = 0.2
	train_paths = labeled_paths
	train_labels = labels
	val_paths: list[Path] = []
	val_labels = np.array([], dtype=np.int64)

	if not USE_FULL_DATASET_EACH_EPOCH:
		n_samples = len(labeled_paths)
		n_classes = len(set(labels.tolist()))
		n_val = max(1, int(round(n_samples * val_ratio)))
		if n_val < n_classes:
			val_ratio = min(0.5, n_classes / n_samples + 0.02)
			n_val = max(1, int(round(n_samples * val_ratio)))
		can_stratify = n_val >= n_classes

		if can_stratify:
			train_paths, val_paths, train_labels, val_labels = train_test_split(
				labeled_paths,
				labels,
				test_size=val_ratio,
				random_state=SEED,
				stratify=labels,
			)
		else:
			print(
				"Warning: too many classes for stratified split at current val ratio; "
				"using random split without stratification."
			)
			train_paths, val_paths, train_labels, val_labels = train_test_split(
				labeled_paths,
				labels,
				test_size=val_ratio,
				random_state=SEED,
				stratify=None,
			)

	eval_transform = transforms.Compose(
		[
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	train_transform = transforms.Compose(
		[
			transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
			transforms.RandomHorizontalFlip(),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
		]
	)
	print(f"Data split -> train: {len(train_paths)} | val: {len(val_paths)}")

	train_dataset = LandmarkDataset(train_paths, np.array(train_labels), transform=train_transform)
	val_dataset = LandmarkDataset(val_paths, np.array(val_labels), transform=eval_transform) if val_paths else None
	eval_dataset = LandmarkDataset(labeled_paths, np.array(labels), transform=eval_transform)

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
	val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2) if val_dataset else None
	eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Training on device: {device}")

	num_classes = len(set(labels.tolist()))
	model = SmallCNN(num_classes=num_classes).to(device)
	if USE_FULL_DATASET_EACH_EPOCH:
		print("Training mode: full dataset each epoch (no validation holdout).")
	model, history = train_model(model, train_loader, val_loader, device)

	torch.save(
		{
			"model_state_dict": model.state_dict(),
			"num_classes": num_classes,
			"seed": SEED,
			"class_to_landmark": class_to_landmark,
			"note": "Model trained with supervised labels from train.csv.",
		},
		MODEL_PATH,
	)

	if val_loader is not None:
		final_acc = evaluate(model, val_loader, device)
		print(f"Final validation accuracy: {final_acc * 100:.2f}%")
	else:
		final_acc = evaluate(model, eval_loader, device)
		print(f"Final full-dataset accuracy: {final_acc * 100:.2f}%")
	print(f"Saved model to: {MODEL_PATH.resolve()}")

	prediction_dataset = val_dataset if val_dataset is not None else eval_dataset
	plot_predictions(model, prediction_dataset, device, OUTPUT_DIR / "prediction_results.png")
	plot_accuracy_curves(history, OUTPUT_DIR / "accuracy_curve.png")
	print(f"Saved sample visualization to: {(OUTPUT_DIR / 'dataset_samples.png').resolve()}")
	print(f"Saved prediction visualization to: {(OUTPUT_DIR / 'prediction_results.png').resolve()}")
	print(f"Saved accuracy curve to: {(OUTPUT_DIR / 'accuracy_curve.png').resolve()}")


if __name__ == "__main__":
	main()