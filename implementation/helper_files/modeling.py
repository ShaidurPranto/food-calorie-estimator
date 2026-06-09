from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torchvision import models, transforms

from .image_utils import tight_crop_foreground


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class TrainMetrics:
    train_count: int
    val_count: int
    threshold: float
    accepted_val_ratio: float


class FingerOneClassDetector:
    """One-class finger detector using pretrained visual embeddings + OneClassSVM."""

    def __init__(
        self,
        *,
        backbone: str = "resnet18",
        image_size: int = 224,
        device: str | None = None,
    ) -> None:
        self.backbone = backbone
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._extractor = self._build_extractor(backbone).to(self.device)
        self._extractor.eval()

        self.scaler: StandardScaler | None = None
        self.svm: OneClassSVM | None = None
        self.threshold: float | None = None

        self._train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=12),
                transforms.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self._infer_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _build_extractor(backbone: str) -> nn.Module:
        if backbone != "resnet18":
            raise ValueError("Only resnet18 is currently supported.")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        return model

    @staticmethod
    def list_images(folder: str | Path) -> List[Path]:
        folder_path = Path(folder)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")
        images = [p for p in folder_path.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        images.sort()
        return images

    def _load_pil(self, image_path: str | Path) -> Image.Image:
        bgr = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if bgr is None:
            raise FileNotFoundError(f"Could not open image: {image_path}")
        crop = tight_crop_foreground(bgr)

        if crop.ndim == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        elif crop.shape[2] == 4:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _embed(self, pil_images: Sequence[Image.Image], use_train_aug: bool) -> np.ndarray:
        transform = self._train_transform if use_train_aug else self._infer_transform

        tensors = [transform(img) for img in pil_images]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            feats = self._extractor(batch).detach().cpu().numpy()
        return feats

    def extract_embeddings(
        self,
        image_paths: Sequence[str | Path],
        *,
        repeats: int = 1,
        use_train_aug: bool = False,
    ) -> np.ndarray:
        all_feats: List[np.ndarray] = []
        for path in image_paths:
            pil = self._load_pil(path)
            for _ in range(max(repeats, 1)):
                feats = self._embed([pil], use_train_aug=use_train_aug)
                all_feats.append(feats[0])
        return np.vstack(all_feats)

    def fit(
        self,
        image_paths: Sequence[str | Path],
        *,
        nu: float = 0.08,
        false_reject_rate: float = 0.1,
        val_ratio: float = 0.2,
        seed: int = 42,
        train_aug_repeats: int = 8,
    ) -> TrainMetrics:
        paths = [Path(p) for p in image_paths]
        if len(paths) < 20:
            raise ValueError("Need at least 20 finger images for stable one-class training.")

        train_paths, val_paths = train_test_split(paths, test_size=val_ratio, random_state=seed)

        train_emb = self.extract_embeddings(train_paths, repeats=train_aug_repeats, use_train_aug=True)
        val_emb = self.extract_embeddings(val_paths, repeats=1, use_train_aug=False)

        self.scaler = StandardScaler()
        train_scaled = self.scaler.fit_transform(train_emb)
        val_scaled = self.scaler.transform(val_emb)

        self.svm = OneClassSVM(kernel="rbf", gamma="scale", nu=nu)
        self.svm.fit(train_scaled)

        val_scores = self.svm.decision_function(val_scaled)
        self.threshold = float(np.quantile(val_scores, false_reject_rate))

        accepted_val = (val_scores >= self.threshold).mean()
        return TrainMetrics(
            train_count=len(train_paths),
            val_count=len(val_paths),
            threshold=self.threshold,
            accepted_val_ratio=float(accepted_val),
        )

    def _check_ready(self) -> None:
        if self.scaler is None or self.svm is None or self.threshold is None:
            raise RuntimeError("Model is not trained/loaded. Call fit() or load().")

    def score(self, image_path: str | Path) -> float:
        self._check_ready()
        emb = self.extract_embeddings([image_path], repeats=1, use_train_aug=False)
        scaled = self.scaler.transform(emb)
        score = float(self.svm.decision_function(scaled)[0])
        return score

    def is_finger(self, image_path: str | Path) -> bool:
        score = self.score(image_path)
        return score >= float(self.threshold)

    def batch_scores(self, image_paths: Iterable[str | Path]) -> List[tuple[str, float]]:
        out: List[tuple[str, float]] = []
        for path in image_paths:
            s = self.score(path)
            out.append((str(path), s))
        out.sort(key=lambda x: x[1], reverse=True)
        return out

    def save(self, path: str | Path) -> None:
        self._check_ready()
        payload = {
            "backbone": self.backbone,
            "image_size": self.image_size,
            "threshold": self.threshold,
            "scaler": self.scaler,
            "svm": self.svm,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "FingerOneClassDetector":
        payload = joblib.load(path)
        model = cls(
            backbone=payload["backbone"],
            image_size=payload["image_size"],
            device=device,
        )
        model.threshold = float(payload["threshold"])
        model.scaler = payload["scaler"]
        model.svm = payload["svm"]
        return model
