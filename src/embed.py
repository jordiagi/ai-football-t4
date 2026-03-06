from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torchreid.utils import FeatureExtractor


_EXTRACTOR: Optional[FeatureExtractor] = None


def _device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_extractor() -> FeatureExtractor:
    global _EXTRACTOR
    if _EXTRACTOR is None:
        _EXTRACTOR = FeatureExtractor(
            model_name="osnet_x1_0",
            model_path="",
            device=_device(),
        )
    return _EXTRACTOR


def compute_embedding(image_crop: np.ndarray) -> np.ndarray:
    """Compute an L2-normalized OSNet embedding for one image crop."""
    if image_crop is None or image_crop.size == 0:
        raise ValueError("image_crop is empty")

    rgb = cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB)
    extractor = _get_extractor()
    with torch.no_grad():
        feat = extractor([rgb])[0]

    vec = np.asarray(feat, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(vec) + 1e-12
    return vec / norm


def _read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def load_target_embeddings(
    ref_current_game: Optional[str] = None,
    ref_gallery: Optional[str] = None,
    ref_current_game_list: Optional[List[str]] = None,
) -> List[Tuple[float, np.ndarray, str]]:
    """
    Load target references as weighted embeddings.
    Returns list of (weight, embedding, source_path).

    Current-game images share 0.7 weight equally.
    Gallery images (ref_gallery dir) share 0.3 equally.
    """
    # Resolve current-game image list
    current_paths: List[Path] = []
    if ref_current_game_list:
        for p in ref_current_game_list:
            cp = Path(p)
            if cp.exists():
                current_paths.append(cp)
    if not current_paths and ref_current_game:
        cp = Path(ref_current_game)
        if not cp.exists():
            raise FileNotFoundError(f"Missing ref_current_game image: {ref_current_game}")
        current_paths.append(cp)
    if not current_paths:
        raise ValueError("No valid current-game reference images provided")

    entries: List[Tuple[float, np.ndarray, str]] = []

    per_current = 0.7 / len(current_paths)
    for cp in current_paths:
        emb = compute_embedding(_read_image(cp))
        entries.append((per_current, emb, str(cp)))

    gallery_embs: List[Tuple[float, np.ndarray, str]] = []
    if ref_gallery:
        gpath = Path(ref_gallery)
        if gpath.exists() and gpath.is_dir():
            for p in sorted(gpath.iterdir()):
                if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                emb = compute_embedding(_read_image(p))
                gallery_embs.append((0.0, emb, str(p)))

    if gallery_embs:
        per = 0.3 / len(gallery_embs)
        entries.extend((per, emb, src) for _, emb, src in gallery_embs)

    return entries


def weighted_similarity(embedding: np.ndarray, target_embeddings: Sequence[Tuple[float, np.ndarray, str]]) -> float:
    if not target_embeddings:
        return 0.0
    score = 0.0
    for w, ref, _ in target_embeddings:
        score += float(w) * float(np.dot(embedding, ref))
    return score
