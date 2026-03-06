from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from embed import weighted_similarity


@dataclass
class PrecisionMatchConfig:
    similarity_threshold: float = 0.80
    stable_steps: int = 3
    confusable_margin: float = 0.05
    max_missing_steps: int = 10


@dataclass
class PrecisionTargetMatcher:
    target_embeddings: list
    config: PrecisionMatchConfig = field(default_factory=PrecisionMatchConfig)

    _streak_track: Optional[int] = None
    _streak_len: int = 0
    _active_track: Optional[int] = None
    _missing_steps: int = 0

    def _scores(self, track_embeddings: Dict[int, np.ndarray]) -> Dict[int, float]:
        return {
            tid: weighted_similarity(emb, self.target_embeddings)
            for tid, emb in track_embeddings.items()
        }

    def update(self, track_embeddings: Dict[int, np.ndarray]) -> Optional[int]:
        """
        Precision-first matching.
        Returns confirmed target track ID or None.
        """
        if not track_embeddings:
            self._missing_steps += 1
            if self._missing_steps > self.config.max_missing_steps:
                self._active_track = None
            return self._active_track

        scores = self._scores(track_embeddings)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_track, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else -1.0

        passed_threshold = best_score >= self.config.similarity_threshold
        passed_margin = (best_score - second_score) >= self.config.confusable_margin

        if passed_threshold and passed_margin:
            if self._streak_track == best_track:
                self._streak_len += 1
            else:
                self._streak_track = best_track
                self._streak_len = 1

            if self._streak_len >= self.config.stable_steps:
                self._active_track = best_track
                self._missing_steps = 0
                return self._active_track
        else:
            # Precision-first: skip rather than guess.
            self._streak_track = None
            self._streak_len = 0
            self._missing_steps += 1
            if self._missing_steps > self.config.max_missing_steps:
                self._active_track = None

        # Keep returning active track only if it still exists in current candidates.
        if self._active_track is not None and self._active_track not in track_embeddings:
            self._missing_steps += 1
            if self._missing_steps > self.config.max_missing_steps:
                self._active_track = None

        return self._active_track
