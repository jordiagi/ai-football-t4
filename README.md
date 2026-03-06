# AI Football T4 — Player ID & Highlights Pipeline
### Kaggle T4 GPU Edition (16GB VRAM)

Multi-player identification and highlight extraction for full 2-hour soccer matches.
Forked and extended from `ai-football-highlights-cloudflare`.

---

## Architecture

```
Input Video (2hr MP4)
        │
        ▼
┌──────────────────┐
│  1. DETECTION    │  YOLOv8x / YOLO11x — person detection every N frames
│     (T4 fast)    │  ~30–60 FPS on T4 @ 720p
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  2. TRACKING     │  ByteTrack — assign track IDs across frames
│                  │  Handles occlusion, camera cuts
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  3. PLAYER ID    │  ReID (OSNet-x1-0 / BoT-SORT embeddings)
│                  │  Jersey number OCR (PaddleOCR / EasyOCR)
│                  │  Team classification (k-means on shirt color)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  4. BALL         │  Separate YOLO model (football-trained)
│     TRACKING     │  Associate ball possession to nearby players
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  5. HIGHLIGHTS   │  Score segments by:
│     SCORING      │  - Target player involvement
│                  │  - Ball proximity / possession
│                  │  - Motion intensity + speed
│                  │  - Audio RMS peaks (crowd reactions)
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  6. EXPORT       │  FFmpeg clip extraction
│                  │  highlight_player.mp4 (target player only)
│                  │  highlight_team.mp4 (full team moments)
│                  │  highlight_all.mp4 (best moments)
└──────────────────┘
```

---

## Kaggle Setup

### Session strategy (12hr limit)
- Phase 1 (Session A, ~3hr): Detection + Tracking → save `tracks.pkl`
- Phase 2 (Session B, ~2hr): Player ID + ReID → save `player_ids.json`
- Phase 3 (Session C, ~1hr): Highlight scoring + export

### Checkpointing
All phases write checkpoints every 5-minute chunk:
```
checkpoints/
  chunk_000_tracks.pkl
  chunk_001_tracks.pkl
  ...
  player_ids.json
  segments.json
```
Resume from any failed chunk automatically.

---

## New vs Cloudflare version

| Feature | Cloudflare | T4 |
|---|---|---|
| Target players | 1 | Multiple |
| Jersey OCR | ❌ | ✅ |
| Team classification | ❌ | ✅ |
| Ball tracking | ❌ | ✅ |
| Multi-GPU | ❌ | Planned |
| VRAM | CPU/MPS | 16GB T4 |
| Session | Persistent | 12hr chunks |

---

## Usage

```bash
# On Kaggle notebook:
!git clone https://github.com/jordiagi/ai-football-t4
%cd ai-football-t4
!pip install -r requirements.txt

# Phase 1: Detection + Tracking
!python src/pipeline.py --video /kaggle/input/match.mp4 --phase detect

# Phase 2: Player identification
!python src/pipeline.py --phase identify --target "Marc" --jersey 10

# Phase 3: Export highlights
!python src/pipeline.py --phase export --player "Marc"
```

---

## Models used (all free)
- `ultralytics/yolov8x.pt` — person detection
- `mikel-brostrom/yolo_tracking` — ByteTrack
- `kaiyangzhou/torchreid` — OSNet ReID embeddings
- `PaddlePaddle/PaddleOCR` — jersey number recognition
- Custom ball detection: `keremberke/yolov8m-football-player-detection`
