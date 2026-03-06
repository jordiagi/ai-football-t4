# Player Reference Images

## Target Player
- **Jersey number:** 23
- **Kit:** Yellow jersey, dark shorts
- **Files:** player1.jpg (close-up), player2.jpg (wider angle)
- **Video:** https://youtu.be/Flm3Ki1TCXY

## Pipeline notes
- OSNet re-ID: use both images (weighted average embedding, 0.7/0.3 split)
- Cleat-based tracking: crop bottom 25% of bbox for cleat color/pattern matching
  (useful when jersey number is obscured or player is far from camera)
- Jersey OCR: target "23" as secondary confirmation signal
