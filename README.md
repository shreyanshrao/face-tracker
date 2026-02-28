# 👁 Face Tracker — Visitor Identification System

Identifies and tracks unique individuals across multiple video clips using **YuNet** face detection, **SFace** embeddings, and **PostgreSQL pgvector** for similarity search.

---

## Architecture

```
Videos → YuNet (detect faces) → SFace (128-dim embeddings)
                                        ↓
                              pgvector cosine similarity search
                                        ↓
                        Assign face_id (new) or match existing
                                        ↓
                           PostgreSQL visitor log table
                                        ↓
                              React UI — visitor dashboard
```

---

## Quick Start (Docker — Recommended)

### Prerequisites
- Docker + Docker Compose
- Your demo videos (MP4, AVI, MOV, MKV)

### 1. Clone and Start
```bash
cd face-tracker
docker-compose up --build
```

This starts:
- **PostgreSQL + pgvector** on port 5432
- **FastAPI backend** on port 8000
- **React frontend** on port 3000

### 2. Open UI
Visit **http://localhost:3000**

### 3. Upload Videos
- Click the upload zone in the left sidebar
- Upload your demo videos (can select multiple at once)
- Processing runs in the background
- The UI polls every 3s while processing

### 4. View Results
- Select any processed video from the sidebar
- See all unique faces detected as cards
- **Green badge** = first-time visitor
- **Purple badge** = returning visitor (seen in previous video)
- Click any face card to see their full visit timeline

---

## Quick Start (Standalone Python Script)

If you prefer not to use Docker for the processing:

### Prerequisites
```bash
pip install opencv-python psycopg2-binary pgvector numpy

# Start just the database:
docker run -d --name pgvec \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Process Videos
```bash
cd scripts

# Process 3 videos
python process_videos.py video1.mp4 video2.mp4 video3.mp4

# Reset database and process fresh
python process_videos.py --reset video1.mp4 video2.mp4

# Adjust matching sensitivity (default 0.35, lower = stricter)
python process_videos.py --threshold 0.30 video1.mp4 video2.mp4
```

The script prints a full **Visitor Log** to the console after processing.

---

## Database Schema

### `persons` table
| Column | Type | Description |
|--------|------|-------------|
| face_id | VARCHAR(36) | Unique ID (e.g. `A3F9BC21`) |
| thumbnail_path | TEXT | Path to first face image |
| total_visits | INTEGER | Number of video appearances |
| created_at | TIMESTAMP | When first seen |

### `visits` table
| Column | Type | Description |
|--------|------|-------------|
| face_id | VARCHAR(36) | FK → persons |
| video_name | VARCHAR | Which video |
| frame_number | INTEGER | Frame of first full frontal detection |
| timestamp_seconds | FLOAT | Time in video |
| face_embedding | vector(128) | SFace embedding for similarity search |
| face_image_path | TEXT | Saved face crop URL |

### `video_processing_log` table
| Column | Type | Description |
|--------|------|-------------|
| video_name | VARCHAR | Video filename |
| status | VARCHAR | pending / processing / completed |
| total_faces_detected | INTEGER | Unique faces in this video |

---

## How It Works

### Face Detection — YuNet
- Lightweight ONNX model by OpenCV
- Detects multiple faces per frame (handles group entry scenarios)
- Filters to faces with confidence ≥ 0.8 for quality
- Samples 2 frames/second for efficiency

### Face Recognition — SFace
- Produces 128-dimensional embeddings
- Aligns face before embedding for consistency
- Robust to lighting and angle variations

### Matching — pgvector Cosine Similarity
```sql
SELECT face_id, 1 - (face_embedding <=> $1::vector) AS similarity
FROM visits
WHERE 1 - (face_embedding <=> $1::vector) > 0.65  -- threshold
ORDER BY face_embedding <=> $1::vector
LIMIT 1
```
- `<=>` = cosine distance operator
- IVFFlat index for fast approximate nearest neighbor search
- Default threshold: 0.65 similarity (0.35 distance)

### Multiple People Entering Together
Each frame is processed independently — all detected faces are extracted and matched/created simultaneously. No limit on faces per frame.

---

## Configuration

### Matching Threshold
In `backend/main.py`:
```python
MATCH_THRESHOLD = 0.35  # Cosine distance (lower = stricter matching)
```
- `0.25` — very strict, might not match same person across different angles
- `0.35` — balanced (default)
- `0.45` — lenient, may get false matches

### Sample Rate
```python
sample_interval = max(1, int(fps / 2))  # 2 frames/second
```
Increase divisor for more thorough detection, decrease for speed.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/upload` | Upload and process a video, photo archive (ZIP) or individual image |
| GET | `/api/sources` | List all processed sources (videos or photo sets) |
| GET | `/api/sources/{name}/persons` | Persons in a source (video or photo folder) |
| GET | `/api/persons` | All known persons |
| GET | `/api/persons/{face_id}/visits` | Visit history for a person |
| GET | `/api/stats` | Dashboard statistics |
| DELETE | `/api/reset` | Clear all data |

---

## UI Features

- **Dashboard stats** — total persons, appearances, returning visitors, videos
- **Sources sidebar** — click a video or photo set to see its visitor log
- **Person cards** — face thumbnail, ID, frame number, new/returning badge
- **Previous visit info** — shown inline on returning visitor cards
- **Timeline modal** — click any face to see full visit history across all videos
- **Auto-polling** — UI refreshes while videos are being processed
- **Reset button** — clear all data for fresh demo

---

## Notes for Demo Videos

For best results:
1. Record people walking **toward the camera** (frontal face capture)
2. Good lighting — avoid strong backlighting
3. At least one frame where each person is **within ~3m** of camera
4. People should be at least partially separated (not fully occluded by each other)
5. Suggested: 1920×1080 or 1280×720, 25-30fps
