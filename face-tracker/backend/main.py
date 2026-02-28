import os
import uuid
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
import numpy as np
import cv2

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Face Tracker API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FACE_IMAGES_DIR = Path("/app/face_images")
VIDEOS_DIR = Path("/app/videos")
PHOTOS_DIR = Path("/app/photos")  # for CCTV image folders (zipped or unzipped)
MODELS_DIR = Path("/app/models")
FACE_IMAGES_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)
PHOTOS_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/facetracker")

# Similarity threshold for matching faces (cosine distance - lower = more similar)
MATCH_THRESHOLD = 0.35


def get_db():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
    return conn


def get_face_models():
    yunet_path = str(MODELS_DIR / "face_detection_yunet_2023mar.onnx")
    sface_path = str(MODELS_DIR / "face_recognition_sface_2021dec.onnx")
    
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    return detector, recognizer


def detect_and_embed_faces(frame: np.ndarray, detector, recognizer):
    """Detect faces and compute embeddings. Returns list of (bbox, embedding, aligned_face)."""
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    
    results = []
    for face in faces:
        # Only process faces with reasonable frontal visibility
        # face[4] is the score, face[0:4] is bbox
        score = face[-1]
        if score < 0.8:
            continue
        
        # Align face
        aligned = recognizer.alignCrop(frame, face)
        
        # Get embedding
        feature = recognizer.feature(aligned)
        embedding = feature.flatten()
        
        bbox = face[:4].astype(int).tolist()  # x, y, w, h
        results.append((bbox, embedding, aligned, score))
    
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def find_or_create_person(conn, embedding: np.ndarray) -> tuple[str, bool, Optional[dict]]:
    """
    Search pgvector for similar face. 
    Returns (face_id, is_new, previous_visit_info)
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    embedding_list = embedding.tolist()
    
    # Vector similarity search - find closest match
    cursor.execute("""
        SELECT v.face_id, v.video_name, v.frame_number, v.timestamp_seconds,
               1 - (v.face_embedding <=> %s::vector) AS similarity
        FROM visits v
        WHERE 1 - (v.face_embedding <=> %s::vector) > %s
        ORDER BY v.face_embedding <=> %s::vector
        LIMIT 1
    """, (embedding_list, embedding_list, 1 - MATCH_THRESHOLD, embedding_list))
    
    match = cursor.fetchone()
    
    if match:
        face_id = match['face_id']
        previous_visit = {
            'video_name': match['video_name'],
            'frame_number': match['frame_number'],
            'timestamp_seconds': match['timestamp_seconds'],
            'similarity': float(match['similarity'])
        }
        return face_id, False, previous_visit
    else:
        # New person
        face_id = str(uuid.uuid4())[:8].upper()
        cursor.execute("""
            INSERT INTO persons (face_id, total_visits) VALUES (%s, 0)
            ON CONFLICT (face_id) DO NOTHING
        """, (face_id,))
        conn.commit()
        return face_id, True, None


def save_face_image(face_img: np.ndarray, face_id: str, video_name: str, frame_no: int) -> str:
    """Save cropped face image and return relative path."""
    filename = f"{face_id}_{video_name}_{frame_no}.jpg"
    filepath = FACE_IMAGES_DIR / filename
    cv2.imwrite(str(filepath), face_img)
    return f"/face_images/{filename}"


def _log_start(source_name: str):
    """Helper used by both video and photo pipelines to mark start in the log."""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("""
        INSERT INTO video_processing_log (video_name, status) 
        VALUES (%s, 'processing')
        ON CONFLICT (video_name) DO UPDATE SET status='processing'
    """, (source_name,))
    conn.commit()
    conn.close()


def process_video_file(video_path: str, video_name: str):
    """Main video processing pipeline."""
    conn = get_db()
    detector, recognizer = get_face_models()
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing video: {video_name}, {total_frames} frames @ {fps}fps")
    
    # Track faces seen in this video to avoid duplicates
    seen_faces_this_video = {}  # face_id -> first frame info
    
    # Sample every N frames for efficiency
    sample_interval = max(1, int(fps / 2))  # 2 frames per second
    
    frame_num = 0
    faces_detected_total = 0
    
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Update log to processing
    _log_start(video_name)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % sample_interval == 0:
            faces = detect_and_embed_faces(frame, detector, recognizer)
            
            for bbox, embedding, aligned_face, score in faces:
                face_id, is_new, previous_visit = find_or_create_person(conn, embedding)
                
                # Only log first appearance per person per video
                if face_id not in seen_faces_this_video:
                    seen_faces_this_video[face_id] = True
                    faces_detected_total += 1
                    
                    # Save face image
                    face_image_path = save_face_image(aligned_face, face_id, 
                                                       video_name.replace('.', '_'), frame_num)
                    
                    # Save visit
                    embedding_list = embedding.tolist()
                    cursor.execute("""
                        INSERT INTO visits 
                        (face_id, video_name, frame_number, timestamp_seconds, face_embedding, face_image_path)
                        VALUES (%s, %s, %s, %s, %s::vector, %s)
                    """, (face_id, video_name, frame_num, frame_num / fps, 
                          embedding_list, face_image_path))
                    
                    # Update person thumbnail if new
                    if is_new:
                        cursor.execute("""
                            UPDATE persons SET thumbnail_path=%s WHERE face_id=%s
                        """, (face_image_path, face_id))
                    
                    # Increment visit count
                    cursor.execute("""
                        UPDATE persons SET total_visits = total_visits + 1 WHERE face_id=%s
                    """, (face_id,))
                    
                    conn.commit()
                    
                    logger.info(f"Face {face_id} | {'NEW' if is_new else 'RETURNING'} | "
                               f"Frame {frame_num} | Video {video_name}"
                               + (f" | Prev: {previous_visit['video_name']}" if previous_visit else ""))
        
        frame_num += 1
    
    cap.release()
    
    # Update log
    cursor.execute("""
        UPDATE video_processing_log 
        SET status='completed', total_faces_detected=%s, processed_at=NOW()
        WHERE video_name=%s
    """, (faces_detected_total, video_name))
    conn.commit()
    conn.close()
    
    logger.info(f"Done processing {video_name}. {faces_detected_total} unique faces detected.")
    return faces_detected_total


# ─── Processing helper for photo folders ─────────────────────────────────────────

def process_photo_folder(folder_path: str, folder_name: str):
    """Walk through each image in the folder and run the face pipeline similarly to a video.
    The folder may be a plain directory or a temporary extraction of a ZIP archive.
    """
    conn = get_db()
    detector, recognizer = get_face_models()

    # gather image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = sorted(
        [p for p in Path(folder_path).iterdir() if p.suffix.lower() in image_extensions]
    )
    total_frames = len(files)
    logger.info(f"Processing photo folder: {folder_name}, {total_frames} images")

    seen_faces_this_video = {}
    faces_detected_total = 0
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    _log_start(folder_name)

    for frame_num, img_path in enumerate(files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        faces = detect_and_embed_faces(frame, detector, recognizer)
        for bbox, embedding, aligned_face, score in faces:
            face_id, is_new, previous_visit = find_or_create_person(conn, embedding)
            if face_id not in seen_faces_this_video:
                seen_faces_this_video[face_id] = True
                faces_detected_total += 1
                face_image_path = save_face_image(aligned_face, face_id,
                                                   folder_name.replace('.', '_'), frame_num)
                embedding_list = embedding.tolist()
                cursor.execute("""
                    INSERT INTO visits 
                    (face_id, video_name, frame_number, timestamp_seconds, face_embedding, face_image_path)
                    VALUES (%s, %s, %s, %s, %s::vector, %s)
                """, (face_id, folder_name, frame_num, frame_num, embedding_list, face_image_path))
                if is_new:
                    cursor.execute("""
                        UPDATE persons SET thumbnail_path=%s WHERE face_id=%s
                    """, (face_image_path, face_id))
                cursor.execute("""
                    UPDATE persons SET total_visits = total_visits + 1 WHERE face_id=%s
                """, (face_id,))
                conn.commit()
                logger.info(f"Face {face_id} | {'NEW' if is_new else 'RETURNING'} | "
                           f"Image {frame_num} | Folder {folder_name}" +
                           (f" | Prev: {previous_visit['video_name']}" if previous_visit else ""))

    # Update log
    cursor.execute("""
        UPDATE video_processing_log 
        SET status='completed', total_faces_detected=%s, processed_at=NOW()
        WHERE video_name=%s
    """, (faces_detected_total, folder_name))
    conn.commit()
    conn.close()

    logger.info(f"Done processing {folder_name}. {faces_detected_total} unique faces detected.")
    return faces_detected_total

# ─── API Routes ───────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "Face Tracker API running"}


@app.post("/api/upload")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload and process either a video or a set of photos (zip or images).
    - Video formats: mp4, avi, mov, mkv
    - Photo folders must be uploaded as a ZIP archive containing images
      or as individual image files (jpg/png).
    """
    filename = file.filename or "unnamed"
    lower = filename.lower()
    content = await file.read()

    # video case
    if lower.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        video_path = VIDEOS_DIR / filename
        with open(video_path, 'wb') as f:
            f.write(content)
        background_tasks.add_task(process_video_file, str(video_path), filename)
        return {"message": f"Video '{filename}' uploaded. Processing started.",
                "source_name": filename}

    # zip archive containing images
    if lower.endswith('.zip'):
        zip_path = PHOTOS_DIR / filename
        with open(zip_path, 'wb') as f:
            f.write(content)
        # extract into a temp folder
        import zipfile, tempfile
        tmpdir = tempfile.mkdtemp(dir=PHOTOS_DIR)
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(tmpdir)
        folder_name = filename.rsplit('.', 1)[0]
        background_tasks.add_task(process_photo_folder, tmpdir, folder_name)
        return {"message": f"Photo archive '{filename}' uploaded. Processing started.",
                "source_name": folder_name}

    # single image
    if lower.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        img_path = PHOTOS_DIR / filename
        with open(img_path, 'wb') as f:
            f.write(content)
        # process single image as a folder with one file
        background_tasks.add_task(process_photo_folder, str(img_path.parent), filename)
        return {"message": f"Image '{filename}' uploaded. Processing started.",
                "source_name": filename}

    raise HTTPException(400, "Unsupported upload format")


@app.get("/api/sources")
def list_sources():
    """List all processed sources (videos or photo sets)."""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    cursor.execute("SELECT * FROM video_processing_log ORDER BY processed_at DESC")
    sources = cursor.fetchall()
    conn.close()
    return sources


@app.get("/api/sources/{source_name}/persons")
def get_persons_in_source(source_name: str):
    """Get unique persons detected in a source (video or photo set) with visit history."""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    # Get all visits in this source
    cursor.execute("""
        SELECT v.face_id, v.frame_number, v.timestamp_seconds, v.face_image_path,
               p.total_visits, p.thumbnail_path
        FROM visits v
        JOIN persons p ON v.face_id = p.face_id
        WHERE v.video_name = %s
        ORDER BY v.frame_number ASC
    """, (source_name,))
    
    current_visits = cursor.fetchall()
    
    result = []
    for visit in current_visits:
        face_id = visit['face_id']
        
        # Get previous appearances (other videos)
        cursor.execute("""
            SELECT video_name, frame_number, timestamp_seconds, face_image_path, detected_at
            FROM visits
            WHERE face_id = %s AND video_name != %s
            ORDER BY detected_at ASC
        """, (face_id, video_name))
        
        prev_visits = cursor.fetchall()
        
        result.append({
            "face_id": face_id,
            "frame_number": visit['frame_number'],
            "timestamp_seconds": round(float(visit['timestamp_seconds']), 2),
            "face_image_path": visit['face_image_path'],
            "total_visits": visit['total_visits'],
            "is_returning": len(prev_visits) > 0,
            "previous_visits": [dict(v) for v in prev_visits]
        })
    
    conn.close()
    return result


@app.get("/api/persons")
def list_all_persons():
    """List all known persons with their visit summary."""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT p.face_id, p.thumbnail_path, p.total_visits, p.created_at,
               array_agg(DISTINCT v.video_name) as videos_appeared
        FROM persons p
        LEFT JOIN visits v ON p.face_id = v.face_id
        GROUP BY p.face_id, p.thumbnail_path, p.total_visits, p.created_at
        ORDER BY p.total_visits DESC, p.created_at ASC
    """)
    
    persons = cursor.fetchall()
    conn.close()
    return [dict(p) for p in persons]


@app.get("/api/persons/{face_id}/visits")
def get_person_visits(face_id: str):
    """Get full visit history for a person."""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("""
        SELECT * FROM visits WHERE face_id = %s ORDER BY detected_at ASC
    """, (face_id,))
    
    visits = cursor.fetchall()
    conn.close()
    
    if not visits:
        raise HTTPException(404, f"No visits found for face_id {face_id}")
    
    return [dict(v) for v in visits]


@app.get("/api/stats")
def get_stats():
    """Dashboard statistics."""
    conn = get_db()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    cursor.execute("SELECT COUNT(*) as total_persons FROM persons")
    total_persons = cursor.fetchone()['total_persons']
    
    cursor.execute("SELECT COUNT(*) as total_visits FROM visits")
    total_visits = cursor.fetchone()['total_visits']
    
    cursor.execute("""
        SELECT COUNT(DISTINCT face_id) as returning_visitors
        FROM persons WHERE total_visits > 1
    """)
    returning = cursor.fetchone()['returning_visitors']
    
    cursor.execute("SELECT COUNT(*) as total_videos FROM video_processing_log WHERE status='completed'")
    total_videos = cursor.fetchone()['total_videos']
    
    conn.close()
    
    return {
        "total_persons": total_persons,
        "total_visits": total_visits,
        "returning_visitors": returning,
        "total_videos": total_videos
    }


@app.delete("/api/reset")
def reset_database():
    """Reset all data (for testing)."""
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("TRUNCATE visits, persons, video_processing_log RESTART IDENTITY CASCADE")
    conn.commit()
    conn.close()
    
    # Clear face images
    for f in FACE_IMAGES_DIR.glob("*.jpg"):
        f.unlink()
    
    return {"message": "Database reset complete"}


# Mount face images as static files
app.mount("/face_images", StaticFiles(directory=str(FACE_IMAGES_DIR)), name="face_images")
