#!/usr/bin/env python3
"""
Standalone face tracker - processes video files directly.
Use this if you don't want to run Docker.

Prerequisites:
    pip install opencv-python psycopg2-binary pgvector numpy

PostgreSQL with pgvector must be running:
    docker run -d -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
"""

import os
import sys
import uuid
import logging
import argparse
from pathlib import Path

import numpy as np
import cv2
import psycopg2
from psycopg2.extras import RealDictCursor
from pgvector.psycopg2 import register_vector

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/facetracker")
MODELS_DIR = Path(__file__).parent / "models"
FACE_IMAGES_DIR = Path(__file__).parent / "face_images"
MATCH_THRESHOLD = 0.35  # Cosine distance threshold (lower = stricter)

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"


def download_models():
    """Download YuNet and SFace models if not present."""
    import urllib.request
    MODELS_DIR.mkdir(exist_ok=True)
    
    yunet = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
    sface = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
    
    if not yunet.exists():
        logger.info("Downloading YuNet model...")
        urllib.request.urlretrieve(YUNET_URL, yunet)
    if not sface.exists():
        logger.info("Downloading SFace model...")
        urllib.request.urlretrieve(SFACE_URL, sface)
    
    return str(yunet), str(sface)


def init_db(conn):
    """Initialize database schema."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id SERIAL PRIMARY KEY,
                face_id VARCHAR(36) UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT NOW(),
                thumbnail_path TEXT,
                total_visits INTEGER DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS visits (
                id SERIAL PRIMARY KEY,
                face_id VARCHAR(36) NOT NULL REFERENCES persons(face_id),
                video_name VARCHAR(255) NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp_seconds FLOAT NOT NULL,
                face_embedding vector(128),
                face_image_path TEXT,
                detected_at TIMESTAMP DEFAULT NOW()
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS video_processing_log (
                id SERIAL PRIMARY KEY,
                video_name VARCHAR(255) UNIQUE NOT NULL,
                processed_at TIMESTAMP DEFAULT NOW(),
                total_faces_detected INTEGER DEFAULT 0,
                status VARCHAR(50) DEFAULT 'pending'
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS visits_embedding_idx 
            ON visits USING ivfflat (face_embedding vector_cosine_ops)
            WITH (lists = 100)
        """)
    conn.commit()
    logger.info("Database initialized.")


def get_models(yunet_path, sface_path):
    detector = cv2.FaceDetectorYN.create(yunet_path, "", (320, 320), 0.6, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(sface_path, "")
    return detector, recognizer


def detect_faces(frame, detector, recognizer):
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    results = []
    for face in faces:
        score = face[-1]
        if score < 0.8:
            continue
        aligned = recognizer.alignCrop(frame, face)
        feat = recognizer.feature(aligned).flatten()
        bbox = face[:4].astype(int).tolist()
        results.append((bbox, feat, aligned, float(score)))
    return results


def find_or_create_person(conn, embedding):
    """Returns (face_id, is_new, previous_visit_or_None)"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        emb_list = embedding.tolist()
        cur.execute("""
            SELECT face_id, video_name, frame_number, timestamp_seconds,
                   1 - (face_embedding <=> %s::vector) AS similarity
            FROM visits
            WHERE 1 - (face_embedding <=> %s::vector) > %s
            ORDER BY face_embedding <=> %s::vector
            LIMIT 1
        """, (emb_list, emb_list, 1 - MATCH_THRESHOLD, emb_list))
        match = cur.fetchone()
        
        if match:
            return match['face_id'], False, dict(match)
        
        # New person
        face_id = str(uuid.uuid4())[:8].upper()
        cur.execute("INSERT INTO persons (face_id) VALUES (%s)", (face_id,))
        conn.commit()
        return face_id, True, None


def process_video(video_path: str, conn, detector, recognizer):
    video_name = Path(video_path).name
    FACE_IMAGES_DIR.mkdir(exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing: {video_name} ({total} frames @ {fps:.1f}fps)")
    
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO video_processing_log (video_name, status)
            VALUES (%s, 'processing')
            ON CONFLICT (video_name) DO UPDATE SET status='processing'
        """, (video_name,))
    conn.commit()
    
    seen = {}  # face_id -> True (deduplicate per video)
    sample_interval = max(1, int(fps / 2))
    frame_num = 0
    total_detected = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_num % sample_interval == 0:
            faces = detect_faces(frame, detector, recognizer)
            for bbox, embedding, aligned, score in faces:
                face_id, is_new, prev = find_or_create_person(conn, embedding)
                
                if face_id not in seen:
                    seen[face_id] = True
                    total_detected += 1
                    
                    # Save face crop
                    img_filename = f"{face_id}_{video_name.replace('.','_')}_{frame_num}.jpg"
                    img_path = FACE_IMAGES_DIR / img_filename
                    cv2.imwrite(str(img_path), aligned)
                    
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO visits 
                            (face_id, video_name, frame_number, timestamp_seconds, face_embedding, face_image_path)
                            VALUES (%s, %s, %s, %s, %s::vector, %s)
                        """, (face_id, video_name, frame_num, frame_num/fps, 
                              embedding.tolist(), f"/face_images/{img_filename}"))
                        
                        if is_new:
                            cur.execute("""
                                UPDATE persons SET thumbnail_path=%s WHERE face_id=%s
                            """, (f"/face_images/{img_filename}", face_id))
                        
                        cur.execute("""
                            UPDATE persons SET total_visits = total_visits + 1 WHERE face_id=%s
                        """, (face_id,))
                    conn.commit()
                    
                    if is_new:
                        logger.info(f"New face {face_id}")
                    else:
                        logger.info(f"Returning face {face_id}")
        
        frame_num += 1

    # finished reading
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE video_processing_log
            SET status='completed', total_faces_detected=%s, processed_at=NOW()
            WHERE video_name=%s
        """, (total_detected, video_name))
    conn.commit()
    logger.info(f"Done processing: {video_name} ({total_detected} unique faces)")


def process_photo_dir(folder_path: str, conn, detector, recognizer):
    """Process a folder of images (JPEG/PNG) in the same way as a video."""
    folder_name = Path(folder_path).name
    FACE_IMAGES_DIR.mkdir(exist_ok=True)
    image_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    files = sorted([p for p in Path(folder_path).iterdir() if p.suffix.lower() in image_exts])
    total = len(files)
    logger.info(f"Processing photo folder: {folder_name} ({total} images)")

    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO video_processing_log (video_name, status)
            VALUES (%s, 'processing')
            ON CONFLICT (video_name) DO UPDATE SET status='processing'
        """, (folder_name,))
    conn.commit()

    seen = {}
    total_detected = 0
    for frame_num, img_path in enumerate(files):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        faces = detect_faces(frame, detector, recognizer)
        for bbox, embedding, aligned, score in faces:
            face_id, is_new, prev = find_or_create_person(conn, embedding)
            if face_id not in seen:
                seen[face_id] = True
                total_detected += 1
                img_filename = f"{face_id}_{folder_name.replace('.','_')}_{frame_num}.jpg"
                img_out = FACE_IMAGES_DIR / img_filename
                cv2.imwrite(str(img_out), aligned)
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO visits 
                        (face_id, video_name, frame_number, timestamp_seconds, face_embedding, face_image_path)
                        VALUES (%s, %s, %s, %s, %s::vector, %s)
                    """, (face_id, folder_name, frame_num, frame_num,
                          embedding.tolist(), f"/face_images/{img_filename}"))
                    if is_new:
                        cur.execute("""
                            UPDATE persons SET thumbnail_path=%s WHERE face_id=%s
                        """, (f"/face_images/{img_filename}", face_id))
                    cur.execute("""
                        UPDATE persons SET total_visits = total_visits + 1 WHERE face_id=%s
                    """, (face_id,))
                conn.commit()
                logger.info(f"Face {face_id} | {'NEW' if is_new else 'RETURNING'} | image {frame_num} | folder {folder_name}")

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE video_processing_log
            SET status='completed', total_faces_detected=%s, processed_at=NOW()
            WHERE video_name=%s
        """, (total_detected, folder_name))
    conn.commit()
    logger.info(f"Done processing folder {folder_name}: {total_detected} unique faces")
    


def print_visitor_log(conn):
    """Print a summary visitor log table."""
    print("\n" + "="*80)
    print(" VISITOR LOG")
    print("="*80)
    
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT p.face_id, p.total_visits, p.thumbnail_path,
                   array_agg(v.video_name || ' (frame ' || v.frame_number || ')' ORDER BY v.detected_at) as appearances
            FROM persons p
            JOIN visits v ON p.face_id = v.face_id
            GROUP BY p.face_id, p.total_visits, p.thumbnail_path
            ORDER BY p.total_visits DESC
        """)
        persons = cur.fetchall()
    
    for p in persons:
        tag = "↩ RETURNING" if p['total_visits'] > 1 else "✦ FIRST VISIT"
        print(f"\n  Face ID: #{p['face_id']}  [{tag}]  ({p['total_visits']} appearances)")
        for a in p['appearances']:
            print(f"    → {a}")
    
    print("\n" + "="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Face Tracker - Process videos or photo folders and identify returning visitors")
    parser.add_argument("videos", nargs="+", help="Paths to video files, directories of images, or zip archives")
    parser.add_argument("--threshold", type=float, default=0.35, help="Match threshold (0-1, lower=stricter)")
    parser.add_argument("--reset", action="store_true", help="Reset database before processing")
    args = parser.parse_args()
    
    global MATCH_THRESHOLD
    MATCH_THRESHOLD = args.threshold
    
    # Setup
    yunet_path, sface_path = download_models()
    
    conn = psycopg2.connect(DB_URL)
    register_vector(conn)
    init_db(conn)
    
    if args.reset:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE visits, persons, video_processing_log RESTART IDENTITY CASCADE")
        conn.commit()
        logger.info("Database reset.")
    
    detector, recognizer = get_models(yunet_path, sface_path)
    
    # Process each input path (video file, directory of images, or zip archive)
    import zipfile, tempfile
    for video_path in args.videos:
        p = Path(video_path)
        if not p.exists():
            logger.error(f"Path not found: {video_path}")
            continue
        if p.is_dir():
            # treat directory as collection of images
            process_photo_dir(str(p), conn, detector, recognizer)
        elif p.suffix.lower() == '.zip':
            # extract to temp and process
            tmpdir = tempfile.mkdtemp()
            with zipfile.ZipFile(str(p), 'r') as z:
                z.extractall(tmpdir)
            process_photo_dir(tmpdir, conn, detector, recognizer)
        else:
            process_video(str(p), conn, detector, recognizer)
    
    # Print summary
    print_visitor_log(conn)
    conn.close()


if __name__ == "__main__":
    main()
