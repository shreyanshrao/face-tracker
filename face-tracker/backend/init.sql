CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS persons (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(36) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    thumbnail_path TEXT,
    total_visits INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS visits (
    id SERIAL PRIMARY KEY,
    face_id VARCHAR(36) NOT NULL REFERENCES persons(face_id),
    video_name VARCHAR(255) NOT NULL,
    frame_number INTEGER NOT NULL,
    timestamp_seconds FLOAT NOT NULL,
    face_embedding vector(128),
    face_image_path TEXT,
    detected_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS video_processing_log (
    id SERIAL PRIMARY KEY,
    video_name VARCHAR(255) UNIQUE NOT NULL,
    processed_at TIMESTAMP DEFAULT NOW(),
    total_faces_detected INTEGER DEFAULT 0,
    status VARCHAR(50) DEFAULT 'pending'
);

-- Index for fast vector similarity search
CREATE INDEX IF NOT EXISTS visits_embedding_idx 
    ON visits USING ivfflat (face_embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS visits_face_id_idx ON visits(face_id);
CREATE INDEX IF NOT EXISTS visits_video_name_idx ON visits(video_name);
