import React, { useState, useEffect, useCallback } from 'react';

const API = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const styles = {
  root: {
    fontFamily: "'Inter', -apple-system, sans-serif",
    background: '#0a0a0f',
    minHeight: '100vh',
    color: '#e2e8f0',
    margin: 0,
  },
  header: {
    background: 'linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%)',
    borderBottom: '1px solid #334155',
    padding: '0 32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 64,
    position: 'sticky',
    top: 0,
    zIndex: 100,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: 700,
    color: '#818cf8',
    display: 'flex',
    alignItems: 'center',
    gap: 10,
  },
  statBar: {
    display: 'flex',
    gap: 24,
    padding: '20px 32px',
    borderBottom: '1px solid #1e293b',
    background: '#0d1117',
  },
  statCard: {
    background: '#131929',
    border: '1px solid #1e293b',
    borderRadius: 12,
    padding: '14px 24px',
    flex: 1,
    textAlign: 'center',
  },
  statNum: {
    fontSize: 32,
    fontWeight: 800,
    color: '#818cf8',
    lineHeight: 1,
  },
  statLabel: {
    fontSize: 12,
    color: '#64748b',
    marginTop: 4,
    textTransform: 'uppercase',
    letterSpacing: '0.05em',
  },
  layout: {
    display: 'grid',
    gridTemplateColumns: '300px 1fr',
    gap: 0,
    minHeight: 'calc(100vh - 130px)',
  },
  sidebar: {
    background: '#0d1117',
    borderRight: '1px solid #1e293b',
    padding: 20,
    overflowY: 'auto',
  },
  sidebarTitle: {
    fontSize: 11,
    fontWeight: 600,
    color: '#475569',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    marginBottom: 12,
  },
  videoItem: {
    background: '#131929',
    border: '1px solid #1e293b',
    borderRadius: 10,
    padding: '12px 16px',
    marginBottom: 8,
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  videoItemActive: {
    border: '1px solid #818cf8',
    background: '#1e1b4b22',
  },
  videoName: {
    fontSize: 13,
    fontWeight: 600,
    color: '#e2e8f0',
    wordBreak: 'break-all',
  },
  videoMeta: {
    fontSize: 11,
    color: '#64748b',
    marginTop: 4,
  },
  statusBadge: (status) => ({
    display: 'inline-block',
    padding: '2px 8px',
    borderRadius: 20,
    fontSize: 10,
    fontWeight: 600,
    marginTop: 6,
    background: status === 'completed' ? '#052e16' : status === 'processing' ? '#1c1917' : '#0f172a',
    color: status === 'completed' ? '#4ade80' : status === 'processing' ? '#fb923c' : '#94a3b8',
    border: `1px solid ${status === 'completed' ? '#166534' : status === 'processing' ? '#92400e' : '#334155'}`,
  }),
  mainContent: {
    padding: 24,
    overflowY: 'auto',
  },
  uploadZone: {
    border: '2px dashed #334155',
    borderRadius: 16,
    padding: 40,
    textAlign: 'center',
    marginBottom: 24,
    background: '#0d1117',
    transition: 'all 0.2s',
    cursor: 'pointer',
  },
  uploadBtn: {
    background: 'linear-gradient(135deg, #6366f1, #818cf8)',
    color: '#fff',
    border: 'none',
    borderRadius: 8,
    padding: '10px 24px',
    fontSize: 14,
    fontWeight: 600,
    cursor: 'pointer',
    marginTop: 12,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: 700,
    color: '#e2e8f0',
    marginBottom: 16,
    display: 'flex',
    alignItems: 'center',
    gap: 8,
  },
  personGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: 16,
  },
  personCard: {
    background: '#0d1117',
    border: '1px solid #1e293b',
    borderRadius: 14,
    overflow: 'hidden',
    transition: 'border-color 0.15s',
    cursor: 'pointer',
  },
  personCardReturning: {
    border: '1px solid #6366f1',
    boxShadow: '0 0 20px #6366f110',
  },
  personImg: {
    width: '100%',
    aspectRatio: '1',
    objectFit: 'cover',
    background: '#131929',
    display: 'block',
  },
  personInfo: {
    padding: 12,
  },
  faceId: {
    fontSize: 14,
    fontWeight: 700,
    color: '#818cf8',
    fontFamily: 'monospace',
  },
  frameBadge: {
    fontSize: 11,
    color: '#64748b',
    marginTop: 4,
  },
  returningBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    background: '#1e1b4b',
    border: '1px solid #6366f1',
    color: '#818cf8',
    fontSize: 10,
    fontWeight: 700,
    padding: '3px 8px',
    borderRadius: 20,
    marginTop: 6,
  },
  newBadge: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: 4,
    background: '#052e16',
    border: '1px solid #166534',
    color: '#4ade80',
    fontSize: 10,
    fontWeight: 700,
    padding: '3px 8px',
    borderRadius: 20,
    marginTop: 6,
  },
  prevVisit: {
    marginTop: 8,
    padding: '6px 8px',
    background: '#131929',
    borderRadius: 6,
    fontSize: 11,
    color: '#94a3b8',
  },
  modal: {
    position: 'fixed',
    inset: 0,
    background: '#000000cc',
    zIndex: 1000,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 20,
  },
  modalBox: {
    background: '#0d1117',
    border: '1px solid #334155',
    borderRadius: 20,
    padding: 32,
    maxWidth: 600,
    width: '100%',
    maxHeight: '80vh',
    overflowY: 'auto',
  },
  modalClose: {
    float: 'right',
    background: 'none',
    border: 'none',
    color: '#64748b',
    fontSize: 24,
    cursor: 'pointer',
  },
  timeline: {
    marginTop: 20,
    position: 'relative',
  },
  timelineItem: {
    display: 'flex',
    gap: 16,
    marginBottom: 20,
    alignItems: 'flex-start',
  },
  timelineDot: {
    width: 12,
    height: 12,
    borderRadius: '50%',
    background: '#6366f1',
    marginTop: 4,
    flexShrink: 0,
    boxShadow: '0 0 8px #6366f180',
  },
  timelineContent: {
    flex: 1,
    background: '#131929',
    border: '1px solid #1e293b',
    borderRadius: 10,
    padding: '10px 14px',
  },
  emptyState: {
    textAlign: 'center',
    padding: '60px 20px',
    color: '#475569',
  },
  resetBtn: {
    background: '#1e0000',
    border: '1px solid #7f1d1d',
    color: '#f87171',
    borderRadius: 8,
    padding: '6px 16px',
    fontSize: 12,
    cursor: 'pointer',
  },
  processingMsg: {
    background: '#1c1917',
    border: '1px solid #78350f',
    borderRadius: 8,
    padding: '12px 16px',
    color: '#fb923c',
    fontSize: 13,
    marginBottom: 16,
  },
};

function FaceImage({ path, style = {} }) {
  const [errored, setErrored] = useState(false);
  if (!path || errored) {
    return (
      <div style={{ ...styles.personImg, display: 'flex', alignItems: 'center', 
                    justifyContent: 'center', fontSize: 32, color: '#334155', ...style }}>
        👤
      </div>
    );
  }
  return (
    <img
      src={`${API}${path}`}
      alt="face"
      style={{ ...styles.personImg, ...style }}
      onError={() => setErrored(true)}
    />
  );
}

function PersonCard({ person, onClick }) {
  const returning = person.is_returning;
  return (
    <div
      style={{ ...styles.personCard, ...(returning ? styles.personCardReturning : {}) }}
      onClick={() => onClick(person)}
    >
      <FaceImage path={person.face_image_path} />
      <div style={styles.personInfo}>
        <div style={styles.faceId}>#{person.face_id}</div>
        <div style={styles.frameBadge}>
          Frame {person.frame_number} &nbsp;·&nbsp; {person.timestamp_seconds}s
        </div>
        {returning ? (
          <div style={styles.returningBadge}>
            ↩ Returning Visitor
          </div>
        ) : (
          <div style={styles.newBadge}>
            ✦ New
          </div>
        )}
        {returning && person.previous_visits[0] && (
          <div style={styles.prevVisit}>
            Last seen: <strong style={{ color: '#818cf8' }}>
              {person.previous_visits[0].video_name}
            </strong>
            <br />Frame {person.previous_visits[0].frame_number}
          </div>
        )}
      </div>
    </div>
  );
}

function PersonModal({ person, onClose }) {
  if (!person) return null;
  return (
    <div style={styles.modal} onClick={onClose}>
      <div style={styles.modalBox} onClick={e => e.stopPropagation()}>
        <button style={styles.modalClose} onClick={onClose}>✕</button>
        <div style={{ display: 'flex', gap: 20, marginBottom: 20 }}>
          <FaceImage path={person.face_image_path} style={{ width: 100, height: 100, borderRadius: 12 }} />
          <div>
            <div style={{ fontSize: 22, fontWeight: 800, color: '#818cf8', fontFamily: 'monospace' }}>
              #{person.face_id}
            </div>
            <div style={{ color: '#64748b', marginTop: 4 }}>
              {person.is_returning ? '↩ Returning Visitor' : '✦ First Visit'}
            </div>
            <div style={{ color: '#94a3b8', marginTop: 8, fontSize: 13 }}>
              {person.total_visits} total appearance{person.total_visits !== 1 ? 's' : ''}
            </div>
          </div>
        </div>

        <div style={{ fontSize: 13, fontWeight: 700, color: '#475569', 
                      textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>
          Visit Timeline
        </div>

        <div style={styles.timeline}>
          {person.previous_visits.map((v, i) => (
            <div key={i} style={styles.timelineItem}>
              <div style={styles.timelineDot} />
              <div style={styles.timelineContent}>
                <div style={{ fontWeight: 600, color: '#e2e8f0', fontSize: 13 }}>
                  📹 {v.video_name}
                </div>
                <div style={{ color: '#64748b', fontSize: 11, marginTop: 4 }}>
                  Frame {v.frame_number} · {parseFloat(v.timestamp_seconds).toFixed(1)}s
                </div>
              </div>
            </div>
          ))}
          <div style={styles.timelineItem}>
            <div style={{ ...styles.timelineDot, background: '#4ade80', boxShadow: '0 0 8px #4ade8080' }} />
            <div style={{ ...styles.timelineContent, borderColor: '#166534' }}>
              <div style={{ fontWeight: 600, color: '#4ade80', fontSize: 13 }}>
                📹 Current video
              </div>
              <div style={{ color: '#64748b', fontSize: 11, marginTop: 4 }}>
                Frame {person.frame_number} · {person.timestamp_seconds}s
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [persons, setPersons] = useState([]);
  const [stats, setStats] = useState({});
  const [uploading, setUploading] = useState(false);
  const [selectedPerson, setSelectedPerson] = useState(null);
  const [polling, setPolling] = useState(false);

  const fetchStats = useCallback(() => {
    fetch(`${API}/api/stats`).then(r => r.json()).then(setStats).catch(() => {});
  }, []);

  const fetchVideos = useCallback(() => {
    // renamed to sources on backend
    fetch(`${API}/api/sources`).then(r => r.json()).then(vids => {
      setVideos(vids);
      const hasProcessing = vids.some(v => v.status === 'processing');
      setPolling(hasProcessing);
    }).catch(() => {});
  }, []);

  const fetchPersons = useCallback((videoName) => {
    // videoName here actually is generic source name
    fetch(`${API}/api/sources/${encodeURIComponent(videoName)}/persons`)
      .then(r => r.json())
      .then(setPersons)
      .catch(() => {});
  }, []);

  useEffect(() => {
    fetchStats();
    fetchVideos();
  }, [fetchStats, fetchVideos]);

  // Poll while videos are processing
  useEffect(() => {
    if (!polling) return;
    const interval = setInterval(() => {
      fetchVideos();
      fetchStats();
      if (selectedVideo) fetchPersons(selectedVideo);
    }, 3000);
    return () => clearInterval(interval);
  }, [polling, selectedVideo, fetchVideos, fetchStats, fetchPersons]);

  const handleVideoSelect = (video) => {
    setSelectedVideo(video.video_name);
    setPersons([]);
    fetchPersons(video.video_name);
  };

  const handleUpload = async (e) => {
    const files = e.target.files;
    if (!files.length) return;
    setUploading(true);
    for (const file of files) {
      const form = new FormData();
      form.append('file', file);
      // decide endpoint based on type or extension
      await fetch(`${API}/api/upload`, { method: 'POST', body: form }).catch(() => {});
    }
    setUploading(false);
    fetchVideos();
    fetchStats();
    setPolling(true);
  };

  const handleReset = async () => {
    if (!window.confirm('Reset all data?')) return;
    await fetch(`${API}/api/reset`, { method: 'DELETE' });
    setVideos([]); setPersons([]); setSelectedVideo(null); setStats({});
  };

  const returningCount = persons.filter(p => p.is_returning).length;

  return (
    <div style={styles.root}>
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerTitle}>
          <span>👁</span> Face Tracker — Visitor Log
        </div>
        <button style={styles.resetBtn} onClick={handleReset}>⚠ Reset All Data</button>
      </div>

      {/* Stats */}
      <div style={styles.statBar}>
        {[
          { num: stats.total_persons || 0, label: 'Unique Persons' },
          { num: stats.total_visits || 0, label: 'Total Appearances' },
          { num: stats.returning_visitors || 0, label: 'Returning Visitors' },
          { num: stats.total_videos || 0, label: 'Videos Processed' },
        ].map((s) => (
          <div key={s.label} style={styles.statCard}>
            <div style={styles.statNum}>{s.num}</div>
            <div style={styles.statLabel}>{s.label}</div>
          </div>
        ))}
      </div>

      {/* Layout */}
      <div style={styles.layout}>
        {/* Sidebar - video list */}
        <div style={styles.sidebar}>
          <div style={styles.sidebarTitle}>Sources</div>

          {/* Upload */}
          <label>
            <div style={styles.uploadZone}>
              <div style={{ fontSize: 32 }}>🎬</div>
              <div style={{ fontSize: 13, color: '#64748b', marginTop: 8 }}>
                {uploading ? 'Uploading...' : 'Click to upload video(s) or photo zip/image(s)'}
              </div>
              <div style={{ fontSize: 11, color: '#475569', marginTop: 4 }}>
                MP4, AVI, MOV, MKV, JPG, PNG, ZIP
              </div>
            </div>
            <input type="file" accept="video/*,image/*,.zip" multiple style={{ display: 'none' }}
              onChange={handleUpload} disabled={uploading} />
          </label>

          {videos.length === 0 && (
            <div style={{ color: '#475569', fontSize: 12, textAlign: 'center', padding: 20 }}>
              No videos yet. Upload some!
            </div>
          )}

          {videos.map(v => (
            <div
              key={v.id}
              style={{
                ...styles.videoItem,
                ...(selectedVideo === v.video_name ? styles.videoItemActive : {}),
              }}
              onClick={() => handleVideoSelect(v)}
            >
              <div style={styles.videoName}>{v.video_name}</div>
              <div style={styles.videoMeta}>{v.total_faces_detected} faces detected</div>
              <div style={styles.statusBadge(v.status)}>{v.status}</div>
            </div>
          ))}
        </div>

        {/* Main content */}
        <div style={styles.mainContent}>
          {!selectedVideo ? (
            <div style={styles.emptyState}>
              <div style={{ fontSize: 56 }}>🎥</div>
              <div style={{ fontSize: 18, color: '#334155', marginTop: 16 }}>
                Select a video to view visitor log
              </div>
              <div style={{ fontSize: 13, color: '#1e293b', marginTop: 8 }}>
                Upload videos using the panel on the left
              </div>
            </div>
          ) : (
            <>
              {/* Video header */}
              <div style={{ display: 'flex', justifyContent: 'space-between', 
                            alignItems: 'center', marginBottom: 20 }}>
                <div>
                  <div style={styles.sectionTitle}>
                    📹 {selectedVideo}
                  </div>
                  <div style={{ fontSize: 13, color: '#64748b' }}>
                    {persons.length} unique person{persons.length !== 1 ? 's' : ''} detected
                    {returningCount > 0 && (
                      <span style={{ color: '#818cf8', marginLeft: 8 }}>
                        · {returningCount} returning
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {polling && (
                <div style={styles.processingMsg}>
                  ⏳ Video is still processing... refreshing automatically.
                </div>
              )}

              {persons.length === 0 && !polling ? (
                <div style={styles.emptyState}>
                  <div style={{ fontSize: 40 }}>🔍</div>
                  <div style={{ marginTop: 12, color: '#334155' }}>
                    No faces detected yet
                  </div>
                </div>
              ) : (
                <div style={styles.personGrid}>
                  {persons.map(p => (
                    <PersonCard key={p.face_id} person={p} onClick={setSelectedPerson} />
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Modal */}
      {selectedPerson && (
        <PersonModal person={selectedPerson} onClose={() => setSelectedPerson(null)} />
      )}
    </div>
  );
}
