import { API_BASE_URL } from '../services/api';

function Results({ loading, result, error, mediaType }) {
  return (
    <section className="glass-card panel fade-in-up delay-2">
      <h2>Detection Results</h2>

      {loading && <p className="notice notice-info">Analyzing media with YOLOv8...</p>}

      {error && !loading && (
        <p className="notice notice-error">
          {error}
        </p>
      )}

      {!loading && !error && !result && (
        <p className="notice notice-neutral">
          Upload an image or video to view annotated outputs and confidence values.
        </p>
      )}

      {!loading && result && (
        <div className="stack">
          {mediaType === 'image' && result.processed_image_base64 && (
            <img
              src={`data:image/jpeg;base64,${result.processed_image_base64}`}
              alt="Processed detection"
              className="media-preview"
            />
          )}

          {mediaType === 'video' && result.processed_video_path && (
            <video
              controls
              className="media-preview"
              src={`${API_BASE_URL}${result.processed_video_path}`}
            />
          )}

          <div className="meta-grid">
            <div className="metric-card">
              <p className="metric-label">File</p>
              <p className="metric-value small-text">{result.file_name}</p>
            </div>
            <div className="metric-card">
              <p className="metric-label">Detections</p>
              <p className="metric-value">{result.detections_count}</p>
            </div>
          </div>

          {Array.isArray(result.detections) && result.detections.length > 0 && (
            <div className="table-wrap">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Class</th>
                    <th>Confidence</th>
                    <th>BBox [x1, y1, x2, y2]</th>
                  </tr>
                </thead>
                <tbody>
                  {result.detections.map((det, index) => (
                    <tr key={`${det.class_id}-${index}`}>
                      <td>{det.class_name}</td>
                      <td>{(det.confidence * 100).toFixed(2)}%</td>
                      <td>{det.bbox.map((n) => n.toFixed(1)).join(', ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {Array.isArray(result.timestamps) && result.timestamps.length > 0 && (
            <div className="timestamp-card">
              <p className="timestamp-title">Detection Timestamps (seconds)</p>
              <p>{result.timestamps.join(', ')}</p>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

export default Results;
