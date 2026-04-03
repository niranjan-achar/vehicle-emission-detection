function Dashboard({ summary }) {
  const safeSummary = summary || {
    total_uploads: 0,
    total_polluting_detections: 0,
    image_uploads: 0,
    video_uploads: 0,
  };

  const maxValue = Math.max(
    1,
    safeSummary.image_uploads,
    safeSummary.video_uploads,
    safeSummary.total_polluting_detections
  );

  const bars = [
    { label: 'Images', value: safeSummary.image_uploads, color: 'bar-cyan' },
    { label: 'Videos', value: safeSummary.video_uploads, color: 'bar-amber' },
    {
      label: 'Polluting detections',
      value: safeSummary.total_polluting_detections,
      color: 'bar-coral',
    },
  ];

  return (
    <section className="glass-card panel fade-in-up delay-3">
      <h2>Monitoring Dashboard</h2>

      <div className="metrics-grid">
        <div className="metric-card">
          <p className="metric-label">Total uploads</p>
          <p className="metric-value">{safeSummary.total_uploads}</p>
        </div>
        <div className="metric-card">
          <p className="metric-label">Polluting detections</p>
          <p className="metric-value">{safeSummary.total_polluting_detections}</p>
        </div>
        <div className="metric-card">
          <p className="metric-label">Last update</p>
          <p className="metric-value small-text">
            {safeSummary.last_updated
              ? new Date(safeSummary.last_updated).toLocaleString()
              : 'No uploads yet'}
          </p>
        </div>
      </div>

      <div className="chart-card">
        <p className="chart-title">Detection Activity Chart</p>
        <div className="chart-list">
          {bars.map((bar) => (
            <div key={bar.label} className="chart-row">
              <div className="chart-head">
                <span>{bar.label}</span>
                <span className="strong">{bar.value}</span>
              </div>
              <div className="chart-track">
                <div
                  className={`chart-fill ${bar.color}`}
                  style={{ width: `${(bar.value / maxValue) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default Dashboard;
