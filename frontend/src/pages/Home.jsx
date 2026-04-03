import { useEffect, useState } from 'react';
import axios from 'axios';

import Dashboard from '../components/Dashboard';
import Results from '../components/Results';
import Upload from '../components/Upload';
import { detectImage, detectVideo, fetchHealth, fetchSummary } from '../services/api';

function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [mediaType, setMediaType] = useState(null);
  const [error, setError] = useState('');
  const [summary, setSummary] = useState(null);
  const [health, setHealth] = useState(null);

  useEffect(() => {
    loadDashboard();
    loadHealth();
  }, []);

  async function loadDashboard() {
    try {
      const data = await fetchSummary();
      setSummary(data);
    } catch {
      setSummary(null);
    }
  }

  async function loadHealth() {
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    }
  }

  async function handleUpload({ file, confidence, mediaType: selectedType }) {
    setLoading(true);
    setError('');

    try {
      const data = selectedType === 'image'
        ? await detectImage(file, confidence)
        : await detectVideo(file, confidence);

      setResult(data);
      setMediaType(selectedType);
      await Promise.all([loadDashboard(), loadHealth()]);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail || err.message || 'Detection failed.');
      } else {
        setError('Unexpected error occurred while processing the file.');
      }
      setResult(null);
      setMediaType(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <div className="app-grid" />
      <div className="app-aurora app-aurora-top" />
      <div className="app-aurora app-aurora-bottom" />

      <div className="layout">
        <header className="glass-card hero-card fade-in-up">
          <p className="eyebrow">AI Monitoring Suite</p>
          <h1>Vehicle Emission Detection Dashboard</h1>
          <p className="hero-copy">
            Detect smoke-emitting vehicles from images and videos using YOLOv8, inspect confidence scores,
            and monitor pollution trends from one unified control center.
          </p>

          <div className={`status-pill ${health?.model_loaded ? 'status-ok' : 'status-warn'}`}>
            Backend health: {health ? `${health.status} (${health.model_loaded ? 'model loaded' : 'model unavailable'})` : 'unknown'}
          </div>
        </header>

        <section className="panel-grid">
          <Upload loading={loading} onUpload={handleUpload} onError={setError} />
          <Results loading={loading} result={result} error={error} mediaType={mediaType} />
        </section>

        <Dashboard summary={summary} />
      </div>
    </main>
  );
}

export default Home;
