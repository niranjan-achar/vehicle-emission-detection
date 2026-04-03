import { useMemo, useState } from 'react';

const MAX_MB = 100;
const IMAGE_EXT = ['jpg', 'jpeg', 'png', 'bmp', 'webp'];
const VIDEO_EXT = ['mp4', 'avi', 'mov', 'mkv', 'webm'];

function getExtension(name) {
  const parts = name.toLowerCase().split('.');
  return parts.length > 1 ? parts.at(-1) : '';
}

function Upload({ loading, onUpload, onError }) {
  const [file, setFile] = useState(null);
  const [confidence, setConfidence] = useState(0.2);

  const fileMeta = useMemo(() => {
    if (!file) return null;

    const ext = getExtension(file.name);
    const isImage = IMAGE_EXT.includes(ext);
    const isVideo = VIDEO_EXT.includes(ext);

    return {
      ext,
      isImage,
      isVideo,
      sizeMb: file.size / (1024 * 1024),
    };
  }, [file]);

  function validateSelection(nextFile) {
    const ext = getExtension(nextFile.name);
    const isImage = IMAGE_EXT.includes(ext);
    const isVideo = VIDEO_EXT.includes(ext);

    if (!isImage && !isVideo) {
      onError('Unsupported file type. Upload an image or video.');
      return false;
    }

    if (nextFile.size > MAX_MB * 1024 * 1024) {
      onError(`File is too large. Maximum allowed size is ${MAX_MB}MB.`);
      return false;
    }

    return true;
  }

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0];
    if (!nextFile) return;

    if (!validateSelection(nextFile)) {
      setFile(null);
      event.target.value = '';
      return;
    }

    setFile(nextFile);
  }

  async function handleSubmit(event) {
    event.preventDefault();
    if (!file) {
      onError('Please select an image or video file first.');
      return;
    }

    await onUpload({
      file,
      confidence,
      mediaType: fileMeta?.isImage ? 'image' : 'video',
    });
  }

  return (
    <section className="glass-card panel fade-in-up delay-1">
      <div className="panel-head">
        <h2>Upload Vehicle Media</h2>
        <span className="chip">YOLOv8 Inference</span>
      </div>

      <form className="stack" onSubmit={handleSubmit}>
        <div className="dropzone">
          <input
            type="file"
            accept="image/*,video/*"
            onChange={handleFileChange}
            className="file-input"
          />
          <p className="support-text">
            Supported image: jpg, jpeg, png, bmp, webp | Supported video: mp4, avi, mov, mkv, webm
          </p>
        </div>

        <div className="input-card">
          <label className="field-label">
            Confidence threshold: {confidence.toFixed(2)}
          </label>
          <input
            type="range"
            min="0.1"
            max="0.9"
            step="0.01"
            value={confidence}
            onChange={(event) => setConfidence(Number(event.target.value))}
            className="range-input"
          />
        </div>

        {fileMeta && (
          <div className="meta-card">
            <p>
              Selected: <span className="strong">{file.name}</span>
            </p>
            <p>Type: {fileMeta.isImage ? 'Image' : 'Video'}</p>
            <p>Size: {fileMeta.sizeMb.toFixed(2)} MB</p>
          </div>
        )}

        <button
          type="submit"
          disabled={loading}
          className="primary-btn"
        >
          {loading ? 'Processing...' : 'Run Emission Detection'}
        </button>
      </form>
    </section>
  );
}

export default Upload;
