import axios from 'axios';

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
});

export async function detectImage(file, confidence) {
  const form = new FormData();
  form.append('file', file);

  const response = await client.post('/detect/image', form, {
    params: { confidence },
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return response.data;
}

export async function detectVideo(file, confidence) {
  const form = new FormData();
  form.append('file', file);

  const response = await client.post('/detect/video', form, {
    params: { confidence },
    headers: { 'Content-Type': 'multipart/form-data' },
  });

  return response.data;
}

export async function fetchSummary() {
  const response = await client.get('/detect/summary');
  return response.data;
}

export async function fetchHealth() {
  const response = await client.get('/health');
  return response.data;
}
