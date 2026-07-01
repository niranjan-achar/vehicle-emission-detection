import React, { useEffect, useState } from 'react';
import { getNotificationSettings, updateNotificationSettings, testNotification, getNotificationHistory } from '../services/api';

export default function NotificationSettings() {
  const [settings, setSettings] = useState(null);
  const [history, setHistory] = useState([]);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    load();
  }, []);

  async function load() {
    const s = await getNotificationSettings();
    setSettings(s);
    const h = await getNotificationHistory();
    setHistory(h.history || []);
  }

  function setNested(path, value) {
    const next = { ...settings };
    const parts = path.split('.');
    let cur = next;
    for (let i = 0; i < parts.length - 1; i++) {
      cur = cur[parts[i]] = { ...(cur[parts[i]] || {}) };
    }
    cur[parts[parts.length - 1]] = value;
    setSettings(next);
  }

  async function save() {
    setSaving(true);
    try {
      await updateNotificationSettings(settings);
      await load();
    } finally {
      setSaving(false);
    }
  }

  async function sendTest() {
    await testNotification();
    // refresh history
    const h = await getNotificationHistory();
    setHistory(h.history || []);
  }

  if (!settings) return null;

  return (
    <div className="card">
      <h3>Notification Settings</h3>
      <label>
        <input type="checkbox" checked={settings.enabled} onChange={(e)=>setSettings({...settings, enabled: e.target.checked})} /> Enable notifications
      </label>

      <div>
        <label>Min detections to notify</label>
        <input type="number" value={settings.min_detections} min={1} onChange={(e)=>setSettings({...settings, min_detections: Number(e.target.value)})} />
      </div>

      <div>
        <label>Webhook URL</label>
        <input type="text" value={settings.webhook_url || ''} onChange={(e)=>setSettings({...settings, webhook_url: e.target.value})} />
      </div>

      <h4>Email</h4>
      <label>
        <input type="checkbox" checked={settings.email.enabled} onChange={(e)=>setNested('email.enabled', e.target.checked)} /> Enable email
      </label>
      <div>
        <label>SMTP Host</label>
        <input value={settings.email.smtp_host || ''} onChange={(e)=>setNested('email.smtp_host', e.target.value)} />
      </div>
      <div>
        <label>SMTP Port</label>
        <input type="number" value={settings.email.smtp_port || 587} onChange={(e)=>setNested('email.smtp_port', Number(e.target.value))} />
      </div>
      <div>
        <label>SMTP User</label>
        <input value={settings.email.smtp_user || ''} onChange={(e)=>setNested('email.smtp_user', e.target.value)} />
      </div>
      <div>
        <label>SMTP Password</label>
        <input value={settings.email.smtp_password || ''} onChange={(e)=>setNested('email.smtp_password', e.target.value)} />
      </div>
      <div>
        <label>From Email</label>
        <input value={settings.email.from_email || ''} onChange={(e)=>setNested('email.from_email', e.target.value)} />
      </div>
      <div>
        <label>To Emails (comma-separated)</label>
        <input value={settings.email.to_emails || ''} onChange={(e)=>setNested('email.to_emails', e.target.value)} />
      </div>

      <div style={{marginTop:10}}>
        <button onClick={save} disabled={saving}>Save</button>
        <button onClick={sendTest} style={{marginLeft:8}}>Send Test</button>
      </div>

      <h4>Recent Notification History</h4>
      <div style={{maxHeight:200, overflow:'auto'}}>
        {history.length===0 && <div>No history</div>}
        <ul>
          {history.slice().reverse().map((h, idx)=> (
            <li key={idx}>{h.status} - attempts:{h.attempts} - {h.record?.media_type} - {h.record?.file_name}</li>
          ))}
        </ul>
      </div>
    </div>
  );
}
