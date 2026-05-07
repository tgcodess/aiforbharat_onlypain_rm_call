import React, {
  useState, useEffect, useRef, useCallback,
} from 'react';
import {
  Phone, PhoneOff, Clock, TrendingUp,
  X, RefreshCw, CheckCircle, AlertCircle, User,
} from 'lucide-react';

// ── Config ─────────────────────────────────────────────────────────────────
const WS_URL   = `ws://${window.location.hostname}:8579`;
const API_BASE = `http://${window.location.hostname}:5000`;

// ── Types ──────────────────────────────────────────────────────────────────
type LeadStatus = 'pending' | 'active' | 'hot' | 'warm' | 'cold';
type Lang       = 'hi' | 'en' | 'ta' | 'te' | 'mr' | 'gu' | 'bn';
type Gender     = 'male' | 'female';

interface Lead {
  id:         string;
  name:       string;
  phone:      string;
  language:   Lang;
  gender:     Gender;
  status:     LeadStatus;
  city:       string;
  profession: string;
  summary?:   CallSummary | null;
}

interface CallSummary {
  lead_name:           string;
  city:                string;
  network_size:        string | number;
  current_broker:      string;
  phone:               string;
  profession:          string;
  lead_score:          'Hot' | 'Warm' | 'Cold';
  objections_raised:   string[];
  objections_resolved: string[];
  intent_confirmed:    boolean;
  recommended_action:  string;
  topics_covered:      string[];
  call_duration?:      string;
}

interface TxEntry { role: 'user' | 'bot'; text: string; }

// ── Static maps ─────────────────────────────────────────────────────────────
const LANG_LABEL: Record<Lang, string> = {
  hi: 'Hindi', en: 'English', ta: 'Tamil', te: 'Telugu',
  mr: 'Marathi', gu: 'Gujarati', bn: 'Bengali',
};

const STATUS_CFG: Record<LeadStatus, { label: string; tw: string }> = {
  pending: { label: 'Pending', tw: 'bg-gray-100 text-gray-600' },
  active:  { label: '📞 Active',  tw: 'bg-blue-100 text-blue-700' },
  hot:     { label: '🔥 HOT',    tw: 'bg-red-100  text-red-700' },
  warm:    { label: '🌡 WARM',   tw: 'bg-amber-100 text-amber-700' },
  cold:    { label: '❄ COLD',   tw: 'bg-sky-100  text-sky-700' },
};

const OBJ_LABEL: Record<string, string> = {
  obj_already_broker: 'Already with another broker',
  obj_no_contacts:    'Not enough contacts',
  obj_support:        'Client support concerns',
  obj_trust:          'Trust / legitimacy concern',
  obj_think_later:    'Wants to think it over',
};

const DEMO_LEADS: Lead[] = [
  { id: 'lead_001', name: 'Rajesh Kumar',   phone: '9876543210', language: 'hi', gender: 'male',   status: 'pending', city: 'Delhi',     profession: 'Financial Advisor' },
  { id: 'lead_002', name: 'Priya Sharma',   phone: '9123456780', language: 'hi', gender: 'female', status: 'pending', city: 'Mumbai',    profession: 'Mutual Fund Distributor' },
  { id: 'lead_003', name: 'Ankit Mehta',    phone: '9988776655', language: 'en', gender: 'male',   status: 'pending', city: 'Bangalore', profession: 'Insurance Agent' },
  { id: 'lead_004', name: 'Sunita Reddy',   phone: '8877665544', language: 'te', gender: 'female', status: 'pending', city: 'Hyderabad', profession: 'Finance Influencer' },
  { id: 'lead_005', name: 'Vikram Patel',   phone: '7766554433', language: 'gu', gender: 'male',   status: 'pending', city: 'Ahmedabad', profession: 'Sub-Broker' },
  { id: 'lead_006', name: 'Meena Joshi',    phone: '9955112233', language: 'mr', gender: 'female', status: 'pending', city: 'Pune',      profession: 'CA / Tax Consultant' },
  { id: 'lead_007', name: 'Arun Nair',      phone: '9944556677', language: 'en', gender: 'male',   status: 'pending', city: 'Kochi',     profession: 'Stock Sub-Broker' },
  { id: 'lead_008', name: 'Karthik Iyer',   phone: '9871234560', language: 'ta', gender: 'male',   status: 'pending', city: 'Chennai',   profession: 'Financial Advisor' },
  { id: 'lead_009', name: 'Sneha Banerjee', phone: '8899776655', language: 'bn', gender: 'female', status: 'pending', city: 'Kolkata',   profession: 'Mutual Fund Distributor' },
  { id: 'lead_010', name: 'Rahul Verma',    phone: '9988123456', language: 'hi', gender: 'male',   status: 'pending', city: 'Lucknow',   profession: 'Financial Advisor' },
];

// ── Helpers ─────────────────────────────────────────────────────────────────
function initials(name: string) {
  return name.split(' ').map(n => n[0]).join('').slice(0, 2).toUpperCase();
}

function float32ToInt16(f32: Float32Array): Int16Array {
  const i16 = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) {
    i16[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32768)));
  }
  return i16;
}

const DONE_BYTES = new Uint8Array([68, 79, 78, 69]); // ASCII "DONE"

// ── Sub-components ───────────────────────────────────────────────────────────
function StatusBadge({ status }: { status: LeadStatus }) {
  const { label, tw } = STATUS_CFG[status];
  return (
    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${tw}`}>
      {label}
    </span>
  );
}

function WaveAnim({ color = 'teal' }: { color?: string }) {
  return (
    <span className="flex items-end gap-0.5 h-4">
      {[0, 1, 2, 3].map(i => (
        <span
          key={i}
          className={`w-0.5 bg-${color}-500 rounded-full animate-bounce`}
          style={{ height: `${8 + (i % 2) * 4}px`, animationDelay: `${i * 0.1}s` }}
        />
      ))}
    </span>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  // Dashboard state
  const [leads, setLeads]       = useState<Lead[]>([]);
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState<string | null>(null);

  // Call state (UI)
  const [activeLead, setActiveLead]     = useState<Lead | null>(null);
  const [isConnected, setIsConnected]   = useState(false);
  const [isRecording, setIsRecording]   = useState(false);
  const [isBotSpeaking, setIsBotSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [transcript, setTranscript]     = useState<TxEntry[]>([]);
  const [callSummary, setCallSummary]   = useState<CallSummary | null>(null);
  const [callEnded, setCallEnded]       = useState(false);
  const [duration, setDuration]         = useState('0:00');
  const [wsError, setWsError]           = useState<string | null>(null);

  // Stable refs (not triggering re-renders)
  const wsRef          = useRef<WebSocket | null>(null);
  const playCtxRef     = useRef<AudioContext | null>(null);
  const recCtxRef      = useRef<AudioContext | null>(null);
  const processorRef   = useRef<ScriptProcessorNode | null>(null);
  const sourceRef      = useRef<MediaStreamAudioSourceNode | null>(null);
  const streamRef      = useRef<MediaStream | null>(null);
  const nextPlayRef    = useRef(0);
  const timerRef       = useRef<ReturnType<typeof setInterval> | null>(null);
  const callStartRef   = useRef(0);
  const durationRef    = useRef('0:00');
  const activeLeadRef  = useRef<Lead | null>(null);
  const callEndedRef   = useRef(false);
  const speakingRef    = useRef(false);
  const txEndRef       = useRef<HTMLDivElement>(null);

  // Keep refs in sync with state
  useEffect(() => { activeLeadRef.current  = activeLead;  }, [activeLead]);
  useEffect(() => { callEndedRef.current   = callEnded;   }, [callEnded]);
  useEffect(() => { speakingRef.current    = isBotSpeaking; }, [isBotSpeaking]);

  // Auto-scroll transcript
  useEffect(() => { txEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [transcript]);

  // ── Fetch leads ─────────────────────────────────────────────────────────
  const fetchLeads = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/leads`, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data: Lead[] = await res.json();
      setLeads(data);
    } catch {
      setLeads(DEMO_LEADS);
      setError('Backend offline — showing demo data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchLeads(); }, [fetchLeads]);

  // ── Audio playback ───────────────────────────────────────────────────────
  const playAudio = useCallback(async (buf: ArrayBuffer) => {
    try {
      if (!playCtxRef.current || playCtxRef.current.state === 'closed') {
        playCtxRef.current = new AudioContext();
      }
      const ctx = playCtxRef.current;
      if (ctx.state === 'suspended') await ctx.resume();
      const decoded = await ctx.decodeAudioData(buf.slice(0));
      const src = ctx.createBufferSource();
      src.buffer = decoded;
      src.connect(ctx.destination);

      const when = Math.max(nextPlayRef.current, ctx.currentTime + 0.05);
      src.start(when);
      nextPlayRef.current = when + decoded.duration;

      setIsBotSpeaking(true);
      setIsProcessing(false);

      src.onended = () => {
        // Only clear speaking flag when the whole queue has drained
        if (nextPlayRef.current <= (playCtxRef.current?.currentTime ?? 0) + 0.15) {
          setIsBotSpeaking(false);
        }
      };
    } catch (e) {
      console.warn('[Audio] decode error', e);
      setIsBotSpeaking(false);
    }
  }, []);

  // ── Call end helper ──────────────────────────────────────────────────────
  const finaliseCall = useCallback((lead: Lead, status: LeadStatus, summary: CallSummary | null) => {
    if (callEndedRef.current) return;
    callEndedRef.current = true;
    setCallEnded(true);
    if (timerRef.current) clearInterval(timerRef.current);

    // Update dashboard
    setLeads(prev =>
      prev.map(l => l.id === lead.id ? { ...l, status, summary: summary ?? l.summary } : l),
    );

    // POST summary to API
    if (summary) {
      fetch(`${API_BASE}/api/leads/${lead.id}/summary`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ ...summary, lead_score: summary.lead_score }),
      }).catch(() => {/* offline – ignore */});
    }
  }, []);

  // ── Stop recording ────────────────────────────────────────────────────────
  const stopRecording = useCallback(() => {
    processorRef.current?.disconnect();
    sourceRef.current?.disconnect();
    streamRef.current?.getTracks().forEach(t => t.stop());
    processorRef.current = null;
    sourceRef.current    = null;
    streamRef.current    = null;
    recCtxRef.current?.close();
    recCtxRef.current = null;
    setIsRecording(false);
  }, []);

  // ── Continuous mic stream with sliding-window VAD + pre/post buffer ─────
  // The mic is opened once when the call connects and stays open until End
  // Call. Every frame is fed into a rolling pre-buffer; when VAD flips on,
  // the pre-buffer is flushed first (so Whisper hears the soft onset of the
  // utterance), then voiced frames stream live. After SILENCE_FRAMES_THRESHOLD
  // of trailing silence, a post-buffer of POST_MS is collected and sent,
  // followed by an ASCII "DONE" sentinel — same wire protocol the BE expects.
  // Frames are dropped while the bot is speaking so its TTS doesn't echo back.
  const startMicStream = useCallback(async () => {
    // Audio framing
    const PRE_MS                  = 1784;      // 892 * 2 — pre-voice context for Whisper
    const POST_MS                 = 892;       // trailing silence captured after VAD off
    const FRAME_MS                = 30;
    const FRAME_SAMPLES           = 16000 * FRAME_MS / 1000;        // 480
    const PRE_SAMPLES             = Math.ceil(PRE_MS  / FRAME_MS) * FRAME_SAMPLES;
    const POST_SAMPLES            = Math.ceil(POST_MS / FRAME_MS) * FRAME_SAMPLES;
    const VAD_WINDOW_SAMPLES      = 6144;      // ~384 ms rolling VAD window
    const SILENCE_FRAMES_THRESHOLD = 8;        // ≈8 * 256 ms script-processor frames of silence ends utterance

    // VAD energy thresholds
    const AUDIO_SPEAKER_THRESHOLD = 0.1;       // for interrupt detection while bot is talking
    const MAX_NOISE_SAMPLES       = 25;
    const NOISE_HISTORY_LENGTH    = 10;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const ctx  = new AudioContext({ sampleRate: 16000 });
      recCtxRef.current = ctx;
      const src  = ctx.createMediaStreamSource(stream);
      const proc = ctx.createScriptProcessor(4096, 1, 1);
      sourceRef.current    = src;
      processorRef.current = proc;

      // Per-call VAD + buffer state (closure)
      const noiseSamples: number[] = [];
      const noiseHistory: number[] = [];
      let   avgNoise            = 0;
      let   userSpeakerThreshold = 0.03;
      let   preBuffer:  number[] = [];
      let   postBuffer: number[] = [];
      let   postSamplesCnt   = 0;
      let   isVoiceActive    = false;
      let   silenceFrameCount = 0;
      let   vadWindow:  number[] = [];

      const computeZCR = (a: number[]): number => {
        let zc = 0;
        for (let i = 1; i < a.length; i++) {
          if ((a[i-1] >= 0 && a[i] < 0) || (a[i-1] < 0 && a[i] >= 0)) zc++;
        }
        return zc / a.length;
      };

      const detectVoiceActivity = (a: number[]): boolean => {
        if (!a || a.length === 0) return false;
        let sumSq = 0;
        for (let i = 0; i < a.length; i++) sumSq += a[i] * a[i];
        const avg = Math.sqrt(sumSq / a.length);

        // Adaptive noise floor — only sample when bot isn't playing
        if (!speakingRef.current && avg < 0.055) {
          noiseSamples.push(avg);
          if (noiseSamples.length > MAX_NOISE_SAMPLES) noiseSamples.shift();
          const mean = noiseSamples.reduce((s, x) => s + x, 0) / noiseSamples.length;
          if (mean > 0) avgNoise = mean;
          userSpeakerThreshold = Math.max(avgNoise + 0.02, 0.03);
        }

        noiseHistory.push(avgNoise);
        if (noiseHistory.length > NOISE_HISTORY_LENGTH) noiseHistory.shift();

        const threshold       = speakingRef.current ? AUDIO_SPEAKER_THRESHOLD : userSpeakerThreshold;
        const zcr             = computeZCR(a);
        const noiseHistorySum = noiseHistory.reduce((s, x) => s + x, 0);

        if (speakingRef.current) {
          // While bot speaks: stricter ZCR floor to suppress false-interrupts from speaker echo
          return avg > threshold && zcr > 0.03;
        }
        // Normal case: avoid high-freq noise (zcr cap) and require some noise history
        return avg > threshold && zcr < 0.14 && noiseHistorySum !== 0;
      };

      proc.onaudioprocess = (e) => {
        const ws = wsRef.current;
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const frameF32 = e.inputBuffer.getChannelData(0);
        const frame    = Array.from(frameF32);

        // Always maintain rolling pre-buffer — captures audio BEFORE VAD fires
        preBuffer.push(...frame);
        if (preBuffer.length > PRE_SAMPLES) preBuffer.splice(0, preBuffer.length - PRE_SAMPLES);

        // Don't decide / send while bot is talking through the speakers
        if (speakingRef.current) return;

        // Keep VAD window fresh
        vadWindow.push(...frame);
        if (vadWindow.length > VAD_WINDOW_SAMPLES) vadWindow.splice(0, vadWindow.length - VAD_WINDOW_SAMPLES);

        const voiced = detectVoiceActivity(vadWindow);

        if ((voiced && silenceFrameCount < SILENCE_FRAMES_THRESHOLD) ||
            (silenceFrameCount < SILENCE_FRAMES_THRESHOLD && isVoiceActive)) {
          // Active utterance — voice or short silence dip mid-utterance
          if (!voiced && isVoiceActive) silenceFrameCount++;
          if (voiced) silenceFrameCount = 0;

          if (!isVoiceActive) {
            // First detection: flush pre-buffer
            isVoiceActive = true;
            const pcm = float32ToInt16(new Float32Array(preBuffer));
            ws.send(pcm.buffer.slice(0) as ArrayBuffer);
          } else {
            // Streaming a voiced (or briefly silent) frame
            const pcm = float32ToInt16(frameF32);
            ws.send(pcm.buffer.slice(0) as ArrayBuffer);
          }
          preBuffer.length  = 0;
          postBuffer.length = 0;
          postSamplesCnt    = 0;
        } else if (isVoiceActive && silenceFrameCount === SILENCE_FRAMES_THRESHOLD) {
          // Silence threshold just hit — collect the post-buffer tail
          postBuffer.push(...frame);
          postSamplesCnt += frame.length;

          if (postSamplesCnt >= POST_SAMPLES) {
            // Flush post-buffer + DONE in that order
            const pcm = float32ToInt16(new Float32Array(postBuffer));
            ws.send(pcm.buffer.slice(0) as ArrayBuffer);
            ws.send(DONE_BYTES.buffer.slice(0) as ArrayBuffer);

            isVoiceActive    = false;
            postBuffer.length = 0;
            postSamplesCnt   = 0;
            silenceFrameCount = 0;
            setIsProcessing(true);
          }
        }
      };

      src.connect(proc);
      proc.connect(ctx.destination);
      setIsRecording(true);
    } catch {
      alert('Microphone access denied — please allow mic access and try again.');
    }
  }, []);

  // ── Start call ────────────────────────────────────────────────────────────
  const startCall = useCallback((lead: Lead) => {
    // Reset all call state
    setActiveLead(lead);
    activeLeadRef.current = lead;
    setTranscript([]);
    setCallSummary(null);
    setCallEnded(false);
    callEndedRef.current = false;
    setIsRecording(false);
    setIsBotSpeaking(false);
    setIsProcessing(false);
    setWsError(null);
    nextPlayRef.current = 0;

    // Create AudioContext inside the user gesture so it starts un-suspended
    if (!playCtxRef.current || playCtxRef.current.state === 'closed') {
      playCtxRef.current = new AudioContext();
    }
    playCtxRef.current.resume().catch(() => {/* ignore */});

    // Optimistically mark as active in the list
    setLeads(prev => prev.map(l => l.id === lead.id ? { ...l, status: 'active' } : l));

    // Duration timer
    callStartRef.current = Date.now();
    durationRef.current  = '0:00';
    setDuration('0:00');
    timerRef.current = setInterval(() => {
      const s = Math.floor((Date.now() - callStartRef.current) / 1000);
      const fmt = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
      durationRef.current = fmt;
      setDuration(fmt);
    }, 1000);

    // WebSocket
    const sessionId = `${lead.id}_${Date.now()}`;
    const ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
      ws.send(`SESSION_ID:${sessionId}:${lead.id}`);
      // Open the mic right away. While the bot's intro plays, VAD inside
      // startMicStream gates on speakingRef so no frames are sent — the
      // moment the bot stops speaking, VAD takes over and listens.
      startMicStream();
    };

    ws.onmessage = async (ev) => {
      if (ev.data instanceof ArrayBuffer) {
        await playAudio(ev.data);
        return;
      }
      const msg = ev.data as string;

      if (msg.startsWith('TRANSCRIPT:USER:')) {
        setTranscript(p => [...p, { role: 'user', text: msg.slice(16) }]);
        return;
      }
      if (msg.startsWith('TRANSCRIPT:BOT:')) {
        setTranscript(p => [...p, { role: 'bot', text: msg.slice(15) }]);
        return;
      }

      const parseSummary = (raw: string): CallSummary => {
        const s = JSON.parse(raw);
        s.call_duration = durationRef.current;
        return s;
      };

      if (msg.startsWith('HANDOFF_TO_RM:')) {
        const s = parseSummary(msg.slice('HANDOFF_TO_RM:'.length));
        setCallSummary(s);
        finaliseCall(lead, 'hot', s);
      } else if (msg.startsWith('SCHEDULE_FOLLOWUP:')) {
        const s = parseSummary(msg.slice('SCHEDULE_FOLLOWUP:'.length));
        setCallSummary(s);
        finaliseCall(lead, 'warm', s);
      } else if (msg.startsWith('NURTURE_LEAD:')) {
        const s = parseSummary(msg.slice('NURTURE_LEAD:'.length));
        setCallSummary(s);
        finaliseCall(lead, 'cold', s);
      } else if (msg === 'FLOW_CANCELLED') {
        finaliseCall(lead, 'cold', null);
      }
      // FILLER / PLAY_CONGRATS etc. — ignore for now
    };

    ws.onclose  = () => setIsConnected(false);
    ws.onerror  = () => setWsError('WebSocket error — is the bot server running on port 8579?');
  }, [finaliseCall, playAudio, startMicStream]);

  // ── Hang up ───────────────────────────────────────────────────────────────
  const hangUp = useCallback(() => {
    stopRecording();
    wsRef.current?.close();
    wsRef.current = null;
    if (!callEndedRef.current && activeLeadRef.current) {
      finaliseCall(activeLeadRef.current, 'cold', null);
    }
  }, [finaliseCall, stopRecording]);

  const closeModal = useCallback(() => {
    hangUp();
    setTimeout(() => {
      setActiveLead(null);
      setCallSummary(null);
      setTranscript([]);
      setDuration('0:00');
    }, 100);
  }, [hangUp]);

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-screen bg-slate-50 font-sans">

      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="bg-white border-b border-slate-200 px-6 py-3 flex items-center justify-between sticky top-0 z-10 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-teal-600 flex items-center justify-center shadow">
            <span className="text-white font-extrabold text-base tracking-tight">R</span>
          </div>
          <div>
            <p className="font-bold text-slate-900 leading-tight">Rupeezy RM Dashboard</p>
            <p className="text-xs text-slate-400">AP Partner Telecalling · AI Voice Agent</p>
          </div>
        </div>
        <button
          onClick={fetchLeads}
          className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-lg transition"
        >
          <RefreshCw size={12} />Refresh
        </button>
      </header>

      {/* ── Dashboard ──────────────────────────────────────────────────── */}
      <main className="max-w-6xl mx-auto px-6 py-6">

        {/* Stats row */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          {(['pending','hot','warm','cold'] as LeadStatus[]).map(s => (
            <div key={s} className="bg-white rounded-xl border border-slate-200 px-4 py-3 shadow-sm">
              <p className="text-xs text-slate-400 mb-1">{STATUS_CFG[s].label}</p>
              <p className="text-2xl font-bold text-slate-800">
                {leads.filter(l => l.status === s).length}
              </p>
            </div>
          ))}
        </div>

        {error && (
          <div className="mb-4 flex items-center gap-2 text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-4 py-2 text-sm">
            <AlertCircle size={14} /> {error}
          </div>
        )}

        <div className="flex items-center justify-between mb-3">
          <h2 className="font-semibold text-slate-700 text-sm">Lead Pipeline</h2>
          <span className="text-xs text-slate-400">{leads.length} leads</span>
        </div>

        {loading ? (
          <div className="bg-white rounded-xl border border-slate-200 p-10 text-center text-slate-400 text-sm">
            Loading leads…
          </div>
        ) : (
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-slate-50 text-left text-xs font-semibold text-slate-400 uppercase tracking-wider border-b border-slate-100">
                  <th className="px-5 py-3">Lead</th>
                  <th className="px-5 py-3">Mobile</th>
                  <th className="px-5 py-3">Profession</th>
                  <th className="px-5 py-3">City</th>
                  <th className="px-5 py-3">Language</th>
                  <th className="px-5 py-3">Gender</th>
                  <th className="px-5 py-3">Status</th>
                  <th className="px-5 py-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50">
                {leads.map(lead => (
                  <tr key={lead.id} className="hover:bg-slate-50/70 transition-colors">
                    <td className="px-5 py-3.5">
                      <div className="flex items-center gap-2.5">
                        <div className="w-8 h-8 rounded-full bg-teal-50 border border-teal-100 flex items-center justify-center text-teal-700 font-bold text-xs shrink-0">
                          {initials(lead.name)}
                        </div>
                        <span className="font-medium text-slate-800">{lead.name}</span>
                      </div>
                    </td>
                    <td className="px-5 py-3.5 text-slate-500 font-mono text-xs">{lead.phone}</td>
                    <td className="px-5 py-3.5 text-slate-600">{lead.profession}</td>
                    <td className="px-5 py-3.5 text-slate-500">{lead.city}</td>
                    <td className="px-5 py-3.5">
                      <span className="px-2 py-0.5 bg-slate-100 text-slate-600 rounded text-xs">
                        {LANG_LABEL[lead.language] ?? lead.language}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${lead.gender === 'female' ? 'bg-pink-50 text-pink-700' : 'bg-blue-50 text-blue-700'}`}>
                        {lead.gender === 'female' ? '♀ Female' : '♂ Male'}
                      </span>
                    </td>
                    <td className="px-5 py-3.5">
                      <StatusBadge status={lead.status} />
                    </td>
                    <td className="px-5 py-3.5">
                      <button
                        onClick={() => startCall(lead)}
                        disabled={lead.status === 'active'}
                        className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-teal-600 text-white text-xs font-semibold rounded-lg hover:bg-teal-700 disabled:opacity-40 disabled:cursor-not-allowed transition shadow-sm"
                      >
                        <Phone size={11} /> Contact Lead
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Legend */}
        <div className="mt-4 flex flex-wrap gap-5 text-xs text-slate-400">
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-400 inline-block" />HOT — immediate RM handoff</span>
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-400 inline-block" />WARM — WhatsApp signup link</span>
          <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-sky-400 inline-block" />COLD — nurture pipeline (30 days)</span>
        </div>
      </main>

      {/* ── Call Modal ─────────────────────────────────────────────────── */}
      {activeLead && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
          <div className="bg-white rounded-2xl shadow-2xl w-full max-w-xl flex flex-col max-h-[92vh] border border-slate-200">

            {/* Modal header */}
            <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-teal-50 border border-teal-100 flex items-center justify-center text-teal-700 font-bold text-sm">
                  {initials(activeLead.name)}
                </div>
                <div>
                  <p className="font-bold text-slate-900 leading-tight">{activeLead.name}</p>
                  <p className="text-xs text-slate-400 font-mono">{activeLead.phone} · {LANG_LABEL[activeLead.language]}</p>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {/* Duration */}
                <div className="flex items-center gap-1 text-xs text-slate-400">
                  <Clock size={12} />
                  <span className="font-mono tabular-nums">{duration}</span>
                </div>
                {/* Score badge (once call ends) */}
                {callSummary && <StatusBadge status={callSummary.lead_score.toLowerCase() as LeadStatus} />}
                <button onClick={closeModal} className="p-1.5 text-slate-300 hover:text-slate-600 hover:bg-slate-100 rounded-lg transition">
                  <X size={15} />
                </button>
              </div>
            </div>

            {/* Connection error */}
            {wsError && (
              <div className="mx-5 mt-3 px-3 py-2 bg-red-50 border border-red-200 rounded-lg text-xs text-red-600 flex items-center gap-2">
                <AlertCircle size={12} /> {wsError}
              </div>
            )}

            {/* Transcript area */}
            <div className="flex-1 overflow-y-auto px-5 py-4 space-y-2.5 min-h-[220px]">
              {transcript.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full py-10 text-slate-400">
                  {isBotSpeaking ? (
                    <WaveAnim color="teal" />
                  ) : isConnected ? (
                    <p className="text-xs">Press the mic to start speaking</p>
                  ) : (
                    <p className="text-xs">Connecting to voice agent…</p>
                  )}
                </div>
              )}

              {transcript.map((t, i) => (
                <div key={i} className={`flex ${t.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[78%] rounded-2xl px-3.5 py-2.5 text-sm leading-snug ${
                    t.role === 'user'
                      ? 'bg-teal-600 text-white rounded-tr-sm'
                      : 'bg-slate-100 text-slate-800 rounded-tl-sm'
                  }`}>
                    <p className={`text-[10px] font-semibold mb-0.5 ${t.role === 'user' ? 'text-teal-200' : 'text-slate-400'}`}>
                      {t.role === 'user' ? '👤 Lead' : '🤖 Priya (Rupeezy)'}
                    </p>
                    {t.text}
                  </div>
                </div>
              ))}
              <div ref={txEndRef} />
            </div>

            {/* Post-call summary */}
            {callSummary && (
              <div className="mx-5 mb-3 p-4 bg-slate-50 rounded-xl border border-slate-200 text-xs">
                <p className="font-semibold text-slate-700 mb-2.5 flex items-center gap-1.5">
                  <TrendingUp size={13} /> Post-Call Summary
                </p>
                <div className="grid grid-cols-2 gap-y-1.5 text-slate-500">
                  {[
                    ['Duration',       callSummary.call_duration],
                    ['Network Size',   callSummary.network_size  || '—'],
                    ['Current Broker', callSummary.current_broker|| '—'],
                    ['Intent',         callSummary.intent_confirmed ? '✅ Confirmed' : '—'],
                    ['Topics',         (callSummary.topics_covered ?? []).join(', ') || '—'],
                  ].map(([k, v]) => (
                    <React.Fragment key={String(k)}>
                      <span>{k}</span>
                      <span className="font-medium text-slate-800">{v}</span>
                    </React.Fragment>
                  ))}
                </div>

                {(callSummary.objections_raised ?? []).length > 0 && (
                  <div className="mt-3">
                    <p className="font-semibold text-slate-600 mb-1">Objections</p>
                    <div className="space-y-1">
                      {callSummary.objections_raised.map(obj => {
                        const resolved = callSummary.objections_resolved?.includes(obj);
                        return (
                          <div key={obj} className="flex items-center gap-2">
                            {resolved
                              ? <CheckCircle size={11} className="text-green-500 shrink-0" />
                              : <AlertCircle  size={11} className="text-amber-400 shrink-0" />}
                            <span className={resolved ? 'text-slate-600' : 'text-slate-500'}>
                              {OBJ_LABEL[obj] ?? obj}
                              {resolved && <span className="ml-1 text-green-600 font-medium">✓ Resolved</span>}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                <div className="mt-3 pt-2.5 border-t border-slate-200">
                  <p className="font-semibold text-slate-600 mb-0.5">Next Action</p>
                  <p className="text-slate-700">{callSummary.recommended_action}</p>
                </div>
              </div>
            )}

            {/* Controls */}
            <div className="px-5 py-4 border-t border-slate-100 flex items-center justify-between gap-3">
              {/* Status indicator */}
              <div className="text-xs text-slate-400 flex items-center gap-2 min-w-0">
                {isBotSpeaking && (
                  <span className="flex items-center gap-1.5 text-teal-600 font-medium">
                    <WaveAnim color="teal" /> Priya is speaking…
                  </span>
                )}
                {isProcessing && !isBotSpeaking && (
                  <span className="flex items-center gap-1.5 text-slate-500">
                    <span className="w-2 h-2 rounded-full bg-slate-400 animate-pulse inline-block" />
                    Processing…
                  </span>
                )}
                {isRecording && !isBotSpeaking && !isProcessing && (
                  <span className="flex items-center gap-1.5 text-red-500 font-medium">
                    <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse inline-block" />
                    Recording — click mic to send
                  </span>
                )}
                {isConnected && !isBotSpeaking && !isProcessing && !isRecording && !callEnded && (
                  <span className="text-slate-400">Listening — speak naturally</span>
                )}
                {callEnded && <span className="text-slate-400">Call ended</span>}
              </div>

              {/* End Call is the only control — mic streams hands-free via VAD */}
              <div className="flex items-center gap-3 shrink-0">
                <button
                  onClick={closeModal}
                  className="flex items-center gap-1.5 px-4 py-2 bg-red-500 text-white text-sm font-semibold rounded-xl hover:bg-red-600 transition shadow-sm"
                >
                  <PhoneOff size={14} />
                  {callEnded ? 'Close' : 'End Call'}
                </button>
              </div>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
