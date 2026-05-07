import React, {useState, useRef, useEffect} from 'react';
import {Mic, MicOff} from 'lucide-react';
import {Howl} from 'howler';
// import {congrats_mp3_bytes, filler_mp3_bytes, filler_map} from './Library';
import {congrats_mp3_bytes, filler_map, generate_quotes_bytes, continue_flow_bytes,intro_mp3_bytes} from './Library_elevanLabs';

const WEBSOCKET_URL = 'ws://localhost:8579/ws/audio/';
const SAMPLE_RATE = 16000; 
const AUDIO_SLIDING_WINDOW_MS = 896;
let donestartTime = Date.now(); //10
let websocketEventTime = 0; //11

const AUDIO_SLIDING_WINDOW_ARRAY_SIZE = 4 * SAMPLE_RATE * AUDIO_SLIDING_WINDOW_MS / 1000;
const AUDIO_SLIDING_WINDOW_ARRAY = [];
const POST_SPEAKER_FRAME_COUNT = AUDIO_SLIDING_WINDOW_ARRAY_SIZE / 4; // 4 bytes per float32 sample

/** A global variable to track whether audio is playing. */
let globalIsPlaying = false;

console.log("initializing prebuffer");
let preBuffer = [];

let TTS_ENERGY_BASELINE = 0; // This is calculated, but not used as of now. Can we removed.

let AUDIO_SPEAKER_THRESHOLD = 0.1;
let USER_SPEAKER_THRESHOLD = 0.03;

/** A helper to update the global variable (for logs). */
function setGlobalIsPlaying(value) {
  globalIsPlaying = value;
  let a = new Error();
  console.debug('[Global] isPlaying = ', globalIsPlaying, " -- ", a);
}

function addToSlidingWindow(float32DataToAdd) {

  //let temp = float32ToPCM();
  /* ------------------------------------------------------------ *
   * 1.  Push new samples & keep 0.5 s sliding window
   * ------------------------------------------------------------ */
  AUDIO_SLIDING_WINDOW_ARRAY.push(...float32DataToAdd);
  while (AUDIO_SLIDING_WINDOW_ARRAY.length > AUDIO_SLIDING_WINDOW_ARRAY_SIZE) {
      AUDIO_SLIDING_WINDOW_ARRAY.shift();
  }
}

function cleanSlidingWindow() {
  AUDIO_SLIDING_WINDOW_ARRAY.length = 0;
}

// Returns a random valid index for any non-empty array
function randomIndex(category) {
  let arr = filler_map[category];

  if (!Array.isArray(arr) || arr.length === 0) return -1; // guard
  return arr[Math.floor(Math.random() * arr.length)];
}


/**
* Convert Float32 samples [-1..1] to 16-bit PCM (little-endian).
*/
function float32ToPCM(float32Array) {
  const int16Array = new Int16Array(float32Array.length);
  for (let i = 0; i < float32Array.length; i++) {
      let sample = Math.max(-1, Math.min(1, float32Array[i]));
      int16Array[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
  }
  // console.debug('[PCM Conversion] output length (bytes):', int16Array.byteLength);
  return new Uint8Array(int16Array.buffer);
}



function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isThinking, setIsThinking] = useState(false);

  // Refs
  const websocketRef = useRef(null);
  const audioContextRef = useRef(null);
  const soundRef = useRef(null);
  const soundPlayRef = useRef(null);
  const analyserRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const processorNodeRef = useRef(null);
  const voiceTimeoutRef = useRef(null);
  const ttsMediaDestRef = useRef(null); // for mediaStreamDestination

  const isCleanup = useRef(null);

  // TTS queue
  const audioQueueRef = useRef([]);
  const sendQueueRef = useRef([]);

  // On unmount => cleanup
  useEffect(() => {
      return () => {
          cleanup();
      };
  }, []);

  /**
   * Cleanup resources: close WebSocket, free audio nodes, etc.
   */
  const cleanup = () => {
      
    isCleanup.current = true;
    
    if (websocketRef.current) {
        websocketRef.current.close();
    }
    if (soundRef.current) {
        soundRef.current.unload();
        soundPlayRef.current = null;
    }
    if (processorNodeRef.current) {
        let a = new Error();
        console.log("processorNodeRef reset : " , a);
        processorNodeRef.current.disconnect();
    }
    if (sourceNodeRef.current) {
        sourceNodeRef.current.disconnect();
    }
    if (voiceTimeoutRef.current) {
        clearTimeout(voiceTimeoutRef.current);
    }
    // Close the AudioContext
    if (audioContextRef.current) {
        audioContextRef.current.close();
      }
      setGlobalIsPlaying(false);
  };

  const SIZE_OF_VOICE_DETECTION_ACTIVITY_DATA =  (64 / 64) * 1024;  // 16000 * 0.064  = 1024 -> SAMPLE_RATE * AUDIO_MS 


  /**
   * Simple amplitude-based voice activity detection.
   */
  const noiseSamples = [];
  const MAX_NOISE_SAMPLES = 25;
  const averageHistory = [];
  const MAX_HISTORY_LENGTH = 10; // Number of recent average values to keep for difference calculation
  const NOISE_HISTORY_LENGTH = 10;
  const NOISE_HISTORY = []
  let user_interrupts = false;
  let voiceActiveTimestamp = 0; //148
  /**
   * Simple amplitude-based voice activity detection enhanced with average difference check.
   */

  let avgNoise = 0;
  const detectVoiceActivity = (audioData) => {
      if (!audioData || audioData.length === 0) return false;

      const sum = audioData.reduce((acc, val) => acc + val * val, 0);
      const average = Math.sqrt(sum / audioData.length);

      // Maintain noise samples when not pla6ing and energy is low
      if (!globalIsPlaying && average < 0.055) {
          noiseSamples.push(average);
          if (noiseSamples.length > MAX_NOISE_SAMPLES) {
              noiseSamples.shift();
          }
          let temp = noiseSamples.reduce((a, b) => a + b, 0) / noiseSamples.length;

          if(temp > 0) {
            avgNoise = temp;
          }

          USER_SPEAKER_THRESHOLD = Math.max(avgNoise + 0.02, 0.03); // margin above noise
      }

      averageHistory.push(average);
      if (averageHistory.length > MAX_HISTORY_LENGTH) {
          averageHistory.shift();
      }

      NOISE_HISTORY.push(avgNoise);

      if (NOISE_HISTORY.length > NOISE_HISTORY_LENGTH) {
          NOISE_HISTORY.shift();
      }

      const threshold = globalIsPlaying ? AUDIO_SPEAKER_THRESHOLD : USER_SPEAKER_THRESHOLD;
      const zcr = computeZCR(audioData);
      const noiseHistorySum = NOISE_HISTORY.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
      console.log('Threshold:', threshold.toFixed(5), 'Average:', average.toFixed(5), 'AVGnoise:', avgNoise.toFixed(5), 'ZCR:', zcr, 'noiseHistorySum:', noiseHistorySum, "user_interrupts:", user_interrupts);
      let isVoiceActive = false;

      if (globalIsPlaying) {
        // user interrupts TTS
        isVoiceActive = average > threshold && zcr > 0.03; // for avoiding false positives of interruption           
      } else {
        // normal conversation, no playback
        isVoiceActive = average > threshold && zcr < 0.14 && noiseHistorySum !== 0;
      }

      //if (user_interrupts) {
      //    isVoiceActive = (average > threshold && zcr < 0.09) || (average > threshold && globalIsPlaying);
      //} else {
      //    user_interrupts = false;
      //    isVoiceActive = (average > threshold && zcr < 0.09 && !globalIsPlaying && noiseHistorySum != 0) || (average > threshold && globalIsPlaying);
      //}
      if (isVoiceActive) {
          console.log("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Voice is Active @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@");
      }

      return isVoiceActive;
  };

  const computeZCR = (audioData) => {
      let zeroCrossings = 0;
      for (let i = 1; i < audioData.length; i++) {
          if ((audioData[i - 1] >= 0 && audioData[i] < 0) || (audioData[i - 1] < 0 && audioData[i] >= 0)) {
              zeroCrossings++;
          }
      }
      return zeroCrossings / audioData.length;
  };

  function getSessionId(forceReset) {
      const cookieName = "session_id";

      // Helper to read a cookie
      function getCookie(name) {
          const match = document.cookie.match(new RegExp('(^| )' + name + '=([^;]+)'));
          return match ? match[2] : null;
      }

      // Helper to set a cookie
      function setCookie(name, value, days = 30) {
          const expires = new Date(Date.now() + days * 864e5).toUTCString();
          document.cookie = `${name}=${value}; expires=${expires}; path=/; SameSite=Lax`;
      }

      // Try to fetch existing session_id
      let sessionId = getCookie(cookieName);

      if (!sessionId || forceReset) {
          // Generate new session id
          sessionId = crypto.randomUUID(); // browser-safe UUID
          setCookie(cookieName, sessionId);
          console.log("Generated new session ID:", sessionId);
      } else {
          console.log("Using existing session ID:", sessionId);
      }

      return sessionId;
  }

  /**
   * Establish WebSocket to the server for TTS or ASR results.
   */
  const connectWebSocket = () => {
      if (websocketRef.current) {
          websocketRef.current.close();
      }

      let sessionId = getSessionId();

      websocketRef.current = new WebSocket(WEBSOCKET_URL + "?sessionId=" + sessionId);

      // If your server sends TTS in binary form:
      websocketRef.current.binaryType = 'blob';

      websocketRef.current.onopen = async () => {
          await websocketRef.current.send("SESSION_ID:" + sessionId);
          console.log("sessionId sent : " + sessionId);
      }

      websocketRef.current.onerror = () => {
          console.debug('[WebSocket] Error. Retrying...');
          setTimeout(() => {
              connectWebSocket();
          }, 1000);
      };
      websocketRef.current.onmessage = (event) => {
          console.debug('[WebSocket] Received data from server:', event.data);
          websocketEventTime = Date.now(); //281

          console.log("time fe to be : ", websocketEventTime - donestartTime) ///282

          // Check if it's a string first
          if (typeof event.data === "string") {
            console.log("event received : " + event.data);
            if (event.data === "PLAY_FILLER") {
                playFiller(randomIndex("filler"));
            } else if (event.data === "PLAY_CONGRATS") {
                playFiller(congrats_mp3_bytes);
            } else if (event.data === "GENERATE QUOTES") {
                playFiller(generate_quotes_bytes);
            } else if (event.data === "FILLER:name") {
                playFiller(randomIndex("name"));
            } else if (event.data === "FILLER:age") {
                playFiller(randomIndex("age"));
            } else if (event.data === "FILLER:pincode") {
                playFiller(randomIndex("pincode"));
            } else if (event.data === "FILLER:gender") {
                playFiller(randomIndex("gender"));
            } else if (event.data === "FILLER:ask_again") {
                playFiller(randomIndex("ask_again"));
            } else if (event.data === "FILLER:existing_disease") {
                playFiller(randomIndex("existing_disease"))
            } else if (event.data.startsWith("open_lead_page:")) {
                const visitUrl = event.data.substring("open_lead_page:".length);
                window.open(visitUrl, "_blank");
            } else if (event.data == "PLAY_CONTINUE_FLOW") {
                playFiller(continue_flow_bytes);
            }else if(event.data == "FILLER:age_reask") {
                playFiller(randomIndex("age_reask"))
            }else if(event.data == "FILLER:back_to_buy") {
                playFiller(randomIndex("back_to_buy"))
            }else if(event.data == "FILLER:cancel_intent") {
                playFiller(randomIndex("cancel_intent"))
            }else if(event.data == "FILLER:flow_complete") {
                playFiller(randomIndex("flow_complete"))
            }else if(event.data == "FILLER:re_asking") {
                playFiller(randomIndex("re_asking"))
            }else if(event.data == "FILLER:pincode_reask") {
                playFiller(randomIndex("pincode_reask"))
            }else if(event.data == "FILLER:pincode_registered") {
                playFiller(randomIndex("pincode_registered"))
            }else if(event.data == "FILLER:gender_registered") {
                playFiller(randomIndex("gender_registered"))
            }else if(event.data == "FILLER:follow_up") {
                playFiller(randomIndex("follow_up"))
            }else {
                  console.warn("[WebSocket] Unknown string message:", event.data);
            }
          }
          // Handle binary audio data (like Blob)
          else if (event.data instanceof Blob) {
              console.log("event received is a blob");
              audioQueueRef.current.push(event.data);
              // if (!globalIsPlaying) {
              //   playNextInQueue();
              // }
          }

          // Fallback case for unexpected data types
          else {
              console.warn("[WebSocket] Received unexpected data type:", typeof event.data, event.data);
          }
      };

  };

  setInterval(function() {
      if (!globalIsPlaying) {
          //console.log("Nothing playing. Trying to play now.")
          playNextInQueue();
      } else {
          // do nothing
      }
  }, 30)

  /**
   * Queue-based TTS playback via Howler. 
   * If user speaks mid-play, we stop playback.
   */
  const playNextInQueue = () => {
      const queue = audioQueueRef.current;
      if (queue.length === 0) {
          //console.debug('[Queue] No more audio to play');
          return;
      } else if (globalIsPlaying) {
          console.log('Audio playing. cannot play.');
          return;
      }

      console.log("before playing nextinQueue , globalIsPlaying : " + globalIsPlaying);

      const nextAudioBlob = queue.shift();
      setGlobalIsPlaying(true);
      // console.log('[Queue] Popped next audio blob, queue length:', queue.length);

      // Unload old sound
      if (soundRef.current) {
          soundRef.current.unload();
          soundPlayRef.current = null;
      }

      fetch([URL.createObjectURL(nextAudioBlob)])
          .then(r => r.arrayBuffer())
          .then(async function(mp3data) {
              // AudioContext setup
              let context = new AudioContext();
              const audioBuf = await context.decodeAudioData(mp3data);
              let float32Data = audioBuf.getChannelData(0);
              const sumSq = float32Data.reduce((sum, x) => sum + x * x, 0);
              TTS_ENERGY_BASELINE = Math.sqrt(sumSq / float32Data.length);
              //console.log("audio TTS_ENERGY_BASELINE : " + TTS_ENERGY_BASELINE);

              // ✅ Setup MediaStreamDestinationNode
              ttsMediaDestRef.current = context.createMediaStreamDestination();

              // 🔁 Feed to both: speakers + media destination
              const bufferSource = context.createBufferSource();
              bufferSource.buffer = audioBuf;
              bufferSource.connect(context.destination); // speakers
              bufferSource.connect(ttsMediaDestRef.current); // for echo cancellation

              // Re-create Howl using Blob URL (not affecting above)
              const audioFormat = nextAudioBlob.type === 'audio/mpeg' ? ['mp3'] : ['wav'];
              soundRef.current = new Howl({
                  src: [URL.createObjectURL(nextAudioBlob)],
                  format: audioFormat,
                  html5: true,
                  volume: 0.7,
                  onplay: () => {
                      console.debug("on play");
                      setGlobalIsPlaying(true);
                      startVoiceDetectionDuringPlayback();
                  },
                  onend: () => {
                      setGlobalIsPlaying(false);
                      stopVoiceDetectionDuringPlayback();
                      // playNextInQueue();
                  },
                  onstop: () => {
                      setGlobalIsPlaying(false);
                      stopVoiceDetectionDuringPlayback();
                  },
                  onloaderror: (id, err) => {
                      console.error('[Howl] Load error:', err);
                      setGlobalIsPlaying(false);
                      stopVoiceDetectionDuringPlayback();
                      // playNextInQueue();
                  },
                  onplayerror: (id, err) => {
                      console.error('[Howl] Play error:', err);
                      setGlobalIsPlaying(false);
                      stopVoiceDetectionDuringPlayback();
                      // playNextInQueue();
                  },
              });

              setIsThinking(false);
              soundPlayRef.current = soundRef.current.play();
          })
  };

  async function getFilteredAudioStream() {
    // Create an AudioContext
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContext();
  
    // Get the microphone stream (leave echoCancellation if you still need it)
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: { ideal: true },
        noiseSuppression: true,
        autoGainControl: true
      }
    });
  
    // Pass-through: mic ➜ MediaStreamDestination (no EQ, no compression)
    const source = context.createMediaStreamSource(stream);
    const dest   = context.createMediaStreamDestination();
    source.connect(dest);          // keep the track alive, stay silent locally
  
    return {
      filteredStream: dest.stream, // unchanged API
      audioContext:   context
    };
  }
  
  let voiceDetectionActivityData = [];
  /**
   * During TTS playback, we open a mic stream just to detect user speech.
   * If user speaks => stop TTS, close WS, begin recording for ASR.
   */
  const startVoiceDetectionDuringPlayback = async () => {
      try {
          // const stream = await navigator.mediaDevices.getUserMedia({ audio : {
          //   echoCancellation: {ideal: true}, 
          //   noiseSuppression: {ideal: true}, 
          //   autoGainControl: {ideal: true}
          // }});
          const {filteredStream} = await getFilteredAudioStream();

          let localAudioContext = new AudioContext(),
              localAnalyserRef = localAudioContext.createAnalyser(),
              localSourceNodeRef = localAudioContext.createMediaStreamSource(filteredStream);

          localAnalyserRef.fftSize = 1024;
          const bufferLength = localAnalyserRef.frequencyBinCount;
          const dataArray = new Float32Array(bufferLength);

          localSourceNodeRef.connect(localAnalyserRef);

          //audioContextRef.current = new AudioContext();
          //analyserRef.current = audioContextRef.current.createAnalyser();
          //sourceNodeRef.current = audioContextRef.current.createMediaStreamSource(filteredStream);

          //analyserRef.current.fftSize = 1024;
          //const bufferLength = analyserRef.current.frequencyBinCount;
          //const dataArray = new Float32Array(bufferLength);

          //sourceNodeRef.current.connect(analyserRef.current);

          const checkVoice = () => {
              if (!globalIsPlaying) return; // TTS ended or was stopped
              localAnalyserRef.getFloatTimeDomainData(dataArray);
              voiceDetectionActivityData.push(...dataArray);
              if (voiceDetectionActivityData.length > SIZE_OF_VOICE_DETECTION_ACTIVITY_DATA) {
                voiceDetectionActivityData.splice(0, voiceDetectionActivityData.length - SIZE_OF_VOICE_DETECTION_ACTIVITY_DATA);
              }
              //addToSlidingWindow(dataArray);
              const isVoice = detectVoiceActivity(voiceDetectionActivityData);

              if (isVoice) {
                  console.log('[Playback] Voice detected, stopping TTS');
                  user_interrupts = true;
                  if (soundRef.current) {

                      soundRef.current.fade(0.6, 0, 500, soundPlayRef.current);

                      setTimeout(function() {
                          soundRef.current.stop();
                          soundRef.current.unload();
                          soundPlayRef.current = null;
                      }, 500);
                  }

                  if (websocketRef.current) {
                      websocketRef.current.close();
                  }

                  while (audioQueueRef.current.length > 0) {
                      audioQueueRef.current.pop();
                  }

                  setGlobalIsPlaying(false);
                  stopVoiceDetectionDuringPlayback();
                  //cleanup();
                  isCleanup.current = false; // we dont want to clean everything up.
                  startRecording(); // start capturing user speech
              } else {
                  voiceTimeoutRef.current = requestAnimationFrame(checkVoice);
              }
          };
          checkVoice();
      } catch (err) {
          console.error('[Playback] Error starting voice detection:', err);
      }
  };

  /**
   * Stop analyzing mic for TTS interruption.
   */
  const stopVoiceDetectionDuringPlayback = () => {
      if (voiceTimeoutRef.current) {
          cancelAnimationFrame(voiceTimeoutRef.current);
      }
      //if (sourceNodeRef.current) {
      //    sourceNodeRef.current.disconnect();
      //}
      //if (analyserRef.current) {
      //    analyserRef.current.disconnect();
      //}
  };

  /**
   * Start capturing mic audio in real time with a ScriptProcessorNode.
   * - We only send PCM when voice is detected.
   * - After 500ms of silence, send "DONE" to finalize utterance.
   */

  const startRecording = async (bytesToAdd = []) => {
      try {

          if (bytesToAdd && bytesToAdd.length > 0) {
              preBuffer.push(...bytesToAdd);
          }

          /* ------------------------------------------------------------------ */
          /* Tunables                                                           */
          const PRE_MS = 892 * 2; // 892*2 ms pre-voice
          const POST_MS = 0; // 892 ms post-voice
          const SAMPLE_RATE = 16000; // must match getUserMedia

          const FRAME_MS = 30; // 10 | 20 | 30
          const FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS / 1000; // 320
          const PRE_FRAMES = Math.ceil(PRE_MS / FRAME_MS); // 45
          const POST_FRAMES = Math.ceil(POST_MS / FRAME_MS); // 45
          const PRE_SAMPLES = (PRE_FRAMES * FRAME_SAMPLES); // 14 400
          const POST_SAMPLES = POST_FRAMES * FRAME_SAMPLES; // 14 400

          console.log("preSamples : " + PRE_SAMPLES + ", postSamples : " + POST_SAMPLES + ", prebuffer : " + preBuffer.length + ", cleanup : " + isCleanup.current);
          /* ------------------------------------------------------------------ */

          // Acquire mic
          const stream = await navigator.mediaDevices.getUserMedia({
              audio: {
                  echoCancellation: {
                      ideal: true
                  },
                  noiseSuppression: true, //{ ideal: true },
                  autoGainControl: true //{ ideal: true }
              }
          });

          // Create audio nodes
          if(isCleanup.current) {
            audioContextRef.current = new AudioContext({
              sampleRate: SAMPLE_RATE
            });

            const gainNode = audioContextRef.current.createGain();

            sourceNodeRef.current = audioContextRef.current.createMediaStreamSource(stream);
            processorNodeRef.current = audioContextRef.current.createScriptProcessor(2048, 1, 1);  

            gainNode.gain.value = 1.5;
            sourceNodeRef.current.connect(gainNode).connect(processorNodeRef.current);

            //sourceNodeRef.current.connect(processorNodeRef.current);
            processorNodeRef.current.connect(audioContextRef.current.destination);
          }

          // WebSocket for sending
          connectWebSocket();

          //playFiller(filler_mp3_bytes);

          /* Rolling buffers --------------------------------------------------- */
          //let preBuffer  = [];        // <= PRE_SAMPLES float32
          let postBuffer = []; // collect after VAD silence

          /* VAD state --------------------------------------------------------- */
          let isVoiceActive = false;
          let silenceTimer = null;
          let postSamplesCnt = 0;

          let silenceSampleCountThreshold = 8,
              silenceSampleCount = 0;

          let voiceActivityData = [];

          if(isCleanup.current) {
            processorNodeRef.current.onaudioprocess = (e) => {
              console.log("onaudioprocess initiated : " + preBuffer.length);
              const frame = e.inputBuffer.getChannelData(0); // Float32Array
              voiceActivityData.push(...frame);

              if (voiceActivityData.length > 6144) {
                  voiceActivityData.splice(0, voiceActivityData.length - 6144); // 16000 * 0.384 -> SAMPLE_RATE * AUDIO_MS
              }

              // if(!voiced) {
              //   silenceSampleCount++;
              // }

              /* ---- Always maintain PRE-BUFFER -------------------------------- */
              preBuffer.push(...frame);
              if (preBuffer.length > PRE_SAMPLES) {
                  preBuffer.splice(0, preBuffer.length - PRE_SAMPLES);
              }

              // other than adding to prebuffer, this is not supposed to do anything else when audio is playing
              if(globalIsPlaying) {
                console.log("global audio is playing. Returning. prebuffer length : " + preBuffer.length);
                return;
              }

              console.log("global audio not playing. proceeding as expected. prebuffer size : " + preBuffer.length);

              const voiced = detectVoiceActivity(voiceActivityData);

              if ((voiced && silenceSampleCount < silenceSampleCountThreshold) || (silenceSampleCount < silenceSampleCountThreshold && isVoiceActive)) {

                  if (!voiced && isVoiceActive) {
                      silenceSampleCount++;
                  }

                  if(silenceSampleCount === 1){
                    voiceActiveTimestamp = Date.now();
                  } //677


                  /* -------- Speaker is talking ---------------------------------- */
                  if (silenceTimer) {
                      clearTimeout(silenceTimer);
                      silenceTimer = null;
                  }
                  if (voiced) {
                      silenceSampleCount = 0;
                      console.log("resetting silence sample ((((((((((((((((((((((((((")
                  }


                  if (!isVoiceActive) {
                      // First detection of speech
                      isVoiceActive = true;

                      // Send just preBuffer – current frame is already inside it
                      let chunk = float32ToPCM(preBuffer.slice());
                      const preBufferSentTime = Date.now();
                      console.log("time preBufferSentTime : ", preBufferSentTime); //698

                      
                      sendDataOnSocket(chunk, websocketRef.current);
                      console.log("prebuffer clear 1");
                      preBuffer.length = 0; // clear it AFTER sending

                      postBuffer.length = 0; // reset post-buffer
                      postSamplesCnt = 0;

                      return; // IMPORTANT: don't process current frame again

                  } else {
                      sendDataOnSocket(float32ToPCM(frame), websocketRef.current);
                      const frameSentTime = Date.now();
                      console.log("time frameSentTime : ", frameSentTime, " frame lenegth : " , frame.length); //712

                      console.log("prebuffer clear 2");
                      postBuffer.length = 0; // reset post-buffer
                      postSamplesCnt = 0;
                      preBuffer.length = 0; // clear it AFTER sending
                  }
              }
              //else if(silenceSampleCount < silenceSampleCountThreshold && isVoiceActive){
              //  console.log("**************** silenceSampleCount : ", silenceSampleCount )
              //  silenceSampleCount++;
              //}
              else {
                  /* -------- Detected silence ------------------------------------ */
                  if (isVoiceActive && silenceSampleCount == silenceSampleCountThreshold) {
                      console.log("resetting silence sample-----------------")


                      postBuffer.push(...frame);
                      postSamplesCnt += frame.length;

                      const wantDone = postSamplesCnt >= POST_SAMPLES;
                      if (wantDone && !silenceTimer) {
                          //silenceTimer = setTimeout(() => {
                          /* send trailing 892 ms before DONE ------------------------ */
                          sendDataOnSocket(float32ToPCM(postBuffer), websocketRef.current);
                          const postBufferSentTime = Date.now();
                          console.log("time postBufferTime : ", postBufferSentTime); // 
                          postBuffer.length = 0;
                          postSamplesCnt = 0;
                          isVoiceActive = false;
                          silenceTimer = null;
                          sendDoneSignal();
                          silenceSampleCount = 0
                          // <-- notify server
                          //}, 0);   // fire immediately once POST_MS reached
                      }
                  }
              }
            };
          }

          setIsRecording(true);
      } catch (err) {
          console.error('[Recording] Error starting recording:', err);
      }
  };



  /**
   * Send raw bytes (PCM) or a sentinel message if open.
   */

  let webSocketDataSendingInterval;
  const sendDataOnSocket = (data, websocket) => {
      //if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
      //websocket.send(data);
      console.log("!!!!!!!!!!!!!!!!!!!!!!!sending data to socket!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      sendQueueRef.current.push(data);

      clearTimeout(webSocketDataSendingInterval);
      webSocketDataSendingInterval = setInterval(function() {
          if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
          let popped = sendQueueRef.current.shift();

          if (popped) {
              console.debug("Popped : ", popped);
              websocket.send(popped);
          }
      }, 10)
  };



  /**
   * Tells the server we're done with the current utterance.
   */
  const sendDoneSignal = () => {
      if (!websocketRef.current || websocketRef.current.readyState !== WebSocket.OPEN) return;
      const encoder = new TextEncoder();
      const doneData = encoder.encode('DONE');
      console.log("Sending Request : " + (new Date()));
      donestartTime = Date.now();
      console.log("time donestartTime : ", donestartTime)
      console.log("time voice to done:", donestartTime-voiceActiveTimestamp, "ms"); // 796
      setIsThinking(true);
      //websocketRef.current.send(doneData);
      sendDataOnSocket(doneData, websocketRef.current);
  };

  /**
   * Stop capturing audio, close mic tracks, cleanup.
   */
    const stopRecording = () => {
    if (processorNodeRef.current) {
        let a = new Error();
        console.log("processorNodeRef reset : " , a);
        processorNodeRef.current.disconnect();
    }
    if (sourceNodeRef.current) {
        sourceNodeRef.current.disconnect();
        const tracks = sourceNodeRef.current.mediaStream.getTracks();
        tracks.forEach((track) => track.stop());
    }

    while (audioQueueRef.current.length > 0) {
        audioQueueRef.current.pop();
    }
    // added this for audioContextRef closure------------------------------------------------------------
    if (audioContextRef.current) {                          
      audioContextRef.current.close().catch(err => {
        console.warn("Error closing AudioContext:", err);
      });
      audioContextRef.current = null;
    }
  
    cleanup();
    setIsRecording(false);

    getSessionId(true);
};
  /**
   * Toggle mic button => start or stop.
   */
  const handleVoiceButtonClick = () => {

      if (isRecording) {
          stopRecording();
      } else {
          cleanup();
          startRecording();
          playFiller(intro_mp3_bytes);
      }
  };

  const playFiller = (blobContent) => {
      const mp3Blob = dataURLToBlob(blobContent);
      audioQueueRef.current.push(mp3Blob);

      // if (!globalIsPlaying) {
      //   playNextInQueue();
      // }
  }

//   const playDummyFiller = () => {
//       playFiller(congrats_mp3_bytes);
//   }

  return ( <div className = "min-h-screen bg-gradient-to-b from-[#FF6347] to-[#FFDAB9] flex flex-col items-center justify-center p-4" >
              <div className = "w-full max-w-md bg-white rounded-2xl shadow-xl p-6 space-y-6" >
                <h1 className = "text-3xl font-bold text-center text-[#FF6347]" >VOICE - BOT </h1>
                <div className = "flex justify-center" >
                  {/* <button id = "test" onClick = {playDummyFiller} > Click me </button> */}
                  <button id = "aditya" onClick = {handleVoiceButtonClick} className = {`w-24 h-24 rounded-full flex items-center justify-center transition-all ${isRecording ? 'bg-[#FF6347] text-white' : 'bg-white border-4 border-[#FF6347] text-[#FF6347]'}`} >
                    { isRecording ? < MicOff className = "w-12 h-12" / > : < Mic className = "w-12 h-12" / >} 
          </button>
        </div>

                <div className = "text-center text-gray-600" > {isRecording ? (isThinking ? <p> Thinking.... </p> : <p>Listening... Tap to stop</p > ) : <p> Tap microphone to start </p>} 
              </div>
        </div>
      </div>
  );
}

  function dataURLToBlob(dataURL) {
      const parts = dataURL.split(",");
      const byteString = atob(parts[1]);
      const mimeString = parts[0].split(":")[1].split(";")[0];

      // Create an ArrayBuffer and a typed array (Uint8Array) to hold the binary data
      const ab = new ArrayBuffer(byteString.length);
      const ia = new Uint8Array(ab);
      for (let i = 0; i < byteString.length; i++) {
          ia[i] = byteString.charCodeAt(i);
      }

      return new Blob([ab], {
          type: mimeString
      });
  }

  export default App;