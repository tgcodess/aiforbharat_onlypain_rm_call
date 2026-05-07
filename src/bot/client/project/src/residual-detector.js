class ResidualDetector extends AudioWorkletProcessor {
    constructor() {
      super();
      this.smoothedMicRMS = 0;
      this.smoothedTTSRMS = 0;
      this.scale = 1;
      this.smoothingFactor = 0.05; // EMA smoothing factor
    }
  
    process(inputs) {
      const tts = inputs[0]?.[0]; // TTS audio from speaker
      const mic = inputs[1]?.[0]; // Mic input
  
      if (!tts || !mic) return true;
  
      let micSum = 0, ttsSum = 0;
      for (let i = 0; i < mic.length; i++) {
        micSum += mic[i] * mic[i];
        ttsSum += tts[i] * tts[i];
      }
  
      const micRMS = Math.sqrt(micSum / mic.length);
      const ttsRMS = Math.sqrt(ttsSum / tts.length);
  
      // Smooth both RMS values
      this.smoothedMicRMS = this.smoothingFactor * micRMS + (1 - this.smoothingFactor) * this.smoothedMicRMS;
      this.smoothedTTSRMS = this.smoothingFactor * ttsRMS + (1 - this.smoothingFactor) * this.smoothedTTSRMS;
  
      if (this.smoothedTTSRMS > 0.001) {
        this.scale = this.smoothedMicRMS / this.smoothedTTSRMS;
      }
  
      // Subtract scaled TTS from mic
      let residual = 0;
      for (let i = 0; i < mic.length; i++) {
        const diff = mic[i] - (tts[i] * this.scale);
        residual += diff * diff;
      }
  
      const residualRMS = Math.sqrt(residual / mic.length);
  
      if (residualRMS > 0.015) {
        this.port.postMessage('user_speaking');
      }
  
      return true;
    }
  }
  
  registerProcessor('residual-detector', ResidualDetector);
  