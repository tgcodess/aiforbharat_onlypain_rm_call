// test.js (ES Module)
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// To get __dirname in ES module
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Use absolute path according to your device 
const mp3File = '/home/user/advisor-ai-call-report/elevanlabaudio/gender_registered.mp3';

try {
    const mp3Bytes = fs.readFileSync(mp3File);
    const base64Data = mp3Bytes.toString('base64');
    const dataUrl = `data:audio/mp3;base64,${base64Data}`;

    console.log(dataUrl);
} catch (err) {
    console.error("❌ Error:", err.message);
}
