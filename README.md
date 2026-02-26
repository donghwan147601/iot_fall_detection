# 🚑 IoT Fall Detection System
**Capstone Design Project | Summer 2025**

This project features an automated fall detection system integrated with a Google Cloud AI-powered voice assistant to minimize emergency response times.

## ⚙️ System Workflow
The system operates on an **event-driven architecture** to ensure a rapid and reliable response:

1. **Fall Detection:** Continuous monitoring via the `run_detection` loop. It triggers the `handle_fall_event` callback immediately upon detection.
2. **Voice Prompt (TTS):** Plays a "Are you okay?" message using **Google Cloud Text-to-Speech**. Uses a *blocking call* to prevent audio overlap.
3. **User Response Capture:** Activates the microphone for a **10-second window** to record the user's status.
4. **Speech Analysis (STT):** Processes audio through **Google Cloud Speech-to-Text** to extract transcripts.
5. **Emergency Logic:**
   - ✅ **Safe:** Resumes monitoring if positive keywords (e.g., "Yes", "Fine") are detected.
   - 🚨 **Alert:** Returns an **"ALERT"** status if silence or unrecognizable responses are detected, initiating emergency protocols.

## 🛠 Tech Stack
- **Languages:** Python
- **APIs:** Google Cloud TTS & STT
- **Key Libraries:** `sounddevice`, `numpy`, `google-cloud-speech`
