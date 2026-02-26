<<<<<<< HEAD
# iot_fall_detection
## Capstone Design Project in Summer, 2025
=======
This system operates on an event-driven architecture to ensure rapid response to emergencies. The workflow is as follows:

Fall Detection: The system continuously monitors via the run_detection loop. When a fall event is triggered, it calls the handle_fall_event callback.

Voice Prompt (TTS): The system immediately plays a voice message ("Are you okay?") using Google Cloud Text-to-Speech. It uses a blocking call to ensure the message is fully delivered before listening.

User Response Capture: The microphone activates for a 10-second window to record the user's audio response.

Speech Analysis (STT): The recorded audio is processed through Google Cloud Speech-to-Text to extract a text transcript.

Emergency Logic:

Safe: If keywords like "Yes" or "Fine" are detected, the system resumes monitoring.

Alert: If there is silence, an unrecognizable response, or an error, the system returns an "ALERT" status to trigger emergency protocols.
>>>>>>> 213c72a (Initial commit)
