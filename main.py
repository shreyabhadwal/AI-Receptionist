#AI Receptionist 
# Take down messages, maybe check calendar invites, provide information
# Schedule a meeting/appointment, google calendar, ask for information, is the doctor available

import sounddevice as sd 
import soundfile as sf
import openai
import keyboard
import tempfile
import os
from elevenlabs import generate, play, set_api_key
from execute_ai import call_agent, answer_the_call

duration = 10  # duration of each recording in seconds
fs = 44100  # sample rate
channels = 1  # number of channels
sd.default.samplerate = fs
sd.default.channels = 2

os.environ["OPENAI_API_KEY"] = "" #OpenAI Key
set_api_key("")

def record_audio(duration, fs, channels):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=channels)
    sd.wait()
    print("Finished recording.")
    return recording

def transcribe_audio(recording, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, recording, fs)
        temp_audio.close()
        with open(temp_audio.name, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        os.remove(temp_audio.name)
    return transcript["text"].strip()

def play_generated_audio(text, voice="Bella", model="eleven_monolingual_v1"):
    audio = generate(text=text, voice=voice, model=model)
    play(audio)

if __name__ == '__main__':

    inital_text = "Hi! You have reached Dunder Mifflin's office, this is Pam AI. What can I do for you today?"
    play_generated_audio(inital_text)

    while True:
        print("Press spacebar to start recording.")
        keyboard.wait("space")  # wait for spacebar to be pressed
        recorded_audio = record_audio(duration, fs, channels)
        message = transcribe_audio(recorded_audio, fs)
        print(f"You: {message}")
        agent = call_agent()
        task_info = agent.run(message)
        answer_chain = answer_the_call()
        answer = answer_chain.predict(INFO = task_info)
        play_generated_audio(answer) 