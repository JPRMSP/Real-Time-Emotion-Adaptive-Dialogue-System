import streamlit as st
import numpy as np
import sounddevice as sd
import scipy.signal as signal
import scipy.io.wavfile as wav
import io
import re
import random
import time

st.set_page_config(page_title="Emotion Adaptive Dialogue System", layout="wide")

# ------------------------------
# SPEECH FEATURE EXTRACTION
# ------------------------------
def extract_features(audio, fs):
    # Normalize
    audio = audio / np.max(np.abs(audio))

    # Intensity (RMS)
    intensity = np.sqrt(np.mean(np.square(audio)))

    # Pitch estimation using autocorrelation
    corr = signal.correlate(audio, audio, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]
    peak = np.argmax(corr[start:]) + start
    pitch = fs / peak if peak != 0 else 0

    # Speech rate (zero-crossing-based proxy)
    zc = ((audio[:-1] * audio[1:]) < 0).sum()
    speech_rate = zc / len(audio)

    # Silence ratio
    silence_ratio = np.sum(np.abs(audio) < 0.02) / len(audio)

    return {
        "intensity": float(intensity),
        "pitch": float(pitch),
        "speech_rate": float(speech_rate),
        "silence_ratio": float(silence_ratio)
    }

# ------------------------------
# LINGUISTIC EMOTION CUES
# ------------------------------
emotion_keywords = {
    "happy": ["happy", "great", "awesome", "nice", "wow"],
    "sad": ["sad", "down", "bad", "missing", "upset"],
    "angry": ["angry", "mad", "irritated", "annoyed"],
    "fear": ["scared", "fear", "nervous", "afraid"],
    "surprise": ["shocked", "surprised", "woah"],
}

def linguistic_scores(text):
    text = text.lower()
    scores = {e: 0 for e in emotion_keywords}
    for emo, words in emotion_keywords.items():
        for w in words:
            if w in text:
                scores[emo] += 1
    return scores

# ------------------------------
# SEMI-STOCHASTIC EMOTION ENGINE
# ------------------------------
def emotional_fusion(speech_feat, ling_scores):
    base = {
        "happy": speech_feat["pitch"] * 0.2 + speech_feat["speech_rate"] * 2,
        "sad": speech_feat["silence_ratio"] * 3,
        "angry": speech_feat["intensity"] * 2,
        "fear": speech_feat["speech_rate"] * 1.5 + speech_feat["silence_ratio"] * 2,
        "surprise": abs(speech_feat["pitch"] - 200) * 0.01,
    }

    # Add linguistic influence
    for emo in base:
        base[emo] += ling_scores[emo] * 2

    # Add stochasticity
    for emo in base:
        base[emo] += random.uniform(0, 1)

    total = sum(base.values())
    probs = {emo: base[emo] / total for emo in base}

    predicted = max(probs, key=probs.get)
    return predicted, probs

# ------------------------------
# ADAPTIVE DIALOGUE RESPONSE
# ------------------------------
adaptive_responses = {
    "happy": "You sound cheerful! Let's keep that positive flow going.",
    "sad": "I'm here for you. Take your time — want to talk about it?",
    "angry": "It’s okay to feel frustrated. Want me to help calm things down?",
    "fear": "I understand your concern. Let me guide you safely.",
    "surprise": "That seems unexpected! Tell me more about it.",
}

# ------------------------------
# STREAMLIT UI
# ------------------------------
st.title("🎭 Real-Time Emotion Adaptive Dialogue System (No ML/No Dataset)")
st.write("This system recognizes emotions using rule-based speech + text analysis and a semi-stochastic emotional model.")

col1, col2 = st.columns(2)

with col1:
    st.header("🎤 Record Speech")
    duration = st.slider("Recording duration (seconds)", 2, 10, 4)
    if st.button("Start Recording"):
        st.write("Recording...")
        audio = sd.rec(int(duration * 16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        audio = audio.flatten()
        st.success("Recording complete!")

        features = extract_features(audio, 16000)
        st.subheader("Speech Features")
        st.json(features)

with col2:
    st.header("⌨ Enter Text")
    user_text = st.text_area("Type a message")
    ling = linguistic_scores(user_text)
    st.subheader("Linguistic Emotion Scores")
    st.json(ling)

if st.button("Analyze Emotion"):
    if user_text.strip() == "":
        st.error("Please enter a text message.")
    else:
        # Use dummy speech features if not recorded
        if "features" not in locals():
            features = {"intensity":0.1, "pitch":150, "speech_rate":0.05, "silence_ratio":0.1}

        emotion, probs = emotional_fusion(features, ling)

        st.header("🎯 Predicted Emotion")
        st.success(emotion.upper())

        st.subheader("📊 Emotion Probability Distribution")
        st.json(probs)

        st.subheader("🗣 Adaptive Dialogue Response")
        st.info(adaptive_responses[emotion])
