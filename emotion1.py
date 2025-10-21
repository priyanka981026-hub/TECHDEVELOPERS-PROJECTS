# app_emotion_dashboard.py

import streamlit as st
import torch
import torch.nn as nn
import librosa
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import Wav2Vec2Processor, Wav2Vec2Model
#set_page_config → for page setup (background, layout, tab name)
#   st.title → for on-page heading shown to users
# ------------------- Page Config -------------------
st.set_page_config(page_title="🎙️ Gender & Emotion Voice Classifier", layout="wide")
st.title("🎧 Gender & Emotion Voice Classifier")
st.markdown("Upload an audio file to detect **Gender** and **Emotion** using a deep learning model.")

# ------------------- Upload Audio -------------------
audio_file = st.file_uploader("Upload your voice file (WAV or MP3)", type=["wav", "mp3"])

if audio_file:
    st.audio(audio_file, format="audio/wav")
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    # ------------------- Feature Extraction -------------------
    y, sr = librosa.load("temp_audio.wav", sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    inputs = processor(y, sampling_rate=sr, return_tensors="pt", padding=True)

    # ------------------- Model Definitions -------------------
    class EmotionClassifier(nn.Module):
        def __init__(self, num_emotions=4):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.classifier = nn.Sequential(
                nn.Linear(768, 128),
                nn.ReLU(),
                nn.Linear(128, num_emotions)
            )
        def forward(self, x):
            outputs = self.wav2vec2(**x)
            hidden_state = outputs.last_hidden_state[:, 0, :]
            return self.classifier(hidden_state)

    class GenderClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.classifier = nn.Sequential(
                nn.Linear(768, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
        def forward(self, x):
            outputs = self.wav2vec2(**x)
            hidden_state = outputs.last_hidden_state[:, 0, :]
            return self.classifier(hidden_state)

    # ------------------- Initialize Models -------------------
    emotion_model = EmotionClassifier()
    gender_model = GenderClassifier()
    emotion_model.eval()
    gender_model.eval()

    # ------------------- Predictions -------------------
    with torch.no_grad():
        emotion_logits = emotion_model(inputs)
        gender_logits = gender_model(inputs)

        emotion_probs = torch.softmax(emotion_logits, dim=1).numpy()[0]
        gender_probs = torch.softmax(gender_logits, dim=1).numpy()[0]

        emotion_labels = ['Happy', 'Sad', 'Angry', 'Neutral']
        gender_labels = ['Male', 'Female']

        emotion_pred = emotion_labels[np.argmax(emotion_probs)]
        gender_pred = gender_labels[np.argmax(gender_probs)]

    # ------------------- Dashboard Layout -------------------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🎭 Emotion Prediction")
        st.success(f"**{emotion_pred}**")
        fig1 = px.bar(
            x=emotion_labels, y=emotion_probs,
            title="Emotion Confidence", color=emotion_labels
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("👤 Gender Prediction")
        st.info(f"**{gender_pred}**")
        fig2 = px.bar(
            x=gender_labels, y=gender_probs,
            title="Gender Confidence", color=gender_labels
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ------------------- Spectrogram -------------------
    st.subheader("🎶 Audio Waveform & Spectrogram")
    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    librosa.display.waveshow(y, sr=sr, ax=ax[0], color='purple')
    ax[0].set_title("Waveform")

    
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax[1])
    fig.colorbar(img, ax=ax[1], format='%+2.0f dB')
    ax[1].set_title("Mel Spectrogram")
    st.pyplot(fig)

else:
    st.info("⬆️ Please upload an audio file to begin.")
