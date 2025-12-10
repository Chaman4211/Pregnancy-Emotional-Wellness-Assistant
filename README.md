**🤰 Pregnancy Emotional Wellness Assistant**


![Python](https://img.shields.io/badge/python-3.10-blue)
![AI Model Ready](https://img.shields.io/badge/AI_Model-Ready-green)
![Dataset Included](https://img.shields.io/badge/Dataset-Included-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)


Your AI-powered emotional wellness and pregnancy support system using voice analysis, text mood detection, symptom tracking, and interactive dashboarding — built using Streamlit.

**🌟 Project Overview**
Pregnancy is an emotional journey. This project provides a supportive companion for expecting mothers by analyzing their voice, text, and symptoms to track emotional wellbeing and baby development.

**The system uses:**

🔊 Voice Emotion Detection (librosa-based audio feature extraction)

📝 Text-based Mood Detection

📊 Interactive Dashboard (Plotly visualizations)

📅 Symptom & Daily Check-ins

👶 Baby Week-by-Week Development Information

📄 Automatic Report Generation

💾 Local Data Privacy — nothing stored on server

**🚀 Live Demo (Streamlit Cloud)**

👉 https://pregnancy-emotional-wellness-assistant-rvpslhpnlc3rtzrrycsmk4.streamlit.app/

**🧠 Features**

**🎤 Voice Emotion Analysis**

Upload an audio file (WAV/MP3)

Extract MFCC, pitch, energy, ZCR

Predict simplified emotional state (Calm, Happy, Tired, Anxious, Energetic, Emotional)

Visualize emotion confidence scores

Add results to emotional history

**📝 Text Emotion Analysis**

Type how you feel

Keyword-based emotional scoring

Stores check-ins for history tracking

**📊 Dashboard**

Emotion timeline

Emotion distribution

Quick stats

Recent activities

**👶 Baby Development Tracking**

Week-by-week fetal development info

Size comparison (fruit/vegetable model)

Baby kick counter

Trimester-specific tips

**🩺 Symptom Tracking**

Log symptoms with severity

Plot symptom frequency (bar chart)

Daily mood and energy check-ins

**📄 Exportable Reports**

Generate weekly/monthly/trimester summaries

Download JSON reports

Export complete local dataset

**🔒 Privacy**

Your data is stored ONLY in browser session state — not uploaded anywhere.

**🛠️ Tech Stack**

Frontend / App

Streamlit

Plotly

HTML/CSS (custom styling)

Machine Learning / Audio

Librosa

Numpy / Pandas

Scikit-learn (optional for future model loading)

Visualization

Plotly (line charts, pie charts, bars)


**⚙️ Installation (Local Machine)**

**1️⃣ Clone the repository**

bash

Copy code

git clone https://github.com/Chaman4211/Pregnancy-Emotional-Wellness-Assistant.git

cd Pregnancy-Emotional-Wellness-Assistant

**2️⃣ Install dependencies**

bash

Copy code

pip install -r requirements.txt

**3️⃣ Run the app**

bash

Copy code

streamlit run app.py


**📦 Model Integration (Optional)**

You can plug in your trained emotion-classification model (e.g., CNN, RNN, MFCC-based classifier) by replacing the predict_emotion() function with your model loading + inference code.

If you want help integrating your real model, I can write that code for you.

**👩‍⚕️ Disclaimer**

This tool is for emotional wellness support only.

It does not provide medical advice.

Always consult healthcare professionals for medical concerns.

**🤝 Contributing**

Contributions are welcome!

Feel free to open an issue or pull request.

**📜 License**

MIT License

**💖 Thank You**

Supporting maternal mental health through technology.

