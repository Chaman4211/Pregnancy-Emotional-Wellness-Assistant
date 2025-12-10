# app.py - Corrected Streamlit App
import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import librosa
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import datetime
import tempfile
import json
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pregnancy Wellness Assistant",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/pregnancy-wellness',
        'Report a bug': "https://github.com/yourusername/pregnancy-wellness/issues",
        'About': """
        # Pregnancy Emotional Wellness Assistant
        Supporting maternal mental health through voice analysis and emotional tracking.
        
        **Disclaimer**: This tool provides emotional support only, not medical advice.
        Always consult healthcare providers for medical concerns.
        """
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF69B4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #9370DB;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #FF9800;
        margin: 1rem 0;
    }
    .baby-box {
        background-color: #F3E5F5;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #9C27B0;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(0,0,0,0.1);
    }
    .feature-icon {
        font-size: 2rem;
        margin-right: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'trimester': 2,
        'weeks_pregnant': 20,
        'baby_name': 'Little One',
        'due_date': None,
        'emotion_history': [],
        'symptom_log': [],
        'checkins': [],
        'baby_milestones': []
    }

if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = None

if 'recordings' not in st.session_state:
    st.session_state.recordings = []

# Helper Functions
def extract_features(audio_file):
    """Extract audio features for emotion detection"""
    try:
        # Load audio
        y, sr = librosa.load(audio_file, sr=22050, duration=3)
        
        # Extract basic features
        features = {}
        
        # MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
        
        # Energy
        rms = librosa.feature.rms(y=y)
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        
        return features
        
    except Exception as e:
        st.error(f"Error extracting features: {e}")
        return None

def predict_emotion(features):
    """Predict emotion from audio features"""
    # Simplified emotion prediction for demo
    # In production, load a trained model
    
    # Calculate emotion scores based on features
    emotion_scores = {
        'Calm': 0.3,
        'Happy': 0.2,
        'Tired': 0.15,
        'Anxious': 0.15,
        'Energetic': 0.1,
        'Emotional': 0.1
    }
    
    # Adjust based on features
    if 'pitch_mean' in features:
        if features['pitch_mean'] > 300:
            emotion_scores['Anxious'] += 0.2
            emotion_scores['Energetic'] += 0.1
    
    if 'energy_mean' in features:
        if features['energy_mean'] < 0.05:
            emotion_scores['Tired'] += 0.2
        elif features['energy_mean'] > 0.1:
            emotion_scores['Energetic'] += 0.15
    
    # Normalize scores
    total = sum(emotion_scores.values())
    normalized_scores = {k: v/total for k, v in emotion_scores.items()}
    
    # Get dominant emotion
    dominant_emotion = max(normalized_scores.items(), key=lambda x: x[1])
    
    return dominant_emotion[0], normalized_scores

def get_recommendations(emotion, trimester):
    """Get personalized recommendations"""
    recommendations = {
        'Calm': [
            "Enjoy this peaceful moment",
            "Practice gratitude journaling",
            "Share your calm feelings with your partner"
        ],
        'Happy': [
            "Capture this happy moment in your pregnancy journal",
            "Share the joy with loved ones",
            "Do something special to celebrate"
        ],
        'Tired': [
            "Rest when you can - your body is doing important work",
            "Drink plenty of water and have a healthy snack",
            "Take a short nap if possible"
        ],
        'Anxious': [
            "Practice deep breathing: 4 seconds in, 7 hold, 8 out",
            "Write down your worries in a pregnancy journal",
            "Talk to your partner about how you're feeling"
        ],
        'Energetic': [
            "Use this energy for gentle exercise like walking",
            "Prepare something for the baby's arrival",
            "Enjoy activities you love"
        ],
        'Emotional': [
            "Allow yourself to feel your emotions",
            "Talk to a supportive friend or family member",
            "Write about your feelings in a journal"
        ]
    }
    
    trimester_tips = {
        1: ["Focus on hydration", "Eat small, frequent meals", "Rest often"],
        2: ["Enjoy your increased energy", "Start prenatal yoga", "Begin planning for baby"],
        3: ["Practice relaxation techniques", "Prepare your hospital bag", "Rest frequently"]
    }
    
    base_recs = recommendations.get(emotion, ["Be kind to yourself today"])
    trimester_recs = trimester_tips.get(trimester, [])
    
    return base_recs + trimester_recs[:2]

def get_baby_info(week):
    """Get baby development information"""
    milestones = {
        4: "Size of a poppy seed. Neural tube begins to form.",
        8: "Size of a raspberry. All major organs have begun to form.",
        12: "Size of a lime. Fingers and toes are fully formed.",
        16: "Size of an avocado. Can make sucking motions.",
        20: "Size of a banana. You might feel movement.",
        24: "Size of an ear of corn. Lungs are developing.",
        28: "Size of an eggplant. Eyes can open and close.",
        32: "Size of a squash. Kicking and moving frequently.",
        36: "Size of a head of romaine. Settling into birth position.",
        40: "Size of a small pumpkin. Ready for birth!"
    }
    
    closest_week = min(milestones.keys(), key=lambda x: abs(x - week))
    return milestones[closest_week]

def create_emotion_chart(emotion_history):
    """Create emotion timeline chart"""
    if not emotion_history:
        return None
    
    df = pd.DataFrame(emotion_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    fig = go.Figure()
    
    # Add line for emotion scores
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['confidence'],
        mode='lines+markers',
        name='Emotion Confidence',
        line=dict(color='#FF69B4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Emotional Wellness Timeline",
        xaxis_title="Date",
        yaxis_title="Confidence",
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_symptom_chart(symptom_log):
    """Create symptom tracking chart"""
    if not symptom_log:
        return None
    
    df = pd.DataFrame(symptom_log)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Group by symptom
    symptom_counts = df['symptom'].value_counts().reset_index()
    symptom_counts.columns = ['symptom', 'count']
    
    fig = px.bar(
        symptom_counts,
        x='symptom',
        y='count',
        color='symptom',
        title="Symptom Frequency",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_wellness_summary():
    """Create wellness summary metrics"""
    if not st.session_state.user_data['emotion_history']:
        return None
    
    emotions = pd.DataFrame(st.session_state.user_data['emotion_history'])
    
    summary = {
        'total_checkins': len(emotions),
        'most_common_emotion': emotions['emotion'].mode()[0] if not emotions.empty else "N/A",
        'average_confidence': emotions['confidence'].mean() if not emotions.empty else 0,
        'last_checkin': emotions['timestamp'].max() if not emotions.empty else "N/A"
    }
    
    return summary

# Main App Layout
def main():
    # Header
    st.markdown('<h1 class="main-header">🤰 Pregnancy Emotional Wellness Assistant</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Use an emoji or local image instead of URL
        st.markdown("### 🤱 Your Pregnancy Details")
        
        # User inputs
        trimester = st.selectbox(
            "Current Trimester",
            [1, 2, 3],
            index=st.session_state.user_data['trimester'] - 1
        )
        
        weeks = st.slider(
            "Weeks Pregnant",
            min_value=4,
            max_value=42,
            value=st.session_state.user_data['weeks_pregnant']
        )
        
        baby_name = st.text_input(
            "Baby's Name (optional)",
            value=st.session_state.user_data['baby_name']
        )
        
        # Update session state
        st.session_state.user_data['trimester'] = trimester
        st.session_state.user_data['weeks_pregnant'] = weeks
        st.session_state.user_data['baby_name'] = baby_name
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 📊 Quick Stats")
        summary = create_wellness_summary()
        if summary:
            st.metric("Total Check-ins", summary['total_checkins'])
            st.metric("Most Common Emotion", summary['most_common_emotion'])
            if summary['average_confidence'] > 0:
                st.metric("Average Confidence", f"{summary['average_confidence']:.0%}")
        
        st.markdown("---")
        
        # Emergency Info
        with st.expander("🆘 Emergency Support"):
            st.info("""
            **Seek immediate medical attention for:**
            - Severe abdominal pain
            - Heavy bleeding
            - Decreased fetal movement
            - Signs of preeclampsia
            
            **Emergency: 911** (US) or local emergency number
            """)
        
        # About
        with st.expander("ℹ️ About This App"):
            st.write("""
            This app helps track emotional wellbeing during pregnancy
            through voice analysis and mood tracking.
            
            **Features:**
            - Voice emotion analysis
            - Symptom tracking
            - Baby development info
            - Personalized recommendations
            
            **Privacy:** All data stays in your browser.
            """)
    
    # Main Content
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎤 Voice Check-in", 
        "📊 Dashboard", 
        "👶 Baby Info", 
        "📝 Symptom Tracker",
        "📄 Reports"
    ])
    
    # Tab 1: Voice Check-in
    with tab1:
        st.markdown('<h2 class="sub-header">Voice Emotional Check-in</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-box">
            <strong>How it works:</strong><br>
            1. Record your voice or upload an audio file<br>
            2. The app analyzes your emotional state<br>
            3. Get personalized recommendations<br>
            4. Track your emotional wellness over time
            </div>
            """, unsafe_allow_html=True)
            
            # Audio input options
            input_method = st.radio(
                "Choose input method:",
                ["Upload Audio File", "Describe Your Feelings"]  # Removed Record Audio for now
            )
            
            if input_method == "Upload Audio File":
                uploaded_file = st.file_uploader(
                    "Upload an audio file (WAV, MP3)", 
                    type=['wav', 'mp3', 'm4a']
                )
                
                if uploaded_file:
                    # Save uploaded file
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Analyze audio
                    with st.spinner("Analyzing your emotions..."):
                        features = extract_features(tmp_path)
                        
                        if features:
                            emotion, scores = predict_emotion(features)
                            
                            # Update session state
                            st.session_state.current_emotion = {
                                'emotion': emotion,
                                'confidence': scores[emotion],
                                'scores': scores,
                                'timestamp': datetime.datetime.now().isoformat()
                            }
                            
                            # Add to history
                            st.session_state.user_data['emotion_history'].append(
                                st.session_state.current_emotion
                            )
                            
                            # Clean up temp file
                            os.unlink(tmp_path)
            
            else:  # Describe Your Feelings
                user_input = st.text_area("How are you feeling today?", height=100)
                
                if st.button("Analyze Text", use_container_width=True) and user_input:
                    # Simple text-based emotion analysis
                    emotion_keywords = {
                        'happy': ['happy', 'joy', 'excited', 'good', 'great'],
                        'calm': ['calm', 'peaceful', 'relaxed', 'content'],
                        'tired': ['tired', 'exhausted', 'sleepy', 'fatigued'],
                        'anxious': ['anxious', 'worried', 'nervous', 'stressed'],
                        'emotional': ['emotional', 'sensitive', 'moody', 'tearful']
                    }
                    
                    # Count keyword matches
                    emotion_scores = {}
                    for emotion, keywords in emotion_keywords.items():
                        count = sum(1 for keyword in keywords if keyword in user_input.lower())
                        emotion_scores[emotion] = count
                    
                    if sum(emotion_scores.values()) > 0:
                        # Get dominant emotion
                        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])
                        emotion = dominant_emotion[0].capitalize()
                        confidence = dominant_emotion[1] / len(emotion_keywords[dominant_emotion[0]])
                        
                        # Update session state
                        st.session_state.current_emotion = {
                            'emotion': emotion,
                            'confidence': confidence,
                            'scores': {k.capitalize(): v/sum(emotion_scores.values()) if sum(emotion_scores.values()) > 0 else 0 
                                      for k, v in emotion_scores.items()},
                            'timestamp': datetime.datetime.now().isoformat(),
                            'source': 'text'
                        }
                        
                        # Add to history
                        st.session_state.user_data['emotion_history'].append(
                            st.session_state.current_emotion
                        )
                    else:
                        st.info("Could not detect specific emotions from your text. Please try to describe your feelings in more detail.")
        
        with col2:
            # Display results
            if st.session_state.current_emotion:
                emotion = st.session_state.current_emotion['emotion']
                confidence = st.session_state.current_emotion['confidence']
                
                st.markdown(f"""
                <div class="success-box">
                <h3>🎯 Analysis Complete!</h3>
                <strong>Emotion:</strong> {emotion}<br>
                <strong>Confidence:</strong> {confidence:.0%}<br>
                <strong>Time:</strong> {datetime.datetime.now().strftime('%H:%M')}
                </div>
                """, unsafe_allow_html=True)
                
                # Emotion scores chart
                scores = st.session_state.current_emotion['scores']
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(scores.keys()),
                        y=list(scores.values()),
                        marker_color=['#FF69B4' if k == emotion else '#9370DB' for k in scores.keys()]
                    )
                ])
                
                fig.update_layout(
                    title="Emotion Analysis",
                    height=300,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Get recommendations
                recommendations = get_recommendations(emotion, trimester)
                
                st.markdown("### 💡 Recommendations")
                for i, rec in enumerate(recommendations[:3], 1):
                    st.write(f"{i}. {rec}")
                
                # Baby bonding suggestion
                if baby_name != "Little One":
                    st.markdown(f"""
                    <div class="baby-box">
                    <strong>👶 Baby Connection:</strong><br>
                    Take a moment to talk to {baby_name} about how you're feeling today.
                    </div>
                    """, unsafe_allow_html=True)
    
    # Tab 2: Dashboard
    with tab2:
        st.markdown('<h2 class="sub-header">Wellness Dashboard</h2>', unsafe_allow_html=True)
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        summary = create_wellness_summary()
        if summary:
            with col1:
                st.metric("Total Check-ins", summary['total_checkins'])
            with col2:
                st.metric("Most Common Emotion", summary['most_common_emotion'])
            with col3:
                st.metric("Average Confidence", f"{summary['average_confidence']:.0%}")
            with col4:
                st.metric("Current Week", weeks)
        
        # Charts row
        col1, col2 = st.columns(2)
        
        with col1:
            # Emotion timeline
            fig1 = create_emotion_chart(st.session_state.user_data['emotion_history'])
            if fig1:
                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.info("No emotion data yet. Complete a voice check-in!")
        
        with col2:
            # Emotion distribution
            if st.session_state.user_data['emotion_history']:
                emotions = pd.DataFrame(st.session_state.user_data['emotion_history'])
                emotion_counts = emotions['emotion'].value_counts()
                
                fig2 = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="Emotion Distribution",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                
                fig2.update_layout(height=400)
                st.plotly_chart(fig2, use_container_width=True)
        
        # Recent activity
        st.markdown("### 📝 Recent Activity")
        if st.session_state.user_data['emotion_history']:
            recent = st.session_state.user_data['emotion_history'][-5:]
            for item in reversed(recent):
                date = datetime.datetime.fromisoformat(item['timestamp']).strftime('%b %d, %H:%M')
                st.write(f"**{date}**: {item['emotion']} ({item['confidence']:.0%} confidence)")
        else:
            st.info("Complete your first voice check-in to see activity here!")
    
    # Tab 3: Baby Info
    with tab3:
        st.markdown('<h2 class="sub-header">Baby Development Information</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Baby development info
            milestone = get_baby_info(weeks)
            
            st.markdown(f"""
            <div class="baby-box">
            <h3>👶 Week {weeks} Update</h3>
            <strong>Your baby is about the size of:</strong><br>
            {milestone}
            </div>
            """, unsafe_allow_html=True)
            
            # Week slider for exploring
            explore_week = st.slider(
                "Explore different weeks:",
                min_value=4,
                max_value=40,
                value=weeks
            )
            
            if explore_week != weeks:
                st.write(f"**Week {explore_week}**: {get_baby_info(explore_week)}")
        
        with col2:
            # Baby size comparison
            size_info = {
                4: "Poppy Seed",
                8: "Raspberry", 
                12: "Lime",
                16: "Avocado",
                20: "Banana",
                24: "Corn",
                28: "Eggplant",
                32: "Squash",
                36: "Romaine",
                40: "Pumpkin"
            }
            
            closest_size = min(size_info.keys(), key=lambda x: abs(x - weeks))
            
            st.markdown(f"""
            <div class="metric-card">
            <h3>📏 Size Comparison</h3>
            <p style="font-size: 1.2rem; text-align: center;">
            {size_info[closest_size]}
            </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Trimester info
            trimester_info = {
                1: "Focus on rest and nutrition. Morning sickness common.",
                2: "Energy often returns. You may feel baby movements.",
                3: "Prepare for birth. Rest often as baby grows."
            }
            
            st.markdown(f"""
            <div class="info-box">
            <strong>Trimester {trimester} Tips:</strong><br>
            {trimester_info.get(trimester, "Listen to your body.")}
            </div>
            """, unsafe_allow_html=True)
        
        # Baby kick counter (simplified)
        st.markdown("### 👣 Baby Movement Tracker")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💖 Log Kick", use_container_width=True):
                timestamp = datetime.datetime.now().strftime('%H:%M')
                st.session_state.user_data.setdefault('baby_kicks', []).append(timestamp)
                st.success(f"Kick logged at {timestamp}")
        
        with col2:
            kick_count = len(st.session_state.user_data.get('baby_kicks', []))
            st.metric("Today's Kicks", kick_count)
        
        with col3:
            if st.button("🔄 Reset Counter", use_container_width=True):
                st.session_state.user_data['baby_kicks'] = []
                st.rerun()
        
        # Display recent kicks
        if 'baby_kicks' in st.session_state.user_data and st.session_state.user_data['baby_kicks']:
            st.write("**Recent movements:**")
            for kick in st.session_state.user_data['baby_kicks'][-5:]:
                st.write(f"• {kick}")
    
    # Tab 4: Symptom Tracker
    with tab4:
        st.markdown('<h2 class="sub-header">Symptom & Wellness Tracker</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Symptom logging form
            st.markdown("### 📝 Log Symptoms")
            
            symptom = st.selectbox(
                "Select symptom:",
                ["Nausea", "Fatigue", "Back Pain", "Swelling", 
                 "Headache", "Heartburn", "Other"]
            )
            
            if symptom == "Other":
                symptom = st.text_input("Enter symptom:")
            
            severity = st.slider("Severity (1-10)", 1, 10, 5)
            notes = st.text_area("Notes (optional)")
            
            if st.button("➕ Log Symptom", use_container_width=True):
                symptom_entry = {
                    'symptom': symptom,
                    'severity': severity,
                    'notes': notes,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'week': weeks
                }
                
                st.session_state.user_data['symptom_log'].append(symptom_entry)
                st.success(f"Logged: {symptom} (Severity: {severity}/10)")
                st.rerun()
            
            # Daily check-in
            st.markdown("---")
            st.markdown("### 🌅 Daily Check-in")
            
            mood = st.select_slider(
                "Today's mood:",
                options=["😢 Sad", "😕 Okay", "😊 Good", "😄 Great", "🤩 Excellent"]
            )
            
            energy = st.slider("Energy level (1-10)", 1, 10, 7)
            
            if st.button("✅ Complete Daily Check-in", use_container_width=True):
                checkin = {
                    'date': datetime.datetime.now().strftime('%Y-%m-%d'),
                    'mood': mood,
                    'energy': energy,
                    'week': weeks
                }
                
                st.session_state.user_data['checkins'].append(checkin)
                st.success("Daily check-in completed!")
        
        with col2:
            # Symptom visualization
            fig = create_symptom_chart(st.session_state.user_data['symptom_log'])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No symptoms logged yet")
            
            # Recent symptoms
            if st.session_state.user_data['symptom_log']:
                st.markdown("### 📋 Recent Symptoms")
                recent_symptoms = st.session_state.user_data['symptom_log'][-5:]
                
                for symptom in reversed(recent_symptoms):
                    date = datetime.datetime.fromisoformat(symptom['timestamp']).strftime('%b %d')
                    st.write(f"**{date}**: {symptom['symptom']} (Severity: {symptom['severity']}/10)")
                    if symptom.get('notes'):
                        st.caption(f"Notes: {symptom['notes']}")
    
    # Tab 5: Reports
    with tab5:
        st.markdown('<h2 class="sub-header">Reports & Export</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Generate Report")
            
            report_type = st.selectbox(
                "Select report type:",
                ["Weekly Summary", "Monthly Summary", "Trimester Summary", "Custom Period"]
            )
            
            if st.button("📊 Generate Report", use_container_width=True):
                # Create report
                report_data = {
                    'report_type': report_type,
                    'generated_date': datetime.datetime.now().isoformat(),
                    'pregnancy_info': {
                        'trimester': trimester,
                        'week': weeks,
                        'baby_name': baby_name
                    },
                    'wellness_summary': create_wellness_summary(),
                    'recent_emotions': st.session_state.user_data['emotion_history'][-10:] if st.session_state.user_data['emotion_history'] else [],
                    'recent_symptoms': st.session_state.user_data['symptom_log'][-10:] if st.session_state.user_data['symptom_log'] else []
                }
                
                # Display report
                st.markdown("### 📋 Report Preview")
                st.json(report_data, expanded=False)
                
                # Download button
                report_json = json.dumps(report_data, indent=2, default=str)
                st.download_button(
                    label="⬇️ Download Report (JSON)",
                    data=report_json,
                    file_name=f"pregnancy_report_{datetime.datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.markdown("### 💾 Data Management")
            
            # Export all data
            if st.button("📤 Export All Data", use_container_width=True):
                export_data = {
                    'user_data': st.session_state.user_data,
                    'export_date': datetime.datetime.now().isoformat(),
                    'app_version': '1.0.0'
                }
                
                export_json = json.dumps(export_data, indent=2, default=str)
                
                st.download_button(
                    label="⬇️ Download Full Data",
                    data=export_json,
                    file_name=f"pregnancy_data_export_{datetime.datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
            
            # Reset data
            st.markdown("---")
            st.markdown("### 🗑️ Data Management")
            
            if st.button("🔄 Reset All Data", type="secondary", use_container_width=True):
                st.session_state.user_data = {
                    'trimester': 2,
                    'weeks_pregnant': 20,
                    'baby_name': 'Little One',
                    'due_date': None,
                    'emotion_history': [],
                    'symptom_log': [],
                    'checkins': [],
                    'baby_milestones': []
                }
                st.session_state.current_emotion = None
                st.success("Data reset successfully!")
                st.rerun()
        
        # Data backup reminder
        st.markdown("""
        <div class="warning-box">
        <strong>💡 Important:</strong><br>
        - Reports are generated in your browser<br>
        - Data is stored locally in your browser's session<br>
        - Export data regularly to prevent loss<br>
        - This app does not store data on external servers
        </div>
        """, unsafe_allow_html=True)

# Footer
def footer():
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.caption("🤰 Pregnancy Wellness Assistant")
        st.caption("Version 1.0.0")
    
    with col2:
        st.caption("💖 Supporting maternal mental health")
        st.caption("Emotional support only, not medical advice")
    
    with col3:
        st.caption("🔒 Your data stays in your browser")
        st.caption("Built with Streamlit")

# Run the app
if __name__ == "__main__":
    main()
    footer()