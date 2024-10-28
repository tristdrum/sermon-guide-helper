import streamlit as st
from openai import OpenAI
import tempfile
import os
from dotenv import load_dotenv
import pyperclip  # Add this import

# Load environment variables and setup
load_dotenv()
st.set_page_config(page_title="Audio Transcription App")
st.title("Audio Transcription App")

# Initialize OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OpenAI API key not found. Please check your .env file.")
    st.stop()
client = OpenAI(api_key=api_key)

# Initialize session state
if 'transcription' not in st.session_state:
    st.session_state.transcription = None

# Initialize session state for upload counter
if 'upload_counter' not in st.session_state:
    st.session_state.upload_counter = 0

def transcribe_audio(file_path):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
        return transcript

# File uploader with dynamic key
uploaded_file = st.file_uploader("Upload an audio file", 
    type=['mp3', 'wav', 'mpeg', 'm4a'],
    key=f"uploader_{st.session_state.upload_counter}")

if uploaded_file:
    # Add model selector after successful file upload
    model = st.radio(
        "Select Transcription Model",
        ["Fastest (whisper-1)", "Balanced (whisper-medium)", "Highest Quality (whisper-large-v3)"],
        horizontal=True,
        index=0  # Default to fastest model
    )

    # Convert display name to API model name
    model_map = {
        "Fastest (whisper-1)": "whisper-1",
        "Balanced (whisper-medium)": "whisper-medium",
        "Highest Quality (whisper-large-v3)": "whisper-large-v3"
    }

    # Add transcribe button - full width with updated text
    if st.button("Start Transcribing", use_container_width=True):
        if not st.session_state.transcription:  # Only transcribe if we haven't already
            with st.spinner('Transcribing audio...'):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    # Use selected model for transcription
                    st.session_state.transcription = transcribe_audio(tmp_file_path, model_map[model])
                finally:
                    os.unlink(tmp_file_path)

    # Display transcription
    if st.session_state.transcription:
        st.text_area("Transcription", st.session_state.transcription, height=300)
        
        if st.button("Copy to Clipboard"):
            pyperclip.copy(st.session_state.transcription)
            st.success("✓ Copied to clipboard!")  # Using success for better visual feedback

    # Only show Reset button if a file was uploaded
    if st.button("Reset"):
        # Increment counter to force new file uploader instance
        st.session_state.upload_counter += 1
        # Clear transcription
        st.session_state.transcription = None
        # Rerun the app
        st.rerun()
