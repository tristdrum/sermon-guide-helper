import streamlit as st
from openai import OpenAI
import tempfile
import os
from dotenv import load_dotenv
import pyperclip  # Add this import

# Load environment variables and setup
load_dotenv()
st.set_page_config(page_title="Audio Transcription App")

# Custom CSS for button colors
st.markdown("""
    <style>
        /* Start Transcribing button - Blue */
        .stButton button[kind="primary"] {
            background-color: rgba(0, 123, 255, 0.8);
            color: white;
            border: 1px solid rgba(0, 123, 255, 0.8);
            transition: all 0.3s ease;
        }
        .stButton button[kind="primary"]:hover {
            background-color: rgba(40, 167, 69, 1);
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        /* Copy to Clipboard button - Blue */
        .stButton button[data-testid="baseButton-secondary"] {
            background-color: rgba(0, 123, 255, 0.8);
            color: white;
            transition: all 0.3s ease;
        }
        .stButton button[data-testid="baseButton-secondary"]:hover {
            background-color: rgba(0, 123, 255, 1);
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        /* Reset button - Subtle Red */
        .stButton button[kind="secondary"] {
            background-color: rgba(220, 53, 69, 0.6);
            color: white;
            transition: all 0.3s ease;
        }
        .stButton button[kind="secondary"]:hover {
            background-color: rgba(220, 53, 69, 0.8);
            transform: translateY(-2px);
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }

        /* Text Area - Hide border by default, show blue border on focus */
        .stTextArea textarea,
        div[data-baseweb="textarea"] textarea {
            border: 1px solid transparent !important;
            transition: border-color 0.3s ease !important;
        }
        
        .stTextArea div[data-baseweb="textarea"] {
            border: 1px solid transparent !important;
            transition: border-color 0.3s ease !important;
        }

        .stTextArea textarea:focus,
        div[data-baseweb="textarea"] textarea:focus {
            border-color: rgba(0, 123, 255, 0.8) !important;
        }
        
        .stTextArea div[data-baseweb="textarea"]:focus-within {
            border-color: rgba(0, 123, 255, 0.8) !important;
        }
    </style>
""", unsafe_allow_html=True)

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
if 'current_model' not in st.session_state:
    st.session_state.current_model = None

# Initialize session state for upload counter
if 'upload_counter' not in st.session_state:
    st.session_state.upload_counter = 0

def transcribe_audio(file_path, model="whisper-1"):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="text",  # Changed back to "text"
        )
        paragraphs = [p.strip() for p in str(transcript).split('\n') if p.strip()]
        return paragraphs

# File uploader with dynamic key
uploaded_file = st.file_uploader("Upload an audio file", 
    type=['mp3', 'wav', 'mpeg', 'm4a'],
    key=f"uploader_{st.session_state.upload_counter}")

if uploaded_file:
    # Start transcription immediately after file upload
    if not st.session_state.transcription:
        with st.spinner('Transcribing audio...'):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                # Use whisper-1 model for transcription
                paragraphs = transcribe_audio(tmp_file_path, "whisper-1")
                # Join paragraphs with double newlines
                st.session_state.transcription = "\n\n".join(paragraphs)
            finally:
                os.unlink(tmp_file_path)

    # Display transcription
    if st.session_state.transcription:
        st.text_area("Transcription", st.session_state.transcription, height=300)
        
        # Create two columns with 1:2 ratio
        col1, col2 = st.columns([1, 2])
        
        # Reset button in left column (red)
        with col1:
            if st.button("Reset", 
                        use_container_width=True,
                        type="secondary",  # secondary with custom color
                        help="Clear current transcription and start over"):
                st.session_state.upload_counter += 1
                st.session_state.transcription = None
                st.rerun()

        # Copy button in right column (blue)
        with col2:
            if st.button("Copy to Clipboard", 
                        use_container_width=True,
                        type="primary",  # primary with custom color
                        help="Copy transcription to clipboard"):
                pyperclip.copy(st.session_state.transcription)
                st.success("✓ Copied to clipboard!")
