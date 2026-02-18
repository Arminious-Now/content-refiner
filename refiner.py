import streamlit as st
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from pypdf import PdfReader
import io
import re

# --- PAGE CONFIG ---
st.set_page_config(page_title="Content Refiner", page_icon="💎", layout="wide")

st.title("💎 AI Content Refiner")
st.markdown("""
**Goal:** Ingest raw, messy content (Books/Videos) and output a **structured, high-density Knowledge File**.
*Optimized for preserving details, nuance, and technical steps.*
""")

# --- SIDEBAR: CONFIG ---
with st.sidebar:
    st.header("⚙️ Settings")
    
    if 'GEMINI_KEY' in st.secrets:
        api_key = st.secrets['GEMINI_KEY']
    else:
        api_key = st.text_input("Gemini API Key", type="password")

    # Model Selection
    model_name = st.selectbox(
        "Model", 
        ["gemini-2.0-flash", "gemini-1.5-pro"], 
        index=0,
        help="Flash is faster. Pro is better for very complex reasoning."
    )
    
    st.divider()
    st.info("Tip: This tool uses the 'Long Context' window to process the entire file at once.")

# --- FUNCTIONS ---

def extract_youtube_text(url):
    try:
        video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url).group(1)
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t['text'] for t in transcript]), video_id
    except Exception as e:
        st.error(f"YouTube Error: {e}")
        return None, None

def extract_pdf_text(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return None

def refine_content(raw_text, content_type):
    """
    Sends the full text to Gemini with a 'Refiner' prompt.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # THE ARCHITECT PROMPT
    # This is designed to PRESERVE detail, not just summarize.
    sys_prompt = f"""
    You are an expert Knowledge Architect. Your goal is to convert the following raw {content_type} 
    into a structured, high-density reference document.

    RULES:
    1. DO NOT summarize broadly. Retain specific technical details, numbers, step-by-step instructions, and unique examples.
    2. RESTRUCTURE the content logically. Use H1, H2, and H3 headers.
    3. If there are distinct concepts, separate them into clear sections.
    4. Capture the "Why" and "How", not just the "What".
    5. If the content is a dialogue, extract the key arguments/lessons rather than transcribing the chat.
    6. Output format: Markdown.

    RAW CONTENT STARTS BELOW:
    --------------------------
    {raw_text}
    """
    
    try:
        # Stream the response so you don't have to wait for the whole thing
        response = model.generate_content(sys_prompt, stream=True)
        return response
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return None

# --- MAIN UI ---

tab1, tab2 = st.tabs(["📺 YouTube Video", "Cc📄 PDF Document"])

# TAB 1: YOUTUBE
with tab1:
    yt_url = st.text_input("Enter YouTube URL")
    if st.button("Refine Video", type="primary"):
        if not api_key:
            st.warning("Please enter API Key in sidebar.")
        elif not yt_url:
            st.warning("Please enter a URL.")
        else:
            with st.spinner("Extracting Transcript..."):
                raw_text, vid_id = extract_youtube_text(yt_url)
            
            if raw_text:
                st.info(f"Transcript Extracted ({len(raw_text)} characters). Sending to Gemini...")
                
                output_container = st.empty()
                full_response = ""
                
                # Stream the refinement
                stream = refine_content(raw_text, "Video Transcript")
                if stream:
                    for chunk in stream:
                        full_response += chunk.text
                        output_container.markdown(full_response)
                    
                    # Download Button
                    st.download_button(
                        label="Download Refined Notes",
                        data=full_response,
                        file_name=f"refined_video_{vid_id}.md",
                        mime="text/markdown"
                    )

# TAB 2: PDF FILES
with tab2:
    uploaded_file = st.file_uploader("Upload PDF", type=['pdf'])
    if st.button("Refine Document", type="primary"):
        if not api_key:
            st.warning("Please enter API Key in sidebar.")
        elif not uploaded_file:
            st.warning("Please upload a file.")
        else:
            with st.spinner("Extracting Text..."):
                raw_text = extract_pdf_text(uploaded_file)
            
            if raw_text:
                st.info(f"Text Extracted ({len(raw_text)} characters). Sending to Gemini...")
                
                output_container = st.empty()
                full_response = ""
                
                # Stream the refinement
                stream = refine_content(raw_text, "PDF Document")
                if stream:
                    for chunk in stream:
                        full_response += chunk.text
                        output_container.markdown(full_response)
                    
                    # Download Button
                    st.download_button(
                        label="Download Refined Notes",
                        data=full_response,
                        file_name=f"refined_{uploaded_file.name}.md",
                        mime="text/markdown"
                    )
