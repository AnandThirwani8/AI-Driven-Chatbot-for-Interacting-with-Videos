#---------------------------------------------- Imports ----------------------------------------------

import streamlit as st
import tempfile
from utils import *

def extract_and_display_clip(video_path, timestamp_in_sec, play_before_sec=10, play_after_sec=10):
    """Extract and save a video clip around the specified timestamp."""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        
        # Define start and end times
        start_time = max(timestamp_in_sec - play_before_sec, 0)
        end_time = min(timestamp_in_sec + play_after_sec, duration)
        
        # Extract subclip
        clip = video.subclipped(start_time, end_time)
        
        # Ensure temp directory exists
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate unique filename to avoid conflicts
        clip_path = os.path.join(temp_dir, f"clip_{timestamp_in_sec}.mp4")
        clip.write_videofile(clip_path, codec="libx264", fps=24)
        
        # Clean up resources
        clip.close()
        video.close()
        
        return clip_path
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None

def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]


#---------------------------------------------- Side Bar (Key) ----------------------------------------------

st.set_page_config(page_title="Chat with your Videos", page_icon="🎬 🎥 🖥️")

# Create a sidebar for entering the Gemini API key.
with st.sidebar:
    st.markdown("# Enter your Gemini API Key here:")
    gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")
    st.markdown("# Upload your video file:")
    uploaded_file = st.file_uploader("Upload a video file", type="mp4")

    path_to_video = 'sample_videos/fire2.mp4'
    vid_metadata = get_video_trascript(path_to_video, num_of_extracted_frames_per_second = 1/20)
    vid_trans = [vid['transcript'] for vid in vid_metadata]
    vid_frames = [base64_to_image(vid['extracted_frame_base64']) for vid in vid_metadata]
    embedding_model = BridgeTowerEmbeddings()
    embeddings = embedding_model.embed_image_text_pairs(texts=vid_trans, images=vid_frames)
    retriever = get_vectorstore(vid_metadata, embeddings)

    # if uploaded_file is not None:
    #     # Save the uploaded file to a temporary file
    #     with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
    #         tmp_file.write(uploaded_file.getvalue())
    #         csv_path = tmp_file.name

    #         st.success("CSV file uploaded successfully!")
    #         # Set up the database and the agent using the temporary CSV file
    #         engine = setup_database(csv_path, uploaded_file.name)
    #         if not gemini_api_key:
    #             st.warning('Please enter your API key!', icon='⚠')
    #             st.stop()
    #         agent_executor, query_logger = setup_agent(engine, gemini_api_key)
    #         st.success("Agent is set up and ready to answer your questions!")
    
    "[🔑 Get your Gemini API key](https://ai.google.dev/gemini-api/docs)"
    "[👨‍💻 View the source code](https://github.com/AnandThirwani8/Agentic-AI-Driven-Chat-with-SQL-Database)"
    "[🤝 Let's Connect](https://www.linkedin.com/in/anandthirwani/)"
    
    st.markdown("---")
    st.markdown("# About")
    st.markdown(
        "🚀"
    )
    st.markdown(
        "You can contribute to the project on [GitHub](https://github.com/AnandThirwani8/Agentic-AI-Driven-Chat-with-SQL-Database) "  
        "with your feedback and suggestions💡"
    )

#---------------------------------------------- Chat UI ----------------------------------------------

# Set up the web application interface
st.title("Video Chat Assistant 🎬 🎥 🖥️ ")
st.caption("Ask questions about the video or request to see specific segments")

# Initialize session state
initialize_session_state()

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    # If this message contains a video path, display the video
    if msg.get("video_path") and os.path.exists(msg["video_path"]):
        with st.chat_message(msg["role"]):
            st.video(msg["video_path"])

# Chat input
if query := st.chat_input():
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").write(query)
    
    with st.spinner("Processing..."):
        answer, extracted_trascript, extracted_path, extracted_timestamp = QnA(query, retriever)
        clip_path = extract_and_display_clip(extracted_path, timestamp_in_sec=extracted_timestamp)
        if clip_path:
            msg = {"role": "assistant", "content": answer, "video_path": clip_path}
        else:
            msg = {"role": "assistant", "content": "Sorry, I couldn't process the video clip."}

        # Add assistant's response to chat history
        st.session_state.messages.append(msg)
        
        # Display the response
        st.chat_message("assistant").write(msg["content"])
        if msg.get("video_path"):
            with st.chat_message("assistant"):
                st.video(msg["video_path"])
        