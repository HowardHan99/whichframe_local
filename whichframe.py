import streamlit as st
import cv2
from PIL import Image
import clip as openai_clip
import torch
import math
from humanfriendly import format_timespan
import numpy as np
import time
import os
import yt_dlp
import io
from datetime import datetime
import shutil

EXAMPLE_URL = "https://www.youtube.com/watch?v=zTvJJnoWIPk"
CACHED_DATA_PATH = "cached_data/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model, preprocess = openai_clip.load("ViT-B/32", device=device)


def fetch_video(source):
    """
    Fetch video from either YouTube URL or local file.
    Args:
        source: Either a YouTube URL or local file path
    Returns:
        tuple: (video_path, video_url) where video_path is None for YouTube videos
    """
    if source.startswith('http'):  # YouTube URL
        try:
            ydl_opts = {
                'format': 'bestvideo[height<=360][ext=mp4][vcodec=avc1]/best[height<=360][ext=mp4]',
                'quiet': True,
                'no_warnings': True
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(source, download=False)
                video_url = info['url']
                return None, video_url
                
        except Exception as e:
            st.error(f"Error fetching video: {str(e)}")
            st.error("Try another YouTube video or check if the URL is correct.")
            st.stop()
    else:  # Local video file
        return source, source

def extract_frames(video, status_text, progress_bar):
    """
    Extract frames from a video at regular intervals.
    Args:
        video: Path to video file or video URL
        status_text: Streamlit text element for status updates
        progress_bar: Streamlit progress bar element
    Returns:
        tuple: (frames, fps, frame_indices) containing extracted frames and metadata
    """
    # For local files, use the path directly
    if isinstance(video, str) and os.path.isfile(video):
        video_path = video
    else:
        video_path = video  # For YouTube URLs or other sources
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")
        
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps/2))  # Extract 2 frames per second
    total_frames = frame_count // step
    frame_indices = []
    
    # Create directory if it doesn't exist
    if not os.path.exists("found_frames"):
        os.makedirs("found_frames")
    
    frame_number = 0
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            frame_indices.append(i)
            
            current_frame = len(frames)
            status_text.text(f'Extracting frames... ({min(current_frame, total_frames)}/{total_frames})')
            progress = min(current_frame / total_frames, 1.0)
            progress_bar.progress(progress)
    
    cap.release()
    return frames, fps, frame_indices

def encode_frames(video_frames, status_text):
    """
    Encode frames using CLIP model for semantic search.
    Args:
        video_frames: List of PIL Image frames
        status_text: Streamlit text element for status updates
    Returns:
        torch.Tensor: Encoded features for all frames
    """
    batch_size = 256  # Process frames in batches to manage memory
    batches = math.ceil(len(video_frames) / batch_size)
    video_features = torch.empty([0, 512], dtype=torch.float32).to(device)
    
    for i in range(batches):
        batch_frames = video_frames[i*batch_size : (i+1)*batch_size]
        batch_preprocessed = torch.stack([preprocess(frame) for frame in batch_frames]).to(device)
        with torch.no_grad():
            batch_features = model.encode_image(batch_preprocessed)
            batch_features = batch_features.float()
            batch_features /= batch_features.norm(dim=-1, keepdim=True)  # Normalize features
        video_features = torch.cat((video_features, batch_features))
        status_text.text(f'Encoding frames... ({(i+1)*batch_size}/{len(video_frames)})')
    
    return video_features

def img_to_bytes(img):
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr

def get_youtube_timestamp_url(url, frame_idx, frame_indices, fps):
    """
    Generate YouTube URL with timestamp for a specific frame.
    Args:
        url: YouTube video URL
        frame_idx: Index of the frame in the video_frames list
        frame_indices: List of actual frame numbers in the video
        fps: Frames per second of the video
    Returns:
        tuple: (timestamp_url, seconds) where timestamp_url is the YouTube URL with timestamp
    """
    if frame_indices is None or fps is None:
        return None, None
        
    frame_count = frame_indices[frame_idx]
    seconds = frame_count / fps
    seconds_rounded = int(seconds)
    
    if url == EXAMPLE_URL:
        video_id = "zTvJJnoWIPk"
    else:
        try:
            from urllib.parse import urlparse, parse_qs
            parsed_url = urlparse(url)
            video_id = parse_qs(parsed_url.query)['v'][0]
        except:
            return None, None
    
    return f"https://youtu.be/{video_id}?t={seconds_rounded}", seconds

def display_results(best_photo_idx, video_frames, video_name=None, frame_indices=None, fps=None):
    """
    Display and save search results.
    Args:
        best_photo_idx: Indices of best matching frames
        video_frames: List of video frames
        video_name: Name of the video (for multiple video mode)
        frame_indices: List of frame indices for timestamp calculation
        fps: Frames per second for timestamp calculation
    """
    st.subheader("Top 10 Results")
    
    # Determine the output directory
    if video_name:
        output_dir = os.path.join("found_frames", os.path.splitext(video_name)[0])
    else:
        output_dir = "found_frames"
    
    # Clear the output directory before saving new matches
    if os.path.exists(output_dir):
        for file in os.listdir(output_dir):
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(output_dir)
    
    # Save and display each matching frame
    for idx, frame_id in enumerate(best_photo_idx):
        result = video_frames[frame_id]
        st.image(result, width=400)
        
        # Save the frame if it's from a local file
        if hasattr(st.session_state, 'source') and (
            (isinstance(st.session_state.source, str) and os.path.isfile(st.session_state.source)) or
            isinstance(st.session_state.source, list)
        ):
            frame_path = os.path.join(output_dir, f"match_{idx:02d}.jpg")
            result.save(frame_path)
        
        if video_name:
            st.write(f"From video: {video_name}")
        
        # Only try to get timestamp for YouTube videos
        if hasattr(st.session_state, 'url'):
            timestamp_url, seconds = get_youtube_timestamp_url(st.session_state.url, frame_id, frame_indices, fps)
            if timestamp_url:
                st.markdown(f"[▶️ Play video at {format_timespan(int(seconds))}]({timestamp_url})")

def text_search(search_query, video_features, video_frames, display_results_count=10, video_name=None, frame_indices=None, fps=None):
    """
    Perform text-based semantic search on video frames.
    Args:
        search_query: Text query to search for
        video_features: Encoded features of video frames
        video_frames: List of video frames
        display_results_count: Number of results to display
        video_name: Name of the video (for multiple video mode)
        frame_indices: List of frame indices for timestamp calculation
        fps: Frames per second for timestamp calculation
    """
    display_results_count = min(display_results_count, len(video_frames))
    
    # Encode text query using CLIP
    with torch.no_grad():
        text_tokens = openai_clip.tokenize(search_query).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features.float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarities and find best matches
    video_features = video_features.float()
    similarities = (100.0 * video_features @ text_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)
    display_results(best_photo_idx, video_frames, video_name, frame_indices, fps)

def image_search(query_image, video_features, video_frames, display_results_count=10, video_name=None, frame_indices=None, fps=None):
    """
    Perform image-based semantic search on video frames.
    Args:
        query_image: PIL Image to search for
        video_features: Encoded features of video frames
        video_frames: List of video frames
        display_results_count: Number of results to display
        video_name: Name of the video (for multiple video mode)
        frame_indices: List of frame indices for timestamp calculation
        fps: Frames per second for timestamp calculation
    """
    # Preprocess and encode query image using CLIP
    query_image = preprocess(query_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(query_image)
        image_features = image_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)  # Normalize features
    
    # Calculate similarities and find best matches
    video_features = video_features.float()
    similarities = (100.0 * video_features @ image_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)
    display_results(best_photo_idx, video_frames, video_name, frame_indices, fps)

def text_and_image_search(search_query, query_image, video_features, video_frames, display_results_count=10, video_name=None, frame_indices=None, fps=None):
    """
    Perform combined text and image semantic search on video frames.
    Args:
        search_query: Text query to search for
        query_image: PIL Image to search for
        video_features: Encoded features of video frames
        video_frames: List of video frames
        display_results_count: Number of results to display
        video_name: Name of the video (for multiple video mode)
        frame_indices: List of frame indices for timestamp calculation
        fps: Frames per second for timestamp calculation
    """
    # Encode text query using CLIP
    with torch.no_grad():
        text_tokens = openai_clip.tokenize(search_query).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features.float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Encode image query using CLIP
    query_image = preprocess(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(query_image)
        image_features = image_features.float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # Combine text and image features with equal weights
    combined_features = (text_features + image_features) / 2
    
    # Calculate similarities and find best matches
    video_features = video_features.float()
    similarities = (100.0 * video_features @ combined_features.T)
    values, best_photo_idx = similarities.topk(display_results_count, dim=0)
    display_results(best_photo_idx, video_frames, video_name, frame_indices, fps)

def load_cached_data(url):
    """
    Load cached video data for the example URL.
    Args:
        url: YouTube video URL
    Returns:
        tuple: (video_frames, video_features, fps, frame_indices) or (None, None, None, None) if not cached
    """
    if url == EXAMPLE_URL:
        try:
            video_frames = np.load(f"{CACHED_DATA_PATH}example_frames.npy", allow_pickle=True)
            video_features = torch.load(f"{CACHED_DATA_PATH}example_features.pt")
            fps = np.load(f"{CACHED_DATA_PATH}example_fps.npy")
            frame_indices = np.load(f"{CACHED_DATA_PATH}example_frame_indices.npy")
            return video_frames, video_features, fps, frame_indices
        except:
            return None, None, None, None
    return None, None, None, None

def save_cached_data(url, video_frames, video_features, fps, frame_indices):
    """
    Save video data to cache for the example URL.
    Args:
        url: YouTube video URL
        video_frames: List of video frames
        video_features: Encoded features of video frames
        fps: Frames per second of the video
        frame_indices: List of frame indices
    """
    if url == EXAMPLE_URL:
        os.makedirs(CACHED_DATA_PATH, exist_ok=True)
        np.save(f"{CACHED_DATA_PATH}example_frames.npy", video_frames)
        torch.save(video_features, f"{CACHED_DATA_PATH}example_features.pt")
        np.save(f"{CACHED_DATA_PATH}example_fps.npy", fps)
        np.save(f"{CACHED_DATA_PATH}example_frame_indices.npy", frame_indices)

def clear_cached_data():
    if os.path.exists(CACHED_DATA_PATH):
        try:
            for file in os.listdir(CACHED_DATA_PATH):
                file_path = os.path.join(CACHED_DATA_PATH, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(CACHED_DATA_PATH)
        except Exception as e:
            print(f"Error clearing cache: {e}")

st.set_page_config(page_title="Which Frame? 🎞️🔍", page_icon = "🔍", layout = "centered", initial_sidebar_state = "collapsed")

hide_streamlit_style = """
<style>
/* Hide Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
* {
    font-family: Avenir;
}
.block-container {
    max-width: 800px;
    padding: 2rem 1rem;
}
.stTextInput input {
    border-radius: 8px;
    border: 1px solid #E0E0E0;
    padding: 0.75rem;
    font-size: 1rem;
    color: #333;
}

/* Style for the Video Source radio button */
div[data-testid="stRadio"] > label {
    color: rgb(250, 250, 250) !important;
}
.stRadio [role="radiogroup"] {
    background: rgb(14, 17, 23);
    padding: 1rem;
    border-radius: 12px;
}

h1 {text-align: center;}
.css-gma2qf {display: flex; justify-content: center; font-size: 36px; font-weight: bold;}
a:link {text-decoration: none;}
a:hover {text-decoration: none;}
.st-ba {font-family: Avenir;}
.st-button {text-align: center;}

/* Dark mode specific styles */
@media (prefers-color-scheme: dark) {
    .stTextInput input {
        border-color: rgba(250, 250, 250, 0.2);
    }
    .stRadio [role="radiogroup"] {
        border-color: rgba(250, 250, 250, 0.2);
    }
}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if not os.path.exists("found_frames"):
    os.makedirs("found_frames")

if 'progress' not in st.session_state:
    st.session_state.progress = 1
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = None
if 'video_features' not in st.session_state:
    st.session_state.video_features = None
if 'fps' not in st.session_state:
    st.session_state.fps = None
if 'video_name' not in st.session_state:
    st.session_state.video_name = 'videos/example.mp4'

st.title("Which Frame? 🎞️🔍")
st.markdown("""
Search a video semantically. For example, which frame has "a person with sunglasses"?
Search using text, images, or a mix of text + image. WhichFrame uses [CLIP](https://github.com/openai/CLIP) for zero-shot frame classification.
""")

if 'url' not in st.session_state:
    st.session_state.url = ''

url = st.text_input("Enter a YouTube URL (e.g., https://www.youtube.com/watch?v=zTvJJnoWIPk)", key="url_input")

source_type = st.radio("Video Source", ["YouTube URL", "Local File","Load Video"])

if source_type == "YouTube URL":
    source = url
elif source_type == "Local File":
    folder_path = st.text_input("Enter path to video folder or file")
    source = None
    if folder_path and os.path.exists(folder_path):
        if os.path.isdir(folder_path):
            # If it's a directory, list all video files
            video_files = [f for f in os.listdir(folder_path) 
                         if f.lower().endswith(('.mp4', '.avi', '.mov'))]
            if len(video_files) > 0:
                # Always show processing mode for directories
                process_mode = st.radio("Processing Mode", ["Single Video", "All Videos"])
                st.write(f"Found {len(video_files)} videos in folder")
                
                if process_mode == "Single Video":
                    selected_video = st.selectbox("Select video file", video_files)
                    source = os.path.join(folder_path, selected_video)
                else:  # All Videos
                    source = [os.path.join(folder_path, video) for video in video_files]
            else:
                st.error("No video files found in the specified folder")
        elif os.path.isfile(folder_path) and folder_path.lower().endswith(('.mp4', '.avi', '.mov')):
            # If it's a direct file path
            source = folder_path
        else:
            st.error("Please enter a valid video file path or folder containing videos")
else:  # Load Video
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
    if uploaded_file:
        try:
            # Save the uploaded file temporarily
            temp_path = f"temp_video_{int(time.time())}.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            source = temp_path
            st.session_state.temp_path = temp_path  # Store temp_path in session state
        except Exception as e:
            st.error(f"Error saving uploaded file: {str(e)}")
            source = None
    else:
        source = None

if st.button("Process Video"):
    if not source:
        st.error("Please provide a video source first")
    else:
        try:
            if source_type == "YouTube URL" and not source.startswith('http'):
                st.error("Please enter a valid YouTube URL")
                st.stop()
            
            # Store source in session state before processing
            st.session_state.source = source
            
            # Handle single video or multiple videos
            if isinstance(source, list):  # Multiple videos
                # Initialize lists to store all videos' data
                st.session_state.all_video_frames = []
                st.session_state.all_video_features = []
                st.session_state.all_video_names = []
                st.session_state.all_fps = []
                st.session_state.all_frame_indices = []
                
                total_videos = len(source)
                for video_idx, video_path in enumerate(source):
                    video_name = os.path.basename(video_path)
                    st.write(f"Processing video {video_idx + 1}/{total_videos}: {video_name}")
                    
                    # Create a subfolder for this video's frames
                    video_frames_dir = os.path.join("found_frames", os.path.splitext(video_name)[0])
                    if not os.path.exists(video_frames_dir):
                        os.makedirs(video_frames_dir)
                    
                    # Process the video
                    try:
                        with st.spinner(f'Processing {video_name}...'):
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            # For local files, use the path directly
                            if os.path.isfile(video_path):
                                # Extract frames directly from the video path
                                video_frames, fps, frame_indices = extract_frames(video_path, status_text, progress_bar)
                                
                                # Encode frames
                                video_features = encode_frames(video_frames, status_text)
                                
                                # Store this video's data
                                st.session_state.all_video_frames.append(video_frames)
                                st.session_state.all_video_features.append(video_features)
                                st.session_state.all_video_names.append(video_name)
                                st.session_state.all_fps.append(fps)
                                st.session_state.all_frame_indices.append(frame_indices)
                            else:
                                raise Exception(f"Video file not found: {video_path}")
                            
                            status_text.empty()
                            progress_bar.empty()
                        
                        st.success(f"Finished processing {video_name}")
                    except Exception as e:
                        st.error(f"Error processing {video_name}: {str(e)}")
                        continue  # Continue with next video even if this one fails
                
                if len(st.session_state.all_video_frames) > 0:
                    st.session_state.progress = 2
                    st.success(f"Successfully processed {len(st.session_state.all_video_frames)} out of {total_videos} videos")
                else:
                    st.error("Failed to process any videos")
                
            else:  # Single video
                cached_frames, cached_features, cached_fps, cached_frame_indices = load_cached_data(source)
                
                if cached_frames is not None:
                    st.session_state.video_frames = cached_frames
                    st.session_state.video_features = cached_features
                    st.session_state.fps = cached_fps
                    st.session_state.frame_indices = cached_frame_indices
                    st.session_state.progress = 2
                    st.success("Loaded cached video data!")
                else:
                    with st.spinner('Processing video...'):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Extract frames
                        st.session_state.video_frames, st.session_state.fps, st.session_state.frame_indices = extract_frames(source, status_text, progress_bar)
                        
                        # Encode frames
                        st.session_state.video_features = encode_frames(st.session_state.video_frames, status_text)
                        
                        if source_type == "YouTube URL":  # Only cache YouTube videos
                            save_cached_data(source, st.session_state.video_frames, st.session_state.video_features, st.session_state.fps, st.session_state.frame_indices)
                        
                        status_text.text('Finalizing...')
                        st.session_state.progress = 2
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()
                        st.success("Video processed successfully!")
                    
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            # Clean up temp file if it exists in session state
            if source_type == "Load Video" and hasattr(st.session_state, 'temp_path'):
                try:
                    if os.path.exists(st.session_state.temp_path):
                        os.remove(st.session_state.temp_path)
                except Exception:
                    pass  # Silently ignore cleanup errors
        finally:
            # Always try to clean up temp file after processing
            if source_type == "Load Video" and hasattr(st.session_state, 'temp_path'):
                try:
                    if os.path.exists(st.session_state.temp_path):
                        os.remove(st.session_state.temp_path)
                except Exception:
                    pass  # Silently ignore cleanup errors

if st.session_state.progress == 2:
    search_type = st.radio("Search Method", ["Text Search", "Image Search", "Text + Image Search"], index=0)
    
    if search_type == "Text Search":  # Text Search
        text_query = st.text_input("Type a search query (e.g., 'red car' or 'person with sunglasses')")
        if st.button("Search"):
            if not text_query:
                st.error("Please enter a search query first")
            else:
                if isinstance(st.session_state.source, list):
                    # Process each video in the folder
                    for idx, (video_frames, video_features) in enumerate(zip(st.session_state.all_video_frames, st.session_state.all_video_features)):
                        video_name = st.session_state.all_video_names[idx]
                        frame_indices = st.session_state.all_frame_indices[idx]
                        fps = st.session_state.all_fps[idx]
                        st.subheader(f"Results for {video_name}")
                        text_search(text_query, video_features, video_frames, 
                                  video_name=video_name, frame_indices=frame_indices, fps=fps)
                else:
                    text_search(text_query, st.session_state.video_features, st.session_state.video_frames,
                              frame_indices=st.session_state.frame_indices, fps=st.session_state.fps)
    
    elif search_type == "Image Search":  # Image Search
        uploaded_file = st.file_uploader("Upload a query image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert('RGB')
            st.image(query_image, caption="Query Image", width=200)
        if st.button("Search"):
            if uploaded_file is None:
                st.error("Please upload an image first")
            else:
                if isinstance(st.session_state.source, list):
                    # Process each video in the folder
                    for idx, (video_frames, video_features) in enumerate(zip(st.session_state.all_video_frames, st.session_state.all_video_features)):
                        video_name = st.session_state.all_video_names[idx]
                        frame_indices = st.session_state.all_frame_indices[idx]
                        fps = st.session_state.all_fps[idx]
                        st.subheader(f"Results for {video_name}")
                        image_search(query_image, video_features, video_frames,
                                   video_name=video_name, frame_indices=frame_indices, fps=fps)
                else:
                    image_search(query_image, st.session_state.video_features, st.session_state.video_frames,
                               frame_indices=st.session_state.frame_indices, fps=st.session_state.fps)
    
    else:  # Text + Image Search
        text_query = st.text_input("Type a search query")
        uploaded_file = st.file_uploader("Upload a query image", type=['png', 'jpg', 'jpeg'])
        if uploaded_file is not None:
            query_image = Image.open(uploaded_file).convert('RGB')
            st.image(query_image, caption="Query Image", width=200)
        
        if st.button("Search"):
            if not text_query or uploaded_file is None:
                st.error("Please provide both text query and image")
            else:
                if isinstance(st.session_state.source, list):
                    # Process each video in the folder
                    for idx, (video_frames, video_features) in enumerate(zip(st.session_state.all_video_frames, st.session_state.all_video_features)):
                        video_name = st.session_state.all_video_names[idx]
                        frame_indices = st.session_state.all_frame_indices[idx]
                        fps = st.session_state.all_fps[idx]
                        st.subheader(f"Results for {video_name}")
                        text_and_image_search(text_query, query_image, video_features, video_frames,
                                           video_name=video_name, frame_indices=frame_indices, fps=fps)
                else:
                    text_and_image_search(text_query, query_image, st.session_state.video_features, st.session_state.video_frames,
                                        frame_indices=st.session_state.frame_indices, fps=st.session_state.fps)

st.markdown("---")
st.markdown(
    "By [David Chuan-En Lin](https://chuanenlin.com/). "
    "Play with the code at [https://github.com/chuanenlin/whichframe](https://github.com/chuanenlin/whichframe)."
)
