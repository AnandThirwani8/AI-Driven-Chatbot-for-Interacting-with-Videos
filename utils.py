import os
import cv2
import base64
import torch
import json
import requests
from tqdm import tqdm
import numpy as np
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from typing import List
from langchain import hub
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.embeddings import Embeddings
from pydantic import BaseModel
from transformers import BridgeTowerProcessor, BridgeTowerForContrastiveLearning

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

import ast
import lancedb
import pyarrow as pa
from langchain_community.vectorstores import LanceDB
from moviepy.video.io.VideoFileClip import VideoFileClip
from IPython.display import display, Video

#------------------------------------------- Helper functions for metadata extraction ----------------------------------------------------

# Resizes a image and maintains aspect ratio
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Grab the image size and initialize dimensions
    dim = None
    (h, w) = image.shape[:2]

    # Return original image if no need to resize
    if width is None and height is None:
        return image

    # We are resizing height if width is none
    if width is None:
        # Calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    # We are resizing width if height is none
    else:
        # Calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # Return the resized image
    return cv2.resize(image, dim, interpolation=inter)

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64.b64encode(buffered.getvalue()).decode('utf-8')
    # Encode the binary data to Base64
    base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return base64_string

def base64_to_image(base64_string):
    # Decode the base64 string to binary data
    image_data = base64.b64decode(base64_string)
    # Convert binary data to a PIL Image
    image = Image.open(BytesIO(image_data))
    return image

#----------------------------------------------- Google to describe the images ----------------------------------------------------

def gemini_image_description(image_base64, model_name = "gemini-2.0-flash"):
    model = genai.GenerativeModel(model_name=model_name)
    query = '''"Can you describe the image??"'''
    input_data = [{"text": query},
                  {"mime_type": "image/jpeg", "data": image_base64}]
    response = model.generate_content(input_data)
    return response.text    

#------------------------------------------- Get Transcript for the entire video ----------------------------------------------------

# Split Video into Frames. Describe each frame
def get_video_trascript(path_to_video, num_of_extracted_frames_per_second = 1/20):
    # load video using cv2
    video = cv2.VideoCapture(path_to_video)
    # Get the frames per second
    fps = video.get(cv2.CAP_PROP_FPS)
    # Get hop = the number of frames pass before a frame is extracted
    hop = round(fps / num_of_extracted_frames_per_second) 
    # # Total number of frames that will be processed
    frames_to_process = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) // hop  

    metadatas = []

    curr_frame = 0
    idx = -1
    with tqdm(total=frames_to_process, desc="Processing Frames", unit="frame") as pbar:
        while(True):
            # iterate all frames
            ret, frame = video.read()
            if not ret: 
                break
            if curr_frame % hop == 0:
                idx = idx + 1
        
                # Compute the middle timestamp
                middle_timestamp = (curr_frame + hop / 2) / fps
            
                # if the frame is extracted successfully, resize it
                image = maintain_aspect_ratio_resize(frame, height=350)
                image = Image.fromarray(image)
        
                # get base64 embeddings
                b64_image = image_to_base64(image)
                caption = gemini_image_description(b64_image)
        
                # prepare the metadata
                metadata = {
                    'extracted_frame_base64': b64_image,
                    'transcript': caption,
                    'video_segment_id': idx,
                    'video_path': path_to_video,
                    'timestamp': middle_timestamp 
                }
                metadatas.append(metadata)
                pbar.update(1)
                
            curr_frame += 1
    
    return metadatas

#---------------------------------------------- Multimodal (Image-Text Pair) Embeddings -------------------------------------------------

class BridgeTowerEmbeddings(BaseModel, Embeddings):
    """ BridgeTower embedding model """
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using BridgeTower.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        array = np.zeros((100, 100, 3), dtype=np.uint8) 
        placeholder_image = Image.fromarray(array)

        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm")
        model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

        inputs  = processor([placeholder_image], texts, padding=True, return_tensors="pt")
        outputs = model(**inputs)

        return outputs.text_embeds.tolist()


    def embed_query(self, text: str) -> List[float]:
        """Embed a query using BridgeTower.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]

    def embed_image_text_pairs(self, texts: List[str], images: List[str], batch_size=2) -> List[List[float]]:
        """Embed a list of image-text pairs using BridgeTower.

        Args:
            texts: The list of texts to embed.
            images: The list of path-to-images to embed
            batch_size: the batch size to process, default to 2
        Returns:
            List of embeddings, one for each image-text pairs.
        """

        # the length of texts must be equal to the length of images
        assert len(texts)==len(images), "the len of captions should be equal to the len of images"

        processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-large-itm-mlm")
        model = BridgeTowerForContrastiveLearning.from_pretrained("BridgeTower/bridgetower-large-itm-mlm-itc")

        # images = [Image.open(im) for im in images]
        inputs  = processor(images, texts, padding=True, return_tensors="pt")
        outputs = model(**inputs)

        return outputs.cross_embeds.tolist()

#---------------------------------------------- Vector DB -------------------------------------------------
def get_vectorstore(vid_metadata, embeddings):
    # Create db
    db = lancedb.connect("./video_vectors")
    table_name = "video_vectors"
    schema = pa.schema([pa.field("vector", pa.list_(pa.float32(), list_size=512)),
                        pa.field("text", pa.string()),
                       ])
    table = db.create_table(table_name, schema=schema, mode = 'overwrite') 

    # Add content
    frame_details = [str([vid["transcript"], vid["video_path"], vid["timestamp"]]) for vid in vid_metadata]
    docs_with_embeddings = [{"vector": vector, "text": doc} for doc, vector in zip(frame_details, embeddings)]
    table.add(docs_with_embeddings)

    # Create Retriever
    embedding_model = BridgeTowerEmbeddings()
    vector_store = LanceDB(connection=db, table_name=table_name, embedding=embedding_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    return retriever

def query_vectorstore(query, retriever):
    retrieved_docs = retriever.invoke(query)
    extracted_trascript = ast.literal_eval(retrieved_docs[0].page_content)[0]
    extracted_path = ast.literal_eval(retrieved_docs[0].page_content)[1]
    extracted_timestamp = ast.literal_eval(retrieved_docs[0].page_content)[2]
    return extracted_trascript, extracted_path, extracted_timestamp

def QnA(query, retriever):
    extracted_trascript, extracted_path, extracted_timestamp = query_vectorstore(query, retriever)
    prompt = hub.pull("rlm/rag-prompt")
    prompt = prompt.invoke({"question": query, "context": extracted_trascript})
    llm = GoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.environ["GOOGLE_API_KEY"])
    answer = llm.invoke(prompt)
    return answer, extracted_trascript, extracted_path, extracted_timestamp

# def create_db(vid_metadata, embeddings):
    
#     vid_trans = [vid['transcript'] for vid in vid_metadata]
#     vid_frames_base64 = [vid['extracted_frame_base64'] for vid in vid_metadata]
#     vid_frames = [base64_to_image(vid['extracted_frame_base64']) for vid in vid_metadata]
#     vid_path = [vid['video_path'] for vid in vid_metadata]
#     vid_frames_timestamp = [vid['timestamp'] for vid in vid_metadata]
    
#     table_name = "Video_Embeddings"
#     db = lancedb.connect(table_name) 
#     schema = pa.schema([pa.field("embeddings", pa.list_(pa.float32(), list_size=512)),
#                         pa.field("transcript", pa.string()),
#                         pa.field("image_base64", pa.string()),
#                         pa.field("path", pa.string()),
#                         pa.field("timestamp", pa.float16())
#                        ])
#     table = db.create_table(table_name, schema=schema, mode = 'overwrite')
#     for i in range(len(vid_frames_base64)):
#         data = [{"embeddings": embeddings[i],  
#                  "transcript": vid_trans[i], 
#                  'image_base64': vid_frames_base64[i],
#                  'path': vid_path[i],
#                  'timestamp': vid_frames_timestamp[i]
#                 }]
#         table.add(data)
#     return table


# def query_db(query, table):
#     embedding_model = BridgeTowerEmbeddings()
#     query_embedding = embedding_model.embed_query(query)
#     results = table.search(query_embedding).to_pandas()
#     return results.iloc[0]['path'], results.iloc[0]['timestamp'], results.iloc[0]['transcript']

#---------------------------------------------- Display Clip -------------------------------------------------

def extract_and_display_clip(video_path, timestamp_in_sec, play_before_sec=10, play_after_sec=10):
    
    video = VideoFileClip(video_path)
    duration = video.duration

    # Define start and end times
    start_time = max(timestamp_in_sec - play_before_sec, 0)
    end_time = min(timestamp_in_sec + play_after_sec, duration)
    # Extract subclip
    clip = video.subclipped(start_time, end_time)

    clip_path = os.path.join('temp/', "extracted_clip.mp4")
    clip.write_videofile(clip_path, codec="libx264", fps=24)

    # Display the video in Jupyter Notebook
    display(Video(clip_path, width=400))










    