import re
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv
from pytube import YouTube
import uvicorn
import fastapi
import assemblyai as aai
from googletrans import Translator
from pydantic import BaseModel
import openai
from transformers import pipeline
from summarizer import Summarizer
from langdetect import detect
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import requests
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptAvailable, TranscriptsDisabled
from youtube_transcript_api._errors import NoTranscriptAvailable, TranscriptsDisabled
from gtts import gTTS

import shutil
from language_mappings import language_map


import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
import os
from youtube_transcript_api import YouTubeTranscriptApi


# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI model with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt for summarizing the transcript
prompt = """You are a YouTube video summarizer. You will be taking the 
transcript text and summarizing the entire video and providing the important summary 
in points within 250 words. Please provide the summary of the text given here: """




app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

class URLItem(BaseModel):
    url: str
    language: str

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/result')
def index(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

@app.post("/submit_url", response_class=HTMLResponse)
async def submit_url(request: Request, url: str = Form(...), language: str = Form(...)):     
    # Function to generate content using the generative AI model
    def generate_content(transcript_text, prompt):
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt + transcript_text)
        return response.text

    # Function to extract the transcript from a YouTube video
    def extract_transcript(youtube_url):
        try:
            video_id = youtube_url.split("=")[1]
            transcript_txt = YouTubeTranscriptApi.get_transcript(video_id)
            
            transcript = ""
            for i in transcript_txt:
                transcript += " " + i["text"]

            return transcript     

        except Exception as e:
            raise e     

    transcript_text = extract_transcript(url)

    # Generate and display the summary if the transcript is available
    if transcript_text:
        summary = generate_content(transcript_text, prompt)

    


    #Embedded url formation
    def get_embedded_url(url):
        if "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        elif "watch?v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL format")
        embedded_url = f"https://www.youtube-nocookie.com/embed/{video_id}"
        return embedded_url
        

    # Render the result.html template with the summary and audio file
    context = {
        "request": request,
        "url": get_embedded_url(url),
        "summary_text": summary,
        # "audio_file": audio_file
    }
    
    return templates.TemplateResponse("result.html", context)

def translate_audio(target_language='en'):
    audio_file_path = './output/audio.mp3'
    API_KEY = os.getenv("API_KEY")
    model_id = 'whisper-1'

    with open(audio_file_path, 'rb') as audio_file:
        response = openai.Audio.translate(
            api_key=API_KEY,
            model=model_id,
            file=audio_file,
            target_language=target_language
        )
        text =response.text
    return text
    
def get_transcript(url, target_language='en'):
    audio_file_path = './output/audio.mp3'
    try:
        if "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        elif "watch?v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        else:
            raise ValueError("Invalid YouTube URL format")

        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Extract the languages from the transcript list
        available_languages = [transcript.language for transcript in transcript_list]
        lang = get_language_code(available_languages[0].split('(')[0].strip())
   
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
        text = " ".join(line['text'] for line in transcript)
        if lang!='en':
            text=translate_audio(target_language='en')
        return text

    except (NoTranscriptAvailable, TranscriptsDisabled):
        return None

def get_language_code(language_name):
    # You need to define language_map somewhere in your code
    return language_map.get(language_name)

def translate_text(text , target_language='en'):
    translator = Translator()
    translated_text = translator.translate(text, dest=target_language).text
    return translated_text

def download_audio(youtube_url, output_path, filename="audio"):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_stream.download(output_path)

    # Get the default filename
    default_filename = audio_stream.default_filename

    # Rename the downloaded file
    downloaded_file_path = os.path.join(output_path, default_filename)
    new_file_path = os.path.join(output_path, f"{filename}.mp3")
    os.rename(downloaded_file_path, new_file_path)

# def preprocess_transcript(transcript):
    
#     # Remove punctuation
#     transcript = re.sub(r'[^\w\s]', '', transcript)
        
#     # Remove non-verbal expressions
#     non_verbal_expressions = ["[laughter]", "[music]", "[applause]"]
#     for expression in non_verbal_expressions:
#         transcript = transcript.replace(expression, "")
    
#     # Remove extra whitespace
#     transcript = ' '.join(transcript.split())
    
#     return transcript

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
