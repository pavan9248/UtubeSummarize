import os
import re
import shutil
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from gtts import gTTS  # Import gTTS
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
import uvicorn

from youtube_transcript_api.formatters import TextFormatter
# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI model with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt for summarizing the transcript
prompt = """You are a YouTube video summarizer. You will be taking the 
transcript text and summarizing the entire video and providing the important summary 
in points within 200 words and also important topics in 3 to 4 points. Please provide them for the text given here: """

translate_prompt = """You are a language translator. You will be provided with a text 
in any language, and your task is to translate it into English. Please translate the following text: """


def translate_to_english(transcript_text):
    return generate_content(transcript_text, translate_prompt)

# Function to generate content using the generative AI model
def generate_content(transcript_text, prompt):
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt + transcript_text)
    return response.text

# Function to extract the transcript from a YouTube video
def extract_transcript(youtube_url):
        video_id = youtube_url.split("=")[1]
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try to fetch English transcript
        try:
            transcript_txt = transcripts.find_transcript(['en'])
        except:
            # If English transcript is not available, use the first available one
            transcript_txt = transcripts.find_generated_transcript(transcripts._generated_transcripts.keys())
        
        # Use a formatter to get plain text
        formatter = TextFormatter()
        transcript = formatter.format_transcript(transcript_txt.fetch())
        
        return transcript     


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get('/result')
def index(request: Request):
    return templates.TemplateResponse("result.html", {"request": request})

@app.post("/submit_url", response_class=HTMLResponse)
async def submit_url(request: Request, url: str = Form(...), language: str = Form(...)):     
    # Function to generate content using the generative AI model
    


    transcript_text = extract_transcript(url)

    # Translate the transcript to English if it's not already in English
    if transcript_text:
        if 'en' not in url:  # Here we assume non-English transcript needs translation
            transcript_text = translate_to_english(transcript_text)

        # Generate and display the summary
        summary = generate_content(transcript_text, prompt)

    # Function to convert summary text to audio
    def text_to_audio(summary_text):
        tts = gTTS(text=summary_text, lang="en")
        audio_path = os.path.join("static", "summary.mp3")
        tts.save(audio_path)
        return audio_path

    # Convert summary to audio and save it
    audio_file = None
    if summary:
        audio_file = text_to_audio(summary)

    # Embedded url formation
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
        "audio_file": "/static/summary.mp3" if audio_file else None  # Ensure correct path
    }
    
    return templates.TemplateResponse("result.html", context)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
