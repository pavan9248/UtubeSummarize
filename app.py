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

# Load environment variables from .env file
load_dotenv()

# Configure the Google Generative AI model with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Prompt for summarizing the transcript
prompt = """You are a YouTube video summarizer. You will be taking the 
transcript text and summarizing the entire video and providing the important summary 
in points within 200 words. Please provide the summary of the text given here: """

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
