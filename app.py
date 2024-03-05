import re
from fastapi import FastAPI, Request, File, UploadFile,Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os
from pytube import YouTube
import uvicorn
import fastapi
from googletrans import Translator
import assemblyai as aai
from transformers import pipeline
from summarizer import Summarizer
import torch




app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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

aai.settings.api_key = "d31b46660902421d8b7de5c2fd378c9a"

# URL of the file to transcribe
FILE_URL = "audio/audio.mp3"

transcriber = aai.Transcriber()
transcript = transcriber.transcribe(FILE_URL)
    
print(transcript.text)

def detect_language(text):
    translator = Translator()
    result = translator.detect(text)
    return result.lang

def translate_text(text, target_language='en'):
    translator = Translator()

    words = text.split()
    translated_text = ""

    for word in words:
        detected_lang = detect_language(word)
        if detected_lang != 'en':
            translation = translator.translate(word, src="ta", dest=target_language)
            translated_text += translation.text + " "
        else:
            translated_text += word + " "

    return translated_text.strip()

if __name__ == "__main__":
    # Example text containing both English and Hindi
    mixed_text = """Irunga hello what doing Vanakkam okay nice Epadi irukeenga"""
    # Translate the mixed text to English
    translated_text = translate_text(mixed_text)

    # Print the results
    print("Original Text:", mixed_text)
    print("Translated Text:", translated_text)

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Split the content into chunks of 1000 tokens
chunk_size = 1000
chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]

# Summarize each chunk and store the results
summaries = []
for chunk in chunks:
    summary = summarizer(chunk, max_length=40, min_length=8, length_penalty=2.0, num_beams=4, early_stopping=True)
    summaries.append(summary[0]['summary_text'])

text = '\n'.join(summaries)
print(text)

# Load BertSum model
bertsum_model = Summarizer()

# Summarize the content to 40 lines
summary = bertsum_model(text, num_sentences=40)

# Print the summarized content
print(summary)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
    youtube_url = "https://youtu.be/nOYW31rrkig?si=1vP__LwyLUfXPXKC"
    output_path = "/content/output"
    download_audio(youtube_url, output_path, filename="audio")
    