#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the .env file")

client = OpenAI(api_key=api_key)

audio_files = ["mat.ogg"]

for index, audio_file_path in enumerate(audio_files):
    try:
        print(f"Transcribing audio file: {audio_file_path}...")

        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )

        output_file_name = f"transcription_output_{index}.txt"
        with open(output_file_name, "w", encoding="utf-8") as output_file:
            output_file.write(transcription.text)

        print(f"Transcription saved to {output_file_name}")

    except Exception as e:
        print(f"Error processing {audio_file_path}: {e}")
