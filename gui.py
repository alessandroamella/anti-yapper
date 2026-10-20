#!/usr/bin/env python3

import base64
import glob
from datetime import date
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile

import mistune
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from PyQt6.QtCore import QSettings, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut, QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

# Configure logging to console
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

# Formats accepted by the OpenAI Transcriptions API; anything else is converted with ffmpeg
OPENAI_SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
OPENAI_MAX_FILE_BYTES = 24 * 1024 * 1024  # API limit is 25 MB, keep a small margin
OPENAI_CHUNK_SECONDS = 30 * 60  # ~7 MB per chunk at 32 kbps
# gpt-4o-* transcription models reject audio longer than ~1400 s
GPT4O_MAX_SECONDS = 20 * 60

# Models from https://developers.openai.com/api/docs/guides/speech-to-text
OPENAI_MODELS = {
    "gpt-transcribe": {
        "prompt": True,
        "max_seconds": None,
        "stream": True,
        "price": {"per_minute": 0.0045},
        "info": "Modello consigliato da OpenAI per trascrivere registrazioni nella "
        "lingua originale. Il prompt fa da contesto (argomento, nomi, termini tecnici)."
        "<br><br>Prezzo: $0.0045 al minuto (un'ora costa circa $0.27).",
    },
    "gpt-4o-transcribe-diarize": {
        "prompt": False,
        "diarize": True,
        "max_seconds": GPT4O_MAX_SECONDS,
        "stream": True,
        "price": {"input": 2.50, "output": 10.00, "per_minute": 0.006},
        "info": "Riconosce chi parla ed etichetta ogni intervento (A, B, ...). "
        "Da usare solo se servono gli speaker. Non accetta prompt. Se il file viene "
        "diviso in parti, le etichette ripartono da capo in ogni parte."
        "<br><br>Prezzo: circa $0.006 al minuto ($2.50 / 1M token in input, "
        "$10 / 1M in output).",
    },
    "whisper-1": {
        "prompt": True,
        "max_seconds": None,
        "stream": False,
        "price": {"per_minute": 0.006},
        "info": "Il modello Whisper originale. Il prompt è limitato a 224 token e viene "
        "seguito meno bene. Nell'API offre anche timestamp, sottotitoli e traduzione "
        "in inglese, che questa app non usa.<br><br>Prezzo: $0.006 al minuto "
        "(un'ora costa circa $0.36).",
    },
    "gpt-4o-transcribe": {
        "prompt": True,
        "max_seconds": GPT4O_MAX_SECONDS,
        "stream": True,
        "price": {"input": 2.50, "output": 10.00, "per_minute": 0.006},
        "info": "Trascrizione basata su GPT-4o. Supporta il prompt. OpenAI lo mantiene "
        "per le integrazioni esistenti e consiglia gpt-transcribe per quelle nuove. "
        "In una prova con un dialogo a due voci ha saltato alcune frasi."
        "<br><br>Prezzo: circa $0.006 al minuto ($2.50 / 1M token in input, "
        "$10 / 1M in output).",
    },
    "gpt-4o-mini-transcribe": {
        "prompt": True,
        "max_seconds": GPT4O_MAX_SECONDS,
        "stream": True,
        "price": {"input": 1.25, "output": 5.00, "per_minute": 0.003},
        "info": "Versione più economica di gpt-4o-transcribe. Supporta il prompt. "
        "OpenAI lo mantiene per le integrazioni esistenti e consiglia gpt-transcribe "
        "per quelle nuove. In una prova con un dialogo a due voci ha saltato alcune "
        "frasi.<br><br>Prezzo: circa $0.003 al minuto ($1.25 / 1M token in input, "
        "$5 / 1M in output).",
    },
}

# Gemini models shown in the picker: the newest stable Flash, the newest dedicated
# transcription model and every Gemini 4 model (plus GEMINI_MODEL from .env)
GEMINI_FLASH_PATTERN = re.compile(r"^gemini-(\d+(?:\.\d+)?)-flash$")
GEMINI_TRANSCRIBE_PATTERN = re.compile(r"^gemini-(\d+(?:\.\d+)?)-transcribe$")
GEMINI_4_PATTERN = re.compile(r"^gemini-4(?:\.\d+)?-")
# Gemini 4 variants that don't analyze audio into text
GEMINI_EXCLUDED_KEYWORDS = ("tts", "image", "embedding", "live", "native-audio")

GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
# Descriptions and prices from https://ai.google.dev/gemini-api/docs/pricing
# (paid tier, per 1M tokens); models missing here show the API metadata instead
GEMINI_MODEL_INFO = {
    "gemini-3.8-flash": "Il modello Flash più capace di Google. Va bene per "
    "riassunti, analisi e domande sull'audio.<br><br>Prezzo: $0.75 in input e "
    "$3.75 in output per 1M token fino al 31/12/2026, poi $1.50 e $7.50. "
    "Gratis nel free tier.",
    "gemini-3.5-transcribe": "Modello speech-to-text di Google: rileva la lingua "
    "da solo e supporta la separazione degli speaker e un vocabolario "
    "personalizzato. Non supporta la ricerca Google né le domande di follow-up."
    "<br><br>Prezzo: circa $0.005 "
    "al minuto ($0.003/min di audio in input più $0.002/min di testo in output). "
    "Gratis nel free tier.",
    "gemini-4-argon": "Il modello di punta di Google, pensato per ragionamenti "
    "complessi, matematica e scienza. Non è disponibile nel free tier."
    "<br><br>Prezzo: $2 in input e $10 in output per 1M token fino al 30/11/2026, "
    "poi $4 e $20.",
}


# Paid tier rates in USD per 1M tokens, as (valid until, rates); None = no end date
GEMINI_PRICING = {
    "gemini-3.8-flash": [
        (date(2027, 1, 1), {"input": 0.75, "cached": 0.075, "output": 3.75}),
        (None, {"input": 1.50, "cached": 0.15, "output": 7.50}),
    ],
    "gemini-3.5-transcribe": [
        # Google's own blended estimate for this model is ~$0.005 per minute
        (None, {"input": 2.00, "cached": 2.00, "output": 12.00, "per_minute": 0.005}),
    ],
    "gemini-4-argon": [
        (date(2026, 12, 1), {"input": 2.00, "cached": 0.10, "output": 10.00}),
        (None, {"input": 4.00, "cached": 0.20, "output": 20.00}),
    ],
}
GEMINI_AUDIO_TOKENS_PER_SECOND = 32


def gemini_rates(model_id):
    """Return today's rates for a Gemini model, or None if its price is unknown."""
    for valid_until, rates in GEMINI_PRICING.get(model_id, []):
        if valid_until is None or date.today() < valid_until:
            return rates
    return None


def gemini_cost(model_id, response):
    """Return the cost in USD of a Gemini response, or None if unknown."""
    rates = gemini_rates(model_id)
    usage = response.usage_metadata if response else None
    if not rates or not usage:
        return None
    prompt_tokens = usage.prompt_token_count or 0
    cached_tokens = usage.cached_content_token_count or 0
    # Thinking tokens are billed as output
    output_tokens = (usage.candidates_token_count or 0) + (
        usage.thoughts_token_count or 0
    )
    return (
        (prompt_tokens - cached_tokens) * rates["input"]
        + cached_tokens * rates["cached"]
        + output_tokens * rates["output"]
    ) / 1_000_000


def openai_cost(model, usage, audio_path):
    """Return the cost in USD of an OpenAI transcription."""
    price = OPENAI_MODELS[model]["price"]
    if usage is not None and usage.type == "tokens" and "input" in price:
        return (
            usage.input_tokens * price["input"] + usage.output_tokens * price["output"]
        ) / 1_000_000
    if usage is not None and usage.type == "duration":
        seconds = usage.seconds
    else:
        seconds = get_audio_duration(audio_path) or 0
    return seconds / 60 * price["per_minute"]


def estimated_cost_per_minute(backend, model_id):
    """Return the estimated cost of one minute of audio, or None if unknown."""
    if backend == BACKEND_OPENAI:
        return OPENAI_MODELS[model_id]["price"]["per_minute"]
    rates = gemini_rates(model_id)
    if not rates:
        return None
    # Only the audio input can be estimated in advance, the reply length is unknown
    return rates.get("per_minute") or (
        60 * GEMINI_AUDIO_TOKENS_PER_SECOND * rates["input"] / 1_000_000
    )


def format_cost(cost):
    # Keep sub-cent amounts readable
    return f"${cost:.4f}" if 0 < cost < 0.01 else f"${cost:.2f}"


def stream_gemini_message(chat_session, message, on_text):
    """
    Send a chat message with streaming, calling on_text with the text received so far.
    Returns the full text and the last chunk, which carries the final token usage.
    """
    text = ""
    last_chunk = None
    for chunk in chat_session.send_message_stream(message):
        text += response_text(chunk)
        last_chunk = chunk
        on_text(text)
    return text, last_chunk


def is_gemini_transcribe_model(model_id):
    # Dedicated transcription models don't support Google Search or follow-up chats
    return "-transcribe" in model_id


def response_text(response):
    """Return the text of a Gemini response, including audio transcription parts."""
    if not response.candidates or not response.candidates[0].content:
        return ""
    texts = []
    for part in response.candidates[0].content.parts or []:
        if part.text and not part.thought:
            texts.append(part.text)
        elif part.audio_transcription and part.audio_transcription.text:
            texts.append(part.audio_transcription.text)
    return "".join(texts)


def select_gemini_models(model_ids):
    """Pick the Gemini models worth showing from all the ids the API returns."""

    def newest(pattern):
        matches = [(m, pattern.match(m)) for m in model_ids]
        versions = [
            (tuple(int(n) for n in match.group(1).split(".")), m)
            for m, match in matches
            if match
        ]
        return [max(versions)[1]] if versions else []

    gemini_4 = sorted(
        m
        for m in model_ids
        if GEMINI_4_PATTERN.match(m)
        and not any(k in m for k in GEMINI_EXCLUDED_KEYWORDS)
    )
    return (
        newest(GEMINI_FLASH_PATTERN)
        + newest(GEMINI_TRANSCRIBE_PATTERN)
        + gemini_4
    )

BACKEND_GEMINI = "gemini"
BACKEND_OPENAI = "openai"


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class FollowUpThread(QThread):
    """
    Thread to handle follow-up audio queries using the established Gemini chat session.
    """

    followup_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    cost_update = pyqtSignal(float)
    partial_text = pyqtSignal(str)

    def __init__(self, chat_session, gemini_model, user_text):
        super().__init__()
        self.chat_session = chat_session
        self.gemini_model = gemini_model
        self.user_text = user_text

    def run(self):
        logging.info("FollowUpThread started.")
        try:
            self.progress_update.emit(25)
            self.status_update.emit("Invio follow-up in corso...")
            result_text, last_chunk = stream_gemini_message(
                self.chat_session, self.user_text, self.partial_text.emit
            )
            # The whole conversation, audio included, is sent again with each follow-up
            cost = gemini_cost(self.gemini_model, last_chunk)
            if cost is not None:
                self.cost_update.emit(cost)
            self.progress_update.emit(95)
            self.status_update.emit("Risposta ricevuta.")
            self.followup_finished.emit(result_text)
            self.progress_update.emit(100)
            logging.info("FollowUpThread finished successfully.")
        except Exception as e:
            error_message = f"Error sending follow-up: {e}"
            logging.error(error_message, exc_info=True)
            self.error_occurred.emit(error_message)
            self.progress_update.emit(0)


class ProcessingThread(QThread):
    """
    Thread to handle audio processing using the Gemini API.
    This simplified version uses a single user prompt to guide the AI,
    which can handle transcription, summarization, or other tasks in one call.
    """

    processing_finished = pyqtSignal(str, object)
    error_occurred = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    cost_update = pyqtSignal(float)
    partial_text = pyqtSignal(str)

    def __init__(
        self,
        audio_file_paths,
        gemini_api_key,
        gemini_model,
        user_prompt,
        enable_search=False,
    ):
        super().__init__()
        self.audio_file_paths = audio_file_paths
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
        self.user_prompt = user_prompt
        self.enable_search = enable_search

        self.gemini_client = None

    def run(self):
        logging.info("Processing thread started.")
        file_name_for_error = "audio files"
        uploaded_gemini_files = (
            []
        )  # Keep track of all files used (cached + newly uploaded)
        newly_uploaded_files = (
            []
        )  # Keep track of only newly uploaded files for deletion

        try:

            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set in the .env file")

            self.status_update.emit("Inizializzazione client Gemini...")
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            logging.info("Gemini client initialized.")
            self.progress_update.emit(10)

            logging.info(
                "Checking for existing files and uploading new ones to Gemini..."
            )
            gemini_contents = []

            gemini_contents.append(self.user_prompt)

            # Get list of existing files from Gemini to check for duplicates
            self.status_update.emit("Controllo file cache su Gemini...")
            logging.info("Fetching existing Gemini files for cache check...")
            existing_files = {}  # sha256_hash -> gemini_file
            try:
                for existing_file in self.gemini_client.files.list():
                    if (
                        hasattr(existing_file, "sha256_hash")
                        and existing_file.sha256_hash
                    ):
                        existing_files[existing_file.sha256_hash] = existing_file
                    logging.debug(
                        f"Existing file: {existing_file.name}, SHA-256: {existing_file.sha256_hash}"
                    )
                logging.info(f"Found {len(existing_files)} existing files in Gemini")
            except Exception as e:
                logging.warning(f"Could not fetch existing files list: {e}")
                existing_files = {}

            num_files = len(self.audio_file_paths)
            cache_hits = 0
            new_uploads = 0

            for index, file_path in enumerate(self.audio_file_paths):
                file_name_for_error = os.path.basename(file_path)
                progress = 15 + int((75 / num_files) * index)
                self.progress_update.emit(progress)

                # Calculate SHA-256 hash of the local file
                self.status_update.emit(
                    f"File {index + 1}/{num_files}: Calcolo hash per {file_name_for_error}..."
                )
                logging.info(
                    f"Processing file {index + 1}/{num_files}: {file_name_for_error}..."
                )
                try:
                    local_hash = calculate_sha256(file_path)
                    # Convert hex string to base64 to match Gemini's format
                    local_hash_b64 = base64.b64encode(
                        local_hash.encode("utf-8")
                    ).decode("utf-8")

                    logging.info(
                        f"Local SHA-256 for {file_name_for_error}: {local_hash_b64}"
                    )

                    # Check if file already exists in Gemini
                    if local_hash_b64 in existing_files:
                        existing_file = existing_files[local_hash_b64]
                        self.status_update.emit(
                            f"File {file_name_for_error} trovato nella cache."
                        )
                        logging.info(
                            f"File {file_name_for_error} already exists in Gemini (cache hit): {existing_file.name}"
                        )
                        gemini_contents.append(existing_file)
                        uploaded_gemini_files.append(
                            existing_file
                        )  # Track for content generation (but don't delete)
                        cache_hits += 1
                    else:
                        # Upload new file
                        self.status_update.emit(
                            f"Caricamento di {file_name_for_error} ({index + 1}/{num_files})..."
                        )
                        logging.info(f"Uploading new file: {file_name_for_error}...")
                        gemini_file = self.gemini_client.files.upload(file=file_path)
                        gemini_contents.append(gemini_file)
                        uploaded_gemini_files.append(gemini_file)
                        newly_uploaded_files.append(gemini_file)  # Track for deletion
                        logging.info(f"Successfully uploaded {gemini_file.name}")
                        new_uploads += 1

                except Exception as e:
                    logging.warning(f"Error processing file {file_name_for_error}: {e}")
                    # Fallback to direct upload if hash calculation fails
                    logging.info(
                        f"Falling back to direct upload for: {file_name_for_error}"
                    )
                    gemini_file = self.gemini_client.files.upload(file=file_path)
                    gemini_contents.append(gemini_file)
                    uploaded_gemini_files.append(gemini_file)
                    newly_uploaded_files.append(gemini_file)  # Track for deletion
                    new_uploads += 1

            logging.info(
                f"File processing summary: {cache_hits} cached, {new_uploads} newly uploaded"
            )

            self.status_update.emit("Gemini macina...")
            logging.info("All files uploaded. Requesting generation from Gemini...")
            self.progress_update.emit(50)

            # Prepara la configurazione per la generazione dei contenuti
            content_config = None
            if self.enable_search:
                logging.info("Grounding with Google Search is ENABLED.")
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                content_config = types.GenerateContentConfig(tools=[grounding_tool])
            else:
                logging.info("Grounding with Google Search is DISABLED.")

            # Crea la sessione chat con la configurazione se presente
            chat_session = self.gemini_client.chats.create(
                model=self.gemini_model,
                config=content_config,
            )
            # Esegui la chiamata a Gemini inviando primo messaggio col le clip
            result_text, last_chunk = stream_gemini_message(
                chat_session, gemini_contents, self.partial_text.emit
            )
            cost = gemini_cost(self.gemini_model, last_chunk)
            if cost is not None:
                self.cost_update.emit(cost)

            self.status_update.emit("Risposta ricevuta. Finalizzazione...")
            logging.info("Received response from Gemini.")
            self.progress_update.emit(95)

            # The transcription reply can't be sent back as chat history
            if is_gemini_transcribe_model(self.gemini_model):
                chat_session = None
            self.processing_finished.emit(result_text, chat_session)

            self.progress_update.emit(100)
            logging.info("Processing thread finished successfully.")

        except Exception as e:
            error_message = f"Error processing {file_name_for_error}: {e}"
            logging.error(error_message, exc_info=True)
            self.error_occurred.emit(error_message, file_name_for_error)
            self.progress_update.emit(0)
        finally:
            # Files are preserved on Gemini for future use
            # they last 48h and it's free so no need to delete them
            logging.info(
                "Processing completed. All files preserved for future use (48h)."
            )
        logging.info("Processing thread run method completed.")


class GeminiModelsThread(QThread):
    """
    Thread to fetch the list of Gemini models available to the API key.
    """

    models_loaded = pyqtSignal(list)

    def __init__(self, gemini_api_key):
        super().__init__()
        self.gemini_api_key = gemini_api_key

    def run(self):
        logging.info("Fetching available Gemini models...")
        try:
            client = genai.Client(api_key=self.gemini_api_key)
            available = {
                model.name.removeprefix("models/"): model
                for model in client.models.list()
                if "generateContent" in (model.supported_actions or [])
            }
            models = []
            for model_id in select_gemini_models(list(available)):
                model = available[model_id]
                models.append(
                    {
                        "id": model_id,
                        "display_name": model.display_name,
                        "description": model.description,
                        "input_token_limit": model.input_token_limit,
                        "output_token_limit": model.output_token_limit,
                    }
                )
            logging.info(f"Gemini models: {[m['id'] for m in models]}")
            self.models_loaded.emit(models)
        except Exception as e:
            # Keep using the cached list
            logging.warning(f"Could not fetch Gemini models: {e}")


def get_audio_duration(file_path):
    """Return the duration of an audio file in seconds, or None if unknown."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError) as e:
        logging.warning(f"Could not read duration of {file_path}: {e}")
        return None


def format_diarized_segments(segments):
    """Join diarized segments into Markdown, merging consecutive turns of the same speaker."""
    turns = []
    for segment in segments:
        text = segment.text.strip()
        if turns and turns[-1][0] == segment.speaker:
            turns[-1][1].append(text)
        else:
            turns.append((segment.speaker, [text]))
    return "\n\n".join(f"**{speaker}:** {' '.join(texts)}" for speaker, texts in turns)


def run_ffmpeg(args):
    """Run ffmpeg with the given arguments, raising a readable error on failure."""
    try:
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        raise RuntimeError("ffmpeg non trovato: installalo per convertire l'audio.")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg ha fallito: {e.stderr.strip()[-500:]}")


class TranscriptionThread(QThread):
    """
    Thread to handle audio transcription using the OpenAI Transcriptions API.
    Unsupported or too large files are converted to compact mp3 and split into chunks.
    """

    processing_finished = pyqtSignal(str, object)
    error_occurred = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    cost_update = pyqtSignal(float)
    partial_text = pyqtSignal(str)

    def __init__(self, audio_file_paths, openai_api_key, model, context_prompt):
        super().__init__()
        self.audio_file_paths = audio_file_paths
        self.openai_api_key = openai_api_key
        self.model = model
        self.model_options = OPENAI_MODELS[model]
        self.context_prompt = context_prompt

    def prepare_audio(self, file_path, index, work_dir):
        """Return the list of files to send to the API for this audio file."""
        ext = os.path.splitext(file_path)[1].lower()
        max_seconds = self.model_options["max_seconds"]
        too_long = False
        if max_seconds:
            duration = get_audio_duration(file_path)
            too_long = duration is not None and duration > max_seconds
        if (
            ext in OPENAI_SUPPORTED_EXTENSIONS
            and os.path.getsize(file_path) <= OPENAI_MAX_FILE_BYTES
            and not too_long
        ):
            return [file_path]

        # Mono 16 kHz 32 kbps mp3 is plenty for speech and keeps files small
        converted_path = os.path.join(work_dir, f"file{index}.mp3")
        logging.info(f"Converting {file_path} to {converted_path}...")
        run_ffmpeg(
            ["-i", file_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", converted_path]
        )
        if os.path.getsize(converted_path) <= OPENAI_MAX_FILE_BYTES and not too_long:
            return [converted_path]

        logging.info(f"{converted_path} too large or too long, splitting into chunks...")
        chunk_pattern = os.path.join(work_dir, f"file{index}_chunk%03d.mp3")
        run_ffmpeg(
            [
                "-i",
                converted_path,
                "-f",
                "segment",
                "-segment_time",
                str(min(OPENAI_CHUNK_SECONDS, max_seconds or OPENAI_CHUNK_SECONDS)),
                "-reset_timestamps",
                "1",
                "-c",
                "copy",
                chunk_pattern,
            ]
        )
        return sorted(glob.glob(os.path.join(work_dir, f"file{index}_chunk*.mp3")))

    def transcribe_file(self, client, audio_path, extra_args, on_text):
        """
        Transcribe one file, streaming when the model supports it and calling on_text
        with the text received so far. Returns the text and the billed usage.
        """
        diarize = self.model_options.get("diarize")
        with open(audio_path, "rb") as audio_file:
            if not self.model_options["stream"]:
                transcription = client.audio.transcriptions.create(
                    model=self.model, file=audio_file, **extra_args
                )
                if diarize:
                    return (
                        format_diarized_segments(transcription.segments),
                        transcription.usage,
                    )
                return transcription.text.strip(), transcription.usage

            stream = client.audio.transcriptions.create(
                model=self.model, file=audio_file, stream=True, **extra_args
            )
            text = ""
            segments = []
            usage = None
            for event in stream:
                if event.type == "transcript.text.delta" and not diarize:
                    text += event.delta
                    on_text(text.strip())
                elif event.type == "transcript.text.segment":
                    segments.append(event)
                    on_text(format_diarized_segments(segments))
                elif event.type == "transcript.text.done":
                    text = event.text
                    usage = event.usage
        if diarize:
            return format_diarized_segments(segments), usage
        return text.strip(), usage

    def run(self):
        logging.info("Transcription thread started.")
        file_name_for_error = "audio files"
        total_cost = 0.0

        try:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set in the .env file")

            self.status_update.emit("Inizializzazione client OpenAI...")
            client = OpenAI(api_key=self.openai_api_key)
            self.progress_update.emit(5)

            extra_args = {}
            if self.context_prompt and self.model_options["prompt"]:
                extra_args["prompt"] = self.context_prompt
            if self.model_options.get("diarize"):
                extra_args["response_format"] = "diarized_json"
                extra_args["chunking_strategy"] = "auto"

            num_files = len(self.audio_file_paths)
            sections = []

            with tempfile.TemporaryDirectory() as work_dir:
                for index, file_path in enumerate(self.audio_file_paths):
                    file_name_for_error = os.path.basename(file_path)
                    self.progress_update.emit(5 + int((90 / num_files) * index))

                    self.status_update.emit(
                        f"File {index + 1}/{num_files}: preparazione di {file_name_for_error}..."
                    )
                    chunk_paths = self.prepare_audio(file_path, index, work_dir)

                    def file_section(texts, name=file_name_for_error):
                        text = "\n\n".join(texts)
                        return f"## {name}\n\n{text}" if num_files > 1 else text

                    chunk_texts = []
                    for chunk_index, chunk_path in enumerate(chunk_paths):
                        self.status_update.emit(
                            f"File {index + 1}/{num_files}: trascrizione di {file_name_for_error}"
                            f" (parte {chunk_index + 1}/{len(chunk_paths)})..."
                        )
                        logging.info(f"Transcribing {chunk_path} with {self.model}...")
                        text, usage = self.transcribe_file(
                            client,
                            chunk_path,
                            extra_args,
                            lambda partial: self.partial_text.emit(
                                "\n\n".join(
                                    sections
                                    + [file_section(chunk_texts + [partial])]
                                )
                            ),
                        )
                        total_cost += openai_cost(self.model, usage, chunk_path)
                        chunk_texts.append(text)

                    sections.append(file_section(chunk_texts))

            self.status_update.emit("Trascrizione completata. Finalizzazione...")
            self.progress_update.emit(95)
            self.cost_update.emit(total_cost)
            # No chat session: follow-up is not available for plain transcription
            self.processing_finished.emit("\n\n".join(sections), None)
            self.progress_update.emit(100)
            logging.info("Transcription thread finished successfully.")

        except Exception as e:
            error_message = f"Error transcribing {file_name_for_error}: {e}"
            logging.error(error_message, exc_info=True)
            # Parts transcribed before the error are billed anyway
            if total_cost:
                self.cost_update.emit(total_cost)
            self.error_occurred.emit(error_message, file_name_for_error)
            self.progress_update.emit(0)


class AudioSummaryApp(QWidget):

    DEFAULT_PROMPT = "Esegui una trascrizione e riassunto dei file audio forniti."

    # backend -> (settings key, default prompt, label)
    PROMPT_SETTINGS = {
        BACKEND_GEMINI: (
            "userPrompt",
            DEFAULT_PROMPT,
            "Prompt per Gemini (cosa fare con l'audio?):",
        ),
        BACKEND_OPENAI: (
            "transcribePrompt",
            "",
            "Contesto per la trascrizione (opzionale: argomento, nomi, termini tecnici):",
        ),
    }

    def __init__(self):
        logging.info("AudioSummaryApp initialization started.")
        super().__init__()
        self.setWindowTitle("Riassunto audio - messaggi vocali & lezioni")
        self.setGeometry(100, 100, 1300, 1000)

        self.gemini_api_key = None
        self.openai_api_key = None
        self.load_api_key()
        self.processing_thread = None
        self.audio_file_paths = []
        self.last_result_text = ""
        self.result_is_unsaved = False

        self.chat_session = None
        self.chat_model = None
        self.full_chat_markdown = ""
        self.audio_durations = {}  # file path -> seconds, for cost estimates

        # Streamed text is rendered at most a few times per second
        self.pending_markdown = None
        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(150)
        self.render_timer.timeout.connect(self.render_pending_markdown)
        self.last_cost = None

        self.settings = QSettings("BitreyDev", "AudioSummaryApp")

        self.init_ui()
        self.load_settings()
        self.apply_dark_theme()
        logging.info("AudioSummaryApp initialization completed.")

    def apply_dark_theme(self):
        logging.info("Applying dark theme.")
        self.setStyleSheet(
            """
            QWidget {
                background-color: #333333;
                color: #ffffff;
            }
            QPushButton {
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #666666;
                border: 1px solid #777777;
            }
            QPushButton:pressed {
                background-color: #444444;
            }
            QPushButton:disabled {
                background-color: #3a3a3a;
                color: #666666;
                border: 1px solid #444444;
            }
            QLabel {
                color: #ffffff;
            }
            QLabel:disabled {
                color: #888888;
            }
            QTextEdit {
                background-color: #444444;
                color: #ffffff;
                border: 1px solid #666666;
            }
            QTextEdit:disabled {
                background-color: #2a2a2a;
                color: #666666;
                border: 1px solid #444444;
            }
            QFrame[frameShape="4"] {
                color: #555555;
            }
            QProgressBar {
                background-color: #444444;
                color: #ffffff;
                border: 1px solid #666666;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #6a994e; /* Green accent color */
            }
            QCheckBox {
                color: #ffffff;
            }
            QCheckBox:disabled {
                color: #666666;
            }
            QCheckBox::indicator:disabled {
                background-color: #2a2a2a;
                border: 1px solid #444444;
            }
            QComboBox {
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 4px;
                border-radius: 3px;
            }
            QComboBox:disabled {
                background-color: #3a3a3a;
                color: #666666;
            }
            QToolButton {
                background-color: #555555;
                color: #ffffff;
                border: 1px solid #666666;
                padding: 4px 8px;
                border-radius: 3px;
            }
            QToolButton:hover {
                background-color: #666666;
            }
            QComboBox QAbstractItemView {
                background-color: #444444;
                color: #ffffff;
                selection-background-color: #6a994e;
            }
        """
        )
        logging.info("Dark theme applied.")

    def load_api_key(self):
        logging.info("Loading API keys from .env file.")
        load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            warning_msg_gemini = "GEMINI_API_KEY is not set in the .env file."
            logging.warning(warning_msg_gemini)
            QMessageBox.warning(
                self,
                "Chiave API Gemini mancante",
                "O, imposta GEMINI_API_KEY nel tuo file .env.",
            )
        else:
            logging.info("GEMINI_API_KEY loaded successfully.")

        self.gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        logging.info(f"Using Gemini model: {self.gemini_model}")

        # Only needed for transcription with OpenAI, checked when processing starts
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        logging.info("API keys loading process completed.")

    def init_ui(self):
        logging.info("Initializing user interface.")
        self.layout = QVBoxLayout()

        file_selection_layout = QHBoxLayout()
        self.select_file_button = QPushButton("Seleziona file audio", self)
        self.select_file_button.clicked.connect(self.select_audio_file)
        file_selection_layout.addWidget(self.select_file_button)
        self.file_path_label = QLabel("Nessun file audio selezionato", self)
        file_selection_layout.addWidget(self.file_path_label)

        file_selection_layout.addStretch(1)

        self.model_combo = QComboBox(self)
        self.model_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.model_combo.setMaxVisibleItems(30)
        file_selection_layout.addWidget(self.model_combo)

        self.model_info_button = QToolButton(self)
        self.model_info_button.setText("?")
        self.model_info_button.setToolTip("Info sul modello selezionato")
        self.model_info_button.clicked.connect(self.show_model_info)
        file_selection_layout.addWidget(self.model_info_button)

        self.reset_button = QPushButton("Reset Prompt", self)
        self.reset_button.clicked.connect(self.reset_prompt)
        file_selection_layout.addWidget(self.reset_button)

        self.layout.addLayout(file_selection_layout)

        self.prompt_label = QLabel("", self)
        self.layout.addWidget(self.prompt_label)
        self.user_prompt_text_edit = QTextEdit(self)
        self.user_prompt_text_edit.setFixedHeight(180)
        self.layout.addWidget(self.user_prompt_text_edit)

        # Aggiungi la checkbox per la ricerca Google
        self.search_checkbox = QCheckBox(
            "Abilita ricerca Google (per risposte più aggiornate su eventi recenti)",
            self,
        )
        self.search_checkbox.setChecked(False)  # Imposta lo stato predefinito
        self.layout.addWidget(self.search_checkbox)

        self.process_button = QPushButton("Elabora!", self)
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        self.layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.layout.addWidget(QLabel("Risultato:"))
        self.output_text_edit = QTextEdit(self)
        self.output_text_edit.setReadOnly(True)
        self.layout.addWidget(self.output_text_edit)

        # Added Follow-up UI
        self.separator = QFrame(self)
        self.separator.setFrameShape(QFrame.Shape.HLine)
        self.separator.setFrameShadow(QFrame.Shadow.Sunken)
        self.layout.addWidget(self.separator)

        self.followup_label = QLabel("Continua la conversazione:", self)
        self.layout.addWidget(self.followup_label)

        followup_layout = QHBoxLayout()
        self.followup_input = QTextEdit(self)
        self.followup_input.setFixedHeight(70)
        self.followup_input.setPlaceholderText(
            "Es: fammi un riassunto ancora più breve..."
        )
        self.followup_input.setEnabled(False)
        followup_layout.addWidget(self.followup_input)

        self.followup_button = QPushButton("Invia continuo", self)
        self.followup_button.setEnabled(False)
        self.followup_button.clicked.connect(self.send_followup)
        followup_layout.addWidget(self.followup_button)

        self.layout.addLayout(followup_layout)

        status_layout = QHBoxLayout()
        self.status_label = QLabel("", self)
        status_layout.addWidget(self.status_label, 1)

        self.cost_label = QLabel("", self)
        self.cost_label.setToolTip(
            "Costi calcolati con i prezzi del paid tier. Nel free tier di Gemini "
            "non paghi, e la ricerca Google è esclusa dal calcolo."
        )
        status_layout.addWidget(self.cost_label)

        self.reset_cost_button = QToolButton(self)
        self.reset_cost_button.setText("Azzera")
        self.reset_cost_button.setToolTip("Azzera il totale speso")
        self.reset_cost_button.clicked.connect(self.reset_total_cost)
        status_layout.addWidget(self.reset_cost_button)

        self.layout.addLayout(status_layout)

        self.setLayout(self.layout)

        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.handle_save_shortcut)

        logging.info("User interface initialized.")

    def current_backend(self):
        return self.model_combo.currentData().split(":", 1)[0]

    def current_model(self):
        return self.model_combo.currentData().split(":", 1)[1]

    def populate_model_combo(self):
        """Fill the model picker with Gemini and OpenAI models, keeping the selection."""
        selected = self.settings.value(
            "selectedModel", f"{BACKEND_GEMINI}:{self.gemini_model}"
        )
        gemini_models = list(self.gemini_models)
        if not any(m["id"] == self.gemini_model for m in gemini_models):
            gemini_models.insert(0, {"id": self.gemini_model})

        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        combo_model = self.model_combo.model()
        for header, backend, model_ids in (
            ("Gemini", BACKEND_GEMINI, [m["id"] for m in gemini_models]),
            ("OpenAI", BACKEND_OPENAI, list(OPENAI_MODELS)),
        ):
            if self.model_combo.count():
                self.model_combo.insertSeparator(self.model_combo.count())
            self.model_combo.addItem(header)
            if isinstance(combo_model, QStandardItemModel):
                # Section headers can't be selected
                combo_model.item(self.model_combo.count() - 1).setEnabled(False)
            for model_id in model_ids:
                self.model_combo.addItem(model_id, f"{backend}:{model_id}")
                self.model_combo.setItemData(
                    self.model_combo.count() - 1,
                    self.model_info_text(backend, model_id),
                    Qt.ItemDataRole.ToolTipRole,
                )

        index = self.model_combo.findData(selected)
        if index < 0:
            index = self.model_combo.findData(f"{BACKEND_GEMINI}:{self.gemini_model}")
        self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)

    def model_info_text(self, backend, model_id):
        if backend == BACKEND_OPENAI:
            return f"<b>{model_id}</b> (OpenAI)<br><br>{OPENAI_MODELS[model_id]['info']}"

        info = next((m for m in self.gemini_models if m["id"] == model_id), {})
        text = f"<b>{info.get('display_name', model_id)}</b> ({model_id})"
        if model_id in GEMINI_MODEL_INFO:
            text += f"<br><br>{GEMINI_MODEL_INFO[model_id]}"
        else:
            if info.get("description") and info["description"] != info["display_name"]:
                text += f"<br><br>{info['description']}"
            text += (
                f'<br><br>Prezzi: <a href="{GEMINI_PRICING_URL}">{GEMINI_PRICING_URL}</a>'
            )
        if info.get("input_token_limit"):
            text += (
                f"<br><br>Token in input: {info['input_token_limit']:,}"
                f"<br>Token in output: {info['output_token_limit']:,}"
            )
        return text

    def show_model_info(self):
        QMessageBox.information(
            self,
            "Info modello",
            self.model_info_text(self.current_backend(), self.current_model()),
        )

    def load_gemini_models(self):
        """Show the cached Gemini models now and refresh the list in the background."""
        try:
            self.gemini_models = json.loads(self.settings.value("geminiModels", "[]"))
        except json.JSONDecodeError:
            self.gemini_models = []

        if self.gemini_api_key:
            self.gemini_models_thread = GeminiModelsThread(self.gemini_api_key)
            self.gemini_models_thread.models_loaded.connect(self.on_gemini_models_loaded)
            self.gemini_models_thread.start()

    def on_gemini_models_loaded(self, models):
        if models == self.gemini_models:
            return
        self.gemini_models = models
        self.settings.setValue("geminiModels", json.dumps(models))
        self.populate_model_combo()
        self.apply_backend_ui()
        self.update_cost_label()

    def load_settings(self):
        logging.info("Loading settings from QSettings.")
        self.load_gemini_models()
        self.populate_model_combo()
        self.apply_backend_ui()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.user_prompt_text_edit.textChanged.connect(self.save_settings)
        self.update_cost_label()

        saved_response = self.settings.value("lastResponseMarkdown", "")
        if saved_response:
            self.full_chat_markdown = saved_response
            self.last_result_text = saved_response
            self.result_is_unsaved = False
            self.output_text_edit.setHtml(mistune.html(saved_response))
        logging.info("Settings loaded.")

    def apply_backend_ui(self):
        """Show the prompt and options that belong to the selected backend."""
        settings_key, default_prompt, label = self.PROMPT_SETTINGS[self.current_backend()]
        self.prompt_label.setText(label)
        self.user_prompt_text_edit.blockSignals(True)
        self.user_prompt_text_edit.setPlainText(
            self.settings.value(settings_key, default_prompt)
        )
        self.user_prompt_text_edit.blockSignals(False)
        self.user_prompt_text_edit.setPlaceholderText(
            "Es: lezione di analisi 2, prof. Rossi, teorema di Lagrange"
            if self.current_backend() == BACKEND_OPENAI
            else ""
        )
        if not self.prompt_supported():
            self.prompt_label.setText(f"{self.current_model()} non accetta un prompt.")
        self.user_prompt_text_edit.setEnabled(self.prompt_supported())
        self.search_checkbox.setEnabled(self.search_supported())

    def search_supported(self):
        return (
            self.current_backend() == BACKEND_GEMINI
            and not is_gemini_transcribe_model(self.current_model())
        )

    def prompt_supported(self):
        return (
            self.current_backend() == BACKEND_GEMINI
            or OPENAI_MODELS[self.current_model()]["prompt"]
        )

    def on_model_changed(self):
        logging.info(f"Selected model: {self.model_combo.currentData()}")
        self.settings.setValue("selectedModel", self.model_combo.currentData())
        self.apply_backend_ui()
        self.update_cost_label()

    def update_cost_label(self):
        """Show the estimate for the selected files, the last cost and the total."""
        parts = []
        per_minute = estimated_cost_per_minute(
            self.current_backend(), self.current_model()
        )
        if self.audio_file_paths and per_minute is not None:
            for path in self.audio_file_paths:
                if path not in self.audio_durations:
                    self.audio_durations[path] = get_audio_duration(path) or 0
            minutes = sum(self.audio_durations[p] for p in self.audio_file_paths) / 60
            estimate = f"Stima: ~{format_cost(minutes * per_minute)}"
            if self.current_backend() == BACKEND_GEMINI and not gemini_rates(
                self.current_model()
            ).get("per_minute"):
                estimate += " + risposta"
            parts.append(estimate)
        if self.last_cost is not None:
            parts.append(f"Ultima: {format_cost(self.last_cost)}")
        total = float(self.settings.value("totalCost", 0.0))
        parts.append(f"Totale: {format_cost(total)}")
        self.cost_label.setText(" · ".join(parts))

    def add_cost(self, cost):
        logging.info(f"Request cost: {format_cost(cost)}")
        self.last_cost = (self.last_cost or 0) + cost
        total = float(self.settings.value("totalCost", 0.0)) + cost
        self.settings.setValue("totalCost", total)
        self.update_cost_label()

    def reset_total_cost(self):
        self.settings.setValue("totalCost", 0.0)
        self.update_cost_label()

    def save_settings(self):
        logging.info("Saving settings to QSettings.")
        settings_key = self.PROMPT_SETTINGS[self.current_backend()][0]
        self.settings.setValue(settings_key, self.user_prompt_text_edit.toPlainText())
        logging.info("Settings saved.")

    def save_cached_response(self):
        self.settings.setValue("lastResponseMarkdown", self.full_chat_markdown)

    def reset_prompt(self):
        logging.info("Resetting prompt to default.")
        self.user_prompt_text_edit.setPlainText(
            self.PROMPT_SETTINGS[self.current_backend()][1]
        )
        self.save_settings()
        logging.info("Prompt reset to default.")

    def select_audio_file(self):
        logging.info("Select audio file button clicked.")
        file_dialog = QFileDialog()
        last_directory = self.settings.value("lastInputDirectory", "")

        # Gemini supports more audio formats
        supported_formats = "*.ogg *.wav *.mp3 *.flac *.m4a *.aac *.aiff"
        file_paths, _ = file_dialog.getOpenFileNames(
            self,
            "Seleziona file audio",
            last_directory,
            f"File Audio ({supported_formats})",
        )

        if file_paths:
            last_directory = os.path.dirname(file_paths[0])
            self.settings.setValue("lastInputDirectory", last_directory)

            file_paths_with_ctime = [
                (path, os.path.getctime(path)) for path in file_paths
            ]
            file_paths_with_ctime.sort(key=lambda item: item[1])
            self.audio_file_paths = [path for path, ctime in file_paths_with_ctime]

            if len(self.audio_file_paths) == 1:
                display_text = (
                    f"File selezionato: {os.path.basename(self.audio_file_paths[0])}"
                )
            else:
                display_text = f"{len(self.audio_file_paths)} file selezionati"

            self.file_path_label.setText(display_text)
            self.update_cost_label()
            self.process_button.setEnabled(True)
            self.output_text_edit.clear()
            self.status_label.clear()
        else:
            self.file_path_label.setText("Nessun file audio selezionato")
            self.process_button.setEnabled(False)
            self.audio_file_paths = []
            self.update_cost_label()
            logging.info("No audio file selected.")

    def start_processing(self):
        logging.info("Start processing button clicked.")
        self.result_is_unsaved = False
        if not self.audio_file_paths:
            QMessageBox.warning(
                self, "Nessun file selezionato", "O, seleziona prima un file audio."
            )
            return

        backend = self.current_backend()
        required_key = (
            "OPENAI_API_KEY" if backend == BACKEND_OPENAI else "GEMINI_API_KEY"
        )
        api_key = (
            self.openai_api_key if backend == BACKEND_OPENAI else self.gemini_api_key
        )
        if not api_key:
            QMessageBox.warning(
                self,
                "Chiave API mancante",
                f"Imposta {required_key} nel tuo file .env e riavvia.",
            )
            return

        self.status_label.setText("Avvio del processo...")
        self.process_button.setEnabled(False)
        self.user_prompt_text_edit.setEnabled(False)  # Disable prompt during processing
        self.search_checkbox.setEnabled(False)  # Disabilita anche la checkbox
        self.model_combo.setEnabled(False)
        self.followup_input.setEnabled(False)
        self.followup_button.setEnabled(False)
        self.chat_session = None
        self.chat_model = self.current_model()
        self.full_chat_markdown = ""
        self.last_cost = None

        self.progress_bar.setValue(0)
        self.output_text_edit.clear()

        user_prompt = self.user_prompt_text_edit.toPlainText()

        # Leggi lo stato della checkbox
        use_google_search = self.search_checkbox.isChecked() and self.search_supported()

        logging.info("Creating and starting processing thread.")

        if backend == BACKEND_OPENAI:
            self.processing_thread = TranscriptionThread(
                self.audio_file_paths,
                self.openai_api_key,
                self.current_model(),
                user_prompt.strip(),
            )
        else:
            self.processing_thread = ProcessingThread(
                self.audio_file_paths,
                self.gemini_api_key,
                self.current_model(),
                user_prompt,
                enable_search=use_google_search,
            )
        self.processing_thread.processing_finished.connect(self.display_result)
        self.processing_thread.error_occurred.connect(self.display_error)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.status_update.connect(self.update_status)
        self.processing_thread.cost_update.connect(self.add_cost)
        self.processing_thread.partial_text.connect(self.show_partial_markdown)
        self.processing_thread.start()
        logging.info("Processing thread started.")

    def show_partial_markdown(self, markdown):
        self.pending_markdown = markdown
        if not self.render_timer.isActive():
            self.render_timer.start()

    def render_pending_markdown(self):
        if self.pending_markdown is None:
            return
        scroll_bar = self.output_text_edit.verticalScrollBar()
        # Follow the new text only if the user hasn't scrolled up
        at_bottom = scroll_bar.value() >= scroll_bar.maximum() - 20
        self.output_text_edit.setHtml(mistune.html(self.pending_markdown))
        self.pending_markdown = None
        if at_bottom:
            scroll_bar.setValue(scroll_bar.maximum())

    def stop_partial_rendering(self):
        self.render_timer.stop()
        self.pending_markdown = None

    def display_result(self, result_text, chat_session):
        logging.info("Displaying processing result.")
        self.stop_partial_rendering()

        self.chat_session = chat_session
        self.full_chat_markdown = result_text

        html_content = mistune.html(self.full_chat_markdown)
        self.output_text_edit.setHtml(html_content)

        self.status_label.setText("Finito!!")
        self.process_button.setEnabled(True)
        self.user_prompt_text_edit.setEnabled(self.prompt_supported())
        self.model_combo.setEnabled(True)
        self.search_checkbox.setEnabled(self.search_supported())

        # Abilita box di followup (solo con Gemini, che ha una sessione chat)
        self.followup_input.setEnabled(chat_session is not None)
        self.followup_button.setEnabled(chat_session is not None)

        self.progress_bar.setValue(100)

        # Store result for saving
        self.last_result_text = self.full_chat_markdown
        self.result_is_unsaved = bool(result_text)
        self.save_cached_response()

        self.prompt_save_file()

    def send_followup(self):
        follow_up_text = self.followup_input.toPlainText().strip()
        if not follow_up_text or not self.chat_session:
            return

        self.followup_input.setEnabled(False)
        self.followup_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.last_cost = None
        self.followup_thread = FollowUpThread(
            self.chat_session, self.chat_model, follow_up_text
        )
        self.followup_thread.followup_finished.connect(
            lambda text: self.display_followup(follow_up_text, text)
        )
        self.followup_thread.error_occurred.connect(self.display_followup_error)
        self.followup_thread.progress_update.connect(self.update_progress)
        self.followup_thread.status_update.connect(self.update_status)
        self.followup_thread.cost_update.connect(self.add_cost)
        self.followup_thread.partial_text.connect(
            lambda text: self.show_partial_markdown(
                f"{self.full_chat_markdown}\n\n---\n\n{text}"
            )
        )
        self.followup_thread.start()

    def display_followup(self, _user_text, result_text):
        self.stop_partial_rendering()
        self.full_chat_markdown += f"\n\n---\n\n{result_text}"

        html_content = mistune.html(self.full_chat_markdown)
        self.output_text_edit.setHtml(html_content)

        self.status_label.setText("Finito!!")
        self.followup_input.clear()
        self.followup_input.setEnabled(True)
        self.followup_button.setEnabled(True)
        self.progress_bar.setValue(100)

        self.last_result_text = self.full_chat_markdown
        self.result_is_unsaved = True
        self.save_cached_response()

        # Scorri in basso
        scroll_bar = self.output_text_edit.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def display_followup_error(self, error_message):
        logging.error(f"Error in follow-up: {error_message}")
        QMessageBox.critical(self, "Errore di elaborazione", error_message)
        self.status_label.setText("Errore durante il follow-up.")
        self.followup_input.setEnabled(True)
        self.followup_button.setEnabled(True)
        self.progress_bar.setValue(0)
        logging.info("Result displayed and UI updated.")

    def prompt_save_file(self):
        if not self.last_result_text:
            return

        default_suffix = (
            "transcription.md"
            if self.current_backend() == BACKEND_OPENAI
            else "summary.md"
        )
        file_filter = "Markdown Files (*.md)"
        dialog_title = "Salva riassunto/trascrizione come Markdown"
        content_to_save = self.last_result_text

        file_dialog = QFileDialog()
        last_save_directory = self.settings.value("lastSaveDirectory", "")
        # Use the first audio file's name as a base for the output file
        base_name = os.path.splitext(os.path.basename(self.audio_file_paths[0]))[0]
        default_file_name = os.path.join(
            last_save_directory, f"{base_name}_{default_suffix}"
        )

        file_path, _ = file_dialog.getSaveFileName(
            self, dialog_title, default_file_name, file_filter
        )

        if file_path:
            last_save_directory = os.path.dirname(file_path)
            self.settings.setValue("lastSaveDirectory", last_save_directory)
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content_to_save)
                self.status_label.setText(f"File salvato in: {file_path}")
                self.result_is_unsaved = False
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Errore di salvataggio",
                    f"Errore durante il salvataggio del file: {e}",
                )
        else:
            self.status_label.setText(
                "File non salvato - puoi premere Ctrl+S per salvare."
            )

    def handle_save_shortcut(self):
        if self.result_is_unsaved:
            self.prompt_save_file()

    def display_error(self, error_message, file_path):
        logging.error(f"Error occurred: {error_message} for file: {file_path}")
        QMessageBox.critical(self, "Errore di elaborazione", error_message)
        file_name = os.path.basename(file_path)
        self.status_label.setText(f"Errore durante l'elaborazione di {file_name}.")
        self.process_button.setEnabled(True)
        self.user_prompt_text_edit.setEnabled(self.prompt_supported())
        self.model_combo.setEnabled(True)
        self.search_checkbox.setEnabled(self.search_supported())
        # Il follow up rimane disabilitato perche' la chat non e' pronta
        self.progress_bar.setValue(0)
        logging.error(f"Error displayed in UI for file: {file_name}")

    def update_progress(self, progress_value):
        self.progress_bar.setValue(progress_value)
        logging.debug(f"Progress bar updated to: {progress_value}%")

    def update_status(self, message):
        """Updates the status label with a message from the processing thread."""
        self.status_label.setText(message)


if __name__ == "__main__":
    logging.info("Application started.")
    app = QApplication(sys.argv)
    main_window = AudioSummaryApp()
    main_window.show()
    exit_code = app.exec()
    logging.info(f"Application exited with code: {exit_code}")
    sys.exit(exit_code)
