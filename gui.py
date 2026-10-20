#!/usr/bin/env python3

import base64
import glob
import hashlib
import logging
import os
import subprocess
import sys
import tempfile

import mistune
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI
from PyQt6.QtCore import QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
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
    QVBoxLayout,
    QWidget,
)

# Configure logging to console
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

OPENAI_TRANSCRIBE_MODEL = "gpt-transcribe"
# Formats accepted by the OpenAI Transcriptions API; anything else is converted with ffmpeg
OPENAI_SUPPORTED_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
OPENAI_MAX_FILE_BYTES = 24 * 1024 * 1024  # API limit is 25 MB, keep a small margin
OPENAI_CHUNK_SECONDS = 30 * 60  # ~7 MB per chunk at 32 kbps

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

    def __init__(self, chat_session, user_text):
        super().__init__()
        self.chat_session = chat_session
        self.user_text = user_text

    def run(self):
        logging.info("FollowUpThread started.")
        try:
            self.progress_update.emit(25)
            self.status_update.emit("Invio follow-up in corso...")
            response = self.chat_session.send_message(self.user_text)
            self.progress_update.emit(95)
            self.status_update.emit("Risposta ricevuta.")
            self.followup_finished.emit(response.text)
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
            response = chat_session.send_message(gemini_contents)
            result_text = response.text

            self.status_update.emit("Risposta ricevuta. Finalizzazione...")
            logging.info("Received response from Gemini.")
            self.progress_update.emit(95)

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

    def __init__(self, audio_file_paths, openai_api_key, context_prompt):
        super().__init__()
        self.audio_file_paths = audio_file_paths
        self.openai_api_key = openai_api_key
        self.context_prompt = context_prompt

    def prepare_audio(self, file_path, index, work_dir):
        """Return the list of files to send to the API for this audio file."""
        ext = os.path.splitext(file_path)[1].lower()
        if (
            ext in OPENAI_SUPPORTED_EXTENSIONS
            and os.path.getsize(file_path) <= OPENAI_MAX_FILE_BYTES
        ):
            return [file_path]

        # Mono 16 kHz 32 kbps mp3 is plenty for speech and keeps files small
        converted_path = os.path.join(work_dir, f"file{index}.mp3")
        logging.info(f"Converting {file_path} to {converted_path}...")
        run_ffmpeg(
            ["-i", file_path, "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", converted_path]
        )
        if os.path.getsize(converted_path) <= OPENAI_MAX_FILE_BYTES:
            return [converted_path]

        logging.info(f"{converted_path} still too large, splitting into chunks...")
        chunk_pattern = os.path.join(work_dir, f"file{index}_chunk%03d.mp3")
        run_ffmpeg(
            [
                "-i",
                converted_path,
                "-f",
                "segment",
                "-segment_time",
                str(OPENAI_CHUNK_SECONDS),
                "-reset_timestamps",
                "1",
                "-c",
                "copy",
                chunk_pattern,
            ]
        )
        return sorted(glob.glob(os.path.join(work_dir, f"file{index}_chunk*.mp3")))

    def run(self):
        logging.info("Transcription thread started.")
        file_name_for_error = "audio files"

        try:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set in the .env file")

            self.status_update.emit("Inizializzazione client OpenAI...")
            client = OpenAI(api_key=self.openai_api_key)
            self.progress_update.emit(5)

            extra_args = {}
            if self.context_prompt:
                extra_args["prompt"] = self.context_prompt

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

                    chunk_texts = []
                    for chunk_index, chunk_path in enumerate(chunk_paths):
                        self.status_update.emit(
                            f"File {index + 1}/{num_files}: trascrizione di {file_name_for_error}"
                            f" (parte {chunk_index + 1}/{len(chunk_paths)})..."
                        )
                        logging.info(f"Transcribing {chunk_path}...")
                        with open(chunk_path, "rb") as audio_file:
                            transcription = client.audio.transcriptions.create(
                                model=OPENAI_TRANSCRIBE_MODEL,
                                file=audio_file,
                                **extra_args,
                            )
                        chunk_texts.append(transcription.text.strip())

                    text = "\n\n".join(chunk_texts)
                    if num_files > 1:
                        text = f"## {file_name_for_error}\n\n{text}"
                    sections.append(text)

            self.status_update.emit("Trascrizione completata. Finalizzazione...")
            self.progress_update.emit(95)
            # No chat session: follow-up is not available for plain transcription
            self.processing_finished.emit("\n\n".join(sections), None)
            self.progress_update.emit(100)
            logging.info("Transcription thread finished successfully.")

        except Exception as e:
            error_message = f"Error transcribing {file_name_for_error}: {e}"
            logging.error(error_message, exc_info=True)
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
        self.full_chat_markdown = ""

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

        self.reset_button = QPushButton("Reset Prompt", self)
        self.reset_button.clicked.connect(self.reset_prompt)
        file_selection_layout.addWidget(self.reset_button)

        self.layout.addLayout(file_selection_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Modello:", self))
        self.model_combo = QComboBox(self)
        self.model_combo.addItem(
            f"Gemini ({self.gemini_model}) - analisi, riassunto, domande", BACKEND_GEMINI
        )
        self.model_combo.addItem(
            f"OpenAI ({OPENAI_TRANSCRIBE_MODEL}) - trascrizione fedele", BACKEND_OPENAI
        )
        model_layout.addWidget(self.model_combo, 1)
        self.layout.addLayout(model_layout)

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

        self.status_label = QLabel("", self)
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)

        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.handle_save_shortcut)

        logging.info("User interface initialized.")

    def current_backend(self):
        return self.model_combo.currentData()

    def load_settings(self):
        logging.info("Loading settings from QSettings.")
        saved_backend = self.settings.value("selectedBackend", BACKEND_GEMINI)
        backend_index = self.model_combo.findData(saved_backend)
        self.model_combo.setCurrentIndex(max(backend_index, 0))
        self.apply_backend_ui()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        self.user_prompt_text_edit.textChanged.connect(self.save_settings)

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
        self.search_checkbox.setEnabled(self.current_backend() == BACKEND_GEMINI)

    def on_model_changed(self):
        logging.info(f"Selected backend: {self.current_backend()}")
        self.settings.setValue("selectedBackend", self.current_backend())
        self.apply_backend_ui()

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
            self.process_button.setEnabled(True)
            self.output_text_edit.clear()
            self.status_label.clear()
        else:
            self.file_path_label.setText("Nessun file audio selezionato")
            self.process_button.setEnabled(False)
            self.audio_file_paths = []
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
        self.full_chat_markdown = ""

        self.progress_bar.setValue(0)
        self.output_text_edit.clear()

        user_prompt = self.user_prompt_text_edit.toPlainText()

        # Leggi lo stato della checkbox
        use_google_search = self.search_checkbox.isChecked()

        logging.info("Creating and starting processing thread.")

        if backend == BACKEND_OPENAI:
            self.processing_thread = TranscriptionThread(
                self.audio_file_paths,
                self.openai_api_key,
                user_prompt.strip(),
            )
        else:
            self.processing_thread = ProcessingThread(
                self.audio_file_paths,
                self.gemini_api_key,
                self.gemini_model,
                user_prompt,
                enable_search=use_google_search,
            )
        self.processing_thread.processing_finished.connect(self.display_result)
        self.processing_thread.error_occurred.connect(self.display_error)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.status_update.connect(self.update_status)
        self.processing_thread.start()
        logging.info("Processing thread started.")

    def display_result(self, result_text, chat_session):
        logging.info("Displaying processing result.")

        self.chat_session = chat_session
        self.full_chat_markdown = result_text

        html_content = mistune.html(self.full_chat_markdown)
        self.output_text_edit.setHtml(html_content)

        self.status_label.setText("Finito!!")
        self.process_button.setEnabled(True)
        self.user_prompt_text_edit.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.search_checkbox.setEnabled(self.current_backend() == BACKEND_GEMINI)

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

        self.followup_thread = FollowUpThread(self.chat_session, follow_up_text)
        self.followup_thread.followup_finished.connect(
            lambda text: self.display_followup(follow_up_text, text)
        )
        self.followup_thread.error_occurred.connect(self.display_followup_error)
        self.followup_thread.progress_update.connect(self.update_progress)
        self.followup_thread.status_update.connect(self.update_status)
        self.followup_thread.start()

    def display_followup(self, _user_text, result_text):
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
        self.user_prompt_text_edit.setEnabled(True)
        self.model_combo.setEnabled(True)
        self.search_checkbox.setEnabled(self.current_backend() == BACKEND_GEMINI)
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
