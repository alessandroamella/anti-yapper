#!/usr/bin/env python3

import logging  # Import the logging module
import os
import sys
import tempfile
import math

import mistune
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pydub import AudioSegment
from PyQt6.QtCore import QSettings, QThread, pyqtSignal  # Import QSettings
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (  # Import QCheckBox
    QApplication,
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Configure logging to console
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the Whisper API file size limit (25MB). We use 24MB to be safe.
WHISPER_API_LIMIT_BYTES = 24 * 1024 * 1024


class SummarizationThread(QThread):
    """
    Thread to handle audio processing, transcription (OpenAI Whisper),
    and summarization (Gemini) to avoid blocking the GUI.
    Handles single or merged audio files based on 'merge_audio' flag.
    Splits files larger than 25MB and handles multiple files intelligently based on size.
    """

    summary_finished = pyqtSignal(str)
    transcription_finished = pyqtSignal(str)  # Signal for transcription output
    error_occurred = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)

    # --- MODIFY THE __init__ signature ---
    def __init__(
        self,
        audio_file_paths,
        openai_api_key,
        gemini_api_key,
        summarization_prompt,
        merge_audio=True,
        transcription_only=False,
        whisper_language="",  # new, with a default
        whisper_prompt="",  # new, with a default
    ):
        # --- END MODIFICATION ---
        super().__init__()
        self.audio_file_paths = audio_file_paths
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.summarization_prompt = summarization_prompt
        self.merge_audio = merge_audio
        self.transcription_only = transcription_only

        # --- ADD THESE LINES to store the new values ---
        self.whisper_language = whisper_language
        self.whisper_prompt = whisper_prompt
        # --- END ---

        self.openai_client = None
        self.gemini_client = None
        self.temp_files_to_clean = []  # Store all temp file objects for cleanup
        self.final_summary_markdown = ""
        self.temp_transcription_file = None

    def run(self):
        logging.info(
            f"Summarization thread started. Merge audio flag: {self.merge_audio}, Transcription Only: {self.transcription_only}"
        )
        file_name_for_error = "audio files"
        full_transcription_text = ""

        try:
            # --- API Key and Client Initialization ---
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set in the .env file")
            if not self.gemini_api_key and not self.transcription_only:
                raise ValueError("GEMINI_API_KEY is not set in the .env file")

            self.openai_client = OpenAI(api_key=self.openai_api_key)
            if not self.transcription_only:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                logging.info("Gemini client initialized.")
            logging.info("OpenAI client initialized.")
            self.progress_update.emit(10)

            # --- File Processing Logic based on Size ---
            files_to_process = []

            # Case 1: Single file selected
            if len(self.audio_file_paths) == 1:
                file_path = self.audio_file_paths[0]
                file_size = os.path.getsize(file_path)

                if file_size < WHISPER_API_LIMIT_BYTES:
                    logging.info(
                        f"Single file '{os.path.basename(file_path)}' is under the 25MB limit. Processing as is."
                    )
                    files_to_process.append(file_path)
                else:
                    logging.warning(
                        f"File '{os.path.basename(file_path)}' is {file_size / (1024*1024):.2f}MB, which is over the 25MB limit. Splitting into chunks."
                    )
                    self.progress_update.emit(15)
                    audio = AudioSegment.from_file(file_path)
                    duration_ms = len(audio)
                    # Calculate chunk duration to stay under the limit
                    chunk_duration_ms = math.floor(
                        (duration_ms * WHISPER_API_LIMIT_BYTES) / file_size
                    )

                    for i, start_ms in enumerate(
                        range(0, duration_ms, chunk_duration_ms)
                    ):
                        end_ms = start_ms + chunk_duration_ms
                        chunk = audio[start_ms:end_ms]

                        temp_chunk_file = tempfile.NamedTemporaryFile(
                            suffix=".mp3", delete=False
                        )
                        chunk.export(temp_chunk_file.name, format="mp3")
                        files_to_process.append(temp_chunk_file.name)
                        self.temp_files_to_clean.append(temp_chunk_file)
                        logging.info(
                            f"Created chunk {i+1} for transcription: {temp_chunk_file.name}. Length: {chunk.duration_seconds:.2f} Size: {os.path.getsize(temp_chunk_file.name) / (1024*1024):.2f}MB"
                        )
                    logging.info(f"Split file into {len(files_to_process)} chunks.")

            # Case 2: Multiple files selected
            elif len(self.audio_file_paths) > 1:
                total_size = 0
                for path in self.audio_file_paths:
                    size = os.path.getsize(path)
                    if size > WHISPER_API_LIMIT_BYTES:
                        raise ValueError(
                            f"File '{os.path.basename(path)}' is larger than 25MB. Processing multiple files where one is oversized is not supported."
                        )
                    total_size += size

                if total_size < WHISPER_API_LIMIT_BYTES and self.merge_audio:
                    logging.info(
                        f"Total size of {len(self.audio_file_paths)} files is {total_size / (1024*1024):.2f}MB. Merging them."
                    )
                    self.progress_update.emit(20)
                    merged_audio = AudioSegment.from_file(self.audio_file_paths[0])
                    for file_path in self.audio_file_paths[1:]:
                        merged_audio += AudioSegment.from_file(file_path)

                    temp_merged_file = tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False
                    )
                    merged_audio.export(
                        temp_merged_file.name, format="mp3", bitrate="128k"
                    )
                    files_to_process.append(temp_merged_file.name)
                    self.temp_files_to_clean.append(temp_merged_file)
                    logging.info(
                        f"Merged audio saved to temporary file: {temp_merged_file.name}"
                    )
                else:
                    if self.merge_audio:
                        logging.warning(
                            f"Total size of files is {total_size / (1024*1024):.2f}MB, which is over the 25MB limit. Processing files individually instead of merging."
                        )
                    else:
                        logging.info(
                            "Merge option is disabled. Processing files individually."
                        )
                    files_to_process = self.audio_file_paths

            self.progress_update.emit(25)

            # --- Unified Transcription Loop ---
            combined_transcriptions = []
            num_files = len(files_to_process)
            for index, audio_file_path in enumerate(files_to_process):
                file_name_for_error = os.path.basename(audio_file_path)
                logging.info(
                    f"Processing chunk/file {index + 1}/{num_files}: {file_name_for_error}"
                )
                # Calculate progress more dynamically
                self.progress_update.emit(25 + int((55 / num_files) * index))

                with open(audio_file_path, "rb") as audio_file_to_process:
                    # --- PREPARE PARAMS DICTIONARY ---
                    transcription_params = {
                        "model": "whisper-1",
                        "file": audio_file_to_process,
                    }
                    if self.whisper_language:  # Only add if not empty
                        transcription_params["language"] = self.whisper_language
                    if self.whisper_prompt:  # Only add if not empty
                        transcription_params["prompt"] = self.whisper_prompt

                    # --- MAKE THE API CALL WITH THE PARAMS ---
                    transcription_response = (
                        self.openai_client.audio.transcriptions.create(
                            **transcription_params
                        )
                    )

                    # --- The old, hard-coded line to be replaced is below ---
                    # transcription_response = self.openai_client.audio.transcriptions.create(
                    #     model="whisper-1",
                    #     file=audio_file_to_process,
                    #     language="it",  # Set language to Italian (it)
                    #     prompt="Trascrivi in italiano una conversazione tra due persone, provando a capire chi parla.",
                    # )

                    transcribed_text = transcription_response.text
                    combined_transcriptions.append(transcribed_text)

                    # Emit individual transcription parts to the UI for responsiveness
                    separator = "\n\n---\n\n" if index > 0 else ""
                    self.transcription_finished.emit(
                        f"{separator}**{file_name_for_error}:**\n{transcribed_text}"
                    )

            full_transcription_text = "\n\n---\n\n".join(combined_transcriptions)
            logging.info("All transcription tasks finished.")
            logging.info(
                f"Full transcription (first 100 chars): {full_transcription_text[:100]}..."
            )
            self.progress_update.emit(80)

            # --- Save transcription to temp file ---
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".txt", mode="w", delete=False, encoding="utf-8"
                ) as self.temp_transcription_file:
                    self.temp_transcription_file.write(full_transcription_text)
                    temp_transcription_file_name = self.temp_transcription_file.name
                logging.info(
                    f"Transcription saved to temporary file: {temp_transcription_file_name}"
                )
            except Exception as e:
                logging.error(f"Error saving transcription to temporary file: {e}")

            # --- Summarization (if not transcription_only) ---
            if not self.transcription_only:
                logging.info("Starting summarization with Gemini.")
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"{self.summarization_prompt}\n\n{full_transcription_text}",
                )
                summary_text = response.text
                logging.info("Summarization finished.")
                logging.info(f"Summary (first 50 chars): {summary_text[:50]}...")
                self.progress_update.emit(90)
                self.final_summary_markdown = summary_text
                self.summary_finished.emit(summary_text)
            else:
                final_summary_output = (
                    "Transcription completed - summarization skipped as requested."
                )
                self.final_summary_markdown = ""  # No summary to save
                self.summary_finished.emit(final_summary_output)
                logging.info("Summarization skipped due to 'Transcription Only' mode.")

            self.progress_update.emit(100)
            logging.info("Summarization thread finished successfully.")

        except Exception as e:
            error_message = f"Error processing {file_name_for_error}: {e}"
            logging.error(error_message, exc_info=True)
            self.error_occurred.emit(error_message, file_name_for_error)
            self.progress_update.emit(0)
        finally:
            # --- Cleanup all temporary files ---
            for temp_file in self.temp_files_to_clean:
                try:
                    os.remove(temp_file.name)
                    logging.info(f"Temporary file deleted: {temp_file.name}")
                except OSError as e:
                    logging.error(f"Error deleting temp file {temp_file.name}: {e}")

            if self.temp_transcription_file:
                logging.info(
                    f"Temporary transcription file kept: {self.temp_transcription_file.name}"
                )

        logging.info("Summarization thread run method completed.")


class AudioSummaryApp(QWidget):
    DEFAULT_PROMPT = (
        "Riassumi la trascrizione di un messaggio vocale ricevuto dall'utente."
    )

    def __init__(self):
        logging.info("AudioSummaryApp initialization started.")
        super().__init__()
        self.setWindowTitle("Riassunto audio - messaggi vocali & lezioni")
        self.setGeometry(100, 100, 1300, 1000)

        self.openai_api_key = None
        self.gemini_api_key = None
        self.load_api_key()
        self.summarization_thread = None
        self.audio_file_paths = []  # To store multiple file paths
        self.summary_markdown_text = ""  # To store summary in markdown format
        self.summary_is_unsaved = False  # Track if summary exists but is unsaved
        self.settings = QSettings(
            "BitreyDev", "AudioSummaryApp"
        )  # Initialize QSettings

        self.init_ui()
        self.load_settings()  # Load settings after UI is initialized
        self.apply_dark_theme()
        logging.info("AudioSummaryApp initialization completed.")

    def apply_dark_theme(self):
        logging.info("Applying dark theme.")
        # Per il dark theme
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
            QLabel {
                color: #ffffff;
            }
            QTextEdit {
                background-color: #444444;
                color: #ffffff;
                border: 1px solid #666666;
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
        """
        )
        logging.info("Dark theme applied.")

    def load_api_key(self):
        logging.info("Loading API keys from .env file.")
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            warning_msg_openai = (
                "OPENAI_API_KEY is not set in the .env file for transcription."
            )
            logging.warning(warning_msg_openai)
            QMessageBox.warning(
                self,
                "Chiave API OpenAI mancante",
                "O, imposta OPENAI_API_KEY nel tuo file .env per la trascrizione.",
            )
        else:
            logging.info("OPENAI_API_KEY loaded successfully.")

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            warning_msg_gemini = (
                "GEMINI_API_KEY is not set in the .env file for summarization."
            )
            logging.warning(warning_msg_gemini)
            QMessageBox.warning(
                self,
                "Chiave API Gemini mancante",
                "O, imposta GEMINI_API_KEY nel tuo file .env per la riassunzione.",
            )
        else:
            logging.info("GEMINI_API_KEY loaded successfully.")
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

        # Reset button
        self.reset_button = QPushButton("Reset Prompt", self)  # Create reset button
        self.reset_button.clicked.connect(
            self.reset_prompt
        )  # Connect to reset function
        file_selection_layout.addWidget(self.reset_button)  # Add to layout

        self.layout.addLayout(file_selection_layout)

        # Checkbox for merging audio files
        self.merge_audio_checkbox = QCheckBox(
            "Unisci file audio (se < 25MB totali)", self
        )  # Text updated
        self.merge_audio_checkbox.setChecked(True)  # Default to checked
        self.merge_audio_checkbox.stateChanged.connect(
            self.save_settings
        )  # Save setting on change

        # Checkbox for transcription only mode
        self.transcription_only_checkbox = QCheckBox("Esegui SOLO trascrizione", self)
        self.transcription_only_checkbox.setChecked(False)  # Default to unchecked
        self.transcription_only_checkbox.stateChanged.connect(
            self.save_settings
        )  # Save setting on change
        self.transcription_only_checkbox.stateChanged.connect(
            self.update_prompt_textbox_state
        )  # Connect to state change

        # Create a horizontal layout for checkboxes
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(self.merge_audio_checkbox)
        checkbox_layout.addWidget(self.transcription_only_checkbox)

        # Add horizontal layout to the main vertical layout
        self.layout.addLayout(checkbox_layout)

        # Prompt text box
        self.prompt_label = QLabel("Prompt per Gemini (AI):", self)
        self.layout.addWidget(self.prompt_label)
        self.prompt_text_edit = QTextEdit(self)
        self.prompt_text_edit.setFixedHeight(180)  # Adjust height as needed
        self.prompt_text_edit.setPlainText(self.DEFAULT_PROMPT)  # Default prompt
        self.prompt_text_edit.textChanged.connect(
            self.save_settings
        )  # Save setting on change
        self.layout.addWidget(self.prompt_text_edit)

        # --- START: New Whisper Settings UI ---

        # Add a separator or label for clarity
        self.whisper_settings_label = QLabel("--- Impostazioni Whisper ---", self)
        self.layout.addWidget(self.whisper_settings_label)

        # Whisper Language Input
        whisper_language_layout = QHBoxLayout()
        self.whisper_language_label = QLabel("Lingua (es: it, en):", self)
        # We use QLineEdit for a short text input
        self.whisper_language_input = QLineEdit(self)
        self.whisper_language_input.setPlaceholderText("lascia vuoto per auto-detect")
        self.whisper_language_input.setText("it")  # Set default to Italian
        self.whisper_language_input.textChanged.connect(self.save_settings)
        whisper_language_layout.addWidget(self.whisper_language_label)
        whisper_language_layout.addWidget(self.whisper_language_input)
        self.layout.addLayout(whisper_language_layout)

        # Whisper Prompt Input
        self.whisper_prompt_label = QLabel("Prompt per Whisper (opzionale):", self)
        self.layout.addWidget(self.whisper_prompt_label)
        # We use QTextEdit for a potentially longer prompt
        self.whisper_prompt_text_edit = QTextEdit(self)
        self.whisper_prompt_text_edit.setFixedHeight(
            80
        )  # A bit shorter than the Gemini one
        self.whisper_prompt_text_edit.setPlainText(
            "Trascrivi in italiano una conversazione tra due persone, provando a capire chi parla."
        )
        self.whisper_prompt_text_edit.textChanged.connect(self.save_settings)
        self.layout.addWidget(self.whisper_prompt_text_edit)

        # --- END: New Whisper Settings UI ---

        self.process_button = QPushButton("Elabora!", self)  # Text changed
        self.process_button.clicked.connect(self.start_processing)
        self.process_button.setEnabled(False)
        self.layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.transcription_output_text_edit = QTextEdit(
            self
        )  # Added for transcription output
        self.transcription_output_text_edit.setReadOnly(True)
        # Label for transcription
        self.layout.addWidget(QLabel("Trascrizione:"))
        # Add transcription text edit
        self.layout.addWidget(self.transcription_output_text_edit)

        self.summary_output_text_edit = QTextEdit(self)
        self.summary_output_text_edit.setReadOnly(True)
        # Changed label to be more specific
        self.layout.addWidget(QLabel("Riassunto:"))
        self.layout.addWidget(self.summary_output_text_edit)

        self.status_label = QLabel("", self)
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)

        # Add Ctrl+S shortcut for saving unsaved summaries
        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.handle_save_shortcut)

        logging.info("User interface initialized.")

        self.update_prompt_textbox_state()  # Set initial state of prompt textbox
        logging.info(
            "Initial prompt textbox state set based on transcription_only checkbox."
        )

    def update_prompt_textbox_state(self):
        if self.transcription_only_checkbox.isChecked():
            self.prompt_text_edit.setEnabled(False)
        else:
            self.prompt_text_edit.setEnabled(True)

    def load_settings(self):
        logging.info("Loading settings from QSettings.")
        saved_prompt = self.settings.value("promptText", self.DEFAULT_PROMPT)
        saved_checkbox_checked = self.settings.value(
            "mergeAudioChecked", True, type=bool
        )
        saved_transcription_only_checked = self.settings.value(
            "transcriptionOnlyChecked", False, type=bool
        )

        self.prompt_text_edit.setPlainText(saved_prompt)
        self.merge_audio_checkbox.setChecked(saved_checkbox_checked)
        self.transcription_only_checkbox.setChecked(saved_transcription_only_checked)

        # --- ADD THESE LINES ---
        saved_whisper_lang = self.settings.value("whisperLanguage", "it")
        saved_whisper_prompt = self.settings.value(
            "whisperPrompt",
            "Trascrivi in italiano una conversazione tra due persone, provando a capire chi parla.",
        )
        self.whisper_language_input.setText(saved_whisper_lang)
        self.whisper_prompt_text_edit.setPlainText(saved_whisper_prompt)
        # --- END ---

        logging.info("Settings loaded.")

    def save_settings(self):
        logging.info("Saving settings to QSettings.")
        self.settings.setValue("promptText", self.prompt_text_edit.toPlainText())
        self.settings.setValue(
            "mergeAudioChecked", self.merge_audio_checkbox.isChecked()
        )
        self.settings.setValue(
            "transcriptionOnlyChecked", self.transcription_only_checkbox.isChecked()
        )
        # --- ADD THESE LINES ---
        self.settings.setValue("whisperLanguage", self.whisper_language_input.text())
        self.settings.setValue(
            "whisperPrompt", self.whisper_prompt_text_edit.toPlainText()
        )
        # --- END ---
        logging.info("Settings saved.")

    def reset_prompt(self):
        logging.info("Resetting prompt and checkbox to default.")
        self.prompt_text_edit.setPlainText(self.DEFAULT_PROMPT)
        self.merge_audio_checkbox.setChecked(True)
        self.transcription_only_checkbox.setChecked(False)  # Default to unchecked
        self.save_settings()  # Save default settings
        logging.info("Prompt and checkboxes reset to default.")

    def select_audio_file(self):
        logging.info("Select audio file button clicked.")
        file_dialog = QFileDialog()

        # Get the last directory from settings, or use the script directory as default
        last_directory = self.settings.value("lastInputDirectory", "")

        file_paths, _ = file_dialog.getOpenFileNames(
            self,
            "Seleziona file audio",
            last_directory,
            "File Audio (*.ogg *.wav *.mp3 *.flac *.m4a)",  # Added m4a
        )

        if file_paths:
            # Save the directory of the first file as the last directory
            last_directory = os.path.dirname(file_paths[0])
            self.settings.setValue("lastInputDirectory", last_directory)
            logging.info(f"Last directory updated to: {last_directory}")

            # Sort files by creation date
            file_paths_with_ctime = [
                (path, os.path.getctime(path)) for path in file_paths
            ]
            file_paths_with_ctime.sort(
                key=lambda item: item[1]
            )  # Sort by creation time
            # Extract sorted paths
            self.audio_file_paths = [path for path, ctime in file_paths_with_ctime]

            if len(self.audio_file_paths) == 1:
                file_display_text = (
                    f"File selezionato: {os.path.basename(self.audio_file_paths[0])}"
                )
                logging.info(f"Single audio file selected: {self.audio_file_paths[0]}")
            else:
                file_display_text = f"{len(self.audio_file_paths)} file selezionati"
                logging.info(
                    f"Multiple audio files selected: {len(self.audio_file_paths)}, sorted by creation date."
                )

            self.file_path_label.setText(file_display_text)
            self.process_button.setEnabled(True)
            self.summary_output_text_edit.clear()
            self.transcription_output_text_edit.clear()
            self.status_label.clear()
        else:
            self.file_path_label.setText("Nessun file audio selezionato")
            self.process_button.setEnabled(False)
            self.audio_file_paths = []
            logging.info("No audio file selected.")

    def start_processing(self):
        logging.info("Start processing button clicked.")
        self.summary_is_unsaved = False  # Reset flag when starting new processing
        if not self.audio_file_paths:  # Check if file paths list is empty
            warning_msg_no_file = "No audio file selected before processing."
            logging.warning(warning_msg_no_file)
            QMessageBox.warning(
                self, "Nessun file selezionato", "O, seleziona prima un file audio."
            )
            return

        openai_key_needed = True
        gemini_key_needed = not self.transcription_only_checkbox.isChecked()

        if not self.openai_api_key and openai_key_needed:
            QMessageBox.warning(
                self,
                "Chiave API mancante",
                "Imposta OPENAI_API_KEY nel tuo file .env e riavvia.",
            )
            return
        if not self.gemini_api_key and gemini_key_needed:
            QMessageBox.warning(
                self,
                "Chiave API mancante",
                "Imposta GEMINI_API_KEY nel tuo file .env e riavvia.",
            )
            return

        self.status_label.setText("Elaborazione in corso...")
        self.process_button.setEnabled(False)
        self.prompt_text_edit.setEnabled(False)
        self.progress_bar.setValue(0)
        self.summary_output_text_edit.clear()
        self.transcription_output_text_edit.clear()

        merge_audio = self.merge_audio_checkbox.isChecked()
        transcription_only = self.transcription_only_checkbox.isChecked()
        logging.info(f"Merge audio checkbox is checked: {merge_audio}")
        logging.info(f"Transcription only checkbox is checked: {transcription_only}")

        summarization_prompt = (
            self.prompt_text_edit.toPlainText() if not transcription_only else ""
        )

        # --- ADD THESE LINES to get values from the new UI fields ---
        whisper_language = self.whisper_language_input.text()
        whisper_prompt = self.whisper_prompt_text_edit.toPlainText()
        # --- END ---

        logging.info("Creating and starting summarization thread.")
        # --- MODIFY THIS LINE to pass the new variables ---
        self.summarization_thread = SummarizationThread(
            self.audio_file_paths,
            self.openai_api_key,
            self.gemini_api_key,
            summarization_prompt,
            merge_audio,
            transcription_only,
            whisper_language,  # new
            whisper_prompt,  # new
        )
        # --- END MODIFICATION ---
        self.summarization_thread.summary_finished.connect(self.display_result)
        self.summarization_thread.transcription_finished.connect(
            self.display_transcription
        )
        self.summarization_thread.error_occurred.connect(self.display_error)
        self.summarization_thread.progress_update.connect(self.update_progress)
        self.summarization_thread.start()
        logging.info("Summarization thread started.")

    def display_result(self, summary_text):
        logging.info("Displaying summary result.")
        html_summary = mistune.html(summary_text)
        self.summary_output_text_edit.setHtml(html_summary)
        self.status_label.setText("Finito!!")
        self.process_button.setEnabled(True)
        # Re-enable prompt text edit after processing if not in transcription only mode
        if not self.transcription_only_checkbox.isChecked():
            self.prompt_text_edit.setEnabled(True)
        self.progress_bar.setValue(100)
        self.summary_markdown_text = summary_text
        self.summary_is_unsaved = True if summary_text else False

        self.prompt_save_file()
        logging.info("Summary displayed and UI updated.")

    def prompt_save_file(self):
        # Determine content type and default file name
        is_summary = bool(
            self.summary_markdown_text
            and "Transcription completed" not in self.summary_markdown_text
        )
        if is_summary:
            content_to_save = self.summary_markdown_text
            default_suffix = "summary.md"
            file_filter = "Markdown Files (*.md)"
            dialog_title = "Salva riassunto come Markdown"
        else:  # Transcription only
            content_to_save = self.transcription_output_text_edit.toPlainText()
            default_suffix = "transcription.txt"
            file_filter = "Text Files (*.txt)"
            dialog_title = "Salva trascrizione come testo"

        if not content_to_save:
            logging.info("No content to save.")
            return

        logging.info(f"Prompting user to save {default_suffix}.")
        file_dialog = QFileDialog()
        last_save_directory = self.settings.value("lastSaveDirectory", "")
        default_file_name = os.path.join(last_save_directory, default_suffix)

        file_path, _ = file_dialog.getSaveFileName(
            self, dialog_title, default_file_name, file_filter
        )

        if file_path:
            last_save_directory = os.path.dirname(file_path)
            self.settings.setValue("lastSaveDirectory", last_save_directory)
            logging.info(f"Last save directory updated to: {last_save_directory}")
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content_to_save)
                logging.info("File saved successfully.")
                self.status_label.setText(f"File salvato in: {file_path}")
                self.summary_is_unsaved = False
            except Exception as e:
                error_msg = f"Errore durante il salvataggio del file: {e}"
                logging.error(error_msg)
                QMessageBox.critical(self, "Errore di salvataggio", error_msg)
        else:
            logging.info("Save file dialog cancelled by user.")
            self.status_label.setText(
                "File non salvato - puoi premere Ctrl+S per salvare."
            )

    def handle_save_shortcut(self):
        if self.summary_is_unsaved:
            logging.info("Ctrl+S shortcut activated for unsaved content.")
            self.prompt_save_file()
        else:
            logging.info("Ctrl+S shortcut ignored: no unsaved content available.")

    def display_transcription(self, transcription_text):
        logging.info("Displaying transcription part.")
        # We use appendHtml to correctly render the markdown-like bolding
        self.transcription_output_text_edit.append(mistune.html(transcription_text))
        logging.info("Transcription part displayed.")

    def display_error(self, error_message, file_path):
        logging.error(f"Error occurred: {error_message} for file: {file_path}")
        QMessageBox.critical(self, "Errore di elaborazione", error_message)
        file_name = os.path.basename(file_path)
        self.status_label.setText(f"Errore durante l'elaborazione di {file_name}.")
        self.process_button.setEnabled(True)
        if not self.transcription_only_checkbox.isChecked():
            self.prompt_text_edit.setEnabled(True)
        self.progress_bar.setValue(0)
        logging.error(f"Error displayed in UI for file: {file_name}")

    def update_progress(self, progress_value):
        self.progress_bar.setValue(progress_value)
        logging.debug(f"Progress bar updated to: {progress_value}%")


if __name__ == "__main__":
    logging.info("Application started.")
    app = QApplication(sys.argv)
    main_window = AudioSummaryApp()
    main_window.show()
    exit_code = app.exec()
    logging.info(f"Application exited with code: {exit_code}")
    sys.exit(exit_code)
