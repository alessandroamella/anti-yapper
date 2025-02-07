#!/usr/bin/env python3

import os
import logging  # Import the logging module
from dotenv import load_dotenv
from openai import OpenAI
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QFileDialog, QTextEdit, QMessageBox,
                             QProgressBar, QHBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal
from google import genai
import mistune
from pydub import AudioSegment
import tempfile

# Configure logging to console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


class SummarizationThread(QThread):
    """
    Thread to handle audio processing, transcription (OpenAI Whisper),
    and summarization (Gemini) to avoid blocking the GUI.
    Handles single or merged audio files.
    Optimized for smaller merged audio file size.
    """
    summary_finished = pyqtSignal(str)
    transcription_finished = pyqtSignal(str)  # Signal for transcription output
    error_occurred = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)

    def __init__(self, audio_file_paths, openai_api_key, gemini_api_key, summarization_prompt):
        super().__init__()
        self.audio_file_paths = audio_file_paths  # Now accepts a list of file paths
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.summarization_prompt = summarization_prompt  # User defined prompt
        self.openai_client = None
        self.gemini_client = None
        self.temp_merged_file = None  # To store temp file object

    def run(self):
        logging.info("Summarization thread started.")
        # Default value in case of early errors
        file_name_for_error = "unknown audio file"
        try:
            if not self.openai_api_key:
                error_msg = "OPENAI_API_KEY is not set in the .env file"
                logging.error(error_msg)
                raise ValueError(error_msg)
            if not self.gemini_api_key:
                error_msg = "GEMINI_API_KEY is not set in the .env file"
                logging.error(error_msg)
                raise ValueError(error_msg)

            self.openai_client = OpenAI(api_key=self.openai_api_key)
            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            logging.info("OpenAI and Gemini clients initialized.")

            self.progress_update.emit(10)

            if isinstance(self.audio_file_paths, list) and len(self.audio_file_paths) > 1:
                logging.info("Merging multiple audio files.")
                self.progress_update.emit(20)
                merged_audio = AudioSegment.from_file(self.audio_file_paths[0])
                for file_path in self.audio_file_paths[1:]:
                    audio_segment = AudioSegment.from_file(file_path)
                    merged_audio += audio_segment

                # Strategy 1: Export merged audio as MP3 to reduce file size
                # MP3 is a lossy format but significantly smaller than WAV, suitable for voice.
                self.temp_merged_file = tempfile.NamedTemporaryFile(
                    suffix=".mp3", delete=False)  # Changed suffix to .mp3
                merged_audio.export(
                    self.temp_merged_file.name,
                    format="mp3",
                    bitrate="128k"  # Strategy 2: Consider lower bitrate if needed, 128k is generally ok for speech
                )  # Export as MP3
                audio_file_to_process = open(self.temp_merged_file.name, "rb")
                file_name_for_error = "merged audio file"  # For error messages
                logging.info(
                    f"Merged audio saved to temporary MP3 file: {self.temp_merged_file.name}")
                self.progress_update.emit(25)

            else:
                # Process single audio file
                audio_file_path = self.audio_file_paths if not isinstance(
                    self.audio_file_paths, list) else self.audio_file_paths[0]  # Handle single path or list with one path
                audio_file_to_process = open(audio_file_path, "rb")
                file_name_for_error = audio_file_path  # For error messages
                logging.info(
                    f"Processing single audio file: {file_name_for_error}")
                self.progress_update.emit(30)

            # Use the appropriate audio file (merged or single)
            with audio_file_to_process:
                logging.info(
                    "Starting audio transcription with OpenAI Whisper.")
                transcription_response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file_to_process
                )
                transcribed_text = transcription_response.text
                logging.info("Transcription finished.")
                # Log first 50 chars of transcription for info (avoid very long logs)
                logging.info(
                    f"Transcription (first 50 chars): {transcribed_text[:50]}...")
                self.transcription_finished.emit(transcribed_text)
                self.progress_update.emit(60)

            logging.info("Starting summarization with Gemini.")
            model = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                # Use user defined prompt
                contents=f"{self.summarization_prompt} {transcribed_text}"
            )
            summary_text = model.text
            logging.info("Summarization finished.")
            # Log first 50 chars of summary for info (avoid very long logs)
            logging.info(f"Summary (first 50 chars): {summary_text[:50]}...")
            self.progress_update.emit(90)

            self.summary_finished.emit(summary_text)
            self.progress_update.emit(100)
            logging.info("Summarization thread finished successfully.")

        except Exception as e:
            # Use dynamic file name in error
            error_message = f"Error processing {file_name_for_error}: {e}"
            logging.error(error_message)
            # Use dynamic file name in error
            self.error_occurred.emit(error_message, file_name_for_error)
            self.progress_update.emit(0)
        finally:
            if self.temp_merged_file:
                # Ensure temp file is deleted
                os.remove(self.temp_merged_file.name)
                logging.info(
                    f"Temporary merged file deleted: {self.temp_merged_file.name}")
        logging.info("Summarization thread run method completed.")


class AudioSummaryApp(QWidget):
    def __init__(self):
        logging.info("AudioSummaryApp initialization started.")
        super().__init__()
        self.setWindowTitle("Riassunto audio - messaggi vocali")
        self.setGeometry(100, 100, 1300, 1000)

        self.openai_api_key = None
        self.gemini_api_key = None
        self.load_api_key()
        self.summarization_thread = None
        self.audio_file_paths = []  # To store multiple file paths

        self.init_ui()
        self.apply_dark_theme()
        logging.info("AudioSummaryApp initialization completed.")

    def apply_dark_theme(self):
        logging.info("Applying dark theme.")
        # Per il dark theme
        self.setStyleSheet("""
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
        """)
        logging.info("Dark theme applied.")

    def load_api_key(self):
        logging.info("Loading API keys from .env file.")
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            warning_msg_openai = "OPENAI_API_KEY is not set in the .env file for transcription."
            logging.warning(warning_msg_openai)
            QMessageBox.warning(self, "Chiave API OpenAI mancante",
                                "O, imposta OPENAI_API_KEY nel tuo file .env per la trascrizione.")
        else:
            logging.info("OPENAI_API_KEY loaded successfully.")

        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            warning_msg_gemini = "GEMINI_API_KEY is not set in the .env file for summarization."
            logging.warning(warning_msg_gemini)
            QMessageBox.warning(self, "Chiave API Gemini mancante",
                                "O, imposta GEMINI_API_KEY nel tuo file .env per la riassunzione.")
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
        self.layout.addLayout(file_selection_layout)

        # Prompt text box
        self.prompt_label = QLabel("Prompt per Gemini (AI):", self)
        self.layout.addWidget(self.prompt_label)
        self.prompt_text_edit = QTextEdit(self)
        self.prompt_text_edit.setFixedHeight(60)  # Adjust height as needed
        self.prompt_text_edit.setPlainText(
            "Fai un riassunto del seguente messaggio vocale:")  # Default prompt
        self.layout.addWidget(self.prompt_text_edit)

        self.process_button = QPushButton(
            "Riassumi!!", self)
        self.process_button.clicked.connect(
            self.start_processing)
        self.process_button.setEnabled(False)
        self.layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        self.transcription_output_text_edit = QTextEdit(
            self)  # Added for transcription output
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
        logging.info("User interface initialized.")

    def select_audio_file(self):
        logging.info("Select audio file button clicked.")
        file_dialog = QFileDialog()
        file_paths, _ = file_dialog.getOpenFileNames(self, "Seleziona file audio", "",
                                                     "File Audio (*.ogg *.wav *.mp3 *.flac)")
        if file_paths:
            # Sort files by creation date
            file_paths_with_ctime = [(path, os.path.getctime(path))
                                     for path in file_paths]
            file_paths_with_ctime.sort(
                key=lambda item: item[1])  # Sort by creation time
            # Extract sorted paths
            self.audio_file_paths = [path for path,
                                     ctime in file_paths_with_ctime]

            if len(self.audio_file_paths) == 1:
                file_display_text = f"File selezionato: {self.audio_file_paths[0]}"
                logging.info(
                    f"Single audio file selected: {self.audio_file_paths[0]}")
            else:
                file_display_text = f"{len(self.audio_file_paths)} file selezionati"
                logging.info(
                    f"Multiple audio files selected: {len(self.audio_file_paths)}, sorted by creation date.")

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
        if not self.audio_file_paths:  # Check if file paths list is empty
            warning_msg_no_file = "No audio file selected before processing."
            logging.warning(warning_msg_no_file)
            QMessageBox.warning(self, "Nessun file selezionato",
                                "O, seleziona prima un file audio.")
            return

        if not self.openai_api_key or not self.gemini_api_key:
            warning_msg_api_keys = "API keys missing (OpenAI or Gemini or both)."
            logging.warning(warning_msg_api_keys)
            QMessageBox.warning(self, "Chiavi API mancanti",
                                "O, imposta sia OPENAI_API_KEY che GEMINI_API_KEY nel tuo file .env e riavvia.")
            return

        self.status_label.setText("Trascrizione audio e riassunto in corso...")
        self.process_button.setEnabled(False)
        # Disable prompt text edit during processing
        self.prompt_text_edit.setEnabled(False)
        self.progress_bar.setValue(0)
        self.summary_output_text_edit.clear()
        self.transcription_output_text_edit.clear()

        # Get prompt from text edit
        summarization_prompt = self.prompt_text_edit.toPlainText()
        logging.info(f"Using summarization prompt: {summarization_prompt}")

        logging.info("Creating and starting summarization thread.")
        self.summarization_thread = SummarizationThread(
            # Pass prompt to thread
            self.audio_file_paths, self.openai_api_key, self.gemini_api_key, summarization_prompt
        )
        self.summarization_thread.summary_finished.connect(
            self.display_result)
        self.summarization_thread.transcription_finished.connect(
            self.display_transcription)
        self.summarization_thread.error_occurred.connect(
            self.display_error)
        self.summarization_thread.progress_update.connect(
            self.update_progress)
        self.summarization_thread.start()
        logging.info("Summarization thread started.")

    def display_result(self, summary_text):
        logging.info("Displaying summary result.")
        html_summary = mistune.html(summary_text)
        self.summary_output_text_edit.setHtml(html_summary)
        self.status_label.setText("Finito!!")
        self.process_button.setEnabled(True)
        # Re-enable prompt text edit after processing
        self.prompt_text_edit.setEnabled(True)
        self.progress_bar.setValue(100)
        logging.info("Summary displayed and UI updated.")

    def display_transcription(self, transcription_text):
        logging.info("Displaying transcription.")
        self.transcription_output_text_edit.setText(transcription_text)
        logging.info("Transcription displayed.")

    def display_error(self, error_message, file_path):  # file_path can be string or list
        logging.error(f"Error occurred: {error_message} for file: {file_path}")
        QMessageBox.critical(self, "Errore di elaborazione", error_message)
        if isinstance(file_path, list):
            file_name = "merged audio files"
        else:
            file_name = os.path.basename(file_path)
        self.status_label.setText(
            f"Errore durante l'elaborazione di {file_name}.")
        self.process_button.setEnabled(True)
        # Re-enable prompt text edit after error
        self.prompt_text_edit.setEnabled(True)
        self.progress_bar.setValue(0)
        logging.error(f"Error displayed in UI for file: {file_name}")

    def update_progress(self, progress_value):
        self.progress_bar.setValue(progress_value)
        # Use debug level as progress updates can be frequent
        logging.debug(f"Progress bar updated to: {progress_value}%")


if __name__ == '__main__':
    logging.info("Application started.")
    app = QApplication(sys.argv)
    main_window = AudioSummaryApp()
    main_window.show()
    exit_code = app.exec_()
    logging.info(f"Application exited with code: {exit_code}")
    sys.exit(exit_code)
