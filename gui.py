#!/usr/bin/env python3

import hashlib
import logging
import os
import sys

import mistune
from dotenv import load_dotenv
from google import genai
from PyQt6.QtCore import QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
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
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def calculate_sha256(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.digest()


class ProcessingThread(QThread):
    """
    Thread to handle audio processing using the Gemini API.
    This simplified version uses a single user prompt to guide the AI,
    which can handle transcription, summarization, or other tasks in one call.
    """

    processing_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)

    def __init__(
        self,
        audio_file_paths,
        gemini_api_key,
        user_prompt,
    ):
        super().__init__()
        self.audio_file_paths = audio_file_paths
        self.gemini_api_key = gemini_api_key
        self.user_prompt = user_prompt

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

            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            logging.info("Gemini client initialized.")
            self.progress_update.emit(10)

            logging.info(
                "Checking for existing files and uploading new ones to Gemini..."
            )
            gemini_contents = []

            gemini_contents.append(self.user_prompt)

            # Get list of existing files from Gemini to check for duplicates
            logging.info("Fetching existing Gemini files for cache check...")
            existing_files = {}  # sha256_hash -> gemini_file
            try:
                for existing_file in self.gemini_client.files.list():
                    if (
                        hasattr(existing_file, "sha256Hash")
                        and existing_file.sha256Hash
                    ):
                        existing_files[existing_file.sha256Hash] = existing_file
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
                logging.info(
                    f"Processing file {index + 1}/{num_files}: {file_name_for_error}..."
                )
                try:
                    local_hash = calculate_sha256(file_path)
                    # Convert to base64 string to match Gemini's format
                    import base64

                    local_hash_b64 = base64.b64encode(local_hash).decode("utf-8")

                    # Check if file already exists in Gemini
                    if local_hash_b64 in existing_files:
                        existing_file = existing_files[local_hash_b64]
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

            logging.info("All files uploaded. Requesting generation from Gemini...")
            self.progress_update.emit(90)

            # No more conditional logic. The user's prompt dictates the entire task.
            response = self.gemini_client.models.generate_content(
                model="gemini-1.5-flash",  # Using 1.5-flash as it's great for this
                contents=gemini_contents,
            )
            result_text = response.text

            logging.info("Received response from Gemini.")
            self.progress_update.emit(95)

            self.processing_finished.emit(result_text)

            self.progress_update.emit(100)
            logging.info("Processing thread finished successfully.")

        except Exception as e:
            error_message = f"Error processing {file_name_for_error}: {e}"
            logging.error(error_message, exc_info=True)
            self.error_occurred.emit(error_message, file_name_for_error)
            self.progress_update.emit(0)
        finally:
            # Clean up only newly uploaded files, not cached ones
            logging.info(
                f"Cleaning up {len(newly_uploaded_files)} newly uploaded Gemini files..."
            )
            for gemini_file in newly_uploaded_files:
                try:
                    self.gemini_client.files.delete(file=gemini_file)
                    logging.info(f"Deleted remote file from Gemini: {gemini_file.name}")
                except Exception as e:
                    logging.warning(
                        f"Could not delete Gemini file {gemini_file.name}: {e}"
                    )

            if len(newly_uploaded_files) == 0:
                logging.info(
                    "No newly uploaded files to clean up (all files were cached)"
                )
            else:
                logging.info(
                    "Cleanup completed. Cached files were preserved for future use."
                )
        logging.info("Processing thread run method completed.")


class AudioSummaryApp(QWidget):

    DEFAULT_PROMPT = "Esegui una trascrizione e riassunto dei file audio forniti."

    def __init__(self):
        logging.info("AudioSummaryApp initialization started.")
        super().__init__()
        self.setWindowTitle("Riassunto audio - messaggi vocali & lezioni")
        self.setGeometry(100, 100, 1300, 1000)

        self.gemini_api_key = None
        self.load_api_key()
        self.processing_thread = None
        self.audio_file_paths = []
        self.last_result_text = ""
        self.result_is_unsaved = False
        self.settings = QSettings("BitreyDev", "AudioSummaryApp")

        self.init_ui()
        self.load_settings()
        self.apply_dark_theme()
        logging.info("AudioSummaryApp initialization completed.")

    def apply_dark_theme(self):
        # ... (no changes here, keeping the theme)
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
        """
        )
        logging.info("Dark theme applied.")

    def load_api_key(self):
        # ... (no changes here)
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

        self.prompt_label = QLabel("Prompt per Gemini (cosa fare con l'audio?):", self)
        self.layout.addWidget(self.prompt_label)
        self.user_prompt_text_edit = QTextEdit(self)
        self.user_prompt_text_edit.setFixedHeight(180)
        self.user_prompt_text_edit.textChanged.connect(self.save_settings)
        self.layout.addWidget(self.user_prompt_text_edit)

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

        self.status_label = QLabel("", self)
        self.layout.addWidget(self.status_label)

        self.setLayout(self.layout)

        save_shortcut = QShortcut(QKeySequence("Ctrl+S"), self)
        save_shortcut.activated.connect(self.handle_save_shortcut)

        logging.info("User interface initialized.")

    def load_settings(self):
        logging.info("Loading settings from QSettings.")
        self.user_prompt_text_edit.textChanged.disconnect()
        saved_prompt = self.settings.value("userPrompt", self.DEFAULT_PROMPT)
        self.user_prompt_text_edit.setPlainText(saved_prompt)
        self.user_prompt_text_edit.textChanged.connect(self.save_settings)
        logging.info("Settings loaded.")

    def save_settings(self):
        logging.info("Saving settings to QSettings.")
        self.settings.setValue("userPrompt", self.user_prompt_text_edit.toPlainText())
        logging.info("Settings saved.")

    def reset_prompt(self):
        logging.info("Resetting prompt to default.")
        self.user_prompt_text_edit.setPlainText(self.DEFAULT_PROMPT)
        self.save_settings()
        logging.info("Prompt reset to default.")

    def select_audio_file(self):
        # ... (no changes in this method, it's already good)
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

        if not self.gemini_api_key:
            QMessageBox.warning(
                self,
                "Chiave API mancante",
                "Imposta GEMINI_API_KEY nel tuo file .env e riavvia.",
            )
            return

        self.status_label.setText("Elaborazione in corso...")
        self.process_button.setEnabled(False)
        self.user_prompt_text_edit.setEnabled(False)  # Disable prompt during processing
        self.progress_bar.setValue(0)
        self.output_text_edit.clear()

        user_prompt = self.user_prompt_text_edit.toPlainText()

        logging.info("Creating and starting processing thread.")

        self.processing_thread = ProcessingThread(
            self.audio_file_paths,
            self.gemini_api_key,
            user_prompt,
        )
        self.processing_thread.processing_finished.connect(self.display_result)
        self.processing_thread.error_occurred.connect(self.display_error)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.start()
        logging.info("Processing thread started.")

    def display_result(self, result_text):
        logging.info("Displaying result from Gemini.")

        html_content = mistune.html(result_text)
        self.output_text_edit.setHtml(html_content)

        self.status_label.setText("Finito!!")
        self.process_button.setEnabled(True)
        self.user_prompt_text_edit.setEnabled(True)
        self.progress_bar.setValue(100)

        # Store result for saving
        self.last_result_text = result_text
        self.result_is_unsaved = bool(result_text)

        self.prompt_save_file()
        logging.info("Result displayed and UI updated.")

    def prompt_save_file(self):
        if not self.last_result_text:
            return

        default_suffix = "summary.md"
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
