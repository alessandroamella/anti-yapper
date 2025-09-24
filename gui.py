#!/usr/bin/env python3

import logging
import os
import sys
import tempfile

import mistune
from dotenv import load_dotenv
from google import genai
from PyQt6.QtCore import QSettings, QThread, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
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


class SummarizationThread(QThread):
    """
    Thread to handle audio processing using the Gemini API for both
    transcription and summarization. This new version eliminates the need
    for manual audio splitting/merging by leveraging Gemini's native
    support for large and multiple audio files.
    """

    processing_finished = pyqtSignal(str, bool)  # Emits: (result_text, is_summary)
    error_occurred = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)

    def __init__(
        self,
        audio_file_paths,
        gemini_api_key,
        summarization_prompt,
        transcription_prompt,
        transcription_only=False,
    ):
        super().__init__()
        self.audio_file_paths = audio_file_paths
        self.gemini_api_key = gemini_api_key
        self.summarization_prompt = summarization_prompt
        self.transcription_prompt = transcription_prompt
        self.transcription_only = transcription_only

        self.gemini_client = None
        self.final_summary_markdown = ""
        self.temp_transcription_file = None

    def run(self):
        logging.info(
            f"Summarization thread started. Transcription Only: {self.transcription_only}"
        )
        file_name_for_error = "audio files"
        full_transcription_text = ""
        uploaded_gemini_files = []  # Keep track of files to delete later

        try:
            # --- API Key and Client Initialization ---
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set in the .env file")

            self.gemini_client = genai.Client(api_key=self.gemini_api_key)
            logging.info("Gemini client initialized.")
            self.progress_update.emit(10)

            # --- Gemini File Upload and Transcription ---
            # This section replaces all the complex Whisper splitting/merging logic.
            logging.info("Using Gemini for transcription. Uploading files...")
            # The 'contents' list will hold the prompt and file references for the API call
            gemini_contents = []

            # Add the user-defined transcription prompt as the first part of the request
            gemini_contents.append(self.transcription_prompt)

            # Upload each audio file using the Gemini Files API
            num_files = len(self.audio_file_paths)
            for index, file_path in enumerate(self.audio_file_paths):
                file_name_for_error = os.path.basename(file_path)
                # Update progress during the upload process
                progress = 15 + int((55 / num_files) * index)
                self.progress_update.emit(progress)
                logging.info(
                    f"Uploading file {index + 1}/{num_files}: {file_name_for_error}..."
                )

                # The core upload call
                gemini_file = self.gemini_client.files.upload(file=file_path)
                gemini_contents.append(gemini_file)
                uploaded_gemini_files.append(gemini_file)  # Track for cleanup
                logging.info(f"Successfully uploaded {gemini_file.name}")

            logging.info("All files uploaded.")
            self.progress_update.emit(70)

            # --- NEW LOGIC: Conditional API Call ---
            if self.transcription_only:
                # --- SCENARIO 1: TRANSCRIPTION ONLY (Existing Logic) ---
                logging.info("Requesting transcription from Gemini...")

                # Build contents for transcription
                gemini_contents.append(self.transcription_prompt)

                transcription_response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=gemini_contents,
                )
                full_transcription_text = transcription_response.text

                # Emit the new single signal with is_summary=False for transcription only
                self.processing_finished.emit(full_transcription_text, False)
                logging.info("Transcription-only result sent to UI.")

            else:
                # --- SCENARIO 2: TRANSCRIPTION AND SUMMARY (New Single-Call Logic) ---
                logging.info(
                    "Requesting transcription AND summary from Gemini in a single call..."
                )

                # Build contents for the combined task. The summarization_prompt is now the main prompt.
                gemini_contents.append(self.summarization_prompt)

                combined_response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=gemini_contents,
                )
                combined_text = combined_response.text

                # --- PARSE THE RESPONSE ---
                # We expect the model to follow the prompt's structure.
                # Let's use robust splitting to separate the two parts.
                full_transcription_text = ""
                summary_text = ""

                # Use unique markers that we will define in the new default prompt.
                transcription_marker = "**Trascrizione:**"
                summary_marker = "**Riassunto:**"

                if summary_marker in combined_text:
                    parts = combined_text.split(summary_marker, 1)
                    if transcription_marker in parts[0]:
                        # Extract transcription from the first part
                        full_transcription_text = (
                            parts[0].replace(transcription_marker, "").strip()
                        )
                    else:
                        # If the transcription marker is missing, assume the first part is the transcription
                        full_transcription_text = parts[0].strip()
                    summary_text = parts[1].strip()
                else:
                    # Fallback if the model doesn't follow instructions perfectly
                    logging.warning(
                        "Summary marker not found in response. Treating the whole response as the summary."
                    )
                    summary_text = combined_text
                    # We can't reliably extract the transcription in this case
                    full_transcription_text = "Could not be separated from the summary."

                logging.info("Transcription and summary received and parsed.")

                # Create a single, combined markdown string for output
                combined_markdown_output = (
                    f"## Riassunto\n\n{summary_text}\n\n"
                    f"---\n\n"
                    f"## Trascrizione\n\n{full_transcription_text}"
                )

                # Emit the new single signal with is_summary=True for combined content
                self.processing_finished.emit(combined_markdown_output, True)
                logging.info("Combined summary and transcription result sent to UI.")

                # Also update the variable used for saving
                self.final_summary_markdown = combined_markdown_output

            self.progress_update.emit(95)  # A bit of progress left for cleanup

            # --- Save transcription to temp file (Optional but can be useful) ---
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".txt", mode="w", delete=False, encoding="utf-8"
                ) as self.temp_transcription_file:
                    self.temp_transcription_file.write(full_transcription_text)
                    logging.info(
                        f"Transcription saved to temporary file: {self.temp_transcription_file.name}"
                    )
            except Exception as e:
                logging.error(f"Error saving transcription to temporary file: {e}")

            self.progress_update.emit(100)
            logging.info("Summarization thread finished successfully.")

        except Exception as e:
            error_message = f"Error processing {file_name_for_error}: {e}"
            logging.error(error_message, exc_info=True)
            self.error_occurred.emit(error_message, file_name_for_error)
            self.progress_update.emit(0)
        finally:
            # --- Cleanup: Delete uploaded files from the Gemini service ---
            logging.info("Cleaning up uploaded Gemini files...")
            for gemini_file in uploaded_gemini_files:
                try:
                    self.gemini_client.files.delete(file=gemini_file)
                    logging.info(f"Deleted remote file from Gemini: {gemini_file.name}")
                except Exception as e:
                    logging.warning(
                        f"Could not delete Gemini file {gemini_file.name}: {e}"
                    )

            if self.temp_transcription_file:
                logging.info(
                    f"Temporary transcription file kept: {self.temp_transcription_file.name}"
                )

        logging.info("Summarization thread run method completed.")


class AudioSummaryApp(QWidget):
    DEFAULT_PROMPT = "Riassumi il contenuto dei file audio forniti."
    DEFAULT_TRANSCRIPTION_PROMPT = "Genera una trascrizione dei file audio forniti."

    def __init__(self):
        logging.info("AudioSummaryApp initialization started.")
        super().__init__()
        self.setWindowTitle("Riassunto audio - messaggi vocali & lezioni")
        self.setGeometry(100, 100, 1300, 1000)

        self.gemini_api_key = None
        self.load_api_key()
        self.summarization_thread = None
        self.audio_file_paths = []
        self.summary_markdown_text = ""
        self.summary_is_unsaved = False
        self.last_result_is_summary = False
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

        # Checkbox for transcription only mode - Merging checkbox is now removed.
        self.transcription_only_checkbox = QCheckBox("Esegui SOLO trascrizione", self)
        self.transcription_only_checkbox.setChecked(False)
        self.transcription_only_checkbox.stateChanged.connect(self.save_settings)
        self.transcription_only_checkbox.stateChanged.connect(
            self.update_prompt_textbox_state
        )
        self.layout.addWidget(self.transcription_only_checkbox)

        # Prompt text box for transcription
        self.transcription_prompt_label = QLabel("Prompt per la Trascrizione:", self)
        self.layout.addWidget(self.transcription_prompt_label)
        self.transcription_prompt_text_edit = QTextEdit(self)
        self.transcription_prompt_text_edit.setFixedHeight(100)
        # Don't set default text here - let load_settings() handle it
        self.transcription_prompt_text_edit.textChanged.connect(self.save_settings)
        self.layout.addWidget(self.transcription_prompt_text_edit)

        # Prompt text box for summarization
        self.prompt_label = QLabel("Prompt per Gemini (Riassunto):", self)
        self.layout.addWidget(self.prompt_label)
        self.prompt_text_edit = QTextEdit(self)
        self.prompt_text_edit.setFixedHeight(180)
        # Don't set default text here - let load_settings() handle it
        self.prompt_text_edit.textChanged.connect(self.save_settings)
        self.layout.addWidget(self.prompt_text_edit)

        # --- Whisper settings UI has been removed ---

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
        self.update_prompt_textbox_state()
        logging.info("Initial prompt textbox state set.")

    def update_prompt_textbox_state(self):
        is_transcription_only = self.transcription_only_checkbox.isChecked()

        # Summary prompt is enabled ONLY if transcription_only is FALSE
        self.prompt_text_edit.setEnabled(not is_transcription_only)
        self.prompt_label.setEnabled(not is_transcription_only)

        # NEW: Transcription prompt is enabled ONLY if transcription_only is TRUE
        self.transcription_prompt_text_edit.setEnabled(is_transcription_only)
        self.transcription_prompt_label.setEnabled(is_transcription_only)

    def load_settings(self):
        logging.info("Loading settings from QSettings.")

        # Temporarily disconnect textChanged signals to avoid triggering save_settings during load
        self.transcription_prompt_text_edit.textChanged.disconnect()
        self.prompt_text_edit.textChanged.disconnect()

        saved_transcription_prompt = self.settings.value(
            "transcriptionPromptText", self.DEFAULT_TRANSCRIPTION_PROMPT
        )
        self.transcription_prompt_text_edit.setPlainText(saved_transcription_prompt)

        saved_prompt = self.settings.value("promptText", self.DEFAULT_PROMPT)
        self.prompt_text_edit.setPlainText(saved_prompt)

        saved_transcription_only_checked = self.settings.value(
            "transcriptionOnlyChecked", False, type=bool
        )

        self.transcription_only_checkbox.setChecked(saved_transcription_only_checked)

        # Reconnect the textChanged signals
        self.transcription_prompt_text_edit.textChanged.connect(self.save_settings)
        self.prompt_text_edit.textChanged.connect(self.save_settings)

        logging.info("Settings loaded.")

    def save_settings(self):
        logging.info("Saving settings to QSettings.")
        self.settings.setValue(
            "transcriptionPromptText", self.transcription_prompt_text_edit.toPlainText()
        )
        self.settings.setValue("promptText", self.prompt_text_edit.toPlainText())
        self.settings.setValue(
            "transcriptionOnlyChecked", self.transcription_only_checkbox.isChecked()
        )
        logging.info("Settings saved.")

    def reset_prompt(self):
        logging.info("Resetting prompt and checkbox to default.")
        self.transcription_prompt_text_edit.setPlainText(
            self.DEFAULT_TRANSCRIPTION_PROMPT
        )
        self.prompt_text_edit.setPlainText(self.DEFAULT_PROMPT)
        self.transcription_only_checkbox.setChecked(False)
        self.save_settings()
        logging.info("Prompt and checkbox reset to default.")

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
        self.summary_is_unsaved = False
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
        self.update_prompt_textbox_state()  # Ensure prompt is disabled if needed
        self.progress_bar.setValue(0)
        self.output_text_edit.clear()

        transcription_only = self.transcription_only_checkbox.isChecked()
        summarization_prompt = (
            self.prompt_text_edit.toPlainText() if not transcription_only else ""
        )
        transcription_prompt = self.transcription_prompt_text_edit.toPlainText()

        logging.info("Creating and starting summarization thread.")
        self.summarization_thread = SummarizationThread(
            self.audio_file_paths,
            self.gemini_api_key,
            summarization_prompt,
            transcription_prompt,
            transcription_only,
        )
        self.summarization_thread.processing_finished.connect(self.display_result)
        self.summarization_thread.error_occurred.connect(self.display_error)
        self.summarization_thread.progress_update.connect(self.update_progress)
        self.summarization_thread.start()
        logging.info("Summarization thread started.")

    def display_result(self, result_text, is_summary):
        logging.info(f"Displaying result. Is summary: {is_summary}")

        if is_summary:
            # If it's a summary, it contains Markdown. Render it as HTML.
            html_content = mistune.html(result_text)
            self.output_text_edit.setHtml(html_content)
        else:
            # Otherwise, it's a plain text transcription.
            self.output_text_edit.setPlainText(result_text)

        self.status_label.setText("Finito!!")
        self.process_button.setEnabled(True)
        self.update_prompt_textbox_state()
        self.progress_bar.setValue(100)

        # This variable is used for saving the file later
        self.last_result_is_summary = is_summary
        self.summary_markdown_text = result_text
        self.summary_is_unsaved = bool(result_text)

        self.prompt_save_file()
        logging.info("Result displayed and UI updated.")

    def prompt_save_file(self):
        # The new flag makes this logic cleaner and more reliable.
        if self.last_result_is_summary:
            content_to_save = self.summary_markdown_text
            default_suffix = "summary.md"
            file_filter = "Markdown Files (*.md)"
            dialog_title = "Salva riassunto e trascrizione come Markdown"
        else:
            # This handles the transcription-only case
            content_to_save = self.output_text_edit.toPlainText()
            default_suffix = "transcription.txt"
            file_filter = "Text Files (*.txt)"
            dialog_title = "Salva trascrizione come testo"

        if not content_to_save:
            return

        file_dialog = QFileDialog()
        last_save_directory = self.settings.value("lastSaveDirectory", "")
        default_file_name = os.path.join(last_save_directory, default_suffix)
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
                self.summary_is_unsaved = False
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
        if self.summary_is_unsaved:
            self.prompt_save_file()

    def display_error(self, error_message, file_path):
        logging.error(f"Error occurred: {error_message} for file: {file_path}")
        QMessageBox.critical(self, "Errore di elaborazione", error_message)
        file_name = os.path.basename(file_path)
        self.status_label.setText(f"Errore durante l'elaborazione di {file_name}.")
        self.process_button.setEnabled(True)
        self.update_prompt_textbox_state()
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
