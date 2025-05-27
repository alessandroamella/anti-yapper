#!/usr/bin/env python3

import logging  # Import the logging module
import os
import sys
import tempfile

import mistune
from dotenv import load_dotenv
from google import genai
from openai import OpenAI
from pydub import AudioSegment
from PyQt5.QtCore import QSettings, QThread, pyqtSignal  # Import QSettings
from PyQt5.QtWidgets import (  # Import QCheckBox
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
    Thread to handle audio processing, transcription (OpenAI Whisper),
    and summarization (Gemini) to avoid blocking the GUI.
    Handles single or merged audio files based on 'merge_audio' flag.
    Optimized for smaller merged audio file size.
    """

    summary_finished = pyqtSignal(str)
    transcription_finished = pyqtSignal(str)  # Signal for transcription output
    error_occurred = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)

    def __init__(
        self,
        audio_file_paths,
        openai_api_key,
        gemini_api_key,
        summarization_prompt,
        merge_audio=True,
        transcription_only=False,
    ):
        super().__init__()
        self.audio_file_paths = audio_file_paths  # Now accepts a list of file paths
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.summarization_prompt = summarization_prompt  # User defined prompt
        self.merge_audio = merge_audio  # Flag to control audio merging
        self.transcription_only = transcription_only  # Flag for transcription only mode
        self.openai_client = None
        self.gemini_client = None
        self.temp_merged_file = None  # To store temp file object
        self.final_summary_markdown = ""  # Store markdown summary
        self.temp_transcription_file = None  # To store temp transcription file

    def run(self):
        logging.info(
            f"Summarization thread started. Merge audio: {self.merge_audio}, Transcription Only: {self.transcription_only}"
        )
        # Default value in case of early errors
        file_name_for_error = "unknown audio file"
        final_summary_output = ""  # To accumulate summaries if not merging
        full_transcription_text = (
            ""  # Accumulate transcription text for saving and display
        )

        try:
            if not self.openai_api_key:
                error_msg = "OPENAI_API_KEY is not set in the .env file"
                logging.error(error_msg)
                raise ValueError(error_msg)
            if (
                not self.gemini_api_key and not self.transcription_only
            ):  # Gemini key is needed only if not transcription_only
                error_msg = "GEMINI_API_KEY is not set in the .env file"
                logging.error(error_msg)
                raise ValueError(error_msg)

            self.openai_client = OpenAI(api_key=self.openai_api_key)
            if not self.transcription_only:
                self.gemini_client = genai.Client(api_key=self.gemini_api_key)
                logging.info("Gemini client initialized.")
            logging.info("OpenAI client initialized.")

            self.progress_update.emit(10)

            if self.merge_audio:
                if (
                    isinstance(self.audio_file_paths, list)
                    and len(self.audio_file_paths) > 1
                ):
                    logging.info("Merging multiple audio files.")
                    self.progress_update.emit(20)
                    merged_audio = AudioSegment.from_file(self.audio_file_paths[0])
                    for file_path in self.audio_file_paths[1:]:
                        audio_segment = AudioSegment.from_file(file_path)
                        merged_audio += audio_segment

                    # Strategy 1: Export merged audio as MP3 to reduce file size
                    # MP3 is a lossy format but significantly smaller than WAV, suitable for voice.
                    self.temp_merged_file = tempfile.NamedTemporaryFile(
                        suffix=".mp3", delete=False
                    )  # Changed suffix to .mp3
                    merged_audio.export(
                        self.temp_merged_file.name,
                        format="mp3",
                        bitrate="128k",  # Strategy 2: Consider lower bitrate if needed, 128k is generally ok for speech
                    )  # Export as MP3
                    audio_file_to_process = open(self.temp_merged_file.name, "rb")
                    file_name_for_error = "merged audio file"  # For error messages
                    logging.info(
                        f"Merged audio saved to temporary MP3 file: {self.temp_merged_file.name}"
                    )
                    self.progress_update.emit(25)

                else:
                    # Process single audio file
                    audio_file_path = (
                        self.audio_file_paths
                        if not isinstance(self.audio_file_paths, list)
                        else self.audio_file_paths[0]
                    )  # Handle single path or list with one path
                    audio_file_to_process = open(audio_file_path, "rb")
                    file_name_for_error = audio_file_path  # For error messages
                    logging.info(f"Processing single audio file: {file_name_for_error}")
                    self.progress_update.emit(30)

                # Use the appropriate audio file (merged or single)
                with audio_file_to_process:
                    logging.info("Starting audio transcription with OpenAI Whisper.")
                    transcription_response = (
                        self.openai_client.audio.transcriptions.create(
                            model="whisper-1", file=audio_file_to_process
                        )
                    )
                    transcribed_text = transcription_response.text
                    logging.info("Transcription finished.")
                    # Log first 50 chars of transcription for info (avoid very long logs)
                    logging.info(
                        f"Transcription (first 50 chars): {transcribed_text[:50]}..."
                    )
                    full_transcription_text = (
                        transcribed_text  # Assign for saving and display
                    )
                    self.transcription_finished.emit(transcribed_text)
                    self.progress_update.emit(60)

            else:  # If merge_audio is False, process each file individually
                logging.info("Processing each audio file individually.")
                self.progress_update.emit(20)
                combined_transcriptions = []
                final_transcription_output = (
                    ""  # Initialize for accumulating markdown transcriptions
                )
                for index, audio_file_path in enumerate(self.audio_file_paths):
                    file_name_for_error = audio_file_path
                    logging.info(f"Processing file: {audio_file_path}")
                    self.progress_update.emit(
                        20 + (60 / len(self.audio_file_paths)) * index
                    )  # Incremental progress

                    with open(audio_file_path, "rb") as audio_file_to_process:
                        logging.info(f"Starting transcription for: {audio_file_path}")
                        transcription_response = (
                            self.openai_client.audio.transcriptions.create(
                                model="whisper-1", file=audio_file_to_process
                            )
                        )
                        transcribed_text = transcription_response.text
                        logging.info(f"Transcription finished for: {audio_file_path}")
                        current_file_transcription_markdown = f"**{os.path.basename(audio_file_path)} Trascrizione:**\n{transcribed_text}\n\n"
                        final_transcription_output += current_file_transcription_markdown  # Accumulate markdown for display
                        self.transcription_finished.emit(
                            current_file_transcription_markdown
                        )  # Emit individual transcription to append on UI
                        combined_transcriptions.append(
                            transcribed_text
                        )  # append just the text, not the markdown

                        self.progress_update.emit(
                            40 + (60 / len(self.audio_file_paths)) * index
                        )
                full_transcription_text = "\n\n".join(
                    combined_transcriptions
                )  # Assign concatenated text for saving

                # Emit combined transcriptions (with markdown) AFTER loop to show full result at the end if needed. But we are emitting in loop already for incremental display.
                # self.transcription_finished.emit(final_transcription_output) # No need to emit again here as we are emitting in loop for incremental display

                self.progress_update.emit(
                    60
                )  # Move progress update here after all transcriptions

            # Save transcription to temp file
            try:
                self.temp_transcription_file = tempfile.NamedTemporaryFile(
                    suffix=".txt", mode="w", delete=False, encoding="utf-8"
                )
                self.temp_transcription_file.write(full_transcription_text)
                temp_transcription_file_name = (
                    self.temp_transcription_file.name
                )  # Capture name before closing
                self.temp_transcription_file.close()  # Close to ensure content is written
                logging.info(
                    f"Transcription saved to temporary file: {temp_transcription_file_name}"
                )
            except Exception as e:
                logging.error(f"Error saving transcription to temporary file: {e}")

            if (
                not self.transcription_only
            ):  # Perform summarization only if transcription_only is False
                if self.merge_audio:  # Summarize merged transcription
                    transcription_text_for_summary = full_transcription_text
                else:  # Summarize concatenated transcriptions
                    transcription_text_for_summary = full_transcription_text

                logging.info("Starting summarization with Gemini.")
                model = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    # Use user defined prompt
                    contents=f"{self.summarization_prompt} {transcription_text_for_summary}",
                )
                summary_text = model.text
                logging.info("Summarization finished.")
                # Log first 50 chars of summary for info (avoid very long logs)
                logging.info(f"Summary (first 50 chars): {summary_text[:50]}...")
                self.progress_update.emit(90)
                final_summary_output = summary_text  # Assign for final output
                self.final_summary_markdown = (
                    final_summary_output  # Store markdown summary for saving
                )
                self.summary_finished.emit(
                    final_summary_output
                )  # Emit the final summary output (merged or combined)
            else:  # If transcription_only, skip summarization and just emit a 'finished' signal (can reuse summary_finished for simplicity)
                final_summary_output = (
                    "Transcription completed - summarization skipped as requested."
                )
                self.final_summary_markdown = final_summary_output
                self.summary_finished.emit(
                    final_summary_output
                )  # Emit to signal completion even if no summary
                logging.info("Summarization skipped due to 'Transcription Only' mode.")

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
                    f"Temporary merged file deleted: {self.temp_merged_file.name}"
                )
            if self.temp_transcription_file:
                # Temporary transcription file is deleted in finally block, but it should be kept for user reference in case of summary failure.
                # Let's NOT delete the temp transcription file. User can delete it manually if needed.
                # os.remove(self.temp_transcription_file.name)
                logging.info(
                    f"Temporary transcription file kept: {temp_transcription_file_name}"
                )  # Log that it's kept, not deleted

        logging.info("Summarization thread run method completed.")


class AudioSummaryApp(QWidget):
    DEFAULT_PROMPT = "Riassumi la trascrizione di un messaggio vocale ricevuto dall'utilizzatore del softwarede. Deve essere un riassunto in terza persona rispetto all'interlocutore. Fai sì che sia comprensibile la comprensione del 'mood' del messaggio, in modo che il lettore possa comprendere il tono e il contenuto di esso, oltre allo stato emotivo dell'interlocutore. Di seguito la trascrizione:"

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
        self.merge_audio_checkbox = QCheckBox("Unisci file audio", self)
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

        self.process_button = QPushButton("Riassumi!!", self)
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
            "File Audio (*.ogg *.wav *.mp3 *.flac)",
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
                file_display_text = f"File selezionato: {self.audio_file_paths[0]}"
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
        if not self.audio_file_paths:  # Check if file paths list is empty
            warning_msg_no_file = "No audio file selected before processing."
            logging.warning(warning_msg_no_file)
            QMessageBox.warning(
                self, "Nessun file selezionato", "O, seleziona prima un file audio."
            )
            return

        openai_key_needed = True
        gemini_key_needed = (
            True if not self.transcription_only_checkbox.isChecked() else False
        )  # Gemini key needed only if not transcription only mode

        if not self.openai_api_key and openai_key_needed:
            warning_msg_api_keys = "API keys missing (OpenAI)."
            logging.warning(warning_msg_api_keys)
            QMessageBox.warning(
                self,
                "Chiavi API mancanti",
                "O, imposta OPENAI_API_KEY nel tuo file .env e riavvia.",
            )
            return
        if not self.gemini_api_key and gemini_key_needed:
            warning_msg_api_keys = "API keys missing (Gemini)."
            logging.warning(warning_msg_api_keys)
            QMessageBox.warning(
                self,
                "Chiavi API mancanti",
                "O, imposta GEMINI_API_KEY nel tuo file .env e riavvia.",
            )
            return

        self.status_label.setText("Trascrizione audio e riassunto in corso...")
        self.process_button.setEnabled(False)
        # Disable prompt text edit during processing
        self.prompt_text_edit.setEnabled(False)
        self.progress_bar.setValue(0)
        self.summary_output_text_edit.clear()
        self.transcription_output_text_edit.clear()

        # Get the state of the checkboxes
        merge_audio = self.merge_audio_checkbox.isChecked()
        transcription_only = self.transcription_only_checkbox.isChecked()
        logging.info(f"Merge audio checkbox is checked: {merge_audio}")
        logging.info(f"Transcription only checkbox is checked: {transcription_only}")

        # Get prompt from text edit
        if transcription_only:
            summarization_prompt = "-- Solo trascrizione audio --"
        else:
            summarization_prompt = self.prompt_text_edit.toPlainText()
            logging.info(f"Using summarization prompt: {summarization_prompt}")

        logging.info("Creating and starting summarization thread.")
        self.summarization_thread = SummarizationThread(
            # Pass prompt and merge_audio flag to thread
            self.audio_file_paths,
            self.openai_api_key,
            self.gemini_api_key,
            summarization_prompt,
            merge_audio,
            transcription_only,  # Pass transcription_only flag
        )
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
        # Re-enable prompt text edit after processing
        self.prompt_text_edit.setEnabled(True)
        self.progress_bar.setValue(100)
        self.summary_markdown_text = summary_text  # Store markdown text

        self.prompt_save_file()  # Call save file dialog after displaying result
        logging.info("Summary displayed and UI updated.")

    def prompt_save_file(self):
        logging.info("Prompting user to save file.")
        file_dialog = QFileDialog()

        # Get the last save directory from settings, or use the current directory as default
        last_save_directory = self.settings.value("lastSaveDirectory", "")
        default_file_name = os.path.join(
            last_save_directory,
            "summary.md" if self.summary_markdown_text else "transcription.txt"
        ) # Default file name based on summary or transcription

        file_path, _ = file_dialog.getSaveFileName(
            self,
            "Salva riassunto come Markdown",
            default_file_name,
            "Markdown Files (*.md)",
        )

        if file_path:
            # Save the directory as the last save directory
            last_save_directory = os.path.dirname(file_path)
            self.settings.setValue("lastSaveDirectory", last_save_directory)
            logging.info(f"Last save directory updated to: {last_save_directory}")

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    # write markdown text if summary, else write transcription
                    if self.summary_markdown_text:
                        f.write(self.summary_markdown_text)
                    else:
                        f.write(self.transcription_output_text_edit.toPlainText())
                logging.info("Summary saved successfully.")
                self.status_label.setText(f"Riassunto salvato in: {file_path}")
            except Exception as e:
                error_msg = f"Errore durante il salvataggio del file: {e}"
                logging.error(error_msg)
                QMessageBox.critical(self, "Errore di salvataggio", error_msg)
                self.status_label.setText(
                    "Errore durante il salvataggio del riassunto."
                )
        else:
            logging.info("Save file dialog cancelled by user.")
            self.status_label.setText("Riassunto non salvato.")

    def display_transcription(self, transcription_text):
        logging.info("Displaying transcription.")
        current_text = self.transcription_output_text_edit.toPlainText()
        if (
            current_text
        ):  # Append only if there's existing text to avoid extra newline at start
            self.transcription_output_text_edit.append(transcription_text)
        else:
            self.transcription_output_text_edit.setText(
                transcription_text
            )  # Set text for the first time
        logging.info("Transcription displayed.")

    def display_error(
        self, error_message, file_path
    ):  # file_path can be string or list
        logging.error(f"Error occurred: {error_message} for file: {file_path}")
        QMessageBox.critical(self, "Errore di elaborazione", error_message)
        if isinstance(file_path, list):
            file_name = "merged audio files"
        else:
            file_name = os.path.basename(file_path)
        self.status_label.setText(f"Errore durante l'elaborazione di {file_name}.")
        self.process_button.setEnabled(True)
        # Re-enable prompt text edit after error
        self.prompt_text_edit.setEnabled(True)
        self.progress_bar.setValue(0)
        logging.error(f"Error displayed in UI for file: {file_name}")

    def update_progress(self, progress_value):
        self.progress_bar.setValue(progress_value)
        # Use debug level as progress updates can be frequent
        logging.debug(f"Progress bar updated to: {progress_value}%")


if __name__ == "__main__":
    logging.info("Application started.")
    app = QApplication(sys.argv)
    main_window = AudioSummaryApp()
    main_window.show()
    exit_code = app.exec_()
    logging.info(f"Application exited with code: {exit_code}")
    sys.exit(exit_code)
