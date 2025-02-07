#!/usr/bin/env python3

import os
from dotenv import load_dotenv
from openai import OpenAI
import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QFileDialog, QTextEdit, QMessageBox,
                             QProgressBar, QHBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal
from google import genai
import mistune


class SummarizationThread(QThread):
    """
    Thread to handle audio processing, transcription (OpenAI Whisper),
    and summarization (Gemini) to avoid blocking the GUI.
    """
    summary_finished = pyqtSignal(str)
    transcription_finished = pyqtSignal(str)  # Signal for transcription output
    error_occurred = pyqtSignal(str, str)
    progress_update = pyqtSignal(int)

    def __init__(self, audio_file_path, openai_api_key, gemini_api_key):
        super().__init__()
        self.audio_file_path = audio_file_path
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.openai_client = None
        self.gemini_client = None

    def run(self):
        try:
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set in the .env file")
            if not self.gemini_api_key:
                raise ValueError("GEMINI_API_KEY is not set in the .env file")

            self.openai_client = OpenAI(api_key=self.openai_api_key)

            self.gemini_client = genai.Client(api_key=self.gemini_api_key)

            self.progress_update.emit(10)

            with open(self.audio_file_path, "rb") as audio_file:
                self.progress_update.emit(30)

                transcription_response = self.openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                transcribed_text = transcription_response.text
                self.transcription_finished.emit(
                    transcribed_text)  # Emit transcription
                self.progress_update.emit(60)

            model = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",

                contents=f"Fai il riassunto del seguente messaggio vocale: {transcribed_text}"
            )
            summary_text = model.text
            self.progress_update.emit(90)

            self.summary_finished.emit(summary_text)
            self.progress_update.emit(100)

        except Exception as e:
            error_message = f"Error processing {self.audio_file_path}: {e}"
            self.error_occurred.emit(error_message, self.audio_file_path)
            self.progress_update.emit(0)


class AudioSummaryApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Riassunto audio - messaggi vocali")
        self.setGeometry(100, 100, 800, 500)

        self.openai_api_key = None
        self.gemini_api_key = None
        self.load_api_key()
        self.summarization_thread = None

        self.init_ui()
        self.apply_dark_theme()

    def apply_dark_theme(self):
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

    def load_api_key(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            QMessageBox.warning(self, "Chiave API OpenAI mancante",
                                "O, imposta OPENAI_API_KEY nel tuo file .env per la trascrizione.")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key:
            QMessageBox.warning(self, "Chiave API Gemini mancante",
                                "O, imposta GEMINI_API_KEY nel tuo file .env per la riassunzione.")

    def init_ui(self):
        self.layout = QVBoxLayout()

        file_selection_layout = QHBoxLayout()
        self.select_file_button = QPushButton("Seleziona file audio", self)
        self.select_file_button.clicked.connect(self.select_audio_file)
        file_selection_layout.addWidget(self.select_file_button)
        self.file_path_label = QLabel("Nessun file audio selezionato", self)
        file_selection_layout.addWidget(self.file_path_label)
        self.layout.addLayout(file_selection_layout)

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

    def select_audio_file(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(self, "Seleziona file", "",
                                                   "File Audio (*.ogg *.wav *.mp3 *.flac)")
        if file_path:
            self.file_path_label.setText(f"File selezionato: {file_path}")
            self.audio_file_path = file_path
            self.process_button.setEnabled(True)
            self.summary_output_text_edit.clear()
            self.transcription_output_text_edit.clear()  # Clear transcription as well
            self.status_label.clear()
        else:
            self.file_path_label.setText("Nessun file audio selezionato")
            self.process_button.setEnabled(False)
            self.audio_file_path = None

    def start_processing(self):
        if not hasattr(self, 'audio_file_path') or not self.audio_file_path:
            QMessageBox.warning(self, "Nessun file selezionato",
                                "O, seleziona prima un file audio.")
            return

        if not self.openai_api_key or not self.gemini_api_key:
            QMessageBox.warning(self, "Chiavi API mancanti",
                                "O, imposta sia OPENAI_API_KEY che GEMINI_API_KEY nel tuo file .env e riavvia.")
            return

        self.status_label.setText("Trascrizione audio e riassunto in corso...")
        self.process_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.summary_output_text_edit.clear()
        self.transcription_output_text_edit.clear()  # Clear transcription output

        self.summarization_thread = SummarizationThread(
            self.audio_file_path, self.openai_api_key, self.gemini_api_key
        )
        self.summarization_thread.summary_finished.connect(
            self.display_result)
        self.summarization_thread.transcription_finished.connect(  # Connect new signal
            self.display_transcription)
        self.summarization_thread.error_occurred.connect(
            self.display_error)
        self.summarization_thread.progress_update.connect(
            self.update_progress)
        self.summarization_thread.start()

    def display_result(self, summary_text):

        html_summary = mistune.html(summary_text)

        self.summary_output_text_edit.setHtml(html_summary)
        self.status_label.setText("Finito!!")
        self.process_button.setEnabled(True)
        self.progress_bar.setValue(100)

    # New method to display transcription
    def display_transcription(self, transcription_text):
        self.transcription_output_text_edit.setText(transcription_text)

    def display_error(self, error_message, audio_file_path):
        QMessageBox.critical(self, "Errore di elaborazione", error_message)
        self.status_label.setText(
            f"Errore durante l'elaborazione di {os.path.basename(audio_file_path)}.")
        self.process_button.setEnabled(True)
        self.progress_bar.setValue(0)

    def update_progress(self, progress_value):
        self.progress_bar.setValue(progress_value)


if __name__ == '__main__':
    app = QApplication(sys.argv)

    main_window = AudioSummaryApp()
    main_window.show()
    sys.exit(app.exec_())
