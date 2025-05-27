# AI-Powered Audio Summarizer (Whisper & Gemini)

A desktop application built with PyQt5 that transcribes audio files using OpenAI's Whisper API and summarizes the transcriptions using Google's Gemini API. It's designed to help users quickly get summaries of voice notes, lectures, or other audio content, with options for merging multiple audio files and a "transcription-only" mode.

## Table of Contents

-   [Features](#features)
-   [Screenshots](#screenshots)
-   [Installation](#installation)
    -   [Prerequisites](#prerequisites)
    -   [API Keys Setup](#api-keys-setup)
    -   [Install Dependencies](#install-dependencies)
-   [Usage](#usage)
-   [How it Works](#how-it-works)
-   [Contributing](#contributing)
-   [License](#license)

## Features

*   **Audio Transcription:** Transcribes audio (MP3, WAV, OGG, FLAC) using OpenAI Whisper.
*   **AI Summarization:** Summarizes transcribed text with Google Gemini (using a customizable prompt).
*   **Intuitive GUI:** User-friendly graphical interface built with PyQt5.
*   **Multiple File Support:** Select and process multiple audio files.
*   **Audio Merging:** Option to merge selected audio files into a single large file before transcription and summarization (useful for sequential voice notes or lectures).
*   **"Transcription Only" Mode:** Skip summarization and only get the audio transcription.
*   **Customizable Prompt:** Define your own summarization prompt for Gemini to tailor the output.
*   **Progress Bar:** Real-time progress updates during processing.
*   **Output Saving:** Automatically prompts to save the transcription and summary to a Markdown (`.md`) file.
*   **Persistent Settings:** Saves your last used prompt, checkbox states, and file directories.
*   **Dark Theme:** Aesthetic dark mode for comfortable viewing.
*   **Robust Error Handling:** Provides user-friendly error messages and logging for troubleshooting.

## Screenshots

![Screenshot](https://github.com/alessandroamella/riassunto-audio/raw/master/screenshot.png "Screenshot dell'app")

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.x**: Download from [python.org](https://www.python.org/downloads/).
2.  **pip**: Python's package installer (usually comes with Python).
3.  **FFmpeg**: `pydub`, which handles audio processing, requires FFmpeg to be installed on your system.
    *   **Windows**: Download a build from [ffmpeg.org](https://ffmpeg.org/download.html#build-windows) and add its `bin` directory to your system's PATH.
    *   **macOS**: `brew install ffmpeg` (using [Homebrew](https://brew.sh/)).
    *   **Linux**: `sudo apt update && sudo apt install ffmpeg` (Debian/Ubuntu) or `sudo dnf install ffmpeg` (Fedora).

### API Keys Setup

This application requires API keys from OpenAI and Google AI Studio.

1.  **Get your OpenAI API Key:**
    *   Visit [platform.openai.com](https://platform.openai.com/).
    *   Sign up or log in.
    *   Go to `API keys` and create a new secret key.
    *   *Note: Using Whisper costs money based on usage.*

2.  **Get your Google Gemini API Key:**
    *   Visit [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).
    *   Sign up or log in.
    *   Create a new API key.
    *   *Note: Using Gemini Flash is generally very cheap or free within generous limits.*

3.  **Create a `.env` file:** In the root directory of the project, create a file named `.env` and add your keys like this:

    ```env
    OPENAI_API_KEY=your_openai_api_key_here
    GEMINI_API_KEY=your_gemini_api_key_here
    ```
    Replace `your_openai_api_key_here` and `your_gemini_api_key_here` with your actual keys.

### Install Dependencies

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/audio-ai-summarizer.git
    cd audio-ai-summarizer
    ```
    *(Replace `https://github.com/yourusername/audio-ai-summarizer.git` with the actual URL of your repository)*

2.  **Create a virtual environment (recommended):**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

    If you don't have a `requirements.txt` yet, you can create one by running:
    ```bash
    pip freeze > requirements.txt
    ```
    Or manually install the listed dependencies:
    ```bash
    pip install PyQt5 openai google-generativeai mistune python-dotenv pydub
    ```

## Usage

1.  **Run the application:**

    ```bash
    python3 main.py
    ```
    *(Assuming your main script file is named `main.py`)*

2.  **Interface Overview (Italian terms translated):**

    *   **"Seleziona file audio"** (Select audio file): Click to choose one or multiple audio files (`.ogg`, `.wav`, `.mp3`, `.flac`). The selected files will be displayed next to the button.
    *   **"Reset Prompt"**: Resets the summarization prompt and checkboxes to their default values.
    *   **"Unisci file audio"** (Merge audio files):
        *   **Checked (Default):** If you select multiple audio files, they will be merged into a single file before transcription and summarization. This is useful for processing long audio split into parts.
        *   **Unchecked:** Each selected audio file will be transcribed and summarized individually.
    *   **"Esegui SOLO trascrizione"** (Perform ONLY transcription):
        *   **Checked:** The application will only perform transcription using Whisper and skip the summarization step with Gemini. The "Prompt per Gemini (AI)" field will be disabled.
        *   **Unchecked (Default):** Both transcription and summarization will be performed.
    *   **"Prompt per Gemini (AI)"** (Prompt for Gemini (AI)): A text area where you can customize the prompt given to Google Gemini for summarization. The default prompt is in Italian and is suitable for voice messages.
    *   **"Riassumi!!"** (Summarize!!): Click this button to start the processing. It will be disabled until at least one audio file is selected.
    *   **Progress Bar:** Shows the progress of transcription and summarization.
    *   **"Trascrizione:"** (Transcription:): Displays the raw transcribed text from Whisper.
    *   **"Riassunto:"** (Summary:): Displays the summarized text from Gemini.
    *   **Status Label:** Shows real-time messages about the application's state (e.g., "Processing...", "Finished!").

3.  **Workflow:**
    *   Click "Seleziona file audio" and choose your desired audio files.
    *   Adjust the "Unisci file audio" and "Esegui SOLO trascrizione" checkboxes based on your needs.
    *   Modify the "Prompt per Gemini (AI)" if you want a custom summary style (this field is disabled if "Esegui SOLO trascrizione" is checked).
    *   Click "Riassumi!!" to start the process.
    *   The application will display the transcription and summary.
    *   Once finished, a file dialog will automatically pop up, prompting you to save the output (summary or transcription) as a Markdown (`.md`) file. Choose your desired location and filename.

## How it Works

1.  **Audio Selection:** The user selects one or more audio files.
2.  **Audio Processing (Pydub):**
    *   If "Merge audio files" is checked, Pydub concatenates all selected audio files into a single temporary MP3 file (to optimize size for API upload).
    *   If not merged, each file is processed individually.
3.  **Transcription (OpenAI Whisper):** The audio (or merged audio) is sent to the OpenAI Whisper API for highly accurate speech-to-text transcription.
4.  **Summarization (Google Gemini):**
    *   If "Transcription Only" is unchecked, the full transcription text is combined with the user-defined prompt.
    *   This combined input is then sent to Google's Gemini-2.0-flash model for summarization.
5.  **GUI Update (PyQt5):** The transcription and summary results are displayed in the application's text areas. Progress is shown via a progress bar.
6.  **Output Saving:** The final summary (or transcription if summarization was skipped) is offered to be saved as a Markdown file.
7.  **Temporary File Cleanup:** Any temporary audio files created during the merging process are automatically deleted.

## Contributing

Feel free to open issues or submit pull requests. Any contributions are welcome!

## License

This project is open-source and available under the [MIT License](LICENSE).
