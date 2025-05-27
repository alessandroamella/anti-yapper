# Anti-Yapper 🗣️➡️📝

Programma GUI per trascrivere e riassumere file audio. Così puoi avere lezioni o messaggi vocali trascritti (con OpenAI Whisper) e riassunti (con Google Gemini).

![Anti-Yapper Screenshot](https://github.com/alessandroamella/riassunto-audio/raw/master/screenshot.png "Screenshot dell'app")

## ✨ Caratteristiche

- **Trascrizione Audio:** Converte l'audio parlato in testo usando OpenAI Whisper.
- **Supporto Multi-File:** Elabora più file audio, unendoli per un riassunto unico o trascrivendo/riassumendo ogni file individualmente.
- **Modalità Solo Trascrizione:** Ottieni solo la trascrizione, senza riassunto.
- **Prompt Personalizzabile:** Definisci il tuo prompt per Gemini e personalizza lo stile del riassunto.
- **GUI:** Interfaccia semplice creata con PyQt5.
- **Tema Scuro:**.
- **Salvataggio Output:** Salvataggio automatico del riassunto (o trascrizione) in formato Markdown (`.md`).
- **Formati Supportati:** Compatibile con OGG, WAV, MP3 e FLAC.

## 🚀 Installazione

Segui questi passaggi per avviare Anti-Yapper sulla tua macchina.

### Prerequisiti

- Python 3.x
- Una Chiave API OpenAI (per la trascrizione)
- Una Chiave API Google Gemini (per il riassunto)

### Passaggi

1.  **Clona il repository:**

    ```bash
    git clone https://github.com/alessandroamella/riassunto-audio.git
    cd riassunto-audio
    ```

    _(Sostituisci `https://github.com/alessandroamella/riassunto-audio.git` con l'URL del tuo repository, se diverso)_

2.  **Crea un ambiente virtuale (consigliato):**

    ```bash
    python3 -m venv venv
    ```

3.  **Attiva l'ambiente virtuale:**

    - **Linux/macOS:**
      ```bash
      source venv/bin/activate
      ```
    - **Windows:**
      ```bash
      .\venv\Scripts\activate
      ```

4.  **Installa le dipendenze:**

    ```bash
    pip install -r requirements.txt
    ```

5.  **Configura le Chiavi API:**
    Crea un file `.env` nella directory principale del progetto (dove si trova `gui.py`) e aggiungi le tue chiavi API:

    ```ini
    OPENAI_API_KEY="la_tua_chiave_api_openai_qui"
    GEMINI_API_KEY="la_tua_chiave_api_gemini_qui"
    ```

    - Puoi ottenere una Chiave API OpenAI da [platform.openai.com](https://platform.openai.com/).
    - Puoi ottenere una Chiave API Google Gemini da [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey).

6.  **File Desktop (Opzionale):**
    È incluso un file `anti-yapper.desktop` che puoi personalizzare e utilizzare per creare un collegamento sul desktop o nel menu delle applicazioni di Linux. Modifica il percorso nel file per puntare alla tua installazione.
    - Copia il file nella directory `~/.local/share/applications/` per renderlo disponibile nel menu delle applicazioni.

## 💡 Utilizzo

1.  **Avvia l'applicazione:**

    ```bash
    python gui.py
    ```

2.  **Seleziona i File Audio:**
    Clicca su "Seleziona file audio" e scegli uno o più file (OGG, WAV, MP3, FLAC).

3.  **Configura le Opzioni:**

    - **Unisci file audio:** Seleziona questa casella se hai più file e vuoi che vengano uniti per una trascrizione e un riassunto combinati. Altrimenti, ogni file sarà elaborato individualmente.
    - **Esegui SOLO trascrizione:** Seleziona questa casella per ottenere solo la trascrizione. Il prompt di Gemini sarà disabilitato in questa modalità.
    - **Prompt per Gemini (AI):** Modifica il testo per guidare Gemini nel riassunto. Il prompt predefinito è in italiano e ottimizzato per i messaggi vocali.

4.  **Avvia l'Elaborazione:**
    Clicca su "Riassumi!!". L'applicazione mostrerà una barra di progresso e visualizzerà trascrizione e riassunto.

5.  **Salva l'Output:**
    Al termine dell'elaborazione, apparirà una finestra di dialogo per salvare il riassunto (o la trascrizione in modalità solo trascrizione) come file Markdown.
