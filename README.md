# Anti-Yapper 🗣️➡️📝

Programma GUI per trascrivere e riassumere file audio, come lezioni e messaggi vocali. Per riassunti e analisi usa Google Gemini, per trascrizioni fedeli OpenAI `gpt-transcribe`.

![Anti-Yapper Screenshot](https://github.com/alessandroamella/riassunto-audio/raw/master/screenshot.png "Screenshot dell'app")

## Caratteristiche

- Un menu a tendina sceglie il modello: Gemini per analisi, riassunti e domande di follow-up, oppure OpenAI `gpt-transcribe` per una trascrizione fedele.
- La trascrizione usa OpenAI `gpt-transcribe`. I file in formati non supportati o più grandi di 25 MB vengono convertiti e divisi automaticamente con `ffmpeg`.
- Si possono elaborare più file insieme. Gemini li riceve tutti in un'unica richiesta; con OpenAI ogni file viene trascritto sotto un titolo con il suo nome.
- Il prompt per Gemini è modificabile, quindi puoi decidere tu cosa fare con l'audio e lo stile del riassunto. Con OpenAI lo stesso campo serve per dare contesto alla trascrizione (argomento, nomi, termini tecnici).
- Dopo una risposta di Gemini puoi continuare la conversazione con domande di follow-up.
- Con Gemini puoi attivare la ricerca Google, utile per domande su eventi recenti.
- I file caricati su Gemini restano disponibili per 48 ore, quindi rielaborare lo stesso file non lo ricarica.
- L'interfaccia è semplice ed è fatta con PyQt6.
- Tema scuro? Sì.
- Il risultato viene salvato in Markdown (`.md`), e l'ultima risposta riappare al riavvio dell'app.
- Formati supportati: OGG, WAV, MP3, FLAC, M4A, AAC e AIFF.

## Installazione

### Prerequisiti

- Python 3.x
- Una chiave API OpenAI (per la trascrizione)
- `ffmpeg` (per convertire e dividere l'audio da trascrivere con OpenAI)
- Una chiave API Google Gemini (per il riassunto)

### Passaggi

1.  Clona il repository:

    ```bash
    git clone https://github.com/alessandroamella/riassunto-audio.git
    cd riassunto-audio
    ```

    Se usi un fork, metti l'URL del tuo repository al posto di `https://github.com/alessandroamella/riassunto-audio.git`.

2.  Crea un ambiente virtuale (consigliato):

    ```bash
    python3 -m venv venv
    ```

3.  Attiva l'ambiente virtuale.

    Su Linux/macOS:

    ```bash
    source venv/bin/activate
    ```

    Su Windows:

    ```bash
    .\venv\Scripts\activate
    ```

4.  Installa le dipendenze:

    ```bash
    pip install -r requirements.txt
    ```

5.  Configura le chiavi API. Crea un file `.env` nella cartella del progetto (quella con `gui.py`) e scrivici le tue chiavi:

    ```ini
    OPENAI_API_KEY="la_tua_chiave_api_openai_qui"
    GEMINI_API_KEY="la_tua_chiave_api_gemini_qui"
    GEMINI_MODEL="gemini-3-flash-preview"
    ```

    La chiave OpenAI si ottiene da [platform.openai.com](https://platform.openai.com/), quella Gemini da [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey). `GEMINI_MODEL` è opzionale: se manca, l'app usa `gemini-3-flash-preview`.

6.  File desktop (opzionale). Il repository include `anti-yapper.desktop`, che puoi usare come collegamento sul desktop o nel menu delle applicazioni di Linux. Modifica il percorso al suo interno in modo che punti alla tua installazione, poi copialo in `~/.local/share/applications/` per vederlo nel menu.

## Utilizzo

1.  Avvia l'applicazione:

    ```bash
    python gui.py
    ```

2.  Clicca su "Seleziona file audio" e scegli uno o più file.

3.  Scegli il modello dal menu "Modello":

    - Gemini, se vuoi un riassunto o un'analisi. Nel campo del prompt scrivi cosa fare con l'audio; quello predefinito chiede trascrizione e riassunto. Se serve, attiva la casella della ricerca Google.
    - OpenAI `gpt-transcribe`, se vuoi solo la trascrizione. Il campo del prompt è opzionale e serve come contesto.

    L'app ricorda un prompt diverso per ciascun modello. "Reset Prompt" riporta quello del modello selezionato al valore predefinito.

4.  Clicca su "Elabora!". Una barra di progresso mostra l'avanzamento, poi compare il risultato.

5.  Alla fine si apre una finestra per salvare il risultato come file Markdown. Se la chiudi, puoi salvare dopo con Ctrl+S.

6.  Con Gemini puoi scrivere altre domande nel campo "Continua la conversazione" e inviarle con "Invia continuo". Le risposte vengono aggiunte sotto il risultato.
