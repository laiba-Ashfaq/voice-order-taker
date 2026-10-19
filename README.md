# Voice Order Taker

Turn WhatsApp voice notes in Urdu, Roman Urdu, Pashto or English into a clean order sheet and a daily prep list. Everything runs on your own computer. No cloud, no API key, no per-order cost.

Built for the Hacktoberfest Open-Source AI Challenge (Week 2: Side Hustle).

<img width="1279" height="539" alt="image" src="https://github.com/user-attachments/assets/7a808f8f-cfbc-42ae-a47b-3b40abae2660" />



## The problem

Home bakers, tiffin services and small clothing sellers take most of their orders as WhatsApp voice notes. Someone says "do kilo chocolate cake aur teen dozen samosay kal shaam tak, house 12 Jinnah Road", and the seller has to replay it and write it down by hand. With 20 or more orders a week, this takes hours and mistakes happen.

## What it does

1. You upload one or more voice notes (`.ogg`, `.opus`, `.m4a`, `.mp3`, `.wav`).
2. Whisper transcribes each note locally.
3. A Gemma model running in Ollama extracts structured orders: customer, item, quantity, unit, delivery date, address and notes.
4. You review and edit the table, download a CSV, and see a daily prep list that adds up quantities per item and date.

## Why open source

- **Private.** Customer names, phone details and addresses never leave the seller's laptop.
- **Free to run.** No per-minute transcription fees and no per-token API costs, so a small monthly price can still make a profit.
- **Works offline.** Useful when the internet or power is unreliable.
- **Adjustable.** You can change the prompt, give Whisper local dish names, swap in a different model, or fine-tune later.

## Tech stack

| Part | Tool |
| --- | --- |
| Speech to text | faster-whisper (open-source Whisper) |
| Order extraction | Gemma via Ollama |
| Interface | Streamlit with a custom CSS 3D robot mascot |
| Data handling | pandas |

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) installed and running
- About 4 GB of free disk space and 8 GB of RAM recommended (less works with the smallest models)

## Setup

```bash
git clone https://github.com/hfaq/voice-order-taker.git
cd voice-order-taker

python -m venv venv
# Windows
venv\Scripts\activate
# Mac or Linux
source venv/bin/activate

pip install -r requirements.txt
```

Download a model:

```bash
ollama pull gemma3:4b      # better accuracy
# or
ollama pull gemma3:1b      # lighter, for low-memory laptops
```

Run the app:

```bash
streamlit run app.py
```

Open `http://localhost:8501`, pick the model name in the sidebar, upload voice notes and press **Process orders**.

## Configuration

- **Speech model:** `base`, `small` or `medium` in the sidebar. Larger models are better at Urdu but slower and use more disk space.
- **Language:** set to `ur` for Urdu, or `auto` for mixed notes.
- **Ollama model:** any chat model you have pulled. Check with `ollama list`.
- **Storage paths:** the top of `app.py` sets cache folders for model downloads. Change or remove those lines if you are not low on disk space on the system drive.

## Tips for better accuracy

- Add your own item names to the Whisper `initial_prompt` in `transcribe()`.
- Add one or two real examples of your sellers' voice notes to the prompt in `extract()`.
- Use a clear recording with little background noise.
- Larger models give better extraction on messy mixed-language speech.

## Known limitations

- Whisper accuracy for Urdu is good but not perfect, and weaker for Pashto and heavy dialects.
- Small models (1B) can miss fields or mix up quantities. Always review the table before acting on it.
- Transcripts are not cached between runs.
- CPU-only processing can take several seconds per note.

## Roadmap

- Automatic import of WhatsApp chat exports
- Customer and payment tracking
- Voice reply confirmations
- Fine-tuned model for local food and clothing vocabulary

## Project structure

```
voice-order-taker/
  app.py
  requirements.txt
  .streamlit/config.toml
  README.md
```

## License

MIT
