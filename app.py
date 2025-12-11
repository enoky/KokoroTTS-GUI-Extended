import gradio as gr
import os
import random
import torch
import time

# --- MONKEY PATCH: Fix for 'EspeakWrapper' has no attribute 'set_data_path' ---
# This block must run BEFORE importing kokoro or misaki.
# It patches a compatibility issue between misaki and newer phonemizer versions.
try:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    if not hasattr(EspeakWrapper, 'set_data_path'):
        print("DEBUG: Patching EspeakWrapper.set_data_path for compatibility...")
        def set_data_path(path):
            # This environment variable is often used by espeak-ng as a fallback
            os.environ["PHONEMIZER_ESPEAK_DATA"] = path
        EspeakWrapper.set_data_path = set_data_path
except ImportError:
    pass # If phonemizer isn't installed, the main import block will catch it.

# --- NEW: Imports and setup for text chunking ---
import re
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    raise ImportError("NLTK not found. Please install it using: pip install nltk")

# Download NLTK 'punkt' model if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("DEBUG: NLTK 'punkt' tokenizer not found. Downloading...")
    nltk.download('punkt')
    print("DEBUG: Download complete.")

# --- NEW: Progress Logging ---
def log_progress(message, level="INFO"):
    """Simple colored and timestamped logger."""
    colors = {
        "INFO": "\033[94m",    # Blue
        "DEBUG": "\033[92m",   # Green
        "WARN": "\033[93m",    # Yellow
        "ERROR": "\033[91m",   # Red
        "ENDC": "\033[0m",     # End color
    }
    color = colors.get(level, colors["INFO"])
    timestamp = time.strftime('%H:%M:%S')
    print(f"{color}[{timestamp} - {level}] {message}{colors['ENDC']}")

# --- LOCAL SETUP: Import Kokoro modules ---
# Ensure the 'kokoro' folder/module is in the same directory or installed
try:
    from kokoro import KModel, KPipeline
    import kokoro
    import misaki
except ImportError as e:
    raise ImportError(f"Could not import kokoro or misaki. Ensure you are in the correct directory with the library files. Error: {e}")

# --- DEVICE DETECTION ---
# Check for NVIDIA GPU (CUDA) on Windows
CUDA_AVAILABLE = torch.cuda.is_available()
device = 'cuda' if CUDA_AVAILABLE else 'cpu'

print(f"DEBUG: Kokoro Version: {kokoro.__version__}")
print(f"DEBUG: Misaki Version: {misaki.__version__}")
print(f"DEBUG: Running on {device.upper()}")

# --- MODEL LOADING ---
# For local use, we simply load one model onto the best available device.
# We keep the dictionary structure for compatibility with existing functions,
# but mapped to the same model instance to save VRAM.
print("Loading model... this may take a moment.")
try:
    model_instance = KModel().to(device).eval()
except Exception as e:
    raise RuntimeError(f"Failed to load KModel. Do you have the model weights (kokoro-v0_19.pth) in this folder? Error: {e}")

models = {True: model_instance, False: model_instance} # True=GPU, False=CPU request (handled same locally)

# --- FORWARD PASS ---
def forward_device(ps, ref_s, speed):
    """Simplified forward pass that uses the globally loaded model."""
    return models[CUDA_AVAILABLE](ps, ref_s, speed)

# --- PIPELINE SETUP ---
pipelines = {lang_code: KPipeline(lang_code=lang_code, model=False) for lang_code in 'ab'}
pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'

# --- TEXT CHUNKING HELPERS (Adapted from Chatter.py) ---

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s{2,}', ' ', text.strip())

def remove_inline_reference_numbers(text):
    # Removes bracketed or superscript-style reference numbers (e.g., [1], .”3)
    return re.sub(r'([.!?,\"\'”’)\]])(\d+)(?=\s|$)', r'\1', text)

def replace_letter_period_sequences(text: str) -> str:
    # Converts "J.R.R." to "J R R"
    def replacer(match):
        cleaned = match.group(0).rstrip('.')
        letters = cleaned.split('.')
        return ' '.join(letters)
    return re.sub(r'\b(?:[A-Za-z]\.){2,}', replacer, text)

def split_long_sentence(sentence, max_len=300, seps=None):
    """
    Recursively split a sentence into chunks of <= max_len using a sequence of separators.
    Tries each separator in order, splitting further as needed.
    """
    if seps is None:
        seps = [';', ':', '-', ',', ' ']

    sentence = sentence.strip()
    if len(sentence) <= max_len:
        return [sentence]

    if not seps:
        # Fallback: force split every max_len chars
        return [sentence[i:i + max_len].strip() for i in range(0, len(sentence), max_len)]

    sep = seps[0]
    parts = sentence.split(sep)

    if len(parts) == 1:
        # Separator not found, try next separator
        return split_long_sentence(sentence, max_len, seps=seps[1:])

    # Recursively process each part, joining the separator back
    chunks = []
    current = parts[0].strip()
    for part in parts[1:]:
        candidate = (current + sep + part).strip()
        if len(candidate) > max_len:
            chunks.extend(split_long_sentence(current, max_len, seps=seps[1:]))
            current = part.strip()
        else:
            current = candidate
    
    # Add the last processed part
    if current:
        chunks.extend(split_long_sentence(current, max_len, seps=seps[1:]))
        
    return [c for c in chunks if c]

def group_sentences(sentences, max_chars=280):
    """
    Groups sentences into chunks of a specified maximum character length.
    """
    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        sentence_len = len(sentence)

        if sentence_len > max_chars:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            
            # Split the oversized sentence and add its parts as separate chunks
            chunks.extend(split_long_sentence(sentence, max_chars))
            
            current_chunk = []
            current_length = 0
        elif current_length + sentence_len + (1 if current_chunk else 0) <= max_chars:
            current_chunk.append(sentence)
            current_length += sentence_len + (1 if current_chunk else 0)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def process_chunk(chunk_text, index, voice, speed):
    """Processes a single chunk of text to generate audio."""
    try:
        pipeline = pipelines[voice[0]]
        pack = pipeline.load_voice(voice)

        chunk_text = chunk_text.strip()
        if not chunk_text:
            return index, None, None

        processed_chunk = next(pipeline(chunk_text, voice, speed), None)
        if processed_chunk is None:
            log_progress(f"Pipeline returned no output for chunk {index+1}.", "WARN")
            return index, None, None

        _, ps, _ = processed_chunk

        ref_s_index = len(ps) - 1
        if ref_s_index >= len(pack):
            log_progress(f"ref_s index {ref_s_index} out of bounds for pack length {len(pack)} in chunk {index+1}. Using last element.", "WARN")
            ref_s_index = len(pack) - 1
        
        ref_s = pack[ref_s_index]

        audio = forward_device(ps, ref_s, speed)
        return index, audio, str(ps)
    except Exception as e:
        log_progress(f"Audio generation failed for chunk {index+1}: {e}", "ERROR")
        return index, None, None

# --- GENERATION FUNCTIONS ---

def generate_first(text, voice, speed, use_gpu, clean_lowercase, clean_whitespace, clean_references, clean_initials, parallel_chunks):
    """
    Generates audio for the given text. For long text, it splits it into chunks,
    generates audio for each in parallel, and concatenates them.
    """
    log_progress("Generation started.", "INFO")
    text = text.strip()
    if not text:
        log_progress("Input text is empty. Aborting.", "WARN")
        return None, ''

    # --- NEW: Apply text cleaning ---
    log_progress("Applying text cleaning options...", "DEBUG")
    if clean_lowercase:
        text = text.lower()
    if clean_whitespace:
        text = normalize_whitespace(text)
    if clean_references:
        text = remove_inline_reference_numbers(text)
    if clean_initials:
        text = replace_letter_period_sequences(text)

    log_progress("Splitting text into sentences and grouping into chunks...")
    sentences = sent_tokenize(text)
    chunks = group_sentences(sentences)
    log_progress(f"Text divided into {len(chunks)} chunks for parallel processing.", "DEBUG")

    results = [None] * len(chunks)
    
    with ThreadPoolExecutor(max_workers=int(parallel_chunks)) as executor:
        log_progress(f"Submitting {len(chunks)} chunks to thread pool ({int(parallel_chunks)} workers)...", "DEBUG")
        futures = [executor.submit(process_chunk, chunk, i, voice, speed) for i, chunk in enumerate(chunks)]
        
        completed_count = 0
        for future in as_completed(futures):
            completed_count += 1
            index, audio, ps = future.result()
            if audio is not None:
                results[index] = (audio, ps)
            log_progress(f"Completed chunk processing: {completed_count}/{len(chunks)}.", "INFO")

    log_progress("Collating results...", "DEBUG")
    all_audio = [res[0] for res in results if res]
    all_ps = [res[1] for res in results if res]

    if not all_audio:
        log_progress("No audio was generated for any chunk.", "ERROR")
        return None, ''

    log_progress("Concatenating audio chunks...", "INFO")
    silence = torch.zeros(int(24000 * 0.25))  # Silence on CPU
    final_audio_list = []
    for i, audio_chunk in enumerate(all_audio):
        final_audio_list.append(audio_chunk.cpu())
        if i < len(all_audio) - 1:
            final_audio_list.append(silence)

    final_audio = torch.cat(final_audio_list)
    final_ps = "\n---\n".join(all_ps)
    
    log_progress("Generation finished successfully.", "INFO")
    return (24000, final_audio.numpy()), final_ps

def tokenize_first(text, voice, clean_lowercase, clean_whitespace, clean_references, clean_initials):
    text = text.strip()
    if not text:
        return ''
    
    log_progress("Tokenizing first chunk for display...", "DEBUG")
    # Apply same cleaning as generation to get accurate tokens
    if clean_lowercase:
        text = text.lower()
    if clean_whitespace:
        text = normalize_whitespace(text)
    if clean_references:
        text = remove_inline_reference_numbers(text)
    if clean_initials:
        text = replace_letter_period_sequences(text)

    sentences = sent_tokenize(text)
    chunks = group_sentences(sentences)
    first_chunk = chunks[0] if chunks else ''

    pipeline = pipelines[voice[0]]
    for _, ps, _ in pipeline(first_chunk, voice):
        return ps
    return ''

# --- FILE HANDLING (Robustness for Local Run) ---
# We use try/except blocks so the app runs even if you don't have the text files.

def load_text_file(filename, default_text):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as r:
                return r.read().strip()
        except Exception:
            return default_text
    return default_text

def get_random_quote():
    if os.path.exists('en.txt'):
        with open('en.txt', 'r', encoding='utf-8') as r:
            lines = [line.strip() for line in r if line.strip()]
            if lines:
                return random.choice(lines)
    return "Kokoro is an open-weight TTS model with 82 million parameters."

def get_gatsby():
    return load_text_file('gatsby5k.md', "The Great Gatsby text file was not found.")

def get_frankenstein():
    return load_text_file('frankenstein5k.md', "The Frankenstein text file was not found.")

# --- UI CONFIGURATION ---

CHOICES = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇺🇸 🚹 Puck': 'am_puck',
    '🇺🇸 🚹 Echo': 'am_echo',
    '🇺🇸 🚹 Eric': 'am_eric',
    '🇺🇸 🚹 Liam': 'am_liam',
    '🇺🇸 🚹 Onyx': 'am_onyx',
    '🇺🇸 🚹 Santa': 'am_santa',
    '🇺🇸 🚹 Adam': 'am_adam',
    '🇬🇧 🚺 Emma': 'bf_emma',
    '🇬🇧 🚺 Isabella': 'bf_isabella',
    '🇬🇧 🚺 Alice': 'bf_alice',
    '🇬🇧 🚺 Lily': 'bf_lily',
    '🇬🇧 🚹 George': 'bm_george',
    '🇬🇧 🚹 Fable': 'bm_fable',
    '🇬🇧 🚹 Lewis': 'bm_lewis',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
}

TOKEN_NOTE = '''💡 Customize pronunciation with Markdown link syntax and /slashes/ like `[Kokoro](/kˈOkəɹO/)`
💬 To adjust intonation, try punctuation `;:,.!?—…"()“”` or stress `ˈ` and `ˌ`
⬇️ Lower stress `[1 level](-1)` or `[2 levels](-2)`
⬆️ Raise stress 1 level `[or](+2)` 2 levels (only works on less stressed, usually short words)'''

# --- GRADIO INTERFACE ---

with gr.Blocks(title="Kokoro TTS Local") as app:
    with gr.Row():
        gr.Markdown("## Kokoro TTS (Local Windows Version)")
    
    with gr.Row():
        with gr.Column():

            text = gr.Textbox(label='Input Text', lines=5, value="Hello, this is a local test of Kokoro TTS on Windows.")
            
            with gr.Row():
                voice = gr.Dropdown(list(CHOICES.items()), value='af_heart', label='Voice')
                speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='Speed')
            
            with gr.Accordion("Text Cleaning Options", open=True):
                clean_lowercase = gr.Checkbox(label="Convert to Lowercase", value=True)
                clean_whitespace = gr.Checkbox(label="Normalize Whitespace", value=True)
                clean_references = gr.Checkbox(label="Remove Reference Numbers (e.g., [1])", value=True)
                clean_initials = gr.Checkbox(label="Format Initials (e.g., J.R.R.)", value=True)

            with gr.Accordion("Parallel Processing", open=True):
                parallel_chunks = gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Chunks to Process in Parallel")

            with gr.Row():
                # Hardware selection is visual only in this local version, logic is auto-handled
                use_gpu = gr.Dropdown(
                    [('GPU (Detected)' if CUDA_AVAILABLE else 'GPU (Not Found)', True), ('CPU', False)],
                    value=CUDA_AVAILABLE,
                    label='Hardware',
                    interactive=False 
                )
            
            with gr.Row():
                generate_btn = gr.Button('Generate', variant='primary')
                stop_generate_btn = gr.Button('Stop', variant='stop')
            out_audio = gr.Audio(label='Output Audio', interactive=False)
            with gr.Accordion('Output Tokens', open=False):
                out_ps = gr.Textbox(interactive=False, show_label=False)
                tokenize_btn = gr.Button('Tokenize', variant='secondary')
                gr.Markdown(TOKEN_NOTE)

            with gr.Row():
                random_btn = gr.Button('🎲 Random Quote', variant='secondary')
                gatsby_btn = gr.Button('🥂 Gatsby', variant='secondary')
                frankenstein_btn = gr.Button('💀 Frankenstein', variant='secondary')

    # Event Handlers
    random_btn.click(fn=get_random_quote, inputs=[], outputs=[text])
    gatsby_btn.click(fn=get_gatsby, inputs=[], outputs=[text])
    frankenstein_btn.click(fn=get_frankenstein, inputs=[], outputs=[text])
    
    generation_inputs = [
        text, 
        voice, 
        speed, 
        use_gpu,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials,
        parallel_chunks
    ]
    
    tokenization_inputs = [
        text,
        voice,
        clean_lowercase,
        clean_whitespace,
        clean_references,
        clean_initials
    ]
    
    generation_event = generate_btn.click(fn=generate_first, inputs=generation_inputs, outputs=[out_audio, out_ps])
    tokenize_btn.click(fn=tokenize_first, inputs=tokenization_inputs, outputs=[out_ps])
    
    stop_generate_btn.click(fn=None, cancels=generation_event)

if __name__ == '__main__':
    # Launch in browser automatically
    app.queue().launch(inbrowser=True)