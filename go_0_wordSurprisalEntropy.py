"""
Author: Nađa
Compute lexical surprisal and entropy using BLOOM-7b (HuggingFace).
"""
# - IMPORTS ------------------------------------------------
import wave
import os, math
import numpy as np
import torch
from tqdm import tqdm
from scipy.io import savemat
from transformers import AutoTokenizer, AutoModelForCausalLM

# - SETUP --------------------------------------------------
FS                  = 100 
FEAT_DIM            = 2     # number of columns in output vector
USE_STORY_CONTEXT   = True  # True for full story context, False for sentence-level context
N_UTTERANCES        = 120 
MuR_STORIES         = ["UEngA", "UEngB", "UFraA", "UFraB"]              # non-linguistic stimuli
STORIES             = ["AEngA","AEngB","AEngC","AEngD","UEngC","UEngD", # linguistic stimuli
                       "AFraA","AFraB","AFraC","AFraD","UFraC","UFraD"]

# - PATHS --------------------------------------------------
BASE        = os.path.expanduser("~/Development/mphil")
WAV_DIR     = os.path.join(BASE, "0_speech_envelope/stimuli/")
STIMULI_DIR = os.path.join(BASE, "0_word_surprisal_entropy/sentence_stimuli")
ENG_TG_DIR  = os.path.join(BASE, "0_morphology/aligned_Eng")
FRA_TG_DIR  = os.path.join(BASE, "0_morphology/aligned_Fra")
OUTPUT      = os.path.join(BASE, "0_word_surprisal_entropy/NS_word_surprisal_entropy.mat")

# - HELPER FUNCTIONS ---------------------------------------

# - load model from HuggingFace ----------------------------
print("Loading BLOOM-7b")
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-7b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-7b1", dtype=torch.float32)
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)
print(f"Model loaded on {device}")

# - helper: read TextGrid file -----------------------------
# returns TG file
def open_textgrid(path):
    for enc in ["utf-8", "utf-16", "latin-1"]:
        # was running into some issues with
        # Praat using different encodings
        try:
            f = open(path, encoding=enc)
            f.read()
            f.seek(0)
            return f
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Unknown encoding: {path}")

# - helper: read the words tier from a TextGrid -----------
# returns utterance duration, words and their onsets
def read_textgrid_words(path):
    f = open_textgrid(path)

    words, onsets = [], []
    duration = 0.0
    header_done = False 
    in_words = False

    with f:
        for line in f:
            line = line.strip()

            if not header_done and line.startswith("xmax ="):
                duration = float(line.split("=")[1]) # get utterance length

            if 'name = "words"' in line: # entering words tier
                in_words = True
                header_done = True
                continue

            if 'name =' in line and 'words' not in line: # exiting words tier
                in_words = False
                header_done = True
                continue

            if in_words: # inside the words tier
                if line.startswith("xmin ="):
                    xmin = float(line.split("=")[1])    # get word onset
                if line.startswith("text ="):           
                    label = line.split('"')[1].strip()  # get the word itself
                    if label: # don't use silences/blanks/whatever
                        words.append(label)
                        onsets.append(xmin)
    return duration, words, onsets


# - helper: tokenize words -------------------------------------
# returns sub-word tokens and (start, end) tuples for each word
# (basically which tokens from the token_ids does each word span)
def tokenize_sentence(words):
    token_ids, spans = [], []

    for i, w in enumerate(words):
        toks  = tokenizer.encode((" " if i > 0 else "") + w, add_special_tokens=False)

        start = len(token_ids)
        token_ids.extend(toks)
        spans.append((start, len(token_ids)))

    return token_ids, spans

# - helper: compute surprisal & entropy ------------------------------
# returns a list of surprisal values () for each word, and a list of
# entropy values for each word
# * surprisal: -log2 P(word | context), summed over sub-word tokens
# * entropy: H over vocabulary just before the first token of the word
def compute_surprisal_entropy(sentences, use_story_context=True):
    all_surp, all_ent = [], []
    context_ids = []
    first_word_of_story = True

    for sentence in sentences:
        # tokenize current utterance
        words = sentence.strip().split()
        sent_ids, sent_spans = tokenize_sentence(words)

        # build model input
        ctx = context_ids if use_story_context else [] # full story vs. sentence context
        n_ctx = len(ctx)
        full_ids = ctx + sent_ids

        # single forward pass
        input_ids = torch.tensor([full_ids], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(input_ids).logits[0] # (seq_len, vocab_size)
            log_probs = torch.log_softmax(logits, dim=-1) # natural log; /log(2) for bits

        for w_idx, (_, (tok_start, tok_end)) in enumerate(zip(words, sent_spans)):
            # first word of story gets a value of 0 for both (as per other literature)
            # first word of each sentence also gets 0 when using sentence-level context
            if first_word_of_story or (not use_story_context and w_idx == 0):
                all_surp.append(0.0)
                all_ent.append(0.0)
                first_word_of_story = False
                continue

            # logit[p] predicts the token at position p+1 in full_ids, so the
            # logit position predicting the first token of this word is
            pred_pos = n_ctx + tok_start - 1

            # compute entropy
            probs = torch.softmax(logits[pred_pos], dim=-1)
            ent = -torch.sum(probs * torch.log2(probs.clamp(min=1e-40))).item()

            # compute surprisal
            surp = 0.0
            for j in range(tok_end - tok_start):
                tok = sent_ids[tok_start + j]
                surp += -log_probs[pred_pos + j, tok].item() / math.log(2)

            all_surp.append(surp)
            all_ent.append(ent)

        if use_story_context:
            context_ids.extend(sent_ids) # extend context

    return all_surp, all_ent


# - helper: build output vector for one utterance ------------
# returns time-aligned (to word onsets) valued impulse vector:
#   col 0: surprisal
#   col 1: entropy
def build_vector(duration, onsets, word_surp, word_ent):
    nSamples = max(1, int(np.ceil(duration * FS)))
    vec = np.zeros((nSamples, 2))

    for i, onset in enumerate(onsets):
        sample = int(round(onset * FS))
        if 0 <= sample < nSamples:
            vec[sample, 0] = word_surp[i]
            vec[sample, 1] = word_ent[i]

    return vec

# - MAIN LOOP -------------------------------------------------

vectors, files = [], []

for story in STORIES:
    tg_dir = ENG_TG_DIR if "Eng" in story else FRA_TG_DIR

    # decompose current story into sentences (one sentence per line, lowercase
    # (except for proper nouns), no punctuation (besides for apostrophes))
    with open(os.path.join(STIMULI_DIR, story + ".txt"), encoding="utf-8") as fh:
        sentences = [l.strip() for l in fh if l.strip()]

    print(f"\n{story}: computing surprisal/entropy ({len(sentences)} sentences)")
    story_surp, story_ent = compute_surprisal_entropy(sentences, USE_STORY_CONTEXT)

    word_idx = 0

    for utt in tqdm(range(1, N_UTTERANCES + 1), desc=f"  {story}", leave=False): # nice progress bar grace a tqdm
        tg_name = f"{story}{utt:03d}.TextGrid"
        tg_path = os.path.join(tg_dir, tg_name)

        duration, tg_words, onsets = read_textgrid_words(tg_path)
        n_words = len(tg_words)

        # slice the next n_words values from the flat surprisal/entropy lists
        utt_surp = story_surp[word_idx : word_idx + n_words]
        utt_ent  = story_ent[word_idx  : word_idx + n_words]
        word_idx += n_words

        vec = build_vector(duration, onsets, utt_surp, utt_ent)

        # pad to minimum 240 samples + 10-sample zero-pad at onset
        if vec.shape[0] < 240:
            vec = np.vstack([vec, np.zeros((240 - vec.shape[0], FEAT_DIM))])
        vec = np.vstack([np.zeros((10, FEAT_DIM)), vec])

        vectors.append(vec)
        files.append(os.path.splitext(tg_name)[0])

# - ZERO VECTORS FOR MuR STIMULI ------------------------
# UEngA, UEngB, UFraA, UFraB have no linguistic features
# so putting zeros for all features exc. speech env
for story in MuR_STORIES:
    for utt in range(1, N_UTTERANCES + 1):
        utt_code = f"{story}{utt:03d}"
        wav_path = os.path.join(WAV_DIR, f"{utt_code}.wav")

        # using .wav file to get duration, since did not run
        # read_textgrid_words on these (where we otherwise get
        # duration from)
        with wave.open(str(wav_path), 'r') as wf:
            duration = wf.getnframes() / wf.getframerate()

        nSamples = int(np.ceil(duration * FS))
        vec = np.zeros((nSamples, FEAT_DIM)) # zero vector

        # same padding as attended entries
        if vec.shape[0] < 240:
            vec = np.vstack([vec, np.zeros((240 - vec.shape[0], FEAT_DIM))])
        vec = np.vstack([np.zeros((10, FEAT_DIM)), vec])

        vectors.append(vec)
        files.append(utt_code)    

# - SAVE OUTPUT MAT FILE ------------------------
print("\nSaving .mat")
file_codes = np.array(files, dtype=object).reshape(-1, 1)  # (1440×1) cell
savemat(OUTPUT, {
    "attended_semantic": vectors,
    "attended_semantic_code": file_codes
})
print(f"Saved {len(vectors)} vectors to {OUTPUT}")