"""
Author: Nađa
Compute syntactic depth and opened/remaining open/closed dependency counts.
"""

# - IMPORTS ------------------------------------------------
import os
import wave
import spacy
import unicodedata
import numpy as np
from tqdm import tqdm
from scipy.io import savemat
from constituent_treelib import ConstituentTree, Language

# - SETUP --------------------------------------------------
FS              = 100
FEAT_DIM        = 4
N_UTTERANCES    = 120
STORIES         = ["AEngA", "AEngB", "AEngC", "AEngD", "UEngC", "UEngD",
                   "AFraA", "AFraB", "AFraC", "AFraD", "UFraC", "UFraD"]
MuR_STORIES     = ["UEngA", "UEngB", "UFraA", "UFraB"]

# - load spaCy + benepar pipelines -------------------------
print("Loading spaCy + benepar pipelines")
nlp_eng_ctl   = ConstituentTree.create_pipeline(Language.English)
nlp_fra_ctl   = ConstituentTree.create_pipeline(Language.French)
nlp_eng_dep   = spacy.load("en_core_web_sm")
nlp_fra_dep   = spacy.load("fr_core_news_sm")
print("Pipelines loaded.")

# - PATHS --------------------------------------------------
BASE        = os.path.expanduser("~/Development/mphil")
STIMULI_DIR = os.path.join(BASE, "0_word_surprisal_entropy/sentence_stimuli")
WAV_DIR            = os.path.join(BASE, "0_speech_envelope/stimuli/")
ENG_TG_DIR  = os.path.join(BASE, "0_morphology/aligned_Eng")
FRA_TG_DIR  = os.path.join(BASE, "0_morphology/aligned_Fra")
OUTPUT      = os.path.join(BASE, "0_syntax/NS_syntactic.mat")

# - HELPER FUNCTIONS ---------------------------------------

# - helper: unicode -> ASCII-compatible --------------------
# returns normalized text
def normalise(txt):
    txt = unicodedata.normalize('NFKC', txt) # ran into some issues with non-breaking spaces so
    txt = ' '.join(txt.split())              # collapses all whitespace variants to single space
    return txt

# - helper: read TextGrid file -----------------------------
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
    in_words    = False
    duration    = 0.0
    header_done = False

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
                if line.startswith("text ="):           # get the word itself
                    label = line.split('"')[1].strip()
                    if label: # don't use silences/blanks/whatever
                        words.append(label)
                        onsets.append(xmin)
    return duration, words, onsets

# - helper: compute syntactic depths --------------------
# returns a list of syntactic depths for each word
# * depth: number of non-unary nodes between root (0) and the word
def compute_constituency_depths(tree, doc):
    depths = [0] * len(doc)

    word_to_indices = {}
    for token in doc:
        word_to_indices.setdefault(token.text.lower(), []).append(token.i)

    def traverse(t, current_depth):
        if len(t) == 0:             
            return
        
        if isinstance(t[0], str):   # leaf node
            word = t[0].lower()
            if word in word_to_indices and word_to_indices[word]:
                idx = word_to_indices[word].pop(0)
                depths[idx] = current_depth
            return
        
        # only increment if branching node (non-unary), see ASCII output
        new_depth = current_depth + 1 if len(t) > 1 else current_depth 
        
        for child in t:
            traverse(child, new_depth)

    traverse(tree.nltk_tree, 0)
    return depths

# - helper: compute dependency counts -------------------
# returns three lists, corresponding to counts of opened,
# remaining open and closed dependencies at each word
def compute_dependency_metrics(doc):
    opened, remaining, closed = [0] * len(doc), [0] * len(doc), [0] * len(doc)

    for token in doc:
        head = token.head.i
        dep  = token.i

        if head < dep:
            opened[head] += 1
            closed[dep]  += 1
        elif head > dep:
            opened[dep]  += 1
            closed[head] += 1
        # else head == dep (ROOT) ignored

    current_open = 0
    for i in range(len(doc)):
        current_open -= closed[i]
        remaining[i] = current_open
        current_open += opened[i] # opened at i not counted in remaining[i]

    return opened, remaining, closed

# - helper: exctract syntactic features for a story ------
# can't extract on utterance-level because syntax spans
# entire sentence so call feature funcs on sentences, also
# compile sentence features into a story
def extract_story_features(sentences, nlp_ctl, nlp_dep):
    all_depths, all_opened, all_remaining, all_closed = [], [], [], []

    for sentence in sentences:
        sentence = sentence.strip()

        doc = nlp_dep(sentence)
        tree = ConstituentTree(sentence, nlp_ctl)

        depths = compute_constituency_depths(tree, doc)
        opened, remaining, closed = compute_dependency_metrics(doc)
        
        all_depths.extend(depths)
        all_opened.extend(opened)
        all_remaining.extend(remaining)
        all_closed.extend(closed)

    return all_depths, all_opened, all_remaining, all_closed

# - helper: build output vector for one utterance ------------
# returns time-aligned (to word onsets) valued impulse vector:
#   col 0: syntactic depths
#   col 1: counts of opened dependencies
#   col 2: counts of remaining open dependencies
#   col 3: counts of closed dependencies
def build_vector(duration, onsets, depths, opened, remaining, closed):
    nSamples = int(np.ceil(duration * FS))
    vec = np.zeros((nSamples, FEAT_DIM))

    for i, onset in enumerate(onsets):
        sample = int(round(onset * FS))
        if 0 <= sample < nSamples:
            vec[sample, 0] = depths[i]
            vec[sample, 1] = opened[i]
            vec[sample, 2] = remaining[i]
            vec[sample, 3] = closed[i]

    return vec

# - MAIN LOOP -------------------------------------------------

vectors, files = [], []

for story in STORIES:
    # set language-dependent context
    is_english = "Eng" in story

    tg_dir = ENG_TG_DIR  if is_english else FRA_TG_DIR
    nlp_ctl = nlp_eng_ctl if is_english else nlp_fra_ctl
    nlp_dep = nlp_eng_dep if is_english else nlp_fra_dep

    # decompose text files into sentences (one sentence per line, lowercase
    # (except for proper nouns), no punctuation (besides for apostrophes))
    with open(os.path.join(STIMULI_DIR, story + ".txt"), encoding="utf-8") as fh:
        sentences = [normalise(l.strip()) for l in fh if l.strip()]

    print(f"\n{story}: extracting syntactic features ({len(sentences)} sentences)")
    story_depths, story_opened, story_remaining, story_closed = extract_story_features(sentences, nlp_ctl, nlp_dep)

    word_idx = 0

    # for alignment check down below
    story_words = [w for s in sentences for w in s.split()]

    for utt in tqdm(range(1, N_UTTERANCES + 1), desc=f"  {story}", leave=False): # nice progress bar grace a tqdm
        tg_name = f"{story}{utt:03d}.TextGrid"
        tg_path = os.path.join(tg_dir, tg_name)

        duration, tg_words, onsets = read_textgrid_words(tg_path)
        n_words = len(tg_words)

        # for alignment check down below
        for j, tg_word in enumerate(tg_words):
            txt_word = story_words[word_idx + j] if (word_idx + j) < len(story_words) else "<END>"
            if tg_word.lower() != txt_word.lower():
                print(f"WARNING | {tg_name} word {j+1}: TextGrid='{tg_word}' vs. transcript='{txt_word}'")

        # slice the next n_words values from the flat syntactic features lists
        utt_depths = story_depths[word_idx : word_idx + n_words]
        utt_opened = story_opened[word_idx : word_idx + n_words]
        utt_remaining = story_remaining[word_idx : word_idx + n_words]
        utt_closed = story_closed[word_idx : word_idx + n_words]
        word_idx += n_words

        vec = build_vector(duration, onsets, utt_depths, utt_opened, utt_remaining, utt_closed)

        # pad to minimum 240 samples + 10-sample zero-pad at onset
        if vec.shape[0] < 240:
            vec = np.vstack([vec, np.zeros((240 - vec.shape[0], FEAT_DIM))])
        vec = np.vstack([np.zeros((10, FEAT_DIM)), vec])

        vectors.append(vec)
        files.append(os.path.splitext(tg_name)[0])

    # alignment check
    n_text_words = sum(len(s.split()) for s in sentences)
    if word_idx != n_text_words:
        print(f"WARNNG |{story}: TextGrids had {word_idx} words and the text file had {n_text_words}")

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
file_codes = np.array(files, dtype=object).reshape(-1, 1) # handle the 1440 x 8 chars instead of 1440 x 1 cell

print("\nSaving .mat")
savemat(OUTPUT, {
    "syntactic": vectors,
    "syntactic_code": file_codes
})
print(f"Saved {len(vectors)} vectors → {OUTPUT}")