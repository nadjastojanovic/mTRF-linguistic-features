"""
Author: Nađa
Compute phonotactic and corpus-based features.
"""

# - IMPORTS ------------------------------------------------
import os
import wave
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import savemat
from collections import defaultdict

from ipatok import tokenise       # IPA tokenization
from phonecodes import phonecodes # IPA -> DISC (CELEX)

# - SETUP --------------------------------------------------
FS              = 100
FEAT_DIM        = 5
N_UTTERANCES    = 120
MuR_STORIES     = ["UEngA", "UEngB", "UFraA", "UFraB"]


MFA_TO_CELEX = {
    "tʰ": ["t"], # aspiration
    "kʰ": ["k"],
    "pʰ": ["p"],
    "cʰ": ["k"],

    "c":  ["k"], # assume
    "ɟ":  ["ɡ"], 
    "ʎ":  ["l"], 
    "ç":  ["h"], 
    "tʲ": ["t"],
    "dʲ": ["d"],
    "fʲ": ["f"],
    "vʲ": ["v"],
    "bʲ": ["b"],
    "pʲ": ["p"],
    "mʲ": ["m"],

    "t̪":  ["t"], # dental diacritics
    "d̪":  ["d"],

    "tʷ": ["t"], # assume
    "kʷ": ["k"],
    "ɫ":  ["l"], 
    "ɫ̩":  ["l"],
    "ɹ":  ["ɻ"],

    "ʉ":  ["uː"], # goose-fronting
    "ʉː": ["uː"], 

    "ɐ":  ["ʌ"], # strut-centering

    "ɒː": ["ɔː"], # lot/cloth/thought-lowering

    "ɑ":  ["ɑː"], # assume
    "i":  ["ɪ"],
    "ɛː": ["ɛ"],

    "aj": ["a", "ɪ"], # diphthongs
    "aw": ["a", "ʊ"],
    "ej": ["e", "ɪ"],
    "ɔj": ["ɔ", "ɪ"],
    "əw": ["ə", "ʊ"],

    "tʃ": ["t", "ʃ"], # assume
    "dʒ": ["d", "ʒ"],

    "ʔ":  [], # no CELEX equivalent
}

MFA_TO_LEXIQUE = {
    "ɡ":  ["g"], # unicode difference
    "ɟ":  ["g"], # assume
    "ʎ":  ["l"], # assume
    "mʲ": ["m"], # assume
    "ʁ":  ["r"], # MFA: /r/ -> /ʁ/ for French, Lexique: /r/
    "ɑ":  ["a"], # MFA: /a/ before [q sˁ tˁ dˁ ðˁ ɫ r] -> /ɑ/ -> Lexique: /a/
}

FRENCH_ELISIONS = ("j'", "lorsqu'", "m'", "n'", "d'", "l'", "qu'", "s'", "t'", "c'")

# - PATHS --------------------------------------------------
BASE        = os.path.expanduser("~/Development/mphil")
WAV_DIR     = os.path.join(BASE, "0_speech_envelope/stimuli/")
ENG_TG_DIR  = os.path.join(BASE,"0_morphology/aligned_Eng")
FRA_TG_DIR  = os.path.join(BASE,"0_morphology/aligned_Fra")
CELEX       = os.path.join(BASE,"0_phonotactic_prob/CELEX2/english/epw/epw.cd")
LEXIQUE     = os.path.join(BASE,"0_phonotactic_prob/Lexique400/Lexique4.tsv")
OUTPUT      = os.path.join(BASE,"0_phonotactic_prob/NS_phonotactic_probs.mat")

# - HELPER FUNCTIONS ---------------------------------------

# - helper: read TextGrid file -----------------------------
# returns TG file
def open_textgrid(path):
    for enc in ["utf-8", "utf-16", "latin-1"]: # braat
        try:
            f = open(path, encoding=enc)
            f.read()
            f.seek(0)
            return f
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"Unknown encoding: {path}")

# - helper: read the phones tier from a TextGrid -----------
# returns utterance duration, phones and their onsets
def read_textgrid_phones(path):
    f = open_textgrid(path)

    phones, onsets = [], []
    duration = 0.0
    header_done = False 
    in_phones = False

    with f:
        for line in f:
            line = line.strip()

            if not header_done and line.startswith("xmax ="):
                duration = float(line.split("=")[1]) # get utterance length

            if 'name = "phones"' in line: # entering phones tier
                in_phones = True
                header_done = True
                continue

            if 'name =' in line and 'phones' not in line: # exiting phones tier
                in_phones = False
                header_done = True
                continue

            if in_phones: # inside the phones tier
                if line.startswith("xmin ="):
                    xmin = float(line.split("=")[1])    # get phone onset
                if line.startswith("text ="):
                    label = line.split('"')[1]          # get the phone itself
                    if label: # don't use silences/blanks/whatever
                        phones.append(label)
                        onsets.append(xmin)
    return duration, phones, onsets

# - helper: read the words tier from a TextGrid -----------
# returns word labels, onsets and offsets
def read_textgrid_words(path):
    f = open_textgrid(path)

    words, onsets, offsets = [], [], []
    in_words = False

    with f:
        for line in f:
            line = line.strip()

            if 'name = "words"' in line: # entering words tier
                in_words = True
                continue

            if 'name =' in line and 'words' not in line: # exiting words tier
                in_words = False
                continue

            if in_words: # inside the words tier
                if line.startswith("xmin ="):
                    xmin = float(line.split("=")[1])    # get word onset
                if line.startswith("xmax ="):
                    xmax = float(line.split("=")[1])    # get word offset
                if line.startswith("text ="):
                    label = line.split('"')[1].strip()  # get the word itself
                    if label: # don't use silences/blanks/whatever
                        words.append(label)
                        onsets.append(xmin)
                        offsets.append(xmax)

    return words, onsets, offsets

# - helper: load CELEX ------------------------------------
# returns words, frequencies, and phoneme sequences
def load_celex():
    words, freqs, phones = [], [], []

    with open(CELEX) as f:
        for line in f:
            fields = line.strip().split("\\")

            word = fields[1]
            freq = float(fields[2]) + 1 # + 1 bc of Laplace smoothing
            disc = fields[6].replace("-","").replace("'","")

            try:
                ipa = phonecodes.convert(disc,"disc","ipa","eng")
            except:
                print("ERR | failed at converting DISC to IPA")
                continue

            words.append(word)
            freqs.append(freq)
            phones.append(tokenise(ipa))

    return words,freqs,phones


# - helper: load Lexique -----------------------------------
# returns words, frequencies, and phoneme sequences
def load_lexique():
    df = pd.read_csv(LEXIQUE, sep="\t")

    words = df["1_Mot"].astype(str).values
    ipas = df["3_Phono_IPA"].astype(str).values
    freqs = df["10_FreqMot"].astype(float).values + (1_000_000 / 316_000_000)   # Laplace smoothing (frequencies in
                                                                                # Lexique are expressed as occurrences per
    phones = [tokenise(ipa.replace("-", "").replace(".", "")) for ipa in ipas]  # million and Lexique4 corpus size is 316m)

    return words, freqs, phones

# - helper: normalise phone sequence with onsets ----------
# expands MFA phones to corpus-compatible IPA, preserving onsets
def normalise_phones_with_onsets(phones, onsets_in_samples, mapping):
    norm_phones, norm_onsets = [], []
    for p, s in zip(phones, onsets_in_samples):
        if p in mapping:
            for ep in mapping[p]:
                norm_phones.append(ep)
                norm_onsets.append(s)
        else:
            norm_phones.append(p)
            norm_onsets.append(s)
    return norm_phones, norm_onsets

# - helper: build frequency probability tables -------------
# returns positional segment and biphone probability dicts
def build_tables(freqs, phones):
    seg_counts, bi_counts, seg_denom_counts, bi_denom_counts = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)

    for f, ph in zip(freqs, phones):
        log_f = np.log10(f) if f >= 0 else 0 # see Vitevitch & Luce (2004), p. 2 > Positional segment frequency

        # positional segment frequency
        for pos, p in enumerate(ph):
            seg_counts[(p, pos)] += log_f
            seg_denom_counts[pos] += log_f # positional segment freq denominator: all words with length >= position

        # biphone frequency
        for pos in range(len(ph)-1): # iterate over each biphone
            bi = (ph[pos], ph[pos+1])
            bi_counts[(bi, pos)] += log_f
            bi_denom_counts[pos] += log_f # biphone freq denominator: all words with length >= position + 1    

    seg_prob = {k: v / seg_denom_counts[k[1]] for k, v in seg_counts.items() if seg_denom_counts[k[1]] > 0}
    bi_prob  = {k: v / bi_denom_counts[k[1]]  for k, v in bi_counts.items()  if bi_denom_counts[k[1]]  > 0}

    return seg_prob, bi_prob

# - helper: build pronunciation-frequency dictionary -------
# maps each word to a list of (phoneme_tuple, freq) pairs
# *** handles multiple entries for the same word ***
def build_pron_freq_dict(words, phones, freqs):
    temp = defaultdict(lambda: defaultdict(float))
    for w, ph, f in zip(words, phones, freqs):
        temp[w.lower()][tuple(ph)] += f  # sum duplicate pronunciations
    return {w: list(entries.items()) for w, entries in temp.items()}

# - helper: build cohort prefix dictionary -----------------
# maps every phoneme prefix tuple to list of (word, freq) pairs
def build_prefix_dict(words, phones, freqs):
    prefix_dict = defaultdict(list)

    for w, ph, f in zip(words, phones, freqs):
        w = w.lower()
        for end in range(1, len(ph) + 1):
            prefix = tuple(ph[:end])
            prefix_dict[prefix].append((w, f))

    return prefix_dict

# - helper: compute cohort surprisal and entropy -----------
# returns lists of surprisal and entropy at each phoneme
# * surprisal: -log2 P(phoneme_i | cohort_{i - 1})
# * entropy: H over cohort at position i
def compute_cohort_features(word_phones, prefix_dict):
    n = len(word_phones)
    surprisal, entropy = [0.0] * n, [0.0] * n

    for i in range(1, n): # both entropy and surprisal = 0 at first phoneme, as it
                          # initializes the cohort, not updates it in the same way
        cohort_prev = prefix_dict.get(tuple(word_phones[:i]), [])
        cohort_curr = prefix_dict.get(tuple(word_phones[:i+1]), [])

        freq_prev = sum(f for _, f in cohort_prev)
        freq_curr = sum(f for _, f in cohort_curr)

        # phoneme surprisal
        surprisal[i] = np.log2(freq_prev / freq_curr) if freq_prev > 0 and freq_curr > 0 else 0.0
    
        # cohort entropy
        if freq_curr > 0:
            probs = [f / freq_curr for _, f in cohort_curr]
            entropy[i] = -sum(p * np.log2(p) for p in probs if p > 0)
        # else 0.0 anyway since it was initialized with zeros

    return surprisal, entropy

# - helper: look up word frequency -------------------------
# handles possessives (word + s') and French elisions
def lookup_freq(word, pron_freq_dict, is_eng, observed_phones=None):
    if "œ" in word:
        return lookup_freq(word.replace("œ", "oe"), pron_freq_dict, is_eng, observed_phones)

    if word != "'s" and word.endswith("'s"): # guard
        return (lookup_freq(word[:-2], pron_freq_dict, is_eng) + lookup_freq("'s", pron_freq_dict, is_eng))

    for elision in FRENCH_ELISIONS:
        if word.startswith(elision) and len(word) > len(elision): # guard added
            return (lookup_freq(elision, pron_freq_dict, is_eng) + lookup_freq(word[len(elision):], pron_freq_dict, is_eng))

    entries = pron_freq_dict.get(word, [])
    if not entries:
        return 1 if is_eng else 1_000_000 / 316_000_000 # Laplace smoothing: unseen words get freq of 1 (bc they appear in the stimuli)

    if observed_phones is not None:
        target = tuple(observed_phones)
        for ph, f in entries:
            if ph == target:
                return f
        return max(f for _, f in entries)

    return sum(f for _, f in entries)

# - helper: build output vector for one utterance ----------
# returns time-aligned valued impulse vector:
#   col 0: positional segment frequency     (at phoneme onsets)
#   col 1: positional biphone frequency     (at onsets of the first phoneme of the biphone)
#   col 2: log10 word frequency             (at word onsets)
#   col 3: phoneme cohort-based surprisal   (at phoneme onsets)
#   col 4: phoneme cohort-based entropy     (at phoneme onsets)
def build_vector(tg_path, seg_prob, bi_prob, pron_freq_dict, prefix_dict, lang_mapping):
    duration, phones, phone_onsets = read_textgrid_phones(tg_path)
    words, word_onsets, word_offsets = read_textgrid_words(tg_path)

    nSamples = int(np.ceil(duration * FS))
    vec = np.zeros((nSamples, FEAT_DIM))

    is_eng = lang_mapping is MFA_TO_CELEX

    for w_idx, (w_start, w_end) in enumerate(zip(word_onsets, word_offsets)):
        # get phones and their sample onsets for this word
        word_indices = [i for i, o in enumerate(phone_onsets) if w_start <= o < w_end]
        word_phones_raw = [phones[i] for i in word_indices]
        word_samples_raw = [int(round(phone_onsets[i] * FS)) for i in word_indices]

        # normalise
        word_phones, word_samples = normalise_phones_with_onsets(
            word_phones_raw, word_samples_raw, lang_mapping
        )

        # col 0: positional segment frequencies
        for i, (p, sample) in enumerate(zip(word_phones, word_samples)):
            prob = seg_prob.get((p, i), 0)
            vec[sample, 0] = prob # I take log freq up there

        #col 1: positional biphone frequencies
        for i in range(1, len(word_phones)):
            bi = (word_phones[i - 1], word_phones[i])
            prob = bi_prob.get((bi, i - 1), 0)
            sample = word_samples[i - 1]
            vec[sample, 1] = prob # I take log freq up there
        
        # col 2: log10 word frequencies
        sample_word_onset = int(round(w_start * FS))
        word_text = words[w_idx].lower()
        freq = lookup_freq(word_text, pron_freq_dict, is_eng, observed_phones = word_phones)
        vec[sample_word_onset, 2] = np.log10(freq) if freq else 0

        # col 3 & 4: cohort surprisal & entropy
        surprisal, entropy = compute_cohort_features(word_phones, prefix_dict)
        for i, sample in enumerate(word_samples):
            vec[sample, 3] = surprisal[i]
            vec[sample, 4] = entropy[i]

    return vec

# - MAIN LOOP -------------------------------------------------

print("Loading corpora...")

wE, fE, pE = load_celex()
wF, fF, pF = load_lexique()

print("Building frequency and cohort model structures")

pron_freq_dict_E = build_pron_freq_dict(wE, pE, fE)
pron_freq_dict_F = build_pron_freq_dict(wF, pF, fF)

prefix_dict_E = build_prefix_dict(wE, pE, fE)
prefix_dict_F = build_prefix_dict(wF, pF, fF)

print("Building phonotactic probability tables")

segE, biE = build_tables(fE, pE)
segF, biF = build_tables(fF, pF)

print("Processing stimuli")

files = []
vectors = []

for tg_dir, seg, bi, pron_freq_dict, prefix_dict, lang_mapping in [ # use current language corpus
    (ENG_TG_DIR, segE, biE, pron_freq_dict_E, prefix_dict_E, MFA_TO_CELEX),
    (FRA_TG_DIR, segF, biF, pron_freq_dict_F, prefix_dict_F, MFA_TO_LEXIQUE)
]:
    for fname in tqdm(sorted(os.listdir(tg_dir))):
        tg_path = os.path.join(tg_dir, fname)
        vec = build_vector(tg_path, seg, bi, pron_freq_dict, prefix_dict, lang_mapping)

        if vec.shape[0] < 240:
            vec = np.vstack([vec, np.zeros((240 - vec.shape[0], FEAT_DIM))])
        vec = np.vstack([np.zeros((10, FEAT_DIM)), vec])

        vectors.append(vec)
        files.append(os.path.splitext(fname)[0])

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
        vec = np.zeros((nSamples, FEAT_DIM))

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
    "phonotactic": vectors,
    "phonotactic_code": file_codes
})
print(f"Saved {len(vectors)} vectors to {OUTPUT}")