"""
Author: Nađa
Extract phonetic articulatory features from PanPhon.
"""

# - IMPORTS ------------------------------------------------
import os
import wave
import numpy as np
from pathlib import Path
from scipy.io import savemat
from praatio import textgrid
from panphon import FeatureTable

# - SETUP --------------------------------------------------
# pip install panphon

FS              = 100
FEAT_DIM        = 22  # number of articulatory features returned by Panphon (24) - 2 (last two all zero)
N_UTTERANCES    = 120
MuR_STORIES     = ["UEngA", "UEngB", "UFraA", "UFraB"]

# some phones in MFA phoneset do not match the spelling in PanPhon IPA exactly,
# check here https://github.com/dmort27/panphon/blob/master/panphon/data/ipa-xsampa.csv
# and add any mappings if necessary, the script will flag if there is an unknown symbol
ALTERNATE_SPELLINGS = {
    # AFFRICATES
    "dʒ": "d͡ʒ",
    "tʃ": "t͡ʃ",

    # DIPHTHONGS, 
    "aj": ["a", "j"], # panphon currently does not handle diphthongs,
    "əw": ["ə", "w"], # see https://github.com/dmort27/panphon/issues/63,
    "ej": ["e", "j"], # so naively handling them as two separate vowels 
    "aw": ["a", "w"], # with the offset of the first being smack in the 
    "ɔj": ["ɔ", "j"]  # middle of the diphthong's duration 
}

FT = FeatureTable()

# - PATHS --------------------------------------------------
BASE_DIR = Path.home() / "Development/mphil/"
WAV_DIR = BASE_DIR / "0_speech_envelope/stimuli/"
TEXTGRID_DIRS = [
    BASE_DIR / "0_morphology/aligned_Eng/",
    BASE_DIR / "0_morphology/aligned_Fra/"
]
OUTPUT = BASE_DIR / "0_articulatory_features/" / "NS_articulatory_features.mat"

# - HELPER FUNCTIONS ---------------------------------------

# - helper: read phones tier from a TextGrid ---------------
# returns utterance duration, list of (start, end) time tuples,
# and matrix of articulatory features
def read_textgrid_phones(path):
    tg = textgrid.openTextgrid(path, includeEmptyIntervals=False)
    phone_tier = tg.getTier('phones')
    duration = tg.maxTimestamp # this exists in praatio

    times, vecs = [], []

    for start, end, label in phone_tier.entries:
        label = FT.normalize(label) # make sure to use consistent unicode
        
        if label not in FT.seg_dict: # handle unknown symbols
            if label in ALTERNATE_SPELLINGS:
                label = ALTERNATE_SPELLINGS[label] # remap

                if isinstance(label, list): # list -> it's a diphthong
                    mid = (start + end) / 2

                    for ph, (t_start, t_end) in zip(label, [(start, mid), (mid, end)]):
                        times.append([t_start, t_end])      
                        vecs.append(FT.segment_to_vector(ph)[:-2]  )
                    
                    continue;
                # else: string -> it's an affricate, but no need
                # to do anything, because we remapped to the new
                # label above the if statement, and below we add that
            else:
                raise ValueError(f"Unknown IPA segment: '{label}' in {path}")
        
        times.append([start, end])
        vecs.append(FT.segment_to_vector(label)[:-2])

    # convert -/0/+ (PanPhon symbols) to numeric -1/0/1
    vec_mat = [
        [1 if v == '+' else -1 if v == '-' else 0 for v in vec]
        for vec in vecs
    ]

    return duration, times, vec_mat

# - helper: build output vector for one utterance ------------
# returns time-aligned (to phoneme onsets) valued impulse vector:
#   col 0-21: phoneme articulatory features (22 cols)
def build_vector(duration, times, numeric_matrix):
    nSamples = int(np.ceil(duration * FS))
    mat = np.zeros((nSamples, FEAT_DIM))

    for (start, end), vec in zip(times, numeric_matrix):
        start_samp = int(start * FS)
        end_samp = min(int(end * FS), nSamples)

        mat[start_samp:end_samp] = vec # step function NOT a impulse
    return mat

# - MAIN LOOP -------------------------------------------------

vectors, files = [], []

tg_files = []
for DIR in TEXTGRID_DIRS:
    tg_files.extend([DIR / f for f in os.listdir(DIR) if f.endswith(".TextGrid")])
tg_files = sorted(tg_files)

print(f"Found {len(tg_files)} TextGrids in the specified directories.")

for tg_path in tg_files:
    duration, times, numeric_matrix = read_textgrid_phones(tg_path)

    vec = build_vector(duration, times, numeric_matrix)

    # pad to minimum 240 samples + 10-sample zero-pad at onset
    if vec.shape[0] < 240:
        vec = np.vstack([vec, np.zeros((240 - vec.shape[0], FEAT_DIM))])
    vec = np.vstack([np.zeros((10, FEAT_DIM)), vec])

    vectors.append(vec)
    files.append(tg_path.stem)

# - ZERO VECTORS FOR MuR STIMULI ------------------------
# UEngA, UEngB, UFraA, UFraB have no linguistic features
# so putting zeros for all features exc. speech env
for story in MuR_STORIES:
    for utt in range(1, N_UTTERANCES + 1):
        utt_code = f"{story}{utt:03d}"
        wav_path = WAV_DIR / f"{utt_code}.wav"

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
    "attended_articulatory": vectors,
    "attended_articulatory_code": file_codes
})
print(f"Saved {len(vectors)} vectors to {OUTPUT}")