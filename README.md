<div align="center">
  
# Modelling neural tracking of linguistic <br />features in speech via forward mTRF
 
![MATLAB](https://img.shields.io/badge/MATLAB-E36410?style=for-the-badge&logo=mathworks&logoColor=white)
![mTRF Toolbox](https://img.shields.io/badge/mTRF--Toolbox-E36410?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
 
</div>

> Forward mTRF modelling of EEG recorded during a dichotic listening task,
> using speech representations spanning acoustic, phonological, lexical,
> and syntactic levels of the speech processing hierarchy.
 
Supervised by Dr. Mirjana Bozic, Brain, Language & Bilingualism Lab, University of Cambridge.
 
## Repository structure
 
```
stim_features/
  go_0_speechEnv_NS.m               # speech envelope
  go_0_onsets_NS.m                  # phoneme, word, and inflectional morphology onsets from .TextGrids
  go_0_phoneArticulatoryFea_NS.py   # articulatory features extraction (PanPhon)
  go_0_phonotacticProb_NS.py        # phoneme frequency, surprisal, entropy (CELEX2, Lexique4)
  go_0_syntactic.py                 # word surprisal, entropy (BLOOM LLM)
  go_0_wordSurprisalEntropy.py      # syntactic dependency counts (spaCy)

mTRF/
  go_1_FirstStep_NS.m               # EEG data + stim data -> mTRF-ready .mat files
  go_2_mTRFforward_NS.m             # forward mTRF (cv, circular shift permutation testing)
```
