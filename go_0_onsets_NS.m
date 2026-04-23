%% Author: Nada
% Extract phoneme, morpheme, and word onsets from TextGrids.

clc; clear; close all;

%% - SETUP --------------------------------------------------
FS              = 100;
N_UTTERANCES    = 120;
FEAT_DIM        = 3;
MuR_STORIES     = {'UEngA', 'UEngB', 'UFraA', 'UFraB'};

%% - PATHS --------------------------------------------------
TG_DIR  = '/Users/nadastojanovic/Development/mphil/0_morphology';
WAV_DIR = '/Users/nadastojanovic/Development/mphil/0_speech_envelope/stimuli/';
OUTPUT  = fullfile(TG_DIR, 'NS_onsets.mat');

textGrids = uipickfiles( ...
    'FilterSpec', fullfile(TG_DIR, '*.TextGrid'), ...
    'Prompt', 'Select all TextGrid files (both Eng and Fra)');

N_FILES = length(textGrids);

%% - MAIN LOOP -----------------------------------------------

onsets = {};
onsets_code = {};

for i = 1:N_FILES
    fid = fopen(textGrids{i}, 'r');
    
    phoneme_onsets_sec = [];
    morpheme_onsets_sec = [];
    word_onsets_sec = [];
    
    current_xmin = 0;
    total_duration = 0;
    
    inPhoneTier = false;
    inMorphemeTier = false;
    inWordTier = false;
    
    while ~feof(fid)
        line = strtrim(fgetl(fid));
        
        %%% get utterance duration
        if startsWith(line, 'xmax =') && total_duration == 0
            total_duration = str2double(extractAfter(line, '='));
        end
        
        %%% detect tier switches
        if contains(line, 'name = "phones"')    % entering phones tier
            inPhoneTier = true;
            inMorphemeTier = false;
            inWordTier = false;
            continue
        end
        
        if contains(line, 'name = "morphemes"') % entering morpheme tier
            inMorphemeTier = true;
            inPhoneTier = false;
            inWordTier = false;
            continue
        end
        
        if contains(line, 'name = "words"')     % entering words tier
            inWordTier = true;
            inPhoneTier = false;
            inMorphemeTier = false;
            continue
        end
        
        if contains(line,'name =') && ...       % none
           ~contains(line,'"phones"') && ...
           ~contains(line,'"words"') && ...
           ~contains(line,'"morphemes"')
            
            inPhoneTier = false;
            inWordTier = false;
            inMorphemeTier = false;
        end
        
        %%% onset
        if startsWith(line, 'xmin =')
            current_xmin = str2double(extractAfter(line, '='));
        end
        
        %%% label (check non-empty for phonemes and words only)
        if startsWith(line, 'text =')
            label = extractBetween(line, '"', '"');
            label = label{1};
            
            if ~isempty(label)
                if inPhoneTier
                    phoneme_onsets_sec(end+1) = current_xmin;
                end
                
                if inWordTier
                    word_onsets_sec(end+1) = current_xmin;
                end
            end

            if inMorphemeTier
                if current_xmin ~= 0
                    morpheme_onsets_sec(end+1) = current_xmin;
                end
            end
            
            current_xmin = 0;
        end
        
    end
    
    fclose(fid);
        
    %%% build impulse vectors
    nSamples = ceil(total_duration * FS);
    phoneme_vector = zeros(nSamples,1);
    morpheme_vector = zeros(nSamples,1);
    word_vector = zeros(nSamples,1);
    
    for p = 1:length(phoneme_onsets_sec)
        onset_sample = round(phoneme_onsets_sec(p) * FS);
        phoneme_vector(onset_sample+1) = 1;
    end
    
    for m = 1:length(morpheme_onsets_sec)
        onset_sample = round(morpheme_onsets_sec(m) * FS);
        morpheme_vector(onset_sample+1) = 1;
    end
    
    for w = 1:length(word_onsets_sec)
        onset_sample = round(word_onsets_sec(w) * FS);
        word_vector(onset_sample+1) = 1;
    end
    
    %%% build output vector
    %%%     col 0: phoneme onsets
    %%%     col 1: inflectional morphology boundaries
    %%%     col 2: word onsets
    
    vec = [phoneme_vector morpheme_vector word_vector];

    if size(vec, 1) < 240
        vec = [vec; zeros(240 - size(vec, 1), FEAT_DIM)];
    end
    vec = [zeros(10, FEAT_DIM); vec];
    
    [~, name, ~] = fileparts(textGrids{i});
    
    onsets{end + 1} = vec;
    onsets_code{end + 1, 1} = name;
    
end

% - ZERO VECTORS FOR MuR STIMULI ------------------------
% UEngA, UEngB, UFraA, UFraB have no linguistic features
% so putting zeros for all features exc. speech env
 
for s = 1:length(MuR_STORIES)
    for utt = 1:N_UTTERANCES
        utt_code = sprintf('%s%03d', MuR_STORIES{s}, utt);
        wav_file = fullfile(WAV_DIR, [utt_code '.wav']);
 
        info = audioinfo(wav_file);
        
        nSamples = ceil(info.Duration * FS);
        vec = zeros(nSamples, FEAT_DIM);
 
        % same padding as attended entries
        if size(vec, 1) < 240
            vec = [vec; zeros(240 - size(vec, 1), FEAT_DIM)];
        end
        vec = [zeros(10, FEAT_DIM); vec];
 
        onsets{end + 1} = vec;
        onsets_code{end + 1, 1} = utt_code;
 
    end
end
 

%% - SAVE OUTPUT MAT FILE -----------------------------------
fprintf('Saving .mat\n')
save(OUTPUT, 'onsets', 'onsets_code');
fprintf('Saved %d vectors to %s\n', length(onsets), OUTPUT);