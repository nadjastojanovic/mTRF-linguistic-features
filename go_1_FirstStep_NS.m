%% Author: Nađa
% Prepare EEG + speech features for mTRF modelling

clc; close all; clear all;

%% SETUP

Epoch_datapoints = 250; % -100ms to 2400ms

attn = 2; % CHANGE: attended:1, unattended:2

if attn == 1
    attn_label = 'Attended';
    attn_first = 'A';
elseif attn == 2
    attn_label = 'Unattended';
    attn_first = 'U';
end

%% PATHS

%%% EEG data
path_data  = ['/Users/nadastojanovic/Development/mphil/1_mTRF/0a_EEGdata_' attn_label '_v1to12/'];

%%% stim feature mats
path_env    = '/Users/nadastojanovic/Development/mphil/0_speech_envelope/';
path_onset  = '/Users/nadastojanovic/Development/mphil/0_morphology/';
path_artic  = '/Users/nadastojanovic/Development/mphil/0_articulatory_features/';
path_phono  = '/Users/nadastojanovic/Development/mphil/0_phonotactic_prob/';
path_sem    = '/Users/nadastojanovic/Development/mphil/0_word_surprisal_entropy/';
path_synt   = '/Users/nadastojanovic/Development/mphil/0_syntax/';

%%% output (mTRFready) matfiles
path_mat   = ['/Users/nadastojanovic/Development/mphil/1_mTRF/1b_mTRFready_' attn_label '_matfiles/'];

%% STIM FEATURES

% speech envelope
load([path_env 'NS_env.mat'])
% env
% env_code (1x1920 cells, 1 column)

% onset features:
%   col 1: phoneme onsets
%   col 2: inflectional morphology boundaries
%   col 3: word onsets
load([path_onset 'NS_onsets.mat'])
% onsets_code
% onsets (1x1920 cells, 3 columns)

% phoneme articulatory features
%   col 1 - 22: see log for articulatory features list
load([path_artic 'NS_articulatory_features.mat'])
% articulatory_code (1920x1 cells, 1 column)
% articulatory (1x1920 cells, 22 columns)

% cohort-based features:
%   col 1: phonotactic probability (positional segment frequency)
%   col 2: phonotactic probability (biphoneme frequency)
%   col 3: word frequency
%   col 4: phoneme (cohort-based) surprisal
%   col 5: phoneme (cohort-based) entropy
load([path_phono 'NS_phonotactic_probs.mat'])
% phonotactic_code
% phonotactic (1x1440 cells, 5 columns)

% semantic features:
%   col 1: word surprisal
%   col 2: word entropy
load([path_sem 'NS_semantic.mat'])
% semantic_code
% semantic (1x1920 cells, 2 columns)

% syntactic features:
%   col 1: syntactic depth
%   col 2: open dependencies (count)
%   col 3: remaining open dependencies (count)
%   col 4: closed dependencies (count)
load([path_synt 'NS_syntactic.mat'])
% syntactic_code
% syntactic (1x1920 cells, 4 columns)

%% EEG

ssList = uipickfiles('FilterSpec', [path_data, '*.set'], ...
    'Prompt', 'Select the preprocessed .set files');

nID = size(ssList,2);

%% EXP CONDITIONS

conditions = {'EngA|EngB','EngC|EngD','FraA|FraB','FraC|FraD'};
% correspond to    EngMuR     EngSame     FraMuR     FraSame

%% MAIN LOOP

for cond_idx = 1:numel(conditions)

    condition_pattern = conditions{cond_idx}; % e.g. 'FraC|FraD'

    for part_i = 1:nID

        [filepath, name, ext] = fileparts(ssList{part_i});
        EEG = pop_loadset([name ext], filepath);

        participant_id = name(9:11); % e.g. '101'

        %%% normalize EEG
        meanData = mean(EEG.data, 1);                   % Calculate mean along the first dimension (rows)
        stdData = std(EEG.data, 0, 1);                  % Calculate standard deviation along the first dimension
        EEG.data = (EEG.data - meanData) ./ stdData;    % Apply z-score normalization

        %%% identify trials matching current condition pattern
        code_array = {};
        trial_idx  = [];

        for cell_array_loop = 1:size(EEG.data,3)

            q  = EEG.event;
            q2 = find([q.epoch] == cell_array_loop);    % trial index e.g. 1
            tempy = {q(q2).type};                       % trial code e.g. {'UEngB001'}

            % all the matching indices
            idx = CheckList(tempy, [attn_first,'(' condition_pattern ')','[0-1][0-9][0-9]$']);

            if ~isempty(idx)
                % exclude first utterance from both blocks for each story
                if ~endsWith(tempy{1,idx}, '001') && ~endsWith(tempy{1,idx}, '061')
                    code_array{end+1,1} = tempy{1, idx}; % trial code e.g. 'UEngB001'
                    trial_idx(end+1)    = cell_array_loop; % epoch number e.g. 870

                end
            end
        end

        wav_match_env   = {};
        wav_match_onsets= {};
        wav_match_artic = {};
        wav_match_phono = {};
        wav_match_sem   = {};
        wav_match_synt  = {};
        data_match      = {};

        %%% pull out corresponding stim data 

        for trial_i = 1:length(trial_idx)

            trial_name = code_array{trial_i}; % 'UEngB001'

            %%% stim matching indices
            env_idx     = find(strcmp(env_code,             trial_name));
            onset_idx   = find(strcmp(onsets_code,          trial_name));
            artic_idx   = find(strcmp(articulatory_code,    trial_name));
            phono_idx   = find(strcmp(phonotactic_code,     trial_name));
            sem_idx     = find(strcmp(semantic_code,        trial_name));
            synt_idx    = find(strcmp(syntactic_code,       trial_name));

            %%% store corresponding stim data

            wav_match_env{trial_i} = ...
                env{env_idx}(1:Epoch_datapoints, 1);

            wav_match_onsets{trial_i} = ...
                onsets{onset_idx}(1:Epoch_datapoints, :);

            wav_match_artic{trial_i} = ...
                articulatory{artic_idx}(1:Epoch_datapoints, :);

            raw_phono = phonotactic{phono_idx}(1:Epoch_datapoints, :);
            wav_match_phono{trial_i} = raw_phono(:, [3, 1, 2, 4, 5]); % cohort-based features reordered:

            wav_match_sem{trial_i} = ...
                semantic{sem_idx}(1:Epoch_datapoints, :);

            wav_match_synt{trial_i} = ...
                syntactic{synt_idx}(1:Epoch_datapoints, :);

            %%% store corresponding EEG data
            data_match{trial_i} = ...
                EEG.data(:, 1:Epoch_datapoints, trial_idx(trial_i))';

        end

        %% STIM FEATURE MAT

        %%% convert to column cells
        wav_match_env   = wav_match_env';
        wav_match_onsets= wav_match_onsets';
        wav_match_artic = wav_match_artic';
        wav_match_phono = wav_match_phono';
        wav_match_sem   = wav_match_sem';
        wav_match_synt  = wav_match_synt';
        data_match      = data_match';

        %%% add 10 x zero padding at the end of each utterance
        zeropadding = 10;

        %%% feature vectors
        [env_glue, eegdata] = gluedata(wav_match_env,   data_match, zeropadding);
        [onset_glue, ~]     = gluedata(wav_match_onsets,data_match, zeropadding);
        [artic_glue, ~]     = gluedata(wav_match_artic, data_match, zeropadding);
        [phono_glue, ~]     = gluedata(wav_match_phono, data_match, zeropadding);
        [sem_glue, ~]       = gluedata(wav_match_sem,   data_match, zeropadding);
        [synt_glue,  ~]     = gluedata(wav_match_synt,  data_match, zeropadding);

        % final column order:
        %   1:      speech envelope
        %   2:      phoneme onsets
        %   3:      inflectional morphology
        %   4:      word Onsets
        %   5 - 26: articulatory features (22 cols)
        %   27:     word Frequency
        %   28:     phonotactic probability (positional segment frequency)
        %   29:     phonotactic probability (biphoneme frequency)
        %   30:     phoneme cohort-based surprisal
        %   31:     phoneme cohort-based entropy
        %   32:     word surprisal
        %   33:     word entropy
        %   34:     syntactic depth
        %   35:     open dependencies
        %   36:     remaining open dependencies
        %   37:     closed dependencies

        stimulusdata = [...
            env_glue    ...
            onset_glue  ...
            artic_glue  ...
            phono_glue  ...
            sem_glue    ...
            synt_glue   ...
        ];

        %%% z-score stim features (recommended by Crosse et al. papers)
        stimulusdata = zscore(stimulusdata);

        %%% remove last column (REF)
        eegdata(:, 93) = [];

        %% ZERO PADDING
        
        extreme_padding_eeg      = zeros(50, size(eegdata,      2));
        extreme_padding_stimulus = zeros(50, size(stimulusdata, 2));

        eegdata      = [extreme_padding_eeg;      eegdata;      extreme_padding_eeg];
        stimulusdata = [extreme_padding_stimulus; stimulusdata; extreme_padding_stimulus];

        %% SAVE MATFILES

        rand_index = 1:length(wav_match_env);

        participant_filename = ...
            ['/NS_mTRFready_' participant_id '_' attn_first '_' condition_pattern '.mat'];

        save(fullfile(path_mat, participant_filename), ...
            'eegdata', 'stimulusdata', 'code_array', 'rand_index')

    end
end