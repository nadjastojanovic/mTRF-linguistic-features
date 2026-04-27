%% Author: Nađa
% Forward mTRF modelling of EEG using linguistic features.

% Output mat file (per participant per condition):
%   TRF.participant_id              string
%   TRF.condition                   string                    e.g. 'EngMuR'
%   TRF.UsageGroup                  string                    'High' | 'Low' | 'No'
%   TRF.timeLags                    time lags ms              [1 x N_LAGS]
%   TRF.grouping                    band index per feature    [1 x N_FEAT]
%
%   TRF.full.weights                mean TRF weights          [N_FEAT x N_LAGS x N_CHANS]
%   TRF.full.lambdas                optimal lambda per fold   [N_FOLD x 1]
%   TRF.full.r_full                 full model r per fold     [N_FOLD x N_CHANS]
%   TRF.full.r_full_avg             mean r across folds/chans scalar
%   TRF.full.r_null                 null r (all cols shifted) [N_PERM x N_CHANS x N_FOLD]
%   TRF.full.r_null_avg             mean null r               scalar
%
%   TRF.(model).feat_cols           permuted column indices   [1 x N_COLS]
%   TRF.(model).r_null              null r per perm/fold      [N_PERM x N_CHANS x N_FOLD]
%   TRF.(model).r_null_avg          mean null r               scalar
%   TRF.(model).r_corr              r_full - mean(r_null)     [N_FOLD x N_CHANS]
%   TRF.(model).r_corr_avg          mean corrected r          scalar

clc; clear; close all;

%% - SETUP ----------------------------------------------------------------

attn = 1;   % CHANGE: attended:1, unattended:2

if attn == 1
    attn_label = 'Attended';
elseif attn == 2
    attn_label = 'Unattended';
end

MODEL_DIR   = 1;            % forward
N_FOLD      = 10;
TMIN        = -100;
TMAX        = 600;
FS          = 100;
N_CHANS     = 92;           % after REF removal
N_PERM      = 10;
MuR_STORIES = {'EngA|EngB', 'FraA|FraB'};

% min circular shift > longest possible time lag
MIN_SHIFT   = ceil(TMAX / 1000 * FS) + 1; % 61

lags_samp   = floor(TMIN / 1000 * FS) : ceil(TMAX / 1000 * FS);
LAGS        = lags_samp / FS * 1000; % -100:10:600 -> 71 lags
N_LAGS      = numel(lags_samp);

% participant usage groups
P_GROUPS = struct( ...
    'High', [101, 103, 106, 115, 117, 118, 123, 125, 128, 130, 131, 134, 138, 139, 140, 145], ...
    'Low', [102, 104, 105, 107, 108, 109, 114, 119, 121, 122, 129, 132, 133, 136, 137, 143], ...
    'No', [111, 112, 113, 124, 126, 135, 141, 142, 146, 147, 148, 149, 150, 151, 152, 153] ...
);

P_GROUPS_NAMES = fieldnames(P_GROUPS);

% condition pattern groups
COND_MAP = containers.Map( ...
    {'EngA|EngB','EngC|EngD','FraA|FraB','FraC|FraD'}, ...
    {'EngMuR',   'EngSame',  'FraMuR',   'FraSame'} );

%% - FEATURE BANDS --------------------------------------------------------

GROUPING = zeros(1, 37);

% tunes one band at a time & freezes earlier bands, so shared
% variance is attributed to earlier-band features - order matters!!
GROUPING([1, 5:26]) = 1;  % acoustic: env, artic
GROUPING([2, 4, 27]) = 2; % seg/freq: phon onsets, word onsets, word freq
GROUPING([3, 28:37]) = 3; % lexical / syntactic

N_BANDS = 3;

LAMBDAS = 10.^(-7:11);

%% - MODELS ---------------------------------------------------------------

% final column order:
%   1:      speech envelope
%   2:      phoneme onsets
%   3:      inflectional morphology
%   4:      word onsets
%   5–26:   articulatory features (22 cols)
%   27:     word frequency
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

%%% composite models
MODELS_COMPOSITE = {...
    'phonotactic',  ... % pos seg freq + bi freq
    'cohort_based', ... % phon surp + phon ent
    'phoneme_level',... % phonotactic + cohort
    'semantic',     ... % word surp + word ent
    'syntactic',    ... % syntactic depth + dep counts
    'rule_based',   ... % inflectional morph + syntactic
    'word_level',   ... % semantic + syntactic
    'linguistic' }; ... % everything exc. (env + artic + phon onsets + word onsets + word freq)
           
COLS_COMPOSITE  = { ...
    28:29,          ... % phonotactic
    30:31,          ... % cohort_based
    28:31,          ... % phoneme_level
    32:33,          ... % semantic
    34:37,          ... % syntactic
    [3, 34:37],     ... % rule_based
    32:37,          ... % word_level
    [3, 28:37] };   ... % linguistic
            
%%% individual feature models
MODELS_INDIV = { ...
    'env',           ...  % speech envelope
    'phon_onsets',   ...  % phoneme onsets
    'morph',         ...  % inflectional morphology
    'word_onsets',   ...  % word onsets
    'artic',         ...  % articulatory features (one group)
    'word_freq',     ...  % word frequency
    'pos_seg_freq',  ...  % phonotactic prob (positional segment frequency)
    'bi_freq',       ...  % phonotactic prob (biphoneme frequency)
    'phon_surp',     ...  % phoneme surprisal
    'phon_ent',      ...  % phoneme entropy
    'word_surp',     ...  % word surprisal
    'word_ent',      ...  % word entropy
    'synt_depth',    ...  % syntactic depth
    'synt_deps'};         % dependency counts (one group)

COLS_INDIV = {1, 2, 3, 4, 5:26, 27, 28, 29, 30, 31, 32, 33, 34, 35:37};

%%% models used for ~(unattended && MuR)
MODELS  = [MODELS_COMPOSITE, MODELS_INDIV]; % model names
COLS    = [COLS_COMPOSITE, COLS_INDIV];     % corresponding feature columns

%% - PATHS ----------------------------------------------------------------

MAT_DIR = ['/Users/nadastojanovic/Development/mphil/1_mTRF/1b_mTRFready_' attn_label '_matfiles/'];
OUTPUT  = '/Users/nadastojanovic/Development/mphil/1_mTRF/3_results/';

ssList = dir(fullfile(MAT_DIR, '*.mat'));
ssList = ssList(1:2);   % uncomment when testing on a single file
nID = numel(ssList);

fprintf('Found %d mat files for %s.\n\n', nID, attn_label);

%% - MAIN LOOP ------------------------------------------------------------

parfor mat_i = 1:nID % requires Parallel Computing Toolbox

    filename = ssList(mat_i).name;

    N_MODELS = numel(MODELS);

    %%% load mTRFready mat file
    tmp = load(fullfile(MAT_DIR, filename), 'eegdata', 'stimulusdata');
    eegdata = tmp.eegdata;
    stimulusdata = tmp.stimulusdata;
    T = size(eegdata, 1);
    N_FEAT = size(stimulusdata, 2); % 37

    %%% parse filename i.e. NS_mTRFready_101_A_EngA|EngB.mat
    [~, basename, ~] = fileparts(filename);
    tokens = strsplit(basename, '_');
    participant_id = tokens{3};         % '101'
    Part = str2double(participant_id);  %  101

    condition_str = tokens{5};          % 'EngA|EngB'
    switch condition_str
        case 'EngA|EngB';  Cond = 'EngMuR';
        case 'EngC|EngD';  Cond = 'EngSame';
        case 'FraA|FraB';  Cond = 'FraMuR';
        case 'FraC|FraD';  Cond = 'FraSame';
    end

    UsageGroup = P_GROUPS_NAMES{ ...    % 'High'
        find(cellfun(@(g) ismember(Part, P_GROUPS.(g)), P_GROUPS_NAMES)) };

    fprintf('Processing: Participant %d | Condition: %s\n', Part, Cond);

    %%% initialise results struct
    TRF = struct();
    TRF.participant_ID = participant_id;
    TRF.condition = Cond;
    TRF.usage_group = UsageGroup;
    TRF.time_lags = LAGS;
    TRF.feature_bands = GROUPING;

    %%% pre-allocate
    r_full = zeros(N_FOLD, N_CHANS);
    w_sum = zeros(N_FEAT, N_LAGS, N_CHANS);
    fold_lambdas = zeros(N_FOLD, 1);
    r_null_all = zeros(N_MODELS, N_PERM, N_CHANS, N_FOLD);
    r_null_full = zeros(N_PERM, N_CHANS, N_FOLD);

    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');

    for fold_i = 1:N_FOLD
        testtrial = fold_i;
        fprintf('Fold: %d  / %d\n', fold_i, N_FOLD);

        [strain, rtrain, stest, rtest] = mTRFpartition(stimulusdata, eegdata, N_FOLD, testtrial);

        %%% progressive banded lambda CV on training folds only
        % tunes one band at a time & freezes earlier bands, so shared
        % variance is attributed to earlier-band features - order matters!!

        %% full model

        % mTRFcvbandedprogressive
        cv = mTRFcrossval(strain, rtrain, FS, ...
            MODEL_DIR, TMIN, TMAX, LAMBDAS, 'verbose', 0);

        [~, lambda_idx] = max(mean(mean(cv.r(:,:,:),3)));
        lambda = LAMBDAS(lambda_idx);
        fold_lambdas(fold_i) = lambda;

        %%% train with banded lambdas

        % mTRFtrainbanded
        model = mTRFtrain(strain, rtrain, FS, MODEL_DIR, ...
            TMIN, TMAX, lambda, 'verbose', 0);

        %%% test
        [~, mtrftest] = mTRFpredict(stest, rtest, model, ...
            'zeropad', 0, 'verbose', 0);

        %%% store per-fold, full model results
        r_full(fold_i, :) = mtrftest.r;
        w_sum = w_sum + model.w;

        %% control (null) shuffled model
        % circular shift of all feature columns in test stimdata

        r_null_full_fold = zeros(N_PERM, N_CHANS);
        for p = 1:N_PERM
            rnd_shft = randi([MIN_SHIFT, T - MIN_SHIFT]);

            stest_perm = circshift(stest, rnd_shft, 1); % all cols
            [~, testperm] = mTRFpredict(stest_perm, rtest, model, ...
                'zeropad', 0, 'verbose', 0);

            r_null_full_fold(p, :) = testperm.r;
        end
        r_null_full(:, :, fold_i) = r_null_full_fold;

        %% composite + individual feature models
        % circular shift of current model's feature columns in test
        % stimdata, test using the same pre-trained model ^

        for g = 1:N_MODELS
            gcols = COLS{g};

            for p = 1:N_PERM
                rnd_shft = randi([MIN_SHIFT, T - MIN_SHIFT]);

                stest_perm = stest;
                stest_perm(:, gcols) = circshift(stest(:, gcols), rnd_shft, 1);
                [~, testperm] = mTRFpredict(stest_perm, rtest, model, ...
                    'zeropad', 0, 'verbose', 0);

                r_null_all(g, p, :, fold_i) = testperm.r;
            end

        end

    end

    %%% store full model results
    TRF.full.weights = w_sum / N_FOLD;              % [N_FEAT x N_LAGS x N_CHAN]
    TRF.full.lambdas = fold_lambdas;                % [N_FOLD x 1]

    TRF.full.r_null = r_null_full;                  % [N_PERM x N_CHANS x N_FOLD]
    TRF.full.r_full = r_full;                       % [N_FOLD x N_CHAN]
    
    TRF.full.r_null_avg = mean(r_null_full(:));     % scalar
    TRF.full.r_full_avg = mean(r_full(:));          % scalar
    
    %%% store per model results
    for g = 1:N_MODELS
        gname  = MODELS{g};
        TRF.(gname).feat_cols = COLS{g};                % e.g. [28, 29]

        r_null_g = squeeze(r_null_all(g, :, :, :));     % [N_PERM x N_CHAN x N_FOLD]
        r_corr = r_full - squeeze(mean(r_null_g, 1))';  % [N_FOLD x N_CHAN]

        TRF.(gname).r_null = r_null_g;                  % [N_PERM x N_CHAN x N_FOLD]
        TRF.(gname).r_corr = r_corr;                    % [N_FOLD x N_CHAN]

        TRF.(gname).r_null_avg = mean(r_null_g(:));     % scalar
        TRF.(gname).r_corr_avg = mean(r_corr(:));       % scalar
    end

%% - SAVE OUTPUT MAT FILE -------------------------------------------------

    save_path = fullfile(OUTPUT, ['banded_' attn_label '_' basename '.mat']);
    parsave_forward(save_path, TRF);
    fprintf('Saved: %s\n', basename);

end

fprintf('Forward banded mTRF modelling complete.\n');

%% - HELPERS --------------------------------------------------------------

function parsave_forward(fpath, TRF)
    save(fpath, 'TRF', '-v7.3');
end