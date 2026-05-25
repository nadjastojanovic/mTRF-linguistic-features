%% Author: Nađa
% Forward mTRF modelling of EEG using linguistic features.

% Output mat file (per participant per condition):
%   TRF.participant_ID              string
%   TRF.condition                   string                    e.g. 'EngMuR'
%   TRF.usage_group                 string                    'High' | 'Low' | 'No'
%   TRF.time_lags                   time lags ms              [1 x N_LAGS]
%
%   ── Full model (all 12 features jointly) ──────────────────────────────────
%   TRF.full.weights                mean TRF weights          [N_FEAT x N_LAGS x N_CHANS]
%   TRF.full.lambdas                optimal lambda per fold   [N_FOLD x 1]
%   TRF.full.r                      prediction r per fold     [N_FOLD x N_CHANS]
%   TRF.full.r_avg                  mean r (folds x chans)    scalar
%   TRF.full.r_null                 null r (all cols shifted) [N_PERM x N_CHANS x N_FOLD]
%   TRF.full.r_null_avg             mean null r               scalar
%
%   ── Per-model permutation results (full model + circular shift) ───────────
%   TRF.(model).feat_cols           permuted column indices   [1 x N_COLS]
%   TRF.(model).r_null              null r per perm/fold      [N_PERM x N_CHANS x N_FOLD]
%   TRF.(model).r_null_avg          mean null r               scalar
%   TRF.(model).r_corr              r_full - mean(r_null)     [N_FOLD x N_CHANS]
%   TRF.(model).r_corr_avg          mean corrected r          scalar
%
%   ── Individual feature models (solo model per feature) ────────────────────
%   TRF.(feature).weights           mean TRF weights          [N_FEAT x N_LAGS x N_CHANS]
%   TRF.(feature).r                 prediction r per fold     [N_FOLD x N_CHANS]
%   TRF.(feature).r_avg             mean r (folds x chans)    scalar
%   % note: individual features also have permutation fields from above
%   % (feat_cols, r_null, r_null_avg, r_corr, r_corr_avg) stored in the same sub-struct

clc; clear; close all;

%% - SETUP ----------------------------------------------------------------

attn = 2;   % CHANGE: attended:1, unattended:2

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
LAMBDAS     = 10.^(-6:6);

MuR_STORIES = {'EngA|EngB', 'FraA|FraB'};

% min circular shift > longest possible time lag
MIN_SHIFT = ceil(TMAX / 1000 * FS) + 1; % 61

lags_samp = floor(TMIN / 1000 * FS) : ceil(TMAX / 1000 * FS);
LAGS = lags_samp / FS * 1000; % -100:10:600 -> 71 lags
N_LAGS = numel(lags_samp);

% participant usage groups
P_GROUPS = struct( ...
    'High', [101, 103, 106, 115, 117, 118, 123, 125, 128, 130, 131, 134, 138, 139, 140, 145], ...
    'Low', [102, 104, 105, 107, 108, 109, 114, 119, 121, 122, 129, 132, 133, 136, 137, 143], ...
    'No', [111, 112, 113, 124, 126, 135, 141, 142, 146, 147, 148, 149, 150, 151, 152, 153] ...
);

P_GROUPS_NAMES = fieldnames(P_GROUPS);

%% - MODELS ---------------------------------------------------------------

% final column order (12 cols total):
%   1:  speech envelope
%   2:  phoneme onsets
%   3:  inflectional morphology
%   4:  word onsets
%   5:  articulatory complexity
%   6:  word frequency
%   7:  phoneme frequency
%   8:  phoneme surprisal
%   9:  phoneme entropy
%   10: word surprisal
%   11: word entropy
%   12: syntactic complexity

%%% composite models
MODELS_COMPOSITE = {...
    'phonological', ... % phon onsets + artic + phon freq + phon surp + phon ent
    'lexical',      ... % word onsets + word freq + word surp + word ent
    'syntactic',    ... % morph + synt complexity
    'linguistic',   ... % lexical + syntactic
    'full_no_env'}; ... % full model exc. speech envelope

COLS_COMPOSITE  = {              ...
    [2, 5, 7, 8, 9],             ... % phonological
    [4, 6, 10, 11],              ... % lexical
    [3, 12],                     ... % syntactic
    [3, 4, 6, 10, 11, 12],       ... % linguistic
    2:12 };                          % full model exc. speech envelope
            
%%% individual feature models
MODELS_INDIV = { ...
    'env',          ...  % speech envelope
    'phon_onsets',  ...  % phoneme onsets
    'morph',        ...  % inflectional morphology
    'word_onsets',  ...  % word onsets
    'artic',        ...  % articulatory complexity
    'word_freq',    ...  % word frequency
    'phon_freq',    ...  % phonotactic prob (positional segment frequency)
    'phon_surp',    ...  % phoneme surprisal
    'phon_ent',     ...  % phoneme entropy
    'word_surp',    ...  % word surprisal
    'word_ent',     ...  % word entropy
    'synt'};             % syntactic complexity

COLS_INDIV = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

%%% models used for ~(unattended && MuR)
MODELS  = [MODELS_COMPOSITE, MODELS_INDIV]; % model names
COLS    = [COLS_COMPOSITE, COLS_INDIV];     % corresponding feature columns

%% - PATHS ----------------------------------------------------------------

MAT_DIR = ['/Users/nadastojanovic/Development/mphil/2_mTRF/2_matfiles/' attn_label '_new/'];
OUTPUT  = ['/Users/nadastojanovic/Development/mphil/2_mTRF/4_results/' attn_label '_new_new/'];

ssList = dir(fullfile(MAT_DIR, '*.mat'));
%ssList = ssList(1); % uncomment when testing on a single file
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
    N_FEAT = size(stimulusdata, 2); % 12

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

    %%% unattended AND MuR distractor -> env model only
    is_mur = (attn == 2) && ismember(condition_str, {'EngA|EngB', 'FraA|FraB'});

    %%% initialise results struct
    TRF = struct();
    TRF.participant_ID = participant_id;
    TRF.condition = Cond;
    TRF.usage_group = UsageGroup;
    TRF.time_lags = LAGS;

    %%% pre-allocate for full model results
    r_full = zeros(N_FOLD, N_CHANS);
    w_sum = zeros(N_FEAT, N_LAGS, N_CHANS);
    fold_lambdas = zeros(N_FOLD, 1);
    r_null_all = zeros(N_MODELS, N_PERM, N_CHANS, N_FOLD);
    r_null_full = zeros(N_PERM, N_CHANS, N_FOLD);

    %%% pre-allocate for individual feature models results
    N_INDIV = numel(MODELS_INDIV);
    r_indiv = zeros(N_INDIV, N_FOLD, N_CHANS);
    w_indiv_sum  = cell(1, N_INDIV);

    for m = 1:N_INDIV
        w_indiv_sum{m} = zeros(numel(COLS_INDIV{m}), N_LAGS, N_CHANS);
    end

    %%% pre-allocate for individual feature models results
    N_COMP = numel(MODELS_COMPOSITE);
    r_comp = zeros(N_COMP, N_FOLD, N_CHANS);
    w_comp_sum = cell(1, N_COMP);
    for c = 1:N_COMP
        w_comp_sum{c} = zeros(numel(COLS_COMPOSITE{c}), N_LAGS, N_CHANS);
    end

    %%% pre-allocate for env model only (for unattended AND MuR distractor)
    r_env_null = zeros(N_PERM, N_CHANS, N_FOLD);

    warning('off', 'MATLAB:nearlySingularMatrix');
    warning('off', 'MATLAB:singularMatrix');

    for fold_i = 1:N_FOLD
        testtrial = fold_i;
        fprintf('Fold: %d  / %d\n', fold_i, N_FOLD);

        [strain, rtrain, stest, rtest] = mTRFpartition(stimulusdata, eegdata, N_FOLD, testtrial);

        if ~is_mur % attended OR (unattended AND non-MuR)
            %% one full model
            cv = mTRFcrossval(strain, rtrain, FS, MODEL_DIR, TMIN, TMAX, ...
                LAMBDAS, 'zeropad', 0, 'verbose', 0);
    
            [~, lambda_idx] = max(mean(mean(cv.r,3)));
            lambda = LAMBDAS(lambda_idx);
            fold_lambdas(fold_i) = lambda;
    
            %%% train
            model = mTRFtrain(strain, rtrain, FS, MODEL_DIR, TMIN, TMAX, ...
                lambda, 'zeropad', 0, 'verbose', 0);
    
            %%% test
            [~, mtrftest] = mTRFpredict(stest, rtest, model, ...
                'zeropad', 0, 'verbose', 0);
    
            %%% store model weights
            w_sum = w_sum + model.w;
    
            %%% store per-fold r values
            r_full(fold_i, :) = mtrftest.r;
    
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
    
            %% permute composite + individual features
            % circular shift of current model's feature columns in test
            % stimdata, test using the same pre-trained model ^
    
            for g = 1:N_MODELS   
                for p = 1:N_PERM
                    rnd_shft = randi([MIN_SHIFT, T - MIN_SHIFT]);
    
                    stest_perm = stest;
                    stest_perm(:, COLS{g}) = circshift(stest(:, COLS{g}), rnd_shft, 1);

                    [~, testperm] = mTRFpredict(stest_perm, rtest, ...
                        model, 'zeropad', 0, 'verbose', 0);
    
                    r_null_all(g, p, :, fold_i) = testperm.r;
                end
            end

            %% individually trained models
            % train a model for each individual feature & permute it out
            % of the full model to get its unique contribution

            %%% COMPOSITE MODELS
            for c = 1:N_COMP
                strain_c = cellfun(@(x) x(:, COLS_COMPOSITE{c}), strain, 'UniformOutput', false);
                stest_c = stest(:, COLS_COMPOSITE{c});
            
                cv_c = mTRFcrossval(strain_c, rtrain, FS, MODEL_DIR, ...
                    TMIN, TMAX, LAMBDAS, ...
                    'zeropad', 0, 'fast', 1, 'verbose', 0);
                [~, li] = max(mean(mean(cv_c.r, 3), 1));
                lambda_c = LAMBDAS(li);
            
                model_c = mTRFtrain(strain_c, rtrain, FS, MODEL_DIR, ...
                    TMIN, TMAX, lambda_c, 'zeropad', 0, 'verbose', 0);
            
                [~, test_c] = mTRFpredict(stest_c, rtest, model_c, ...
                    'zeropad', 0, 'verbose', 0);
            
                r_comp(c, fold_i, :) = test_c.r;
                w_comp_sum{c} = w_comp_sum{c} + model_c.w;
            end
   
            %%% SOLO FEATURE MODELS
            for m = 1:N_INDIV
                strain_m = cellfun(@(x) x(:, COLS_INDIV{m}), strain, 'UniformOutput', false);
                stest_m = stest(:, COLS_INDIV{m});
            
                cv_m = mTRFcrossval(strain_m, rtrain, FS, MODEL_DIR, ...
                    TMIN, TMAX, LAMBDAS, ...
                    'zeropad', 0, 'fast', 1, 'verbose', 0);
                [~, li] = max(mean(mean(cv_m.r, 3), 1));
                lambda = LAMBDAS(li);
            
                model_m = mTRFtrain(strain_m, rtrain, FS, MODEL_DIR, ...
                    TMIN, TMAX, lambda, 'zeropad', 0, 'verbose', 0);
    
                [~, test_m] = mTRFpredict(stest_m, rtest, model_m, ...
                    'zeropad', 0, 'verbose', 0);
            
                r_indiv(m, fold_i, :) = test_m.r;
                w_indiv_sum{m} = w_indiv_sum{m} + model_m.w;
            end

        else % unattended AND MuR distractor
            %%% env model only
            strain_env = cellfun(@(x) x(:, 1), strain, 'UniformOutput', false);
            stest_env = stest(:, 1);
    
            cv_env = mTRFcrossval(strain_env, rtrain, FS, MODEL_DIR, ...
                TMIN, TMAX, LAMBDAS, 'zeropad', 0, 'fast', 1, 'verbose', 0);
            [~, li]   = max(mean(mean(cv_env.r, 3), 1));
            lambda = LAMBDAS(li);

            model_env = mTRFtrain(strain_env, rtrain, FS, MODEL_DIR, ...
                TMIN, TMAX, lambda, 'zeropad', 0, 'verbose', 0);

            [~, test_env] = mTRFpredict(stest_env, rtest, model_env, ...
                'zeropad', 0, 'verbose', 0);
    
            r_indiv(1, fold_i, :) = test_env.r;
            w_indiv_sum{1} = w_indiv_sum{1} + model_env.w;
    
            %%% env null
            r_env_null_fold = zeros(N_PERM, N_CHANS);
            for p = 1:N_PERM
                rnd_shft = randi([MIN_SHIFT, T - MIN_SHIFT]);

                stest_env_p = circshift(stest_env, rnd_shft, 1);
                [~, testperm] = mTRFpredict(stest_env_p, rtest, ...
                    model_env, 'zeropad', 0, 'verbose', 0);

                r_env_null_fold(p, :) = testperm.r;
            end
            r_env_null(:, :, fold_i) = r_env_null_fold;

        end

    end

    if ~is_mur % attended OR (unattended AND non-MuR)
        %% store full model results
        TRF.full.weights = w_sum / N_FOLD;              % [N_FEAT x N_LAGS x N_CHAN]
        TRF.full.lambdas = fold_lambdas;                % [N_FOLD x 1]
    
        TRF.full.r_null = r_null_full;                  % [N_PERM x N_CHANS x N_FOLD]
        TRF.full.r = r_full;                            % [N_FOLD x N_CHAN]
        
        TRF.full.r_null_avg = mean(r_null_full(:));     % scalar
        TRF.full.r_avg = mean(r_full(:));               % scalar
    
        %% store per model results
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

        %% store individually trained composite model results
        for c = 1:N_COMP
            cname = MODELS_COMPOSITE{c};

            TRF.(cname).weights = w_comp_sum{c} / N_FOLD;
            TRF.(cname).r = squeeze(r_comp(c, :, :));
            TRF.(cname).r_avg   = mean(r_comp(c, :, :), 'all');
        end
    
        %% store individually trained solo feature model results
        for m = 1:N_INDIV
            mname = MODELS_INDIV{m};

            TRF.(mname).weights = w_indiv_sum{m} / N_FOLD;     % [N_FEAT x N_LAGS x N_CHANS]
            TRF.(mname).r = squeeze(r_indiv(m, :, :));         % [N_FOLD x N_CHANS]
            TRF.(mname).r_avg = mean(r_indiv(m, :, :), 'all'); % scalar
        end
    else % unattended AND MuR distractor
        %% store env model + its null only
        TRF.env.feat_cols = 1;

        r_env = squeeze(r_indiv(1, :, :));                  % [N_FOLD x N_CHANS]
    
        TRF.env.r_null = r_env_null;                        % [N_PERM x N_CHAN x N_FOLD]
        TRF.env.r_null_avg = mean(r_env_null(:));           % scalar

        TRF.env.weights = w_indiv_sum{1} / N_FOLD;          % [N_FEAT x N_LAGS x N_CHAN]

        TRF.env.r = r_env;                             % [N_FOLD x N_CHAN]
        TRF.env.r_avg = mean(r_env(:));                % scalar
    end

%% - SAVE OUTPUT MAT FILE -------------------------------------------------

    save_path = fullfile(OUTPUT, [basename '.mat']);
    parsave_forward(save_path, TRF);
    fprintf('Saved: %s\n', basename);

end

fprintf('Forward mTRF modelling complete.\n');

%% - HELPERS --------------------------------------------------------------

function parsave_forward(fpath, TRF)
    save(fpath, 'TRF', '-v7.3');
end