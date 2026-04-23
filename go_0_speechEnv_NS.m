%% Author: Nada
% Extracts the broadband amplitude envelopes and downsamples them to 100 Hz

clc; clear; close all

%% - SETUP --------------------------------------------------
FS           = 100;
N_UTTERANCES = 120;

%% - PATHS --------------------------------------------------
WAV_DIR = '/Users/nadastojanovic/Development/mphil/0_speech_envelope/stimuli/';
OUTPUT = '/Users/nadastojanovic/Development/mphil/0_speech_envelope/NS_env.mat';

wavs = uipickfiles( ...
    'FilterSpec', fullfile(WAV_DIR, '*.wav'), ...
    'Prompt', 'Select the .wav files');

N_FILES = size(wavs, 2);

%% - MAIN LOOP -----------------------------------------------

env = {};
env_code = {};

for i = 1:N_FILES
    %%% load wav file
    [~, name, ~] = fileparts(wavs{i});
    [y, Fs] = audioread(wavs{i});
    stim = y(:, 1); % audio track (use only first channel)
    
    %%% get amplitude envelope
    [env_stim, ~] = convert_hilbert2(stim, Fs);

    %%% build output vector
    %%%     col 0: speech env (downsampled to 100Hz)
    vec = resample(env_stim, FS, Fs);

    if size(vec, 1) < 240
        vec = [vec; zeros(240 - size(vec, 1), 1)];
    end
    vec = [zeros(10, 1); vec];

    env{end + 1} = vec;
    env_code{end + 1, 1} = name;

end

%% - SAVE OUTPUT MAT FILE -----------------------------------
fprintf('Saving .mat\n')
save(OUTPUT, 'env', 'env_code');
fprintf('Saved %d vectors to %s\n', length(env), OUTPUT);