%--------------------------------------------------------------------------
% Data_Convert.m
% This file is about convert raw data file to .mat (converted) file
% Please add the bbci toolbox in the 'Reference_toolbox' folder
%--------------------------------------------------------------------------

%% initialization
clear all; clc; close all;

%% Load file
% Write down where raw data file downloaded (file directory)
dd='file path'; 
% Example: dd='Downlad_folder\SampleData\plotScalp\';

% Write down file name that you want to convert
filelist={'file name'};
% Example: filelist = {'session1_sub13_twist_MI'};

%% Convert
for ff= 1:length(filelist)
    
    file= filelist{ff};
    opt= [];
    
    fprintf('** Processing of %s **\n', file);    
    % load the header file
    try
        hdr= eegfile_readBVheader([dd '\' file]);
    catch
        fprintf('%s/%s not found.\n', dd, file);
        continue;
    end
    
    % filtering with Chev filter
    Wps= [42 49]/hdr.fs*2;
    [n, Ws]= cheb2ord(Wps(1), Wps(2),3, 40);
    [filt.b, filt.a]= cheby2(n, 50, Ws);   
    
    % Load channel information and sampling rate
    [cnt, mrk_orig]= eegfile_loadBV([dd '\' file],  ...
        'filt',filt,'fs',250);
    
    % Setting save directory
    cnt.title= ['save directory path' file];
    
    % Load mrk file, Assign the trigger information into mrk variable
    % If you want to convert another task's data, please check the trigger
    % information into 'Converting_Arrow' function.
    mrk = Converting_Arrow(mrk_orig);
    
    % Assign the channel montage information into mnt variable
    mnt = getElectrodePositions(cnt.clab);
    
    % Assign the sampling rate into fs_orig variable
    fs_orig= mrk_orig.fs;
    
    var_list= {'fs_orig',fs_orig, 'mrk_orig',mrk_orig, 'hdr',hdr};
    
    % Convert the raw data file to .mat file
    eegfile_saveMatlab(cnt.title, cnt, mrk, mnt, ...
        'channelwise',1, ...
        'format','int16', ...
        'resolution', NaN);
end

%%
% If this scentence show in command window, raw data converted well
disp('All Data Converting was Done!');