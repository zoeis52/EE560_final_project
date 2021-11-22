% -------------------------------------------------------------------------
% ACC_ArmReaching.m
% This source code is about to measure the classification accuracy based on the basic machine learning method.
% This classification accuracy was computed by usign the fundamental EEG signal processing such band-pass filtering, time epoch, feature extraction, classification.
% Feature extraction: Common spatial pattern (CSP),
% Classifier: regularized linear discriminant analysis (RLDA)

% If you want to improve the classification accuracies, you could adopt more advanced methods. 
% For example: 
% 1. Independent component analysis(ICA) for artifact rejection
% 2. Common average reference (CAR) filter, Laplacian spatial filter, and band power for feature extraction
% 3. Support vector machine (SVM) or Random forest (RF) for classifier 

% Please add the bbci toolbox in the 'Reference_toolbox' folder
%--------------------------------------------------------------------------

%% Initalization
clc; close all; clear all;
%%
% Directory
% Write down where converted data file downloaded (file directory)
dd='file path\'; 
cd 'file path';
% Example: dd='Downlad_folder\SampleData\plotScalp\';

datedir = dir('*.mat');
filelist = {datedir.name};

% Setting time duration: interval 0~3 s
ival=[0 3001];

%% Performance measurement
for i = 1:length(filelist)
    filelist{i}
    [cnt,mrk,mnt]=eegfile_loadMatlab([dd filelist{i}]);
    
    % Band pass filtering, order of 4, range of [8-30] Hz (mu-, beta-bands)
    filterRange = {[8 30]};
    for filt = 1:length(filterRange)
        filelist{i}
        filterRange{filt}
        
        cnt = proc_filtButter(cnt, 4 ,filterRange{filt});
        epo=cntToEpo(cnt,mrk,ival);
        
        % Select channels
        epo = proc_selectChannels(epo, {'FC5','FC3','FC1','FC2','FC4','FC6',...
            'C5','C3','C1', 'Cz', 'C2', 'C4', 'C6',...
            'CP5','CP3','CP1','CPz','CP2','CP4','CP6'});
        
        classes=size(epo.className,2);
        
        trial=50;
        
        % Set the number of rest trial to the same as the number of other classes trial.
        for ii =1:classes
            if strcmp(epo.className{ii},'Rest')
                epoRest=proc_selectClasses(epo,{epo.className{ii}});
                epoRest.x=datasample(epoRest.x,trial,3,'Replace',false);
                epoRest.y=datasample(epoRest.y,trial,2,'Replace',false);
            else
                epo_check(ii)=proc_selectClasses(epo,{epo.className{ii}});
                
                % Randomization
                epo_check(ii).x=datasample(epo_check(ii).x,trial,3,'Replace',false);
                epo_check(ii).y=datasample(epo_check(ii).y,trial,2,'Replace',false);
            end
        end
        if classes<7
            epo_check(size(epo_check,2)+1)=epoRest;
        end
        
        % concatenate the classes
        for ii=1:size(epo_check,2)
            if ii==1
                concatEpo=epo_check(ii);
            else
                concatEpo=proc_appendEpochs(concatEpo,epo_check(ii));
            end
        end
        
        %% CSP - FEATURE EXTRACTION
        [csp_fv,csp_w,csp_eig]=proc_multicsp(concatEpo,3);
        proc=struct('memo','csp_w');
        
        proc.train= ['[fv,csp_w]=  proc_multicsp(fv, 3); ' ...
            'fv= proc_variance(fv); ' ...
            'fv= proc_logarithm(fv);'];
        
        proc.apply= ['fv= proc_linearDerivation(fv, csp_w); ','fv= proc_variance(fv); ' ,'fv= proc_logarithm(fv);'];
        
        
        %% RLDA - CLASSIFICATION WITH 10-FOLD CROSS-VALIDATION       
        [C_eeg, loss_eeg_std, out_eeg.out, memo] = xvalidation(concatEpo,'RLDAshrink','proc',proc, 'kfold', 10);
        Result(filt)= 1-C_eeg;
        Result_Std(filt)=loss_eeg_std;
        All_csp_w(:,:,filt)=csp_w;
    end   
    % Maximum classification performance of each subject
    maxPerformance(i) = max(Result);
    
end

A = num2cell(maxPerformance);
subPerformance = cat(1, filelist, A);

% Save results of CSP with RLDA in excel file
% total results: 9 bands of accuracies
filename = 'Performance_reaching_ME.xlsx';
writecell((subPerformance)', filename, 'Sheet', 1);

