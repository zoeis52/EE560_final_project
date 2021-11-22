%-------------------------------------------------------------------------
% Data_Analysis.m
% This source code is about sample codes for classification accuracy.
% If you want to execute this sample codes, you should download sampleData.zip in the '5_Scripts' folder.

% Please add the bbci toolbox in the 'Reference toolbox' folder.
%-------------------------------------------------------------------------

%% Data Convert from raw data to converted data
% If you want to use converted data directly, please refer to the .mat files in the '_ConvertedData' folder.
% Drag on below code and follow the instruction for converting.

edit Data_Convert

%% Performance measurment (Arm-reaching, Hand-grasping and Wrist-twisting)
% This code is about for classification accuracy according to each task using a common spatial pattern (CSP) with a regularized linear discriminant analysis (RLDA).
% Each line is about Arm-reaching, Hand-grasping, and Wrist-twisting.
% Please execute each line with press F9.

% Performance measurement of Arm-reaching task
edit ACC_ArmReaching

% Performance measurement of Hand-grasping task
edit ACC_HandGrasping

% Performance measurement of Wrist-twisting task
edit ACC_WristTwisting

%%