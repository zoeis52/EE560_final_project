%% Load a file and plot the channels for visualization
clear all;

%% Load File
folder = 'EMG_ConvertedData';
filename = 'EMG_session1_sub1_multigrasp_realMove.mat';

load(fullfile(folder, filename));


%% Plot all channels in file
% Determine current variables
cur_var = who;
% Plot channels
figure()
chan_indx = 1;
for i = 1:length(cur_var)
    if contains(cur_var{i}, 'ch')
        chan{chan_indx} = eval(cur_var{i});
        plot((0:1:length(chan{chan_indx})-1)/dat.fs, chan{chan_indx});
        hold on
        %xlim([0 (length(chan{chan_indx})-1)/dat.fs]);
        xlabel('Time (seconds)');
        ylabel('Channels');
        xlim(mrk.misc.pos/dat.fs);
        chan_indx = chan_indx +1;
    end
end
%% Plot channels as subplot
figure()
if size(chan,2)== 60
for i = 1:size(chan,2)
    subplot(6,10, i);
    plot((0:1:length(chan{i})-1)/dat.fs, chan{i});
    title(['Channel ', num2str(i)]);
    xlim(mrk.misc.pos/dat.fs);
end
else 
 for i = 1:size(chan,2)
    subplot(size(chan,2),1, i);
    plot((0:1:length(chan{i})-1)/dat.fs, chan{i});
    title(['Channel ', num2str(i)]);
    xlim(mrk.misc.pos/dat.fs);
 end
end

%% Plot presentation times
% First create array identifying grasp type
event_indx = ones(1,size(mrk.y,2));
for i = 1:size(mrk.y,1)
    event_indx(mrk.y(i,:) == 1) = i;
end
plotColors = jet(size(mrk.y,1)); % create an array of N amount of colors
figure()
for i = 1:length(mrk.pos)
    xline(mrk.pos(i)/dat.fs, ...
        'Color',plotColors(event_indx(i),:));
end