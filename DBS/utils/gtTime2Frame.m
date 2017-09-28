% Convert gt files of TH14 to frame-level labels, i.e.,
% 3-5s is a specific action A -(if fps is 5)-> F11-F25 is labeled as A

% ind - class-------------label
% 7 BaseballPitch           1
% 9 BasketballDunk          2
% 12 Billiards              3
% 21 CleanAndJerk           4
% 22 CliffDiving            5
% 23 CricketBowling         6
% 24 CricketShot            7
% 26 Diving                 8
% 31 FrisbeeCatch           9
% 33 GolfSwing              10
% 36 HammerThrow            11
% 40 HighJump               12
% 45 JavelinThrow           13
% 51 LongJump               14
% 68 PoleVault              15
% 79 Shotput                16
% 85 SoccerPenalty          17
% 92 TennisSwing            18
% 93 ThrowDiscus            19
% 97 VolleyballSpiking      20
% 102 Unknow                21

% updated by Zhanning Gao 09/09/2017 --- All for MT&G


subset = 'val'; % 'val' or 'test'

if strcmp(subset,'val')
    gtpath = '../data/TH14evalkit/groundtruth';
elseif strcmp(subset,'test')
    gtpath = '../data/TH14_Temporal_Annotations_Test/annotations/annotation';
end

% load video name
load(['../data/TRUE' subset '_set_name.mat']);
videoCNNPath = ['/data3_alpha/datasets/THUMOS14/CNNfeature-rgb/TRUE' subset];

% load all video meta data
load('../data/validation_set.mat');
load('../data/test_set_meta.mat');

if strcmp(subset,'val')
    video_meta = validation_videos;
elseif strcmp(subset,'test')
    video_meta = test_videos;
else
    error('Could not find meta file')
end

[th14classids,th14classnames]=textread([gtpath '/detclasslist.txt'],'%d%s');

vid_info = [];

for i_vid = 1:length(videoNames)
    
    c_vidName = videoNames(i_vid,:);
    
    load(fullfile(videoCNNPath,[c_vidName, '.mat']));
    
    f_all = size(cnn4v,1);
    vid_labels = (length(th14classnames)+1)*ones(1,f_all);% for TSN, f_ is 2 time due to mirroring
    
    duration = -1;
    for i_meta = 1:length(video_meta)
        
        if strcmp(subset,'val')
            video_meta_name = video_meta(i_meta).video_name(1:end-5);
        elseif strcmp(subset,'test')
            video_meta_name = video_meta(i_meta).video_name;
        end
        
        if strcmp(c_vidName,video_meta_name)
            duration = video_meta(i_meta).video_duration_seconds;
            break;
        end
    end
    if duration<0
        error(['Could not find the meta info of ' c_vidName])
    end
    
    for i=1:length(th14classnames)
        class=th14classnames{i};
        gtfilename=[gtpath '/' class '_' subset '.txt'];
        if exist(gtfilename,'file')~=2
            error(['TH14evaldet: Could not find GT file ' gtfilename])
        end
        [subvideonames,t1,t2]=textread(gtfilename,'%s%f%f');
        for i_subvid = 1:length(subvideonames)
            if strcmp(c_vidName,subvideonames{i_subvid})
                [f_start, f_end] = time2frameIndex(t1(i_subvid),t2(i_subvid), duration, f_all/2);
                % for TSN, f_ shold *2 due to mirroring
                f_start = f_start*2-1;
                f_end = f_end*2;
                
                vid_labels(f_start:f_end) = i;
                plot(vid_labels);drawnow;
                
            end
        end
    end
    
    vid_info(i_vid).name = c_vidName;
    vid_info(i_vid).duration = duration;
    vid_info(i_vid).frames = f_all;
    vid_info(i_vid).labels = vid_labels;
    
    fprintf('video %s %d th is done\n', c_vidName, i_vid);
    
end

save(['../data/vid_info_' subset '.mat'], 'vid_info');
