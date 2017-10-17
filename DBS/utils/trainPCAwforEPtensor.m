% get vector to train PCAw for EPtensor maps

% updated by Zhanning Gao 09/13/2017   --- All for MT&G ---


subset = 'val'; % 'val' or 'test'

tau = 1;

% EPmap path
% EPPath = '/data3_alpha/datasets/TH14/EP-TSN/EP_E48W7Dim128_onInit_incep5a';

% video info path
video_info_path = '../data';

% load video info
load(fullfile(video_info_path, ['vid_info_' subset '.mat']));

% load EP data
load(fullfile(EPPath,['TRUE', subset, '.mat']));

num_vid = length(PIQL);
trainEPset = [];

num_perVID = 500;
alpha = 0.5;

for i_vid = 1:num_vid
    
    c_PIQL = PIQL{i_vid};
    
    f_all = c_PIQL.frames;
    
    E = c_PIQL.E;
    W = c_PIQL.W;
    
    PI = reshape(permute(c_PIQL.pi,[3,1,2]), [size(c_PIQL.pi,3), E*E]);
    
    p_tr = i_vid*num_perVID;
    ind_id = randperm(size(PI,2));
    
    %    CGset = (PCAw_res5(1:Dim,:)*bsxfun(@minus,yael_vecs_normalize(cnnCGset(f_id).CGset.^alpha),meanRes5));
    tmp = PI(:,ind_id(1:num_perVID));
    trainEPset(:,p_tr-num_perVID+1:p_tr) = yael_vecs_normalize(sign(tmp).*abs(tmp).^alpha);

end



idx = find(isnan(trainEPset(1,:))==0);

trainEPset = trainEPset(:,idx);

[PCAw_TH14, meanTH14] = trainingPCA_whitening(trainEPset);

save(fullfile(EPPath,['PCAw_TH14_' subset '_v2.mat']), 'PCAw_TH14', 'meanTH14')