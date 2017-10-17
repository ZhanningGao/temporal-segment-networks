% PCA whitening the val and test EPtensors

% Updated by Zhanning Gao 09/13/2017   --- All for MT&G ---

subset_all = [{'test'}, {'val'}];
for  i = 1:2% 'val' or 'test'
    subset = subset_all{i};
    
    tau = 1;
    
    % EPmap path
    EPPath = '/data3/datasets/TH14/EP-TSN/EP_E48W7Dim128_onInit_incep5a';
    
    % extra data path
    EXTPath = '/data3/datasets/UCF20/EP-TSN/EP_E48W7Dim128_var_0.1_init_incep5a';
    %BGPath = '/data3/datasets/TH14/BG-EP-TSN/EP_E32W7Dim128_onInit';
    
    % load EP data
    load(fullfile(EPPath,['TRUE', subset, '.mat']));
    
    % load EPmaps
    load([EPPath '/EPmaps_info_' subset '.mat']);
    
    % load PCAw mat
    load(fullfile(EPPath,'PCAw_TH14_val_v2.mat'));
    
    num_vid = length(PIQL);
    
    alpha = 0.5;
    Dim = 512;
    
    EPallset = [];
    
    for i_vid = 1:num_vid
        
        c_PIQL = PIQL{i_vid};
        
        f_all = c_PIQL.frames;
        
        E = c_PIQL.E;
        W = c_PIQL.W;
        Z = size(c_PIQL.pi,3);
        
        PI = reshape(permute(c_PIQL.pi,[3,2,1]), [Z, E*E]);
        QL = permute(c_PIQL.ql,[3,2,1]);
        
        mask = reshape(permute(EPmaps(i_vid).mask,[2,1]),[1,E*E]);
        mask = mask(ones(Dim,1),:);
        
        CGset = PCAw_TH14(1:Dim,:)*bsxfun(@minus,yael_vecs_normalize(sign(PI).*abs(PI).^alpha),meanTH14);
        
        CGset = yael_vecs_normalize(CGset(1:Dim,:)).*mask;
        
        CGset(isnan(CGset))=0;
        
        EPmap = EPmaps(i_vid).EPmap;
        
        EPallset(i_vid).CGset = reshape(CGset, [Dim,E,E]);
        EPallset(i_vid).ql    = QL;
        EPallset(i_vid).EPmap = permute(single(EPmap), [3,2,1]);
        
        if mod(i_vid,10) == 0
            fprintf('%d th video done\n', i_vid);
        end
        
    end
    
    if strcmp(subset, 'val')
        
        mat_name = dir(fullfile(EXTPath,'*.mat'));
        if ~isempty(mat_name)
            
            num_mat = length(mat_name);
            for i_mat = 1:num_mat
                
                % load EP data
                load(fullfile(EXTPath,mat_name(i_mat).name));
                
                % load EPmaps
                load([EXTPath '/EPmaps/EPmaps_info_' mat_name(i_mat).name]);
                
                num_vid = length(PIQL);
                
                alpha = 0.5;
                Dim = 512;
                
                for i_vid = 1:num_vid
                    c_PIQL = PIQL{i_vid};
                    
                    f_all = c_PIQL.frames;
                    
                    E = c_PIQL.E;
                    W = c_PIQL.W;
                    Z = size(c_PIQL.pi,3);
                    
                    PI = reshape(permute(c_PIQL.pi,[3,2,1]), [Z, E*E]);
                    QL = permute(c_PIQL.ql,[3,2,1]);
                    
                    mask = reshape(permute(EPmaps(i_vid).mask,[2,1]),[1,E*E]);
                    mask = mask(ones(Dim,1),:);
                    
                    CGset = PCAw_TH14(1:Dim,:)*bsxfun(@minus,yael_vecs_normalize(sign(PI).*abs(PI).^alpha),meanTH14);
                    
                    CGset = yael_vecs_normalize(CGset(1:Dim,:)).*mask;
                    
                    CGset(isnan(CGset))=0;
                    
                    EPmap = EPmaps(i_vid).EPmap;
                    
                    EPallset(end+1).CGset = reshape(CGset, [Dim,E,E]);
                    EPallset(end).ql    = QL;
                    EPallset(end).EPmap = permute(single(EPmap), [3,2,1]);
                    
                    if mod(i_vid,10) == 0
                        fprintf('%d th video done\n', i_vid);
                    end
                end
            end
        end
        
        %mat_name = dir(fullfile(BGPath,'*.mat'));
        if 0%~isempty(mat_name)
            
            num_mat = length(mat_name);
            for i_mat = 1:num_mat
                
                % load EP data
                load(fullfile(BGPath,mat_name(i_mat).name));
                
                % load EPmaps
                load([BGPath '/EPmaps/EPmaps_info_' mat_name(i_mat).name]);
                
                num_vid = length(PIQL);
                
                alpha = 0.5;
                Dim = 512;
                
                for i_vid = 1:5:num_vid
                    c_PIQL = PIQL{i_vid};
                    
                    f_all = c_PIQL.frames;
                    
                    E = c_PIQL.E;
                    W = c_PIQL.W;
                    Z = size(c_PIQL.pi,3);
                    
                    PI = reshape(permute(c_PIQL.pi,[3,2,1]), [Z, E*E]);
                    QL = permute(c_PIQL.ql,[3,2,1]);
                    
                    mask = reshape(permute(EPmaps(i_vid).mask,[2,1]),[1,E*E]);
                    mask = mask(ones(Dim,1),:);
                    
                    CGset = PCAw_TH14(1:Dim,:)*bsxfun(@minus,yael_vecs_normalize(sign(PI).*abs(PI).^alpha),meanTH14);
                    
                    CGset = yael_vecs_normalize(CGset(1:Dim,:)).*mask;
                    
                    CGset(isnan(CGset))=0;
                    
                    EPmap = EPmaps(i_vid).EPmap;
                    
                    EPallset(end+1).CGset = reshape(CGset, [Dim,E,E]);
                    EPallset(end).ql    = QL;
                    EPallset(end).EPmap = permute(single(EPmap), [3,2,1]);
                    
                    if mod(i_vid,10) == 0
                        fprintf('%d th video done\n', i_vid);
                    end
                end
            end
        end
        
    end
    
    mkdir(fullfile(EPPath,'TrTe'));
    save(fullfile(EPPath,'TrTe',['EP_' subset '.mat']), 'EPallset', '-v7.3');
    
end