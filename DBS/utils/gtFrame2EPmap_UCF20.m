% convert frame labels to EP map labels
% updated by Zhanning Gao 09/10/2017 --- All for MT&G ---

% EXTPath = '/data3_alpha/datasets/UCF20/EP-TSN/EP_E48W7Dim128_var_0.1_init_incep5a';

mat_name = dir(fullfile(EXTPath, '*.mat'));

num_mat = length(mat_name);
tau = 2;

for  i_mat = 1:num_mat

    % load EP data
    load(fullfile(EXTPath,mat_name(i_mat).name));
    
    num_vid = length(PIQL);
    num_class = 21;
    
    EPmaps = [];
    
    for i_vid = 1:num_vid
        
        c_PIQL = PIQL{i_vid};
        
        f_all = c_PIQL.frames;
        
        E = c_PIQL.E;
        W = c_PIQL.W;
        
        EPmap = zeros(E,E,num_class);
        
        f_label = i_mat*ones(1,f_all);
        
        
        for i_c = 1:num_class
            ind_c = find(f_label==i_c);
            if isempty(ind_c)
                continue;
            end
            
            c_ql = c_PIQL.ql(:,:,ind_c);
            
            Kmap = zeros(E,E);
            Wmap = sum(c_ql,3);
            Kmap(Wmap>tau)=1;
            mask = zeros((W-1)*2+1);
            mask(W:end,W:end)=1;
            Kmap = circconv2(Kmap,mask,W-1,[E,E]);
            Kmap(Kmap>0)=1;
            
            EPmap(:,:,i_c) = Kmap;
            
        end
        
        c_last = sum(EPmap(:,:,1:end-1),3);
        c_last(c_last>0)=1;
        
        mask = sum(EPmap,3);
        mask(mask>0)=1;
        
        EPmap(:,:,end) = xor(ones(size(c_last)), c_last);
        
        EPmap = EPmap./repmat(sum(EPmap,3),[1,1,num_class]);
        
        EPmaps(i_vid).EPmap = EPmap;
        EPmaps(i_vid).mask = mask;
        EPmaps(i_vid).E = E;
        EPmaps(i_vid).W = W;
        
    end
    
    mkdir([EXTPath, '/EPmaps']);
    save([EXTPath '/EPmaps/EPmaps_info_' mat_name(i_mat).name], 'EPmaps');
    
end