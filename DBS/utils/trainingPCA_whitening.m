%whitening the CG map -- hk
function [PCAw, Xmean] = trainingPCA_whitening(Xo)

D = size(Xo,1);
dout = D;
Xmean = mean(Xo,2);

X = bsxfun(@minus, Xo, Xmean);

covD = X*X'/size(X,2);


% Eigen-decomposition
if 3 * dout < D
  eigopts.issym = true;
  eigopts.isreal = true;
  eigopts.tol = eps;
  eigopts.disp = 0;

  [eigvec, eigval] = eigs (double(covD), dout, 'LM', eigopts);
else
  [eigvec, eigval] = eig (covD);
  eigvec = eigvec (:, end:-1:1);
  eigval = diag (eigval);
  eigval = eigval (end:-1:1);
end

%eigval (1024:end) = eigval (1024);
PCAw = diag(eigval.^-0.5) * eigvec'; % projection matrix
%PCAw = eigvec'; % projection matrix