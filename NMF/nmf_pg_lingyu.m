function [W,H,Out] = nmf_pg_lingyu(X,opts)

% We use H to represent H^T in the theory part
if isfield(opts,'maxit') maxit = opts.maxit; end;
if isfield(opts,'tol')   tol   = opts.tol;   end;
if isfield(opts,'H0')    H0 = opts.H0;       end;
if isfield(opts,'W0')    W0 = opts.W0;       end;
eta = 0.0001;
W=W0;H=H0;
hist_obj = zeros(maxit,1);
for iter=1:maxit
  grad_W = W*(H*H') - X*H';
  grad_H = (W'*W)*H - W'*X;
  W = max(W - eta*grad_W,0);    
  H = max(H - eta*grad_H,0);    
  hist_obj(iter) = 0.5*(norm(X-W*H,'fro')^2);
end
Out.iter = iter; Out.hist_obj = hist_obj(1:iter); 
Out.H = H; Out.W = W; 
