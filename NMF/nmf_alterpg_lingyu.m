function [W,H,Out] = nmf_alterpg_lingyu(X,opts)

% We use H to represent H^T in the theory part
if isfield(opts,'maxit') maxit = opts.maxit; end;
if isfield(opts,'tol')   tol   = opts.tol;   end;
if isfield(opts,'H0')    H0 = opts.H0;       end;
if isfield(opts,'W0')    W0 = opts.W0;       end;
eta = 0.0001;
W=W0;H=H0;
r=17;
hist_obj = zeros(maxit,1);
for iter=1:maxit
  grad_W = W*(H*H') - X*H';
  L_W = norm(H*H');
  W = max(W - grad_W/L_W,0); 
  grad_H = (W'*W)*H - W'*X;
  L_H = norm(W'*W);
  H = max(H - grad_H/L_H,0);    
  hist_obj(iter) = 0.5*(norm(X-W*H,'fro')^2);
end
for sub = 1:r
    sub_img = reshape(W(:,sub),[32,32]);
    imwrite(mat2gray(sub_img),strcat('/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/optimization/HW/hw3_sol_lingyu/output/nmf_alterpg_lingyu/Col_',int2str(sub),'.jpg'));
end
Out.iter = iter; Out.hist_obj = hist_obj(1:iter); 
Out.H = H; Out.W = W; 
