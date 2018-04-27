function [b,w,Out] = sgd_hinge_lingyu(X_ord,y_ord,opts)
D = [X_ord y_ord];
D_shuff = D(randperm(size(D,1)),:);
X = D_shuff(:,1:20958);
y = D_shuff(:,20959);
[n,p] = size(X);
%initialize parameters
delta = 1; lam1 = 0.2; lam2 = 1; lam3 = 0;
maxit = 500; tol = 1e-4; 
b0 = 0; 
%w0 = zeros(p,1);
% modified by Lingyu
w0 = normrnd(0,0.1,[p,1]);
fix_b = 0;
if isfield(opts,'delta') delta = opts.delta; end;
if isfield(opts,'lam1')  lam1  = opts.lam1;  end;
if isfield(opts,'lam2')  lam2  = opts.lam2;  end;
if isfield(opts,'lam3')  lam3  = opts.lam3;  end;
if isfield(opts,'maxit') maxit = opts.maxit; end;
if isfield(opts,'tol')   tol   = opts.tol;   end;
if isfield(opts,'b0')    b0 = opts.b0;       end;
if isfield(opts,'w0')    w0 = opts.w0;       end;
if isfield(opts,'fix_b') fix_b = opts.fix_b; end;
t0 = 1;
b_hat = b0; w_hat = w0; b = b0;
%increase factor of L
eta = 6.0;




hist_obj = zeros(maxit,1); hist_err = zeros(maxit,2);
z0 = 1-y.*(X*w0+b0);
f = max(0,z0);

f = sum(f/n)+lam1*sum(abs(w0))+lam2/2*sum(w0.^2)+lam3/2*b0^2;
f0 = f;

L = 1;

batch_size = 50;
fprintf('start optimization based on hinge loss \n');
for iter = 1:maxit
    batch_idx = randi([1 1000]);
    grad_w = 0;
    grad_b = 0;
    for i = 1:batch_size
        idx = batch_size*(batch_idx-1)+i;
        zi = 1-y(idx).*(X(idx,:)*w_hat+b_hat);
        grad_wi = -sign(max(0,zi))*X(idx,:)'*y(idx);
        grad_bi = -sign(max(0,zi))*y(idx);
        grad_w = grad_w + grad_wi;
        grad_b = grad_b + grad_bi;
    end
    grad_w = grad_w./batch_size + lam1*(sign(w_hat))+lam2*w_hat;
    grad_b = grad_b/batch_size;
    %fprintf('%3.2f grad_w \n',grad_w);
    %fprintf('%3.2f grad_b \n',grad_b);
    w_hat = w_hat - eta*grad_w;
    b_hat = b_hat - eta*grad_b;
    
    % record the objective value based on the whole dataset
    z = 1-y.*(X*w_hat+b_hat);
    f = max(0,z0);
    f = sum(f/n);
    f = f+lam1*sum(abs(w_hat))+lam2/2*sum(w_hat.^2)+lam3/2*b_hat^2;
    hist_obj(iter) = f;

end
b = b_hat; w = w_hat; 
Out.iter = iter; Out.hist_obj = hist_obj(1:iter); 
Out.hist_err = hist_err(1:iter,:);
Out.b = b_hat; Out.w = w_hat; 
Out.L = L;
Out.y_train_gd = y;
Out.y_train_pred = sign(b + X*w_hat);


