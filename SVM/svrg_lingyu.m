function [b,w,Out] = svrg_lingyu(X_ord,y_ord,opts)
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
nstall1 = 0; nstall2 = 0;
t0 = 1;
b_hat = b0; w_hat = w0; b = b0;
%increase factor of L
eta = 1.5;
hist_obj = zeros(maxit,1); hist_err = zeros(maxit,2);
z0 = y.*(X*w0+b0);
f = (1-z0).^2/(2*delta);
id = z0>1; f(id) = 0;
id = z0<1-delta; f(id) = 1-delta/2-z0(id);
f = sum(f/n)+lam1*sum(abs(w0))+lam2/2*sum(w0.^2)+lam3/2*b0^2;
f0 = f;

L = 1;
g = w0*ones(1,n);
w_his = w0;
% h = b0*ones(1,n);
% batch_size = 50;
fprintf('start saga based on Elastic net HSVM loss \n');
for iter = 1:maxit
    fprintf('for %3.2f-th iteration \n', iter);
    w_avg = w_his/iter;
    b_avg = b0;
    %batch_idx = randi([1 1000]);
    grad_w = 0;
    %grad_b = 0;
    m = randi([1 n]);
    % Note, randomly select a index once for each iteration / epoch.
    for i = 1:n
        %fprintf('for %3.2f-th sample \n', i);
        supple_val = zeros(p,1);
        z_avg = y(i).*(b_avg + X(i,:)*w_avg);
        if z_avg>1
            grad_wi = 0;
            %grad_bi = 0;
        elseif z_avg<=1 && z_avg > 1-delta
            grad_wi = ((z_avg-1)/delta)*X(i,:)'*y(i);
            %grad_bi = ((zi-1)/delta)*y(idx);
        else
            grad_wi = (-1)*X(i,:)'*y(i);
            %grad_bi = -1*y(idx);
        end
        if i == m
            z_m_hat = y(m).*(b_hat + X(m,:)*w_hat);
            if z_m_hat>1
                grad_wm = 0;
            elseif z_m_hat<=1 && z_m_hat > 1-delta
                grad_wm = ((z_m_hat-1)/delta)*X(m,:)'*y(m);
            else
                grad_wm = (-1)*X(m,:)'*y(m);
            end
            supple_val = n*(grad_wm-grad_wi);
        end
        grad_w = grad_w + grad_wi + supple_val;
        
    end
    grad_w = grad_w./n + lam1*(sign(w_hat))+lam2*w_hat;
    %grad_b = grad_b/n;
    
    w_hat = w_hat - eta*grad_w;
    %b_hat = b_hat - eta*grad_b;
    w_his = w_his + w_hat;
    % record the objective value based on the whole dataset
    z = y.*(X*w_hat+b_hat);
    f = (1-z).^2/(2*delta);
    id = z>1; f(id) = 0;
    id = z<1-delta; f(id) = 1-delta/2-z(id);
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


