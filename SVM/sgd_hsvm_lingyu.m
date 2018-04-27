function [b,w,Out] = sgd_hsvm_lingyu(X,y,opts)
[n,p] = size(X);
delta = 1; lam1 = 0.2; lam2 = 1; lam3 = 0;
maxit = 500; 
b0 = 0; w0 = zeros(p,1);
fix_b = 0;
if isfield(opts,'delta') delta = opts.delta; end;
if isfield(opts,'lam1')  lam1  = opts.lam1;  end;
if isfield(opts,'lam2')  lam2  = opts.lam2;  end;
if isfield(opts,'lam3')  lam3  = opts.lam3;  end;
if isfield(opts,'maxit') maxit = opts.maxit; end;
if isfield(opts,'b0')    b0 = opts.b0;       end;
if isfield(opts,'w0')    w0 = opts.w0;       end;
if isfield(opts,'accel') accel = opts.accel; end;
if isfield(opts,'fix_b') fix_b = opts.fix_b; end;

t0 = 1;
b_hat = b0; w_hat = w0; b = b0;

%get the Lipschitz constant
L_max = n+norm(X,'fro')^2;
L_max = L_max/(n*delta);
if isfield(opts,'L0') L = opts.L0; else L = 2*L_max/n; end;

%increase factor of L
eta = 1.5;

% do efficient computation for sparse solution
if nnz(w_hat)<=0.5*p
    z_hat = y.*(X*sparse(w_hat)+b_hat);
else
    z_hat = y.*(X*w_hat+b_hat);
end

%compute gradient and evaluate objective
grad_phi = (z_hat-1)/delta; 
f_hat = (1-z_hat).^2/(2*delta);
id = z_hat>1; grad_phi(id) = 0; f_hat(id) = 0;
id = z_hat<1-delta; grad_phi(id) = -1; f_hat(id) = 1-delta/2-z_hat(id);
f_hat = sum(f_hat/n);

grad_f = zeros(p+1,1);
grad_f(1) = y'*grad_phi; grad_f(2:end) = ((y.*grad_phi)'*X)';
grad_f = grad_f/n;

if nnz(w0)<=0.5*p
    z0 = y.*(X*sparse(w0)+b0);
else
    z0 = y.*(X*w0+b0);
end

f = (1-z0).^2/(2*delta);
id = z0>1; f(id) = 0;
id = z0<1-delta; f(id) = 1-delta/2-z0(id);
f = sum(f/n)+lam1*sum(abs(w0))+lam2/2*sum(w0.^2)+lam3/2*b0^2;
f0 = f;

hist_obj = zeros(maxit,1); 

for iter = 1:maxit
    
    if ~fix_b
        b = (L*b_hat-grad_f(1))/(L+lam3);
    end
    w_grad = L*w_hat-grad_f(2:end);
    w = sign(w_grad).*max(0,abs(w_grad)-lam1)/(L+lam2);
    
    % do efficient computation for sparse solution
    if nnz(w)<=0.5*p
        z = y.*(X*sparse(w)+b);
    else
        z = y.*(X*w+b);
    end
    f = (1-z).^2/(2*delta);
    id = z>1; f(id) = 0;
    id = z<1-delta; f(id) = 1-delta/2-z(id);
    f = sum(f/n);

    
    while L<L_max & f>f_hat+grad_f(1)*(b-b_hat)+...
            (w-w_hat)'*grad_f(2:end)+L/2*((b-b_hat)^2+sum((w-w_hat).^2))
        
        %update Lipschitz constant L
        L = min(eta*L,L_max);

        if ~fix_b
            b = (L*b_hat-grad_f(1))/(L+lam3);
        end
        w_grad = L*w_hat-grad_f(2:end);
        w = sign(w_grad).*max(0,abs(w_grad)-lam1)/(L+lam2);

        if nnz(w)<=0.5*p
            z = y.*(X*sparse(w)+b);
        else
            z = y.*(X*w+b);
        end
        
        f = (1-z).^2/(2*delta);
        id = z>1; f(id) = 0;
        id = z<1-delta; f(id) = 1-delta/2-z(id);
        f = sum(f/n);
    end
    
    
    f = f+lam1*sum(abs(w))+lam2/2*sum(w.^2)+lam3/2*b^2;
    hist_obj(iter) = f;        
    
    
    b_hat = b; w_hat = w; z_hat = z;
    b0 = b; w0 = w; f0 = f;
   

    %compute gradient and evaluate objective
    grad_phi = (z_hat-1)/delta; 
    f_hat = (1-z_hat).^2/(2*delta);
    id = z_hat>1; grad_phi(id) = 0; f_hat(id) = 0;
    id = z_hat<1-delta; grad_phi(id) = -1; 
    f_hat(id) = 1-delta/2-z_hat(id);
    f_hat = sum(f_hat/n);
    
    grad_f(1) = y'*grad_phi; grad_f(2:end) = ((y.*grad_phi)'*X)';
    grad_f = grad_f/n;
end
Out.iter = iter; Out.hist_obj = hist_obj(1:iter); 
Out.b = b; Out.w = w; Out.L = L;
