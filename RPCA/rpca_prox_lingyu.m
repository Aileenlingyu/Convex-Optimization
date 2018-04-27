%% set up problem
% min_L,S ||L||* + \lambda ||S||_1 + 0.5*\beta*||L+S-X||_2^2
function [L,S,Out] = rpca_prox_lingyu(X,opts)

% We use H to represent H^T in the theory part
if isfield(opts,'maxit') maxit = opts.maxit; end;
if isfield(opts,'tol')   tol   = opts.tol;   end;
if isfield(opts,'L0')    L0 = opts.L0;       end;
if isfield(opts,'S0')    S0 = opts.S0;       end;
if isfield(opts,'beta')  beta = opts.beta;       end;
if isfield(opts,'lambda')  lambda = opts.lambda;       end;
eta = 0.0001;
L=L0;S=S0;
p = 20800;
q = 200;
hist_obj = zeros(maxit,1);
%% run proximal gradient
alpha = 1/1;
for k = 1:maxit
    gradS = beta*(L+S-X);
    yS = S - alpha*gradS;
    S = sign(yS).*max(0, abs(yS)-lambda*alpha);
    [U,D,V] = svd(L,0);
    gradL = U*V' + beta*(L+S-X);
    yL = L - alpha*gradL;
    L = sign(yL).*max(0, abs(yL)-lambda*alpha);
    hist_obj(k) = trace(sqrt(L'*L)) + lambda* norm(S,1)+ 0.5*beta*(norm(L+S-X,'fro')^2);
end

for sub = 1:q
    sub_img = reshape(L(:,sub),[130,160]);
    imwrite(mat2gray(sub_img),strcat('/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/optimization/HW/hw3_sol_lingyu/output/rpca_prox_lingyu/L_Col_',int2str(sub),'.jpg'));
end
for sub = 1:q
    sub_img = reshape(S(:,sub),[130,160]);
    imwrite(mat2gray(sub_img),strcat('/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/optimization/HW/hw3_sol_lingyu/output/rpca_prox_lingyu/S_Col_',int2str(sub),'.jpg'));
end
L = L; S=S;
Out.iter = k; Out.hist_obj = hist_obj(1:k); 
Out.L = L; Out.S = S; 
