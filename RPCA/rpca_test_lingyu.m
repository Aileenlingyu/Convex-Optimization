clear; close all;
load('escalator_data')
Xtrain = double(X);
maxit = 40; tol = 1e-5;
[p,q] = size(Xtrain);
%%
opts = [];
opts.tol = tol; opts.maxit = maxit;
opts.beta = 1e-15; opts.lambda = 5e-3;
t0 = tic;
opts.L0 = abs(randn(p,q));
opts.S0 = abs(randn(p,q));
[W_prox,H_prox,Out_prox] = rpca_prox_lingyu(Xtrain,opts);
time = toc(t0);
fprintf('Projected Gradient for Robust PCA time = %5.4f, objective value = %20.16f\n\n',time,Out_prox.hist_obj(end));
%%
opts = [];
opts.tol = tol; opts.maxit = maxit;
opts.beta = 1e-15; opts.lambda = 5e-3;
t0 = tic;
opts.L0 = abs(randn(p,q));
opts.S0 = abs(randn(p,q));
[W_am,H_am,Out_am] = rpca_am_lingyu(Xtrain,opts);
time = toc(t0);
fprintf('Alternating minimization for Robust PCA time = %5.4f, objective value = %20.16f\n\n',time,Out_am.hist_obj(end));
%% plot results
figure;
plot(Out_prox.hist_obj,'c-','linewidth',2);
hold on;
plot(Out_am.hist_obj,'r-','linewidth',2);
hold on;
legend('Proximal gradient method','Alternating minimization method','location','best');
xlabel('number of iteration','fontsize',14);
ylabel('objective values','fontsize',14);
title('escalator_data Dataset','fontsize',14);