clear; close all;
load('Swimmer')
Xtrain = reshape(Swimmer,[1024,256]);
lam1 = 0.0001; lam2 = 0.0001;
maxit = 1500; tol = 1e-5;
[m,n] = size(Xtrain);
r=17;
%%
opts = [];
opts.tol = tol; opts.maxit = maxit;
t0 = tic;
opts.H0 = abs(randn(r,n));
opts.W0 = abs(randn(m,r));
[W_pg,H_pg,Out_pg] = nmf_pg_lingyu(Xtrain,opts);
time = toc(t0);
fprintf('Projected Gradient for NMF with r = %3.2f: time = %5.4f, objective value = %20.16f\n\n',r,time,Out_pg.hist_obj(end));
%%
opts = [];
opts.tol = tol; opts.maxit = maxit;
t0 = tic;
opts.H0 = abs(randn(r,n));
opts.W0 = abs(randn(m,r));
[W_am,H_am,Out_am] = nmf_am_lingyu(Xtrain,opts);
time = toc(t0);
fprintf('Alternating minimization for NMF with r = %3.2f: time = %5.4f, objective value = %20.16f\n\n',r,time,Out_am.hist_obj(end));
%%
opts = [];
opts.tol = tol; opts.maxit = maxit;
t0 = tic;
opts.H0 = abs(randn(r,n));
opts.W0 = abs(randn(m,r));
[W_alterpg,H_alterpg,Out_alterpg] = nmf_alterpg_lingyu(Xtrain,opts);
time = toc(t0);
fprintf('Alternating Proximal gradient for NMF with r = %3.2f: time = %5.4f, objective value = %20.16f\n\n',r,time,Out_alterpg.hist_obj(end));
%% plot results
figure;
plot(Out_pg.hist_obj,'c-','linewidth',2);
hold on;
plot(Out_am.hist_obj,'r-','linewidth',2);
hold on;
plot(Out_alterpg.hist_obj,'g-','linewidth',2);
hold on;
% plot(Out_saga.hist_obj,'b-','linewidth',2);
% hold on;
legend('Projected gradient method','Alternating minimization method','Alternating proximal gradient method','location','best');
% set(gca,'fontsize',14);
xlabel('number of iteration','fontsize',14);
ylabel('objective values','fontsize',14);
title('Swimmer Dataset','fontsize',14);