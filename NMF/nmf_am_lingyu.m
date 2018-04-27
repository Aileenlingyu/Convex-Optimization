%% set up problem
% min_W,H 0.5*||W*H-X||_2^2
function [W,H,Out] = nmf_am_lingyu(X,opts)

% We use H to represent H^T in the theory part
if isfield(opts,'maxit') maxit = opts.maxit; end;
if isfield(opts,'tol')   tol   = opts.tol;   end;
if isfield(opts,'H0')    H0 = opts.H0;       end;
if isfield(opts,'W0')    W0 = opts.W0;       end;
eta = 0.0001;
W=W0;H=H0;
hist_obj = zeros(maxit,1);
r=17;n=256;m=1024;
%% For alternating minimization based on accelerated proximal gradient
Hhat =H; H0 = H;
What =W; W0 = W;
t0H = ones(n,1);
t0W = ones(m,1);
L = norm(W)^2;
alpha = 1/L;
hist_obj_apg=zeros(1,256,(maxit+1));
for c=1:n
    hist_obj_apg(:,c,1) = .5*norm(W*H(:,c)-X(:,c))^2;
end
gradH = zeros(r,n);
gradW = zeros(m,r);
t1H = zeros(n,1);
t1W = zeros(m,1);
for k = 1:maxit
    for c =1:n
        %fprintf('Update parameter H, Iteration = %3.2f: coordinate= %5.4f\n\n',k,c);
        gradH(:,c) = W'*(W*Hhat(:,c) - X(:,c));
        yH(:,c) = Hhat(:,c) - alpha*gradH(:,c);
        H(:,c) = sign(yH(:,c)).*max(0, abs(yH(:,c)));
%         obj(c) = .5*norm(W*H(:,c)-X(:,c))^2;
%         size(hist_obj_apg(:,c))
%         size(obj(c))
%         hist_obj_apg(:,c,(k+1)) = obj(c);
        t1H(c,:) = (1+sqrt(1+4*t0H(c,:)^2))/2;
        w(c,:) = (t0H(c,:)-1)/t1H(c,:);
        Hhat(:,c) = H(:,c) + w(c,:)*(H(:,c)-H0(:,c));
        t0H(c,:) = t1H(c,:); H0(:,c) = H(:,c);
    end
    for c =1:m
        %fprintf('Update parameter W, Iteration = %3.2f: coordinate= %5.4f\n\n',k,c);
        gradW(c,:) = (What(c,:)*Hhat - X(c,:))*Hhat';
        yW(c,:) = What(c,:) - alpha*gradW(c,:);
        W(c,:) = sign(yW(c,:)).*max(0, abs(yW(c,:)));
        t1W(c,:) = (1+sqrt(1+4*t0W(c,:)^2))/2;
        w(c,:) = (t0W(c,:)-1)/t1W(c,:);
        What(c,:) = W(c,:) + w(c,:)*(W(c,:)-W0(c,:));
        t0W(c,:) = t1W(c,:); W0(c,:) = W(c,:);
    end
    hist_obj(k) = 0.5*(norm(X-What*Hhat,'fro')^2);
end
for sub = 1:r
    sub_img = reshape(What(:,sub),[32,32]);
    imwrite(mat2gray(sub_img),strcat('/Users/lingyuzhang/Spring17/LuckyLucky18/1Spring/1RPI/Course/optimization/HW/hw3_sol_lingyu/output/nmf_am_lingyu/Col_',int2str(sub),'.jpg'));
end
W = What; H=Hhat;
Out.iter = k; Out.hist_obj = hist_obj(1:k); 
Out.H = Hhat; Out.W = W; 
