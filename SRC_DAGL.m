clear;
clc;

addpath(genpath('function\'));
addpath(genpath('dataset\'));

load("BBC.mat");
load("BBC_missing_0.9.mat");
num_view = size(fea,2);
X = cell(1,num_view);
for i = 1:num_view
    X{i}= fea{i}';
end

%finalconvergenceList = [];
%X = {data{1},data{2},data{3},data{4},data{5},data{6}};
% X = {fea{1}',fea{2}',fea{3}',fea{4}'};
%gt = double(truelabel{1});
ind_folds = folds{1};

clear Y;
cls_num = length(unique(gt));
V = length(X);

for v = 1:V
    %[X{v}] = NormalizeFea(X{v},0);
    X{v} = zscore(X{v});
end

ACCMax = 0;
result = [];
OuterItemTimes = 0;

for p1 = -10:2
    lambda1 = 2^p1;

        for p2 = -10:-8
            lambda2 = 2^p2;

n = size(X{1},2);
I = eye(n,n);
A = zeros(cls_num,cls_num);
for v = 1:V
    K{v} = diag(ones(1,n)./n);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%缺失
G = cell(1,V);                % mask matrix
X_missing = cell(1,V);        % 过滤了缺失部分的数据�?
for v = 1:V
    X_exist_missing = X{v};
    ind_0 = find(ind_folds(:,v) == 0);
    ind_1 = find(ind_folds(:,v) == 1);
    X_exist_missing(:,ind_0) = [];
    X_missing{v} = X_exist_missing;

    W0 = eye(size(X{v},2));
    W0(ind_0,:) = [];
    G{v} = W0;

    X_1 = X{v};
    X_1(:,ind_0) = 0;
    X_change{v} = X_1;
    W1 = eye(size(X_1,2));
    W1(:,ind_1) = [];
    N{v} = W1';

    X_exist{v} = X_1(:,ind_1);
    avg = mean(X_exist{v},2);
    for ij = 1:length(ind_0)
        P{v}(:,ij) = avg;
    end
end

for v = 1:V
    X{v} = X_change{v};
end

for v = 1:V
    X{v} = X{v} + P{v}*N{v};
end

n_ind_1 = size(ind_1,1);
n_ind_0 = size(ind_0,1);
I_N = eye(n_ind_0);

pn = 15;
for v = 1:V
    W_exist{v} = constructW_PKN(X_missing{v}, pn);
    W{v} = W_exist{v};
    D{v} = diag(sum(W{v}));
    L{v} = eye(n_ind_1) - D{v} ^ ( - 0.5) * W{v} * D{v} ^ ( - 0.5);
end
clear W;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%缺失

MAXiter = 200;
REPlic = 20;
F = zeros(n, cls_num);
F_exist = kmeans(X{1}', cls_num, 'start', 'sample', 'maxiter', MAXiter, 'replicates', REPlic, 'EmptyAction', 'singleton');
for i = 1:n
    F(i, F_exist(i)) = 1;
end

Isconverg = 0;
epson = 1e-7;
iter = 1;
Objlist = [];

tic;
convergenceList = [];

while(Isconverg == 0)
   

    %1 update W^v
    for v = 1:V
        Mw = 2*X{v}*(K{v}^2)*F*A';
        [U_Mw, ~, V_Mw] = svd(Mw,'econ');
        W{v} = U_Mw*V_Mw';
    end

    %2 update A^v
    Ma = 0;
    for v = 1:V
        Ma = Ma + 2*W{v}'*X{v}*(K{v}^2)*F;
    end
    [U_Ma, ~, V_Ma] = svd(Ma,'econ');
    A = U_Ma*V_Ma';

    %3 update F^v
    [F] = updateF(A, X, L, W, F, K, G, lambda1);

    %4 update P^v
    for v = 1:V
        temp1 = W{v}*A*F'*(K{v}^2)*N{v}' - X{v}*(K{v}^2)*N{v}';
        temp2 = N{v}*(K{v}^2)*N{v}' + lambda2*I_N;
        P{v} = temp1*pinv(temp2);
    end

    %5 update X^v
    for v = 1:V
        X{v} = X{v} + P{v}*N{v};
    end

    %6 update K^v
    for v = 1:V
        Y{v} = X{v} - W{v}*A*F';
        temp_Y = Y{v}.*Y{v}; 
        temp_Y_sum_of_col  = sum(temp_Y,1);
        temp_Y_sum_of_col_fnorm = sqrt(temp_Y_sum_of_col);
        

        numerator = 1./temp_Y_sum_of_col_fnorm;
        denomerator = sum(numerator);
        
        
        K{v}= diag((numerator./denomerator));
   %     K{v} = pinv(K{v});
    end

    %% coverge condition
    Isconverg = 1;

    for v = 1:V
        tempObj1 = norm((X{v} - W{v}*A*F')*K{v},"fro");
        ObjValue = tempObj1;
      %  Objlist = [Objlist; ObjValue];
    end

    for v = 1:V
        if (norm((X{v} - W{v}*A*F')*K{v},inf) > epson)
            history.norm_1 = norm((X{v} - W{v}*A*F')*K{v},inf);
            %fprintf('    norm_1 %7.10f    \n', history.norm_1);
            Isconverg = 0;
        end
    end
    currentResult = 0;
    tempValue1 = 0;
    tempValue2 = 0;
    tempValue3 = 0;
    for v = 1:V
        tempValue1 = tempValue1 + norm((X{v} - W{v}*A*F')*K{v})*norm((X{v} - W{v}*A*F')*K{v});
        tempValue2 = tempValue2 + lambda1 * trace(F'*G{v}'*L{v}*G{v}*F);
        tempValue3 = tempValue3 + lambda2 *norm(P{v})*norm(P{v});
    end
    currentResult = tempValue1 + tempValue2 + tempValue3;

   convergenceList = [convergenceList;currentResult];

    if (iter > 200)
        Isconverg  = 1;
    end

    iter = iter + 1;
    
end

F = NormalizeFea(F);
groups = kmeans(F, cls_num, 'maxiter', MAXiter, 'replicates', REPlic, 'EmptyAction', 'singleton');
result= Clustering8Measure(gt, groups)*100;
%finalconvergenceList = [finalconvergenceList, convergenceList];

OuterItemTimes = OuterItemTimes + 1;
if(ACCMax < result(1) && OuterItemTimes > 1)
    ACCMax = result(1);
end

fprintf('p1 %.5f  p2 %.5f  ACC %.5f  NMI %.5f  Purity %.5f  Fscore %.5f  Precision %.5f  Recall %.5f  AR %.5f  Entropy %.5f  ACCMax %.5f  \n',p1,p2,result,ACCMax);

% 运行结果写入
fid = fopen('BBCMiSSING09.txt', 'at');
fprintf(fid,'p1 %.5f  [2 %.5f  ACC %.5f  NMI %.5f  Purity %.5f  Fscore %.5f  Precision %.5f  Recall %.5f  AR %.5f  Entropy %.5f  ACCMax %.5f  \n',p1,p2,result,ACCMax);
fclose(fid);

toc;

        end

end
