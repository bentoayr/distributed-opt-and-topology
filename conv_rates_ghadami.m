%% here we are assuming that kappa = 1, W_I = W_O = W/2 and W_i,(ij) = 1/deg_i such that Assumption 1 is satisfied

%% cycle graph
A = zeros(6);
A(1,2) = 1; 
A(2,3) = 1;
A(3,4) = 1;
A(4,5) = 1;
A(5,6) = 1;
A(6,1) = 1;
G = A + A';
%plot(graph(G))
deg = sum(G);
D = zeros(length(A));
for i = 1:length(A)
    D(i,i) = 1 + sum(1./deg(find(G(:,i))));
end
P = diag(1./deg)*G + G*diag(1./deg);
leig = eig(P,D);
l_min = min(leig);
s_max = max(leig(leig < max(leig)));
beta = 1/(  1  +  sqrt( 1  - s_max^2 )  );
alpha_s = 4 / ( 2 -  (l_min +  s_max)*beta + sqrt( (l_min^2)*(beta^2)  - 2*beta + 1  )   );
rho_s = 1/sqrt(1 - s_max);
rate = 1 - 0.5*( 1 -  s_max*beta);

% going to check if the actual math checks out. We implement the algorithm that is in the paper and compare convergence rates
numV = length(A);
numE = sum(sum(G))/2;
X0 = rand(length(A),1);
numiter = 10000;
evol = ADMM_edge_variable(G,numV,numE,numiter,X0,rho_s,alpha_s);
rate_est = estimate_rate(evol, X0);
disp([rate,rate_est]);

%% almost complete
A = zeros(5);
A(1,2) = 1; 
A(2,3) = 1;
A(3,4) = 1;
A(4,1) = 1;
A(5,2) = 1;
A(5,3) = 1;
G = A + A';
%plot(graph(G))
deg = sum(G);
D = zeros(length(A));
for i = 1:length(A)
    D(i,i) = 1 + sum(1./deg(find(G(:,i))));
end
P = diag(1./deg)*G + G*diag(1./deg);
leig = eig(P,D);
l_min = min(leig);
s_max = max(leig(leig < max(leig)));
beta = 1/(  1  +  sqrt( 1  - s_max^2 )  );
alpha_s = 4 / ( 2 -  (l_min +  s_max)*beta + sqrt( (l_min^2)*(beta^2)  - 2*beta + 1  )   );
rho_s = 1/sqrt(1 - s_max);
rate = 1 - 0.5*( 1 -  s_max*beta);

% going to check if the actual math checks out. We implement the algorithm that is in the paper and compare convergence rates
numV = length(A);
numE = sum(sum(G))/2;
X0 = rand(length(A),1);
numiter = 10000;
evol = ADMM_edge_variable(G,numV,numE,numiter,X0,rho_s,alpha_s);
rate_est = estimate_rate(evol, X0);
disp([rate,rate_est]);

%% complete graph
A = ones(4);
A = A - eye(4);
G = A;
%plot(graph(G))
deg = sum(G);
D = zeros(length(A));
for i = 1:length(A)
    D(i,i) = 1 + sum(1./deg(find(G(:,i))));
end
P = diag(1./deg)*G + G*diag(1./deg);
leig = eig(P,D);
l_min = min(leig);
s_max = max(leig(leig < max(leig)));
alpha_s = 4 / (2 - l_min);
rho_s = 1;
rate = (-l_min) / (2 - l_min);

% going to check if the actual math checks out. We implement the algorithm that is in the paper and compare convergence rates
numV = length(A);
numE = sum(sum(G))/2;
X0 = rand(length(A),1);
numiter = 10000;
evol = ADMM_edge_variable(G,numV,numE,numiter,X0,rho_s,alpha_s);
rate_est = estimate_rate(evol, X0);
disp([rate,rate_est]);

%% tree graph
A = zeros(6);
A(1,2) = 1;
A(1,3) = 1;
A(1,4) = 1;
A(4,5) = 1;
A(4,6) = 1;
A(5,6) = 1;
G = A + A';
%plot(graph(G))
deg = sum(G);
D = zeros(length(A));
for i = 1:length(A)
    D(i,i) = 1 + sum(1./deg(find(G(:,i))));
end
P = diag(1./deg)*G + G*diag(1./deg);
leig = eig(P,D);
l_min = min(leig);
s_max = max(leig(leig < max(leig)));
beta = 1/(  1  +  sqrt( 1  - s_max^2 )  );
alpha_s = 4 / ( 2 -  (l_min +  s_max)*beta + sqrt( (l_min^2)*(beta^2)  - 2*beta + 1  )   );
rho_s = 1/sqrt(1 - s_max);
rate = 1 - 0.5*( 1 -  s_max*beta);

% going to check if the actual math checks out. We implement the algorithm that is in the paper and compare convergence rates
numV = length(A);
numE = sum(sum(G))/2;
X0 = rand(length(A),1);
numiter = 10000;
evol = ADMM_edge_variable(G,numV,numE,numiter,X0,rho_s,alpha_s);
rate_est = estimate_rate(evol, X0);
disp([rate,rate_est]);



%% auxiliary functions

function rate = estimate_rate(evol, X0)

    points = log(sqrt(sum(abs(evol - repmat(mean(X0),1,size(evol,2))).^2,1)))';
    t = find(points<0.9*min(points(points>-inf)),1);
    rate = exp((points(t) - points(ceil(t/2)))/(t - ceil(t/2)));

end

% this algorithm is not very well described in 
% E. Ghadimi, A. Teixeira, M. G. Rabbat and M. Johansson, "The ADMM algorithm for distributed averaging: Convergence rates and optimal parameter selection," 2014 48th Asilomar Conference on Signals, Systems and Computers, Pacific Grove, CA, 2014, pp. 783-787.
% it is better described  in
% arXiv:1303.6680 [math.OC]
function evol = ADMM_edge_variable(G,numV,numE,numiter,X0,rho,alpha)

    %rng(1);
    deg = sum(G);
    [E1,E2] = find(triu(G));

    U = randn(numV,numV); %U = (U + U')/2;
    Z = randn(numV,numV); Z = (Z + Z')/2;
    GM = randn(numV,numV); 
    X = randn(numV,1);
    evol = zeros(numV,numiter);

    for t = 1:numiter
        for i = 1:numV
            avg_n = 0;
            avg_d = 0;
            for j = find(G(:,i))'
                avg_n = avg_n + (1/deg(i))*(  Z(i,j) - U(i,j)    );
                avg_d = avg_d + (1/deg(i));
            end
            X(i) = (X0(i) + rho*avg_n)/(1 + rho*avg_d);
        end
        for e = 1:numE

            i = E1(e); j = E2(e);
            GM(i,j) = alpha*X(i) + (1 - alpha)*Z(i,j); 
            GM(j,i) = alpha*X(j) + (1 - alpha)*Z(i,j);
            
            Z(i,j) =  ( (1/deg(i))*(  GM(i,j) + U(i,j) ) + (1/deg(j))*(  GM(j,i) + U(j,i)  ) ) / ( (1/deg(i)) + (1/deg(j)) ); Z(j,i) = Z(i,j);

            U(i,j) = U(i,j) + GM(i,j) - Z(i,j); 
            U(j,i) = U(j,i) + GM(j,i) - Z(i,j);
        end
    
        evol(:,t) = X;
    end

end
