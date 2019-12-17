%% testing different algorithms for distributed optimization on the following problem
%
%   {(1 / |E|) \sum_{ (i,j) \in E } (x_i - x_j)^2 } + { (1/|V|) (delta/2) * \sum_{i \in V} (x_i - t)^2    }
%
%
% where each x_i is a scalar and G = (V, E) is a graph, and delta is a
% parameter that we can choose to control the strong convexity of the
% overall objective


%% set the parameters of our problem, the graph and the delta
%global E1 E2 delta numE target

dim = 30; % number of nodes in the graph
G = rand(dim) > 0.5; % random matrix
G = graph(triu(G,1) + triu(G,1)'); % undirected random E-R graph. This is not a matrix. It is a graph object.

% get a few matrices from the graph G
Adj_G = adjacency(G);
Lap_G = laplacian(G);
Inc_G = abs(incidence(G)); % it seems like Matlab orders edges based on the "lexicographical" indices. So edge (1,2) comes before edge (2,3). Also, there is no edge (2,1) in G. Also, it seems like the incidence matlab function puts signs on the elements, even for undirected graphs. that is why we use the abs() outside of it.
Adj_line_G = Inc_G'*Inc_G - 2*eye(G.numedges); % the relation between the line graph and the incidence matrix is well known. see e.g. https://en.wikipedia.org/wiki/Incidence_matrix#Undirected_and_directed_graphs

line_G = graph(Adj_line_G);
Lap_line_G = laplacian(line_G);

% we make a simple choice for the Gossip matrix, it being equal to the Laplacian of our line graph
W = Lap_line_G;


E1 = G.Edges.EndNodes(:,1);
E2 = G.Edges.EndNodes(:,2);

E1line = line_G.Edges.EndNodes(:,1);
E2line = line_G.Edges.EndNodes(:,2);
%the code above makes sure that E1 always has the smaller indices and that E2 the larger indices


numE = G.numedges;
numEline = line_G.numedges;


delta = 0.001;
delta = delta / dim; % it makes sense to scale delta with 1/dim so that both terms in our objective have the same order of magnitude as the graph grows
target = -0.342; 

%% set some parameters to be used in algorithm 1, 2 and 3 from Scaman et al. 2017 and 2018
% Y = rand(dim, numE)*real(sqrtm(full(W)));
% X = Y;
% 
% Theta = repmat(rand(dim,1),1,numE);
% Theta_old = Theta;
% 
% num_iter = 10000;
% 
% alpha = delta;
% beta  = 2 + delta;
% 
% % the choice of the following values is according to Scaman et al. 2017, "Optimal algorithms for smooth and strongly convexdistributed optimization in networks"
% kappa_l = beta / alpha; 
% specW = sort(eig(W)); 
% eta_1 = alpha/specW(end); 
% gamma = specW(2)/specW(end);
% mu_1 = (sqrt(kappa_l) - sqrt(gamma)) / (sqrt(kappa_l) + sqrt(gamma));
% c1 = (1 - sqrt(gamma))/ (1 + sqrt(gamma));
% c2 = (1 + gamma) / (1 - gamma);
% c3 = 2/ ((1+gamma)*specW(end));
% K = floor(1 / sqrt(gamma));
% eta_2 = alpha*(1 + c1^(2*K))/((1 + c1^K)^2);
% mu_2 = ((1 + c1^K)*sqrt(kappa_l) - 1 + c1^K) / ((1 + c1^K)*sqrt(kappa_l) + 1 - c1^K);
% 
% 
% R = 1; 
% L_is = (2 + delta)*ones(numE , 1);
% L_l = norm(L_is,2)/sqrt(numE);
% 
% fixing_factor = 3;
% 
% eta_3  = ((1 - c1^K)/(1 + c1^K))*(numE*R/L_l);
% sigma = (1/fixing_factor)*(1/eta_3)*(1 + c1^(2*K))/((1 - c1^K)^2); %note that there is a typo in the arxiv paper "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks" in the specificaion of the Alg 2. In the definition of sigma, tau should be eta
% sum_Theta = 0;
% 
% M = num_iter;
% eps = 4*R*L_l/num_iter; % %note that there is a typo in the arxiv paper "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks" in the specificaion of the Alg 2. In the definition of T. It should be T = 4 R L_l / eps
% W_Acc = AccGoss(eye(numE), W, K,c2,c3);


%% choose the algorithm and run it

Alg_name = 6;

evol = [];

if (Alg_name == 0) %Gradient descent
    X = rand(dim,1);
    alf = 1;
    num_iter = 2000000;
    evol = grad_desc_cann_prob(X, alf, num_iter, Lap_G, numE , delta, target);
    plot( [ 1 : num_iter  ] , evol');
    drawnow;    
end

if (Alg_name == 1) %Alg 1: "Optimal algorithms for smooth and strongly convex distributed optimization in networks"
    
    Y_init = rand(dim, numE);
    Theta_init = rand(dim,1);
    alpha = delta;
    beta  = 2 + delta;
    num_iter = 10000;
    
    evol = alg_1_Scaman_17_cann_prob(Y_init, Theta_init, @GradConjF, num_iter,dim, numE , Lap_line_G, delta, E1,E2,target,alpha, beta);
    plot( [ 1 : num_iter  ] , evol');
    drawnow;
 
end

if (Alg_name == 2) %Alg 2: "Optimal algorithms for smooth and strongly convex distributed optimization in networks"
    
    Y_init = rand(dim, numE);
    Theta_init = rand(dim,1);
    alpha = delta;
    beta  = 2 + delta;
    num_iter = 10000;
    
    evol = alg_2_Scaman_17_cann_prob(Y_init, Theta_init , @GradConjF, @AccGoss, num_iter, numE , Lap_line_G, delta, E1,E2,target,alpha, beta);
    
    plot([1:K:num_iter*K],evol'); % each iteration corresponds to K gossip steps
    drawnow;
    
end

if (Alg_name == 3) % Alg 2: "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks"
    
    Y_init = rand(dim, numE);
    Theta_init = rand(dim,1);
    
    L_is = (2 + delta)*ones(numE , 1);
    R = 1;
    num_iter = 10000;
    
    evol = alg_2_Scaman_18_cann_prob(Y_init, Theta_init , @ProxF, @AccGoss, num_iter, numE , Lap_line_G, delta, E1,E2,target,L_is, R);
    
    plot([1:K:num_iter*K],evol'); % each iteration corresponds to K gossip steps
    drawnow;

end

if (Alg_name == 4) % Alg in Table 1: "Distributed Optimization Using the Primal-Dual Method of Multipliers"
   
    X_init = randn(dim,numE,1);
    U_init = randn(dim,numE,numE);
    
    rho = 0.00001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.1;
    num_iter = 1000;
    
    evol = PDMM_cann_prob(X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G, @ProxF ,  E1, E2, delta, target);
    
    plot([1:1:num_iter],evol'); 
    drawnow;
    
end


if (Alg_name == 5) % Consensus ADMM of the form (1/numE)* sum_e f_e(x_e) subject to x_e = Z_(e,e') and x_e' = Z_(e,e') if (e,e') is in the line graph.
   
    X_init = randn(dim,numE,1);
    U_init = randn(dim,numE,numE);
    
    rho = 0.000001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.1;
    num_iter = 1000;
    
    evol = ADMM_edge_Z_edge_cann_prob(X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G, @ProxF ,  E1, E2, delta, target);
    
    plot([1:1:num_iter*1],evol'); 
    drawnow;
        
end


if (Alg_name == 6) % Consensus ADMM of the form (1/numE)* sum_e f_e(x_e) subject to x_e = x_e' if (e,e') is in the line graph. The difference between this algorithm and the one above (Alg 5) is that here we do not use the consensus variable Z_(e,e') in the augmented lagrangian

    X_init = randn(dim,numE);
    U_init = randn(dim,numEline);
    
    rho = 0.00001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 1;
    num_iter = 1000;
   
    evol = ADMM_edge_edge_no_Z_cann_prob(X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G, @ProxF ,numEline, E1line, E2line, E1, E2, delta, target);
    
    plot([1:1:num_iter*1],evol'); 
    drawnow;
        
end


if (Alg_name == 7) % Consensus ADMM of the form (1/numE)* sum_( e = (i,j) \in E) f_e(x_ei,x_ej) subject to x_ei = z_i if i touches edges e in the graph G.
    
    X_init = randn(2,numE);
    Z_init = randn(dim,1);
    U_init = randn(2,numE); 
    
    U_old = U;
    rho = 0.0001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.1;
    num_iter = 1000;
   
    evol = ADMM_node_Z_node_cann_prob(X_init, U_init, Z_init, rho, alp,dim, numE, num_iter, @ProxFPair , E1, E2, delta, target);
    
    plot([1:1:num_iter*1],evol'); 
    drawnow;
        
end


% this function computes the gradient of conjugate of the i-th function in the objective that we are trying to optimize
function [GRAD] = GradConjF(X, i, delta, E1,E2,target)
    %global delta E1 E2 target
    d = delta;
    X = X + target*d*ones(length(X),1);
    GRAD = (X)/delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (1/(2*d + d*d))*(  -X(E1(i)) + X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (1/(2*d + d*d))*(   X(E1(i)) - X(E2(i))  );
end

% this is the gradient for the function
% (0.5 * (x_i - x_j)^2 + 0.5*delta*(X - target)^2 )
function [GRAD] = GradF(X, i,delta,E1,E2,target)
    %global delta E1 E2 target
    d = delta;
    GRAD = (X)*delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (   X(E1(i)) - X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (  -X(E1(i)) + X(E2(i))  );

    GRAD = GRAD - d*target*ones(length(X),1);
end

% this is the proximal operator 
% Prox(N) = argmin_X    (1/numE) (0.5 * (x_i - x_j)^2 + 0.5*delta*(X - target)^2 )+ 0.5*rho* (X - N)^2
function [X_out] = ProxF(N, e, rho, E1, E2, numE, delta, target)
    %global E1 E2 numE delta target
    
    N = N*numE*rho + delta*target;
    
    d = ( rho*numE + delta );
    
    X_out = (N)/d;
    
    X_out(E1(e)) = X_out(E1(e)) + (1/(2*d + d*d))*(  -N(E1(e)) + N(E2(e))  );
    X_out(E2(e)) = X_out(E2(e)) + (1/(2*d + d*d))*(   N(E1(e)) - N(E2(e))  );
end


% this is an approximation for the proximal operator 
% Prox(N) = argmin_X    (1/numE) (0.5 * (x_i - x_j)^2 + 0.5*delta*(X - target)^2 )+ 0.5*rho* (X - N)^2
% using gradient descent
function [X_out] = AppProxF(X, i, rho, M, alpha,numE)
    %global numE

    X_out = X;
    for m = 1:M
        X_out = X_out - (2*alpha/(m+2))*( (1/numE)*GradF( X_out , i, delta,E1,E2,target)  + rho*(X_out - X)     );
    end
    
end

% this is the proximal operator that minimizes over x_i and x_j the
% function (0.5 * (x_i - x_j)^2 + 0.5*delta*numE*inv_deg_i(x_i - target)^2 + 0.5*delta*numE**inv_deg_j(x_j - target)^2 )+ 0.5*rho* (x_i - N_i)^2  + 0.5*rho* (x_j - N_j)^2
function [X_out] = ProxFPair(N, e, rho,E1,E2,numE,delta,target)
    %global E1 E2 numE delta target
    
    i = E1(e); j = E2(e);
    degi = length(find( E1 == i | E2 == i));
    degj = length(find( E1 == j | E2 == j));
    inv_deg_i = 1/degi;
    inv_deg_j = 1/degj;
    
    a = 1 + delta*numE*inv_deg_i + rho;
    b = 1 + delta*numE*inv_deg_j + rho;
    c = rho*N(1) + delta*numE*inv_deg_i*target;
    d = rho*N(2) + delta*numE*inv_deg_j*target;
    
    X_out(1) = (1/(a*b - 1))*( b*c + d );
    X_out(2) = (1/(a*b - 1))*( c + d*a );
    
end



function [Y] = AccGoss(X, W, k, c2, c3)

    I = eye(length(X));
    a_old = 1;
    a = c2;
    x_0 = X;
    x_old = X;
    x = c2*X*(I -  c3*W);

    for t = 1:k-1

        a_old_old = a_old;
        a_old = a;
        a = 2*c2*a - a_old_old;

        x_old_old = x_old;
        x_old = x;
        x = 2*c2*x*(I - c3*W) - x_old_old;

    end

    Y =  x_0 - x/a;
end