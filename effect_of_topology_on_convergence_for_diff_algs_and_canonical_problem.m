%% testing different algorithms for distributed optimization
%

% random graph
global E1 E2 delta numE target

dim = 30;
G = rand(dim) > 0.5;
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

%[E1, E2] = find(triu(Adj_G,1));

E1 = G.Edges.EndNodes(:,1);
E2 = G.Edges.EndNodes(:,2);
%the code above makes sure that E1 always has the smaller indices and that E2 the larger indices


%E1E2 = sort([E1, E2] , 2);
%E1 = E1E2(:,1);E2 = E1E2(:,2); %makes sure E1 always has the smaller indices and E2 the larger indices

numE = G.numedges;
numEline = line_G.numedges;


delta = 0.001;
target = -0.342; 

% Scaman et al. 2017, Algorithm 1 and Algorithm 2
Y = rand(dim, numE)*real(sqrtm(full(W)));
X = Y;

Theta = repmat(rand(dim,1),1,numE);
Theta_old = Theta;

num_iter = 10000;

alpha = delta;
beta  = 2 + delta;

% the choice of the following values is according to Scaman et al. 2017, "Optimal algorithms for smooth and strongly convexdistributed optimization in networks"
kappa_l = beta / alpha; 
specW = sort(eig(W)); 
eta_1 = alpha/specW(end); 
gamma = specW(2)/specW(end);
mu_1 = (sqrt(kappa_l) - sqrt(gamma)) / (sqrt(kappa_l) + sqrt(gamma));
c1 = (1 - sqrt(gamma))/ (1 + sqrt(gamma));
c2 = (1 + gamma) / (1 - gamma);
c3 = 2/ ((1+gamma)*specW(end));
K = floor(1 / sqrt(gamma));
eta_2 = alpha*(1 + c1^(2*K))/((1 + c1^K)^2);
mu_2 = ((1 + c1^K)*sqrt(kappa_l) - 1 + c1^K) / ((1 + c1^K)*sqrt(kappa_l) + 1 - c1^K);


R = 1; 
L_is = (2 + delta)*ones(numE , 1);
L_l = norm(L_is,2)/sqrt(numE);

fixing_factor = 3;

eta_3  = ((1 - c1^K)/(1 + c1^K))*(numE*R/L_l);
sigma = (1/fixing_factor)*(1/eta_3)*(1 + c1^(2*K))/((1 - c1^K)^2); %note that there is a typo in the arxiv paper "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks" in the specificaion of the Alg 2. In the definition of sigma, tau should be eta
sum_Theta = 0;

M = num_iter;
eps = 4*R*L_l/num_iter; % %note that there is a typo in the arxiv paper "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks" in the specificaion of the Alg 2. In the definition of T. It should be T = 4 R L_l / eps
W_Acc = AccGoss(eye(numE), W, K,c2,c3);

% tmp = 0*randn(dim,1);
% [X_out] = ProxF(tmp, 2, 0.4);
% [X_out_2] = AppProxF(tmp, 2, 0.4,1000,eta_3);
% [norm(X_out-X_out_2), norm(X_out-tmp)]
% X_out
% return;


Alg_name = 0;

evol = [];

if (Alg_name == 0) %Gradient descent
    alf = 0.1;
    for t = 1:num_iter
        X = X - alf*(Lap_G*X + delta*(X-target));
        evol = [evol, log(norm( X - target ,1)) ];
        plot( [ 1 : t  ] , evol');
        drawnow;
    end
end

if (Alg_name == 1) %Alg 1: "Optimal algorithms for smooth and strongly convex distributed optimization in networks"
    for t = 1:num_iter
        for e = 1:numE 
            Theta(:,e) = GradConjF( X(:,e) , e );
        end
        Y_old = Y;
        Y = X - eta_1*Theta*W;
        X = (1 + mu_1)*Y - mu_1*Y_old;
        evol = [evol, log(norm( Theta(:,:) - target ,1)) ];
        plot( [ 1 : t  ] , evol');
        drawnow;
    end
end

if (Alg_name == 2) %Alg 2: "Optimal algorithms for smooth and strongly convex distributed optimization in networks"
    for t = 1:num_iter
        for e = 1:numE
            Theta(:,e) = GradConjF( X(:,e) , e );
        end
        Y_old = Y;
        Y = X - eta_2*AccGoss(X, W, K, c2, c3);
        X = (1 + mu_2)*Y - mu_2*Y_old;
        evol = [evol, log(norm( Theta(:,:) - target ,1)) ];
        plot([1:K:t*K],evol'); % each iteration corresponds to K gossip steps
        drawnow;
    end
end

if (Alg_name == 3) % Alg 2: "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks"
    
    if (norm(full(W_Acc),2)*sigma*eta_3 > 1)
        disp(['Convergene condition not met because ', num2str(norm(full(W_Acc),2)*sigma*eta_3), ' is bigger than 1']);
        %return;
    end
       
    
    for t = 1:num_iter

        Y = Y - sigma*AccGoss(2*Theta - Theta_old, W, K,c2,c3);

        Theta_old = Theta;
        Theta_tilde = Theta;
        for e = 1:numE

            Theta_tilde(:,e) = ProxF( eta_3*Y(:,e) + Theta(:,e) , e, 1/eta_3 );
            %Theta_tilde(:,e) = AppProxF( eta_3*Y(:,e) + Theta(:,e) , e, 1/eta_3 , num_iter, eta_3);
            %disp(   norm(   Theta_tilde(:,e) - tmp  )  );

        end
        Theta = Theta_tilde;
        
        sum_Theta = sum_Theta + sum(Theta,2)/numE; % the paper asks to compute the average in space and time
        evol = [evol, log(norm(sum_Theta/t - target)) ];
        plot([1:K:t*K],evol'); % each iteration corresponds to K gossip steps
        drawnow;
    end
end


if (Alg_name == 4) % Alg in Table 1: "Distributed Optimization Using the Primal-Dual Method of Multipliers"
   
    X = randn(dim,numE,1);
    XAve = zeros(dim,numE,1);
    U = randn(dim,numE,numE);
    U_old = U;
    rho = 0.00001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.1;
    
    for t = 1:num_iter
        X_old = X;
        U_old = U;

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            X(:,e,1) =   ProxF(  mean(  X_old(:,Neig_e,1) - U_old(:,Neig_e,e) , 2)   ,   e   , rho*length(Neig_e) );     
        end
        
        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            %U(:,e, Neig_e) = 0.5*U_old(:,e, Neig_e) +  0.5*permute(  X_old(:, Neig_e,1 )   -  repmat(X_old(:,e,1),1,length(Neig_e),1)   -   U_old(:,Neig_e,e)  , [1, 3, 2]);
            U(:,e, Neig_e) = U(:,e, Neig_e) + (- U(:,e, Neig_e) + permute( -U_old(:,Neig_e,e) - repmat(X(:,e,1),1,length(Neig_e),1) +  X_old(:, Neig_e,1 ), [1, 3, 2]));
        end
        
        XAve = XAve + X; % the paper asks to compute the average in space
        
        evol = [evol, log(norm( X    - target)) ];
        plot([1:1:t*1],evol'); % each iteration corresponds to K gossip steps
        drawnow;
    end
    
end


if (Alg_name == 5) % Consensus ADMM of the form sum_e f_e(x_e) subject to x_e = x_e' if (e,e') is in the line graph.
   
    X = randn(dim,numE,1);
    U = randn(dim,numE,numE);
    U_old = U;
    
    rho = 0.00001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.1;
    
    for t = 1:num_iter
        
        X_old = X;
        U_old = U;

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            X(:,e,1) =   ProxF(  mean( -permute(U(:,e,Neig_e),[1,3,2])  + 0.5*(X_old(:,Neig_e,1) + U_old(:,Neig_e,e) + permute(U_old(:,e,Neig_e),[1,3,2]) + repmat(X_old(:,e,1),1,length(Neig_e),1)) , 2)    ,   e   , rho*length(Neig_e) );     
        end
        
        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            U(:,e, Neig_e) = U(:,e, Neig_e) + alp*(0.5*(    -U_old(:,e, Neig_e)  + permute( -X(:,Neig_e,1)    -U_old(:,Neig_e,e) + repmat(X(:,e,1),1,length(Neig_e),1) , [1, 3, 2])    ));
        end
                
        evol = [evol, log(norm( X    - target)) ];
        plot([1:1:t*1],evol'); % each iteration corresponds to K gossip steps
        drawnow;
    end
    
end


if (Alg_name == 6)
    
    X = randn(2,numE);
    Z = randn(dim,1);
    U = randn(2,numE); 
    
    U_old = U;
    rho = 0.0001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.1;
    
    for t = 1:num_iter

        for e = 1:numE
            i = E1(e); j = E2(e);
            [X(:,e)] =  ProxFPair( [Z(i) - U(1,e);Z(j) - U(2,e)]   ,   e   , rho*numE );                
        end
        
        for i = 1:dim
            e1Neigh = find(E1 == i);
            e2Neigh = find(E2 == i);            
            Z(i) = (sum(X(1,e1Neigh) + U(1,e1Neigh),2) + sum(X(2,e2Neigh) + U(2,e2Neigh),2)) / (length(e1Neigh) + length(e2Neigh));
        end
       
        
        for e = 1:numE
            i = E1(e); j = E2(e);
            U(1, e) = U(1, e) + alp*( X(1,e)  - Z(i)  ); 
            U(2, e) = U(2, e) + alp*( X(2,e)  - Z(j)  ); 
        end
        
        err = log(norm( Z    - target));
        
        evol = [evol, err ];
        plot([1:1:t*1],evol'); % each iteration corresponds to K gossip steps
        %imagesc(X(:,:,1));
        drawnow;
    end
    
end


if (Alg_name == 7)
   
    X = randn(dim,numE,1);
    U = randn(dim,numE,2); 
    U_old = U;
    rho = 0.001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    
    for t = 1:num_iter
        X_old = X;

        for e = 1:numE 
            e1 = E1(e); e2 = E2(e); %this assumes that E1 always has the smaller indices and that E2 always contains the larger indices
            Neig_e1 = find( E1 == e1); Neig_e2 = find( E2 == e2); 
            Nm = mean(X_old(:,Neig_e1,1) + U_old(:,Neig_e1,1),2) + mean(X_old(:,Neig_e2,1) + U_old(:,Neig_e2,2),2) - 0.5*(U_old(:,e,1) + U_old(:,e,2)) - X_old(:,e,1);
            X(:,e,1) =   ProxF( Nm   ,   e   , 2*rho* 1);     
        end
        
        U_old = U;
        
        for e = 1:numE
            e1 = E1(e); e2 = E2(e); %this assumes that E1 always has the smaller indices and that E2 always contains the larger indices
            Neig_e1 = find( E1 == e1); Neig_e2 = find( E2 == e2); 
            %U(:,e, Neig_e) = permute(    U_old(:,Neig_e,e) - X_old(:, Neig_e,1 ) + repmat(X(:,e,1),1,length(Neig_e),1)   , [1, 3, 2])  ;
            U(:, Neig_e1,1) = U_old(:,Neig_e1,1) + (X(:,e,1) -  mean(X(:,Neig_e1,1) + U_old(:,Neig_e1,1),2));
            U(:, Neig_e2,2) = U_old(:,Neig_e2,2) + (X(:,e,1) -  mean( X(:,Neig_e2,1) + U_old(:,Neig_e2,2),2));
        end
        
        err = sum(sum(sum(abs(U - U_old))));
        err = log(norm( X(:,1,1)    - 1));
        
        evol = [evol, err ];
        plot([1:1:t*1],evol'); % each iteration corresponds to K gossip steps
        %imagesc(X(:,:,1));
        drawnow;
    end
    
end


% this function computes the gradient of conjugate of the i-th function in the objective that we are trying to optimize
function [GRAD] = GradConjF(X, i)
    global delta E1 E2 target
    d = delta;
    X = X + target*d*ones(length(X),1);
    GRAD = (X)/delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (1/(2*d + d*d))*(  -X(E1(i)) + X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (1/(2*d + d*d))*(   X(E1(i)) - X(E2(i))  );
end

% this is the gradient for the function
% (0.5 * (x_i - x_j)^2 + 0.5*delta*(X - target)^2 )
function [GRAD] = GradF(X, i)
    global delta E1 E2 target
    d = delta;
    GRAD = (X)*delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (   X(E1(i)) - X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (  -X(E1(i)) + X(E2(i))  );

    GRAD = GRAD - d*target*ones(length(X),1);
end

% this is the proximal operator 
% Prox(N) = argmin_X    (1/numE) (0.5 * (x_i - x_j)^2 + 0.5*delta*(X - target)^2 )+ 0.5*rho* (X - N)^2
function [X_out] = ProxF(N, e, rho)
    global E1 E2 numE delta target
    
    N = N*numE*rho + delta*target;
    
    d = ( rho*numE + delta );
    
    X_out = (N)/d;
    
    X_out(E1(e)) = X_out(E1(e)) + (1/(2*d + d*d))*(  -N(E1(e)) + N(E2(e))  );
    X_out(E2(e)) = X_out(E2(e)) + (1/(2*d + d*d))*(   N(E1(e)) - N(E2(e))  );
end


% this is an approximation for the proximal operator 
% Prox(N) = argmin_X    (1/numE) (0.5 * (x_i - x_j)^2 + 0.5*delta*(X - target)^2 )+ 0.5*rho* (X - N)^2
% using gradient descent
function [X_out] = AppProxF(X, i, rho, M, alpha)
    global numE

    X_out = X;
    for m = 1:M
        X_out = X_out - (2*alpha/(m+2))*( (1/numE)*GradF( X_out , i)  + rho*(X_out - X)     );
    end
    
end

% this is the proximal operator that minimizes over x_i and x_j the
% function (0.5 * (x_i - x_j)^2 + 0.5*delta*numE*inv_deg_i(x_i - target)^2 + 0.5*delta*numE**inv_deg_j(x_j - target)^2 )+ 0.5*rho* (x_i - N_i)^2  + 0.5*rho* (x_j - N_j)^2
function [X_out] = ProxFPair(N, e, rho)
    global E1 E2 numE delta target
    
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