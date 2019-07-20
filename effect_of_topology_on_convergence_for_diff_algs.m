%% testing different algorithms for distributed optimization

% random graph
global E1 E2 delta numE

dim = 30;
G = rand(dim) > 0.5;
G = graph(triu(G,1) + triu(G,1)');

Adj_G = adjacency(G);
Lap_G = laplacian(G);
Inc_G = incidence(G);
Adj_line_G = Inc_G'*Inc_G - 2*eye(G.numedges); % this does not seem correct to me.....negative values ?!

line_G = graph(Adj_line_G);
Lap_line_G = laplacian(line_G);
W = Lap_line_G;

[E1, E2] = find(triu(Adj_G,1));
E1E2 = sort([E1, E2] , 2);
E1 = E1E2(:,1);E2 = E1E2(:,2); %makes sure E1 always has the smaller indices and E2 the larger indices

numE = G.numedges;
numEline = line_G.numedges;
delta = 0.01;

% Scaman et al. 2017, Algorithm 1 and Algorithm 2
Y = rand(dim, numE)*real(sqrtm(full(W)));
X = Y;

Theta = repmat(rand(dim,1),1,numE);
Theta_old = Theta;

num_iter = 10000;

alpha = delta;
beta  = 2 + delta;
kappa_l = beta / alpha;
specW = sort(eig(W));
eta = alpha/specW(end);
gamma = specW(2)/specW(end);
mu = (sqrt(kappa_l) - sqrt(gamma)) / (sqrt(kappa_l) + sqrt(gamma));
c1 = (1 - sqrt(gamma))/ (1 + sqrt(gamma));
c2 = (1 + gamma) / (1 - gamma);
c3 = 2/ ((1+gamma)*specW(end));
K = floor(1 / sqrt(gamma));
eta_2 = alpha*(1 + c1^(2*K))/((1 + c1^K)^2);
mu_2 = ((1 + c1^K)*sqrt(kappa_l) - 1 + c1^K) / ((1 + c1^K)*sqrt(kappa_l) + 1 - c1^K);


R = 1; 
L_is = (2+delta)*ones(numE , 1);
L_l = norm(L_is,2)/sqrt(numE);

fixing_factor = 3;

eta_3  = ((1 - c1^K)/(1 + c1^K))*(numE*R/L_l);
sigma = (1/fixing_factor)*(1/eta_3)*(1 + c1^(2*K))/((1 - c1^K)^2);
sum_Theta = 0;

M = num_iter;
eps = 4*R*L_l/num_iter; % this is the target accuracy
W_Acc = AccGoss(eye(numE), W, K,c2,c3);

% tmp = 0*randn(dim,1);
% [X_out] = ProxF(tmp, 2, 0.4);
% [X_out_2] = AppProxF(tmp, 2, 0.4,1000,eta_3);
% [norm(X_out-X_out_2), norm(X_out-tmp)]
% X_out
% return;




Alg_name = 4;

evol = [];
    
if (Alg_name == 1)
    for t = 1:num_iter
        for e = 1:numE
            Theta(:,e) = GradConjF( X(:,e) , e );
        end
        Y_old = Y;
        Y = X - eta*Theta*W;
        X = (1 + mu)*Y - mu*Y_old;
        evol = [evol, log(norm( Theta(:,:) - 1 ,1)) ];
        plot( [ 1 : t  ] , evol');
        drawnow;
    end
end

if (Alg_name == 2)
    for t = 1:num_iter
        for e = 1:numE
            Theta(:,e) = GradConjF( X(:,e) , e );
        end
        Y_old = Y;
        Y = X - eta_2*AccGoss(X, W, K, c2, c3);
        X = (1 + mu_2)*Y - mu_2*Y_old;
        evol = [evol, log(norm( Theta(:,:) - 1 ,1)) ];
        plot([1:K:t*K],evol'); % each iteration corresponds to K gossip steps
        drawnow;
    end
end

if (Alg_name == 3)
    
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
        
        sum_Theta = sum_Theta + sum(Theta,2)/numE;
        evol = [evol, log(norm(sum_Theta/t - 1)) ];
        plot([1:K:t*K],evol'); % each iteration corresponds to K gossip steps
        drawnow;
    end
end


if (Alg_name == 4)
   
    X = randn(dim,numE,1);
    U = randn(dim,numE,numE);
    U_old = U;
    rho = 0.0001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    
    for t = 1:num_iter
        X_old = X;
        U_old = U;

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            X(:,e,1) =   ProxF(  mean(  X_old(:,Neig_e,1) - U_old(:,Neig_e,e) , 2)   ,   e   , rho*length(Neig_e) );     
        end
        
        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            U(:,e, Neig_e) = 0.5*U_old(:,e, Neig_e) +  0.5*permute(  X_old(:, Neig_e,1 )   -  repmat(X_old(:,e,1),1,length(Neig_e),1)   -   U_old(:,Neig_e,e)  , [1, 3, 2]);
        end
        
        evol = [evol, log(norm( X(:,1,1)    - 1)) ];
        plot([1:1:t*1],evol'); % each iteration corresponds to K gossip steps
        drawnow;
    end
    
end


if (Alg_name == 5)
   
    X = randn(dim,numE,1);
    U = randn(dim,numEline,2); 
    U_old = U;
    rho = 0.001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    
    for t = 1:num_iter
        X_old = X;

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));

            e1 = E1(e); e2 = E2(e); %this assumes that E1 always has the smaller indices and that E2 always contains the larger indices
            Neig_e1 = find( E1 == e1); Neig_e2 = find( E2 == e2); 
            Nm = mean(X_old(:,Neig_e1,1) + U_old(:,Neig_e1,1),2) + mean(X_old(:,Neig_e2,1) + U_old(:,Neig_e2,2),2) - 0.5*(U_old(:,e,1) + U_old(:,e,2)) - X_old(:,e,1);
            X(:,e,1) =   ProxF( Nm   ,   e   , 2*rho );     
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


if (Alg_name == 6)
   
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
            X(:,e,1) =   ProxF( Nm   ,   e   , 2*rho* );     
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


% this function computes the gradient of the i-th function in the objective
% that we are trying to optimize
function [GRAD] = GradConjF(X, i)
    global delta E1 E2
    d = delta;
    X = X + d*ones(length(X),1);
    GRAD = (X)/delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (1/(2*d + d*d))*(  -X(E1(i)) + X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (1/(2*d + d*d))*(   X(E1(i)) - X(E2(i))  );
end

% this is the gradient for the function
% (0.5 * (x_i - x_j)^2 + 0.5*delta*(X)^2 )
function [GRAD] = GradF(X, i)
    global delta E1 E2
    d = delta;
    GRAD = (X)*delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (   X(E1(i)) - X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (  -X(E1(i)) + X(E2(i))  );

    GRAD = GRAD - d*ones(length(X),1);
end

% this is the proximal operator 
% Prox(N) = argmin_X    (1/numE) (0.5 * (x_i - x_j)^2 + 0.5*delta*(X)^2 )+ 0.5*rho* (X - N)^2
function [X_out] = ProxF(X, i, rho)
    global E1 E2 numE delta
    
    X = X*numE*rho + delta;
    
    d = ( rho*numE + delta );
    
    X_out = (X)/d;
    
    X_out(E1(i)) = X_out(E1(i)) + (1/(2*d + d*d))*(  -X(E1(i)) + X(E2(i))  );
    X_out(E2(i)) = X_out(E2(i)) + (1/(2*d + d*d))*(   X(E1(i)) - X(E2(i))  );
end


% this is an approximation for the proximal operator 
% Prox(N) = argmin_X    (1/numE) (0.5 * (x_i - x_j)^2 + 0.5*delta*(X)^2 )+ 0.5*rho* (X - N)^2
% using gradient descent
function [X_out] = AppProxF(X, i, rho, M, alpha)
    global numE

    X_out = X;
    for m = 1:M
        X_out = X_out - (2*alpha/(m+2))*( (1/numE)*GradF( X_out , i)  + rho*(X_out - X)     );
    end
    
end



function [Y] = AccGoss(X, W, k, c2, c3)

    I = eye(length(X));
    a_old = 1;
    a = c2;
    x_old = X;
    x = c2*X*(I -  c3*W);

    for t = 1:k-1

        a_tmp = a_old;
        a_old = a;
        a = 2*c2*a - a_tmp;

        x_tmp = x_old;
        x_old = x;
        x = 2*c2*x*(I - c3*W) - x_tmp;

    end

    Y = X - x/a;
end