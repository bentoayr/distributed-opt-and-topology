%% testing different algorithms for distributed optimization

% random graph
global E1 E2 delta numE

dim = 30;
G = rand(dim) > 0.5;
G = graph(triu(G,1) + triu(G,1)');

Adj_G = adjacency(G);
Lap_G = laplacian(G);
Inc_G = incidence(G);
Adj_line_G = Inc_G'*Inc_G - 2*eye(G.numedges);

line_G = graph(Adj_line_G);
Lap_line_G = laplacian(line_G);
W = Lap_line_G;

[E1, E2] = find(triu(Adj_G,1));
numE = G.numedges;
delta = 0.01;

% Scaman et al. 2017, Algorithm 1 and Algorithm 2
Y = 0*rand(dim, numE)*real(sqrtm(full(W)));
X = Y;

Theta = repmat(rand(dim,1),1,numE);
Theta_old = Theta;

num_iter = 500;

alpha = delta;
beta  = 2;
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


R = 2; 
L_is = 2*ones(numE , 1);
L_l = norm(L_is,2)/sqrt(numE);
eta_3  = (numE*R/L_l)*((1 - c1^K)/(1 + c1^K));
sigma = (1 + c1^(2*K))/((1 - c1^K)^2);
Theta_tilde = rand(dim,numE);
sum_Theta = 0;


M = num_iter;
eps = 4*R*L_l/num_iter; % this is the target accuracy


% tmp = randn(dim,1);
% [X_out] = ProxF(tmp, 2, 0.4);
% [X_out_2] = AppProxF(tmp, 2, 0.4,600);
% norm(X_out-X_out_2)
% return;


Alg_name = 3;

evol = [];
    
if (Alg_name == 1)
    for t = 1:num_iter
        for e = 1:numE
            Theta(:,e) = GradConjF( X(:,e) , e );
        end
        Y_old = Y;
        Y = X - eta*Theta*W;
        X = (1 + mu)*Y - mu*Y_old;
        evol = [evol, log(norm(Y(:,1))) ];
        plot( [ 1 : t  ] , evol');
        drawnow;
        %disp(t);
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
        evol = [evol, log(norm(Y(:,1))) ];
        plot([1:K:t*K],evol'); % each iteration corresponds to K gossip steps
        drawnow;
    end
end

if (Alg_name == 3)
    for t = 1:num_iter
        Y = Y - sigma*AccGoss(2*Theta - Theta_old, W, K,c2,c3);
        Theta_tilde(:,:) = Theta;
        for e = 1:numE
            
            %for m = 1:M
            %    Theta_tilde(:,e) = (m/(m+2))*Theta_tilde(:,e) - (2/(2+m))*((eta_3/numE)*GradF( Theta_tilde(:,e)  , e) - eta_3*Y(:,e) - Theta(:,e)  );
            %end
            Theta_tilde(:,e) = ProxF( eta_3*Y(:,e) + Theta(:,e) , e, 1/eta_3);
            
        end
        Theta_old = Theta;
        Theta = Theta_tilde;
        sum_Theta = sum_Theta + sum(Theta,2)/(num_iter*numE);
        evol = [evol, log(norm(sum_Theta)) ];
        plot([1:K:t*K],evol'); % each iteration corresponds to K gossip steps
        drawnow;
    end
end




% this function computes the gradient of the i-th function in the objective
% that we are trying to optimize
function [GRAD] = GradConjF(X, i)
    global delta E1 E2
    d = delta;
    GRAD = (X)/delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (1/(2*d + d*d))*(  -X(E1(i)) + X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (1/(2*d + d*d))*(   X(E1(i)) - X(E2(i))  );

end

function [GRAD] = GradF(X, i)
    global delta E1 E2
    d = delta;
    GRAD = (X)*delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (   X(E1(i)) - X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (  -X(E1(i)) + X(E2(i))  );

end


function [X_out] = ProxF(X, i, rho)
    global E1 E2 numE delta
    
    d = (rho*numE+delta);
    
    X_out = (X)/d;
    
    X_out(E1(i)) = X_out(E1(i)) + (1/(2*d + d*d))*(  -X(E1(i)) + X(E2(i))  );
    X_out(E2(i)) = X_out(E2(i)) + (1/(2*d + d*d))*(   X(E1(i)) - X(E2(i))  );

    X_out = X_out*numE*rho;
    
end

function [X_out] = AppProxF(X, i, rho, M)
    global numE

    X_out = X;
    for m = 1:M
        X_out = X_out - (2/(m+2))*( (1/numE)*GradF( X_out , i)  + rho*(X_out - X)     );
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