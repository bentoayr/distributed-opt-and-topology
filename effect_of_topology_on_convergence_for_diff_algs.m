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

function [GRAD] = GradF(X, i)
    global delta E1 E2
    d = delta;
    GRAD = (X)*delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (   X(E1(i)) - X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (  -X(E1(i)) + X(E2(i))  );

    GRAD = GRAD - d*ones(length(X),1);
end

function [X_out] = ProxF(X, i, rho)
    global E1 E2 numE delta
    
    X = X*numE*rho + delta;
    
    d = ( rho*numE + delta );
    
    X_out = (X)/d;
    
    X_out(E1(i)) = X_out(E1(i)) + (1/(2*d + d*d))*(  -X(E1(i)) + X(E2(i))  );
    X_out(E2(i)) = X_out(E2(i)) + (1/(2*d + d*d))*(   X(E1(i)) - X(E2(i))  );
end

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