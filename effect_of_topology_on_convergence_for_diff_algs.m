%% testing different algorithms for distributed optimization

% random graph
global E1 E2 delta

n = 30;
G = rand(n) > 0.5;
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
delta = 0.001;

% Scaman et al. 2017, Algorithm 1 and Algorithm 2
Y = rand(n, numE)*real(sqrtm(full(W)));
X = Y;

Theta = rand(n,numE);

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
AccGossFlag = true;
num_iter = 1000;

evol = [];
for t = 1:num_iter
    
    for e = 1:numE
       Theta(:,e) = GradConjF( X(:,e) , e );
    end
    Y_old = Y;
    
    if (AccGossFlag == false)
        Y = X - eta*Theta*W;
        X = (1 + mu)*Y - mu*Y_old;
    else
        Y = X - eta_2*AccGoss(X, W, K, c2, c3);
        X = (1 + mu_2)*Y - mu_2*Y_old;
    end
     
    evol = [evol, Y(:,1) ];
    plot(evol');
    drawnow;
end



% this function computes the gradient of the i-th function in the objective
% that we are trying to optimize
function [GRAD] = GradConjF(X, i)
    global delta E1 E2
    d = delta;
    GRAD = X/delta;
    
    GRAD(E1(i)) = GRAD(E1(i)) + (1/(2*d + d*d))*(  -X(E1(i)) + X(E2(i))  );
    GRAD(E2(i)) = GRAD(E2(i)) + (1/(2*d + d*d))*(   X(E1(i)) - X(E2(i))  );

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