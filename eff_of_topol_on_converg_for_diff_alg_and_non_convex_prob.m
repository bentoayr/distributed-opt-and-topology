%% this code solves the problem  0.5*(delta)*(1/numV)\sum_{i \in V} |x_i - target|^2   +  (1/ |E|) sum_{(i,j) \in E} | | x_i - x_j|^p  - d^p_ij  |^q
% for some graph G = (V, E) and for different values of q and p. Each x_i is associated with a node i and has dimension dim
% target is a vector/matrix with all entries equal to "target"

% random graph

numV_range = 5:5:40;
type_range = 1:7;
verbose = 0;

parpool(25);
parfor mem_ix = 1:length(numV_range)*length(type_range) % we use the outer for-loop to search over different graph sizes and repetitions for each graph size

numV_count = 1 + mod(mem_ix - 1 ,length(numV_range));
numV = numV_count*5;
graph_type_count =  1 + floor((mem_ix - 1) / length(numV_range));
graph_type = graph_type_count;


[numV,  numE, numEline, Adj_G,Lap_G, Adj_line_G, Lap_line_G, E1, E2, E1line, E2line ] = generate_graph_data(numV, graph_type);


all_rates_all_graphs{mem_ix}{1} = {numV,  numE, numEline, Adj_G,Lap_G, Adj_line_G, Lap_line_G, E1, E2, E1line, E2line };


for alg_name = 1:7



dim = 2;
D = ones(numV);%rand(numV); % matrix of random distances. Note that only the distances corresponding to pairs that are edges matter. The rest does not matter.
D = (D + D')/2; % This is not really necessary, but makes D more interpertable

delta = 1; % this delta is already rescaled by E
target = 1; % this pushes all of the points to the (1,1,1,1,1,1....,1) region of the space

p = 2;
q = 2;

%% run the different algorithms
% at this point Alg_1 and Alg_2, based on using conjugate gradient descent
% are not working, they are unstable

if (Alg_name == 0)
    
    rng(1);
    X_init = 1+0.01*randn(dim,numV);
    num_iter = 1000;

    alp_range = 0.01:0.05:1;
    min_obj_val = inf;
    best_alp = nan;
    range_counts = 0;
    all_rates = {};
    for alp = alp_range
        range_counts = range_counts + 1;
        [evol_obj, evol_X] = grad_desc_non_conv_prob(X_init, alp, @compute_objective,@GradF,num_iter, Adj_G, D, numE ,E1,E2, delta, target);

        rate_obj = estimate_rate_out_of_plot(log(abs(evol_obj - evol_obj(end))));
        rate_X = estimate_rate_out_of_X_evol(evol_X);
        disp([evol_obj(end),rate_obj, rate_X]);
        
        all_rates{range_counts}{1} = evol_obj; all_rates{range_counts}{2} = evol_X;

        if (evol_obj(end) < min_obj_val)
            min_obj_val = evol_obj(end);
            best_alp = alp;
            best_evol_X = evol_X;
            best_evol_obj = evol_obj;
        end
    end
    if (verbose == 1)
        subplot(1,2,1);
        plot([1:1:num_iter*1],best_evol_obj'); % vizualize the evolution of the error
        
        for r = 1:100
            hold on;
            subplot(1,2,2);
            scatter(best_evol_X(1,:,r)',best_evol_X(2,:,r)','.'); % vizualize the position of the points
            hold off;
        end
    end
    
    all_rates_all_graphs{mem_ix}{2}{alg_name} = {min_obj_val, best_alp, best_evol_X, best_evol_obj, all_rates};

end

% there is some problem with the convergence of this algorithm for the non
% convex problem
if (Alg_name == 1) %Alg 1: "Optimal algorithms for smooth and strongly convex distributed optimization in networks"
    %delta = 100;
    %eta_1 = 0.001*eta_1;
    %mu_1  = 0.5*mu_1;
    
%     alpha = 0.001;%*delta;
%     beta  = 0.1;%*(4 + delta); 
%     
%     %delta = 10;
%     rng(1);
%     Y_init = 1+0.01*randn(dim*numV, numE);
%     Theta_init = 1+0.01*randn(dim,numV,1);
%     
%     num_iter = 1000;
%    
%     [evol_obj, evol_AveX] = alg_1_Scaman_17_non_conv(Y_init, Theta_init , @GradConjF, @compute_objective, num_iter, numE , D, Lap_line_G, delta, E1,E2,target,alpha, beta);
% 
%     rate_obj = estimate_rate_out_of_plot(log(abs(evol_obj - evol_obj(end))));
%     rate_X = estimate_rate_out_of_X_evol(evol_AveX);
%     disp([rate_obj, rate_X]);
%     
%     if (verbose == 1)
%         subplot(1,2,1);
%         plot([1:1:num_iter*1],evol_obj'); % vizualize the evolution of the error
% 
%         for r = 1:100
%             hold on;
%             subplot(1,2,2);
%             scatter(evol_AveX(1,:,r)',evol_AveX(2,:,r)','.'); % vizualize the position of the points
%             hold off;
%         end
%     end
    
end

if (Alg_name == 2)
    
end

if (Alg_name == 3)

    rng(1);
    Y_init = 1+0.01*randn(dim*numV, numE);
    Theta_init = 1+0.01*randn(dim,numV,1);
    
    R = 1; %varying R and L_is is basically the same thing as far as the behaviour of the algorithm goes
    fixing_factor = 3; %this does not seem to make a big difference in some of the experiments
    num_iter = 1000;
    
    min_obj_val = inf;
    L_isval_range = 0.01:0.5:10;
    
    range_counts = 0;
    all_rates = {};
    for L_isval = L_isval_range
        range_counts = range_counts + 1;

        L_is = L_isval*ones(numE , 1);
    
        [evol_obj, K, evol_AveX] = alg_2_Scaman_18_non_conv(Y_init, Theta_init , p,q, @ProxF, @AccGoss, @compute_objective, num_iter, numE , Lap_line_G, D, delta, E1,E2,target,L_is, R,     fixing_factor);

        rate_obj = estimate_rate_out_of_plot(log(abs(evol_obj - evol_obj(end))));
        rate_X = estimate_rate_out_of_X_evol(evol_AveX);
        disp([evol_obj(end), rate_obj, rate_X]);

        all_rates{range_counts}{1} = evol_obj; all_rates{range_counts}{2} = evol_AveX; all_rates{range_counts}{3} = K;

        if (evol_obj(num_iter) < min_obj_val) % depending on what we are measuring, it could make sense to use evol_obj(num_iter/K) here
            min_obj_val = evol_obj(end);
            best_L_isval = L_isval;
            best_evol_AveX = evol_AveX;
            best_evol_obj = evol_obj;
            best_K = K;
        end
    end
    if (verbose == 1)
        subplot(1,2,1);
        plot([1:best_K:num_iter*best_K],best_evol_obj'); % vizualize the evolution of the error
        
        for r = 1:100
            hold on;
            subplot(1,2,2);
            scatter(best_evol_AveX(1,:,r)',best_evol_AveX(2,:,r)','.'); % vizualize the position of the points
            hold off;
        end
    end
    all_rates_all_graphs{mem_ix}{2}{alg_name} = {min_obj_val, best_L_isval, best_evol_AveX, best_evol_obj, best_K, all_rates};

end

if (Alg_name == 4) % Alg in Table 1: "Distributed Optimization Using the Primal-Dual Method of Multipliers"
    
    rng(1);
    X_init = 1+0.01*randn(dim , numV , numE , 1);
    U_init = 1+0.01*randn(dim,numV,numE,numE);
    
    rho_range = 0.1:0.3:3; 
    alp_range = 0.01:0.2:2;
    
    num_iter = 1000;
    
    min_obj_val = inf;
    range_counts = 0;
    all_rates = {};
    for rho = rho_range
        for alp = alp_range
            range_counts = range_counts + 1;
            [evol_obj, evol_AveX] = PDMM_non_conv(p,q,X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G, D, @ProxF , @compute_objective, E1, E2, delta, target);
    
            rate_obj = estimate_rate_out_of_plot(log(abs(evol_obj - evol_obj(end))));
            rate_X = estimate_rate_out_of_X_evol(evol_AveX);
            disp([evol_obj(end), rate_obj, rate_X]);
            
            all_rates{range_counts}{1} = evol_obj; all_rates{range_counts}{2} = evol_AveX;

            if (evol_obj(num_iter) < min_obj_val) % depending on what we are measuring, it could make sense to use evol_obj(num_iter/K) here
                min_obj_val = evol_obj(end);
                best_rho = rho;
                best_alp = alp;
                best_evol_AveX = evol_AveX;
                best_evol_obj = evol_obj;
            end
            
        end
    end
    
    if (verbose == 1)
        subplot(1,2,1);
        plot([1:1:num_iter*1],best_evol_obj'); % vizualize the evolution of the error

        for r = 1:100
            hold on;
            subplot(1,2,2);
            scatter(best_evol_AveX(1,:,r)',best_evol_AveX(2,:,r)','.'); % vizualize the position of the points
            hold off;
        end
    end
    all_rates_all_graphs{mem_ix}{2}{alg_name} = {min_obj_val, best_rho,best_alp, best_evol_AveX, best_evol_obj, all_rates};

end

if (Alg_name == 5) % Consensus ADMM of the form (1/numE)* sum_e f_e(x_e) subject to x_e = Z_(e,e') and x_e' = Z_(e,e') if (e,e') is in the line graph.

    rng(1);
    X_init = 1+0.01*randn(dim,numV,numE,1);
    U_init = 1+0.01*randn(dim,numV,numE,numE);
    
    rho_range = 0.01:0.01:0.2; 
    alp_range = 0.01:0.2:2;
    
    num_iter = 1000;
    
    min_obj_val = inf;

    range_counts = 0;
    all_rates = {};
    for rho = rho_range
        for alp = alp_range
            range_counts = range_counts + 1;
            [evol_obj, evol_AveX] = ADMM_edge_Z_edge_non_conv(p,q,X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G, D, @ProxF , @compute_objective, E1, E2, delta, target);

            rate_obj = estimate_rate_out_of_plot(log(abs(evol_obj - evol_obj(end))));
            rate_X = estimate_rate_out_of_X_evol(evol_AveX);
            disp([evol_obj(end), rate_obj, rate_X]);

            all_rates{range_counts}{1} = evol_obj; all_rates{range_counts}{2} = evol_AveX;
            
            if (evol_obj(num_iter) < min_obj_val) % depending on what we are measuring, it could make sense to use evol_obj(num_iter/K) here
                min_obj_val = evol_obj(end);
                best_rho = rho;
                best_alp = alp;
                best_evol_AveX = evol_AveX;
                best_evol_obj = evol_obj;
            end
        
        end
    end
            
    if (verbose == 1)
        subplot(1,2,1);
        plot([1:1:num_iter*1],best_evol_obj'); % vizualize the evolution of the error

        for r = 1:100
            hold on;
            subplot(1,2,2);
            scatter(best_evol_AveX(1,:,r)',best_evol_AveX(2,:,r)','.'); % vizualize the position of the points
            hold off;
        end
    end
    all_rates_all_graphs{mem_ix}{2}{alg_name} = {min_obj_val, best_rho,best_alp, best_evol_AveX, best_evol_obj, all_rates};

end

if (Alg_name == 6) % Consensus ADMM of the form (1/numE)* sum_e f_e(x_e) subject to x_e = x_e' if (e,e') is in the line graph. The difference between this algorithm and the one above (Alg 5) is that here we do not use the consensus variable Z_(e,e') in the augmented lagrangian

    rng(1);
    X_init = 1 + 0.01*randn(dim, numV, numE);
    U_init = 1 + 0.01*randn(dim, numV, numEline);
    
    rho_range = 0.01:0.05:1; 
    alp_range = [(0.001:0.02:0.2), (0.2:0.2:2)];
    
    num_iter = 1000;
      
    min_obj_val = inf;

    range_counts = 0;
    all_rates = {};
    for rho = rho_range
        for alp = alp_range
            range_counts = range_counts + 1;
            [evol_obj, evol_AveX] = ADMM_edge_edge_no_Z_non_conv(p,q,X_init, U_init, rho, alp, numE, numEline, num_iter,  Adj_line_G, D, @ProxF , @compute_objective, E1, E2, E1line,E2line,delta, target);
    
            rate_obj = estimate_rate_out_of_plot(log(abs(evol_obj - evol_obj(end))));
            rate_X = estimate_rate_out_of_X_evol(evol_AveX);
            disp([evol_obj(end), rate_obj, rate_X]);
            
            all_rates{range_counts}{1} = evol_obj; all_rates{range_counts}{2} = evol_AveX;

            if (evol_obj(num_iter) < min_obj_val) % depending on what we are measuring, it could make sense to use evol_obj(num_iter/K) here
                min_obj_val = evol_obj(end);
                best_rho = rho;
                best_alp = alp;
                best_evol_AveX = evol_AveX;
                best_evol_obj = evol_obj;
            end
        end
    end
    
    if (verbose == 1)
        subplot(1,2,1);
        plot([1:1:num_iter*1],best_evol_obj'); % vizualize the evolution of the error

        for r = 1:100
            hold on;
            subplot(1,2,2);
            scatter(best_evol_AveX(1,:,r)',best_evol_AveX(2,:,r)','.'); % vizualize the position of the points
            hold off;
        end
    end
    all_rates_all_graphs{mem_ix}{2}{alg_name} = {min_obj_val, best_rho,best_alp, best_evol_AveX, best_evol_obj, all_rates};

end

if (Alg_name == 7) % Consensus ADMM of the form (1/numE)* sum_( e = (i,j) \in E) f_e(x_ei,x_ej) subject to x_ei = z_i if i touches edges e in the graph G.

    X_init = 1 + 0.01*randn(dim,2,numE);
    Z_init = 1 + 0.01*randn(dim,numV);
    U_init = 1 + 0.01*randn(dim,2,numE); 
    
    rho_range = 0.01:0.05:1; 
    alp_range = [(0.001:0.02:0.2), (0.2:0.2:2)];
    
    num_iter = 1000;
      
    min_obj_val = inf;
    min_rate_val = inf;
    
    range_counts = 0;
    all_rates = {};
    for rho = rho_range
        for alp = alp_range
            range_counts = range_counts + 1;
            [evol_obj, evol_Z] = ADMM_node_Z_node_non_conv(p,q,X_init, U_init, Z_init, rho, alp, numE, num_iter, @ProxFPair , @compute_objective, Adj_G, D,  E1, E2, delta, target);
    
            rate_obj = estimate_rate_out_of_plot(log(abs(evol_obj - evol_obj(end))));
            rate_Z = estimate_rate_out_of_X_evol(evol_Z);
            disp([evol_obj(end), rate_obj, rate_Z]);
            
            all_rates{range_counts}{1} = evol_obj; all_rates{range_counts}{2} = evol_Z;
            
            if (evol_obj(num_iter) < min_obj_val) % depending on what we are measuring, it could make sense to use evol_obj(num_iter/K) here
                min_obj_val = evol_obj(end);
                best_rho = rho;
                best_alp = alp;
                best_evol_Z = evol_Z;
                best_evol_obj = evol_obj;
            end            
            
        end
    end
    if (verbose == 1)
        subplot(1,2,1);
        plot([1:1:num_iter*1],best_evol_obj'); % vizualize the evolution of the error

        for r = 1:100
            hold on;
            subplot(1,2,2);
            scatter(best_evol_Z(1,:,r)',best_evol_Z(2,:,r)','.'); % vizualize the position of the points
            hold off;
        end
    end
    all_rates_all_graphs{mem_ix}{2}{alg_name} = {min_obj_val, best_rho,best_alp, best_evol_AveX, best_evol_obj, all_rates};

end

end
end

% this tests the Proximal operator by seeing if the solution returned is,
% at least, a local minimum of the objective associated with the proximal
% operator
function test_PO()

    while (1)
        
        rho1 = rand;
        rho2 = rand;

        p = randi(2);
        q = randi(2);
        
        dim = 2;
        N1 = randn(dim,1);
        N2 = randn(dim,1);
        d = rand;

        [xi, xj] = ProxFPair(p,q,N1,N2,rho1,rho2,d);
        val_opt = abs(norm(xi - xj)^p - d^p)^q + (rho1/2)*norm(xi - N1)^2 + (rho2/2)*norm(xj - N2)^2;
            
        for t = 1:100
        
            xit = xi + 0.01*randn(dim,1);
            xjt = xj + 0.01*randn(dim,1);

            val_pert = abs(norm(xit - xjt)^p - d^p)^q + (rho1/2)*norm(xit - N1)^2 + (rho2/2)*norm(xjt - N2)^2;

            if ( sign([-val_opt + val_pert]) ~= 1)
                break;
            end

            disp(-val_opt + val_pert);
        end

    end

end

% this tests the gradient and conjugate gradient functions. They should be
% the inverse of each other. The loop should go forever.
function test_GradF_and_Conj_Grad_F()

    while (1)
        numE = 10;
        delta = rand;
        target = rand;
        d = rand;
        dim = 3;
        numV = 5;
        X = randn(dim,numV);

        E1 = 4;
        E2 = 5;
        e = 1;

        Y = GradConjF(X,e,d,numV,delta,target,E1,E2);
        
        YY = GradFPair(Y,e,d,numV,delta,target,E1,E2);

        disp(norm(YY-X));
        
        if (  norm(YY-X)   > 10^(-4))
            1 == 1;
        end
        
    end
    
end



function obj = compute_objective(X,D,p,q,E1,E2,delta,target)
    numE = length(E1);
    numV = size(X,2);
    obj = 0;
    for e = 1:numE
        i = E1(e); j = E2(e);
        d = D(i,j);
        xi = X(:,i);
        xj = X(:,j);
        obj = obj + (1/numE)*abs(norm(xi - xj)^p - d^p)^q;
    end
    obj = obj + (1/numV)*delta*(norm(X - target,'fro')^2);
end

% this is the gradient with respect to variable xi of the objective
% ((1/numE) sum_(i,j)  | |xi - xj|^2  - d_{i,j}^2 |^2) +  0.5*(delta/numV)*|X - target|^2)
function GradF = GradF(X, i , Adj_G, D,numE, delta,target,numV)
    xi = X(:,i);
    GradF = (delta/numV)*(xi - target);
    
    Neig_i = find(Adj_G(i,:));
    for j = Neig_i
        d = D(i,j);
        xj = X(:,j);
        GradF = GradF + (1/numE)*4*(norm(xi - xj)^2 - d^2)*(xi - xj);
    end
    
end

% this is the gradient with respect to variable xi of the objective
% (| |xi - xj|^p  - d^p |^q + 0.5*(delta/numV)*|X - target|^2)
% notice that the optimization aolgorithms that we are using are minimizing
% the average of functions (that decompose the objective). Therefore, here
% we do not need to use the (1/numE) in the functions
function GradFPair = GradFPair(X,e,d,numV,delta,target, E1,E2)

    delta = delta/numV;

    GradFPair = delta*(X - target);
    
    i = E1(e); j = E2(e);
    
    xi = X(:,i);
    xj = X(:,j);
    
    GradFPair(:,i) = GradFPair(:,i) + 4*(norm(xi - xj)^2 - d^2)*(xi - xj);
    GradFPair(:,j) = GradFPair(:,j) + 4*(norm(xi - xj)^2 - d^2)*(xj - xi);
end


% computes the gradient of the conjugate function  |xi - xj|^2 - d^2)^2 + 0.5*(delta/numV)*|X - target|^2 
% notice that the algorithms that use the conjugate gradient are minimizing
% average of functions
% the conjugate gradient is basically minimizing <X,Y> - f(X) where f(X) is the function above
% notice again that we do not need to divide the objective by (1/numE)
% since the optimization algorithms that use these functions already assume
% that the objecgive is an AVERAGE of functions
function GradConjF = GradConjF( Y , e , d , numV , delta , target , E1 , E2 )

    numE = 1; % we do no need to use this, but the code we wrote was initially generic, so we put numE = 1 here.
    delta = delta/numV; 
    
    % the code below computes the conjugate function (1/E)( |xi - xj|^2 - d^2)^2) + 0.5*delta*|X - target|^2 
    % where the (1/E) does not multiply delta, and was wet to 1 above
    
    GradConjF = target + (Y/delta);

    %return;
    
    i = E1(e); j = E2(e);
    xipxj = (1/delta)*(Y(:,i) + Y(:,j)) + 2*target;

    tmp = zeros(6,1);
    
    % we need to test the different possible solutions of two possible cubic polinomials
    % for numerical reasons we also need to check the objective at the end.
    % it is not enough to simply look for the real root.
    
    r = 1;
    c = -norm(Y(:,i) - Y(:,j));
    a = r*8/numE;
    b = r*(delta - (8*d*d/numE));
    s = sqrt(-1);
    tmp(1) = (((1/18)^(1/3))*(sqrt(3)*sqrt(27*(a^4)*(c^2) + 4*(a^3)*(b^3)) - 9*(a^2)*c)^(1/3))/a - (0.87358*b)/(sqrt(3)*sqrt(27 *(a^4)* (c^2) + 4* (a^3) *(b^3)) - 9 *(a^2) *c)^(1/3);
    tmp(2) = ((((1/12)^(1/3)) + 0.75654*s)*b)/(sqrt(3) *sqrt(27 *(a^4) *(c^2) + 4 *(a^3) *(b^3)) - 9 *(a^2) * c)^(1/3) - ((0.19079 - 0.33045 *s) * (sqrt(3) * sqrt(27* (a^4) * (c^2) + 4 *(a^3) * (b^3)) - 9 *(a^2)* c)^(1/3))/a;
    tmp(3) = ((((1/12)^(1/3)) - 0.75654*s)*b)/(sqrt(3) *sqrt(27 *(a^4)* (c^2) + 4 *(a^3) *(b^3)) - 9 *(a^2) * c)^(1/3) - ((0.19079 + 0.33045 *s) * (sqrt(3) * sqrt(27* (a^4) * (c^2) + 4 *(a^3) * (b^3)) - 9 *(a^2)* c)^(1/3))/a;

    r = -1;
    c = -norm(Y(:,i) - Y(:,j));
    a = r*8/numE;
    b = r*(delta - (8*d*d/numE));
    s = sqrt(-1);
    tmp(4) = (((1/18)^(1/3))*(sqrt(3)*sqrt(27*(a^4)*(c^2) + 4*(a^3)*(b^3)) - 9*(a^2)*c)^(1/3))/a - (0.87358*b)/(sqrt(3)*sqrt(27 *(a^4)* (c^2) + 4* (a^3) *(b^3)) - 9 *(a^2) *c)^(1/3);
    tmp(5) = ((((1/12)^(1/3)) + 0.75654*s)*b)/(sqrt(3) *sqrt(27 *(a^4) *(c^2) + 4 *(a^3) *(b^3)) - 9 *(a^2) * c)^(1/3) - ((0.19079 - 0.33045 *s) * (sqrt(3) * sqrt(27* (a^4) * (c^2) + 4 *(a^3) * (b^3)) - 9 *(a^2)* c)^(1/3))/a;
    tmp(6) = ((((1/12)^(1/3)) - 0.75654*s)*b)/(sqrt(3) *sqrt(27 *(a^4)* (c^2) + 4 *(a^3) *(b^3)) - 9 *(a^2) * c)^(1/3) - ((0.19079 + 0.33045 *s) * (sqrt(3) * sqrt(27* (a^4) * (c^2) + 4 *(a^3) * (b^3)) - 9 *(a^2)* c)^(1/3))/a;

    pos_ix = (abs(imag(tmp)) < 10^(-5)) & (real(tmp) > 0);
    
    % search for the solution that correctly inverts the gradient
    % we can break the for-loop after we find one solution
    for normximxj = real(tmp(pos_ix))'
    
        ximxj = (-1/c)*(Y(:,i) - Y(:,j))*sign(normximxj^2 - d^2 + delta*numE/8)*normximxj;
        GradConjF(:,i) = (ximxj + xipxj)/2;
        GradConjF(:,j) = (-ximxj + xipxj)/2;
        
        if (norm(GradFPair(GradConjF,e,d,numV,delta,target,E1,E2) - Y) < 10^(-5))
            break;
        end
    end

end


% this is the proximal operator min_X  | |xi - xj|^p  - d^p |^q  + 0.5*rho*|X - N|^2
% note that this is a PO for the whole set of vectors {x_i}
function [x_opt] = ProxF(p,q,N,rho,D,e,E1,E2)
    
    x_opt = N; % most components will not change. But the components corresponding to the edge being processed change
    i = E1(e); j = E2(e);
    d = D(i,j);
    N1 = N(:,i);
    N2 = N(:,j);
    [x1_opt,x2_opt] = ProxFPair(p,q,N1,N2,rho,rho,d);
    x_opt(:,i) = x1_opt; x_opt(:,j) = x2_opt;
   

end

% this is the proximal operator min_xi xj  | |xi - xj|^p  - d^p |^q + 0.5*rho1*(xi - Ni)^2 + 0.5*rho2*(xj - Nj)^2
% note that here we allow two different rho's for each of the quadratic penalties
function [x1_opt,x2_opt] = ProxFPair(p,q,N1,N2,rho1,rho2,d)
    
    N_m_tilde = N1 - N2;
    
    x_m_opt = mult_dim_prob(p,q,N_m_tilde,(rho1*rho2)/(rho1 + rho2),d);
    
    x_p_opt = ((2*N1 - x_m_opt)*rho1 + (2*N2 + x_m_opt)*rho2)/(rho1 + rho2);
  
    x1_opt = (x_p_opt + x_m_opt)/2;
    x2_opt = (x_p_opt - x_m_opt)/2;

end

% this finds the minimum of the following function of a vector x for p,q in {1,2} and a vector N:  (abs(||x||.^p - 1).^q) +  0.5*rho*||x - N||.^2
function x_opt = mult_dim_prob(p,q,N,rho,d)

    norm_tilde_N = norm(N/d);
    rho_tilde = rho*d^(2 - p*q);
    x_tilde_tilde = one_dim_prob(p,q,norm_tilde_N,rho_tilde);
    
    x_opt = d*x_tilde_tilde*N/norm(N);

end

% this finds the minimum of the following function of a scalar x for p,q in {1,2} and a scalar N:  (abs(abs(x).^p - 1).^q) +  0.5*rho*(x - N).^2
function x_opt = one_dim_prob(p,q,N,rho)

    if (p == 1 && q == 2 )
        x_p = (rho*N + 2)/(2 + rho); val_p = (abs(abs(x_p).^p - 1).^q) +  0.5*rho*(x_p - N).^2;
        x_m = (rho*N - 2)/(2 + rho); val_m = (abs(abs(x_m).^p - 1).^q) +  0.5*rho*(x_m - N).^2;
        val_0 = (abs(abs(0).^p - 1).^q) +  0.5*rho*(0 - N).^2;
        if ~(x_p > 0)
            val_p = inf;
        end
        if ~(x_m < 0)
            val_m = inf;
        end
        if ~( abs(rho*N/2) < 1 && val_0 <= min([val_p,val_m]) )
            val_0 = inf;
        end

        [~,ix] = min([val_p;val_m;val_0]);
        switch ix
            case 1
                x_opt = x_p;
            case 2
                x_opt = x_m;
            case 3
                x_opt = 0;
        end
    end

    if (p == 2 && q == 1)
        x_p = (rho*N)/(2 + rho);
        x_m = (rho*N)/(-2 + rho);
        al_1 = rho*(N-1)/2; val_1 = (abs(abs(1).^p - 1).^q) +  0.5*rho*(1 - N).^2;
        al_m_1 = -rho*(N+1)/2; val_m_1 = (abs(abs(-1).^p - 1).^q) +  0.5*rho*(-1 - N).^2;
        if (abs(x_p) > 1)
            x_opt = x_p;
        end
        if(abs(x_m) < 1)
            x_opt = x_m;
        end
        if ~(al_1 < 1 && al_1 > -1)
            val_1 = inf;
        end
        if ~(al_m_1 < 1 && al_m_1 > -1)
            val_m_1 = inf;
        end
        [v,ix] = min([val_1;val_m_1]);
        if (v < inf)
            switch ix
                case 1
                    x_opt = 1;
                case 2
                    x_opt = -1;
            end 
        end
    end

    if (p == 2 && q == 2)
        r = rho - 4;
        t = -rho*N;
        s = sqrt(-1);
        x_opt_1 = abs((3^(1/3) * (sqrt(3) * sqrt(r^3 + 27*t^2) - 9*t)^(2/3) - 3^(2/3)*r)/(6*(sqrt(3)*sqrt(r^3 + 27*t^2) - 9*t)^(1/3)));
        val_1 = (abs(abs(x_opt_1).^p - 1).^q) +  0.5*rho*(x_opt_1 - N).^2;
        x_opt_2 = abs((s*3^(1/3)*(sqrt(3) + s)* (sqrt(3) * sqrt(r^3 + 27* t^2) - 9 *t)^(2/3) + 3^(1/6)*(sqrt(3) + 3 *s)* r)/(12 *(sqrt(3) *sqrt(r^3 + 27 *t^2) - 9* t)^(1/3)));
        val_2 = (abs(abs(x_opt_2).^p - 1).^q) +  0.5*rho*(x_opt_2 - N).^2;
        x_opt_3 = abs( (3^(1/3) * (-1 - s * sqrt(3)) * (sqrt(3) * sqrt(r^3 + 27 * t^2) - 9 * t)^(2/3) + 3^(1/6) * (sqrt(3) + -3 *s) * r)/(12 * (sqrt(3) * sqrt(r^3 + 27 * t^2) - 9 * t)^(1/3)));
        val_3 = (abs(abs(x_opt_3).^p - 1).^q) +  0.5*rho*(x_opt_3 - N).^2;
        [~,ix] = min([val_1,val_2,val_3]);
        switch ix
            case 1
                x_opt = x_opt_1;
            case 2
                x_opt = x_opt_2;
            case 3
                x_opt = x_opt_3;
        end
    end


    if (p == 1 && q == 1)
        x_p = N - 1/rho; val_p = (abs(abs(x_p).^p - 1).^q) +  0.5*rho*(x_p - N).^2;
        x_n = N + 1/rho; val_n = (abs(abs(x_n).^p - 1).^q) +  0.5*rho*(x_n - N).^2;
        al_0 = N*rho; val_0 = (abs(abs(0).^p - 1).^q) +  0.5*rho*(0 - N).^2;
        al_1 = (N-1)*rho; val_1 = (abs(abs(1).^p - 1).^q) +  0.5*rho*(1 - N).^2;
        al_m_1 = (N+1)*rho; val_m_1 = (abs(abs(-1).^p - 1).^q) +  0.5*rho*(-1 - N).^2;
        if ~(x_p > 1 || (x_p > -1) && (x_p < 0))
            val_p = inf;
        end
        if ~(x_n < -1 || (x_n > 0) && (x_n < 1))
            val_n = inf;
        end
        if ~(al_0 < 1 && al_0 > -1)
            val_0 = inf;
        end
        if ~(al_1 < 1 && al_1 > -1)
            val_1 = inf;
        end
        if ~(al_m_1 < 1 && al_m_1 > -1)
            val_m_1 = inf;
        end
        [v,ix] = min([val_0;val_1;val_m_1;val_p;val_n]);
        if (v < inf)
            switch ix
                case 1
                    x_opt = 0;
                case 2
                    x_opt = 1;
                case 3
                    x_opt = -1;
                case 4
                    x_opt = x_p;
                case 5
                    x_opt = x_n;
            end 
        end
    end
end

function [Y]= AccGoss(X, W, k, c2, c3)

    I = eye(size(X,2));
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