%test_TA_spec_min()
%return;

numV = 80;
graph_type = 6;

[numV,  numE, numEline, Adj_G,Lap_G, Adj_line_G, Lap_line_G, E1, E2, E1line, E2line ] = generate_graph_data(numV, graph_type);

dim = numV;

delta = 0.001;
delta = delta / dim; % we scale delta with 1/dim so that both terms in our objective have the same order of magnitude as the graph grows
target = -0.342; 



for alg_name = 7


if (alg_name == 0) %Gradient descent
   
    num_iter = 100000;
    num_iter_last_hist = 100;
    
    log_eps = -5;
    verbose = 0;
    
    alf = optimizableVariable('a',[0.1,0.7],'Type','real');
    fun = @(x) grad_desc_cann_prob_time(x.a,numV,log_eps ,verbose, num_iter,num_iter_last_hist, Lap_G, numE , delta, target);
   
    results = bayesopt(fun,[alf],'Verbose',0,'AcquisitionFunctionName','expected-improvement-plus');
    T_best = results.MinObjective;
    alf_best = results.XAtMinObjective.a;
   
    rng(1);
    dim = numV;
    X = 1 + 0.01*randn(dim,1);
    [evol, ~] = grad_desc_cann_prob(X, alf_best*numE, num_iter,num_iter_last_hist, Lap_G, numE , delta, target);
    figure;
    plot(evol);
end

if (alg_name == 1) %Alg 1: "Optimal algorithms for smooth and strongly convex distributed optimization in networks"
    % note that this algorithm is built to minimize the average of (1/n)
    % sum_i f_i for some functions f_i, whose Conjugate Gradient we use
    % bellow. For us i are edges and n is the number of edges, and f_e =
    % 0.5*(xi - xj)^2 + 0.5*(delta/dim)*|X|^2, where, we recall, delta/dim is done above 
    Y_init = rand(dim, numE);
    Theta_init = rand(dim,1);
    alpha = delta;
    beta  = 2 + delta;
    num_iter = 1000;
    
    alpha_range = delta/2: delta/5 :2*delta;
    beta_range = (2 + delta)/5: (2 + delta)/5 :2*(2 + delta);
    rate_evol = [];
    for alpha =  alpha_range
        for beta = beta_range
            evol = alg_1_Scaman_17_cann_prob(Y_init, Theta_init, @GradConjF, num_iter, numE , Lap_line_G, delta, E1,E2,target,alpha, beta);
            plot( [ 1 : num_iter  ] , evol');
            drawnow;
            rate_est = estimate_rate_out_of_plot(evol);
            rate_evol = [rate_evol, rate_est];
            disp(rate_est);
        end
    end
    best_rate_alg_1 = min(rate_evol(rate_evol < 0));
    best_alg_1_ix = find(rate_evol == best_rate_alg_1);
    best_beta_alg_1 = beta_range(mod(best_alg_1_ix-1,length(beta_range))+1);
    best_alf_alg_1 = alpha_range(floor((best_alg_1_ix-1)/length(beta_range))+1);
    
    all_rates_all_graphs{mem_ix}{2}{alg_name_count} = {best_rate_alg_1, best_alg_1_ix, best_beta_alg_1 , best_alf_alg_1};
    
end

if (alg_name == 2) %Alg 2: "Optimal algorithms for smooth and strongly convex distributed optimization in networks"
    % note that this algorithm is built to minimize the average of (1/n)
    % sum_i f_i for some functions f_i, whose Conjugate Gradient we use
    % bellow. For us i are edges and n is the number of edges, and f_e =
    % 0.5*(xi - xj)^2 + 0.5*(delta/dim)*|X|^2, where, we recall, delta/dim is done above 
    
    Y_init = rand(dim, numE);
    Theta_init = rand(dim,1);
    alpha = delta;
    beta  = 2 + delta;
    num_iter = 3000;
    
    alpha_range = delta/2: delta/3 :3*delta;
    beta_range = (2 + delta)/5: (2 + delta)/3 :3*(2 + delta);
    rate_evol = [];
    for alpha =  alpha_range
        for beta = beta_range
            [evol,K] = alg_2_Scaman_17_cann_prob(Y_init, Theta_init , @GradConjF, @AccGoss, num_iter, numE , Lap_line_G, delta, E1,E2,target,alpha, beta);
    
            plot([1:K:num_iter*K],evol'); % each iteration corresponds to K gossip steps
            drawnow;
            rate_est = estimate_rate_out_of_plot(evol);
            rate_evol = [rate_evol, rate_est];
            disp(rate_est);
        end
    end
    best_rate_alg_2 = min(rate_evol(rate_evol < 0));
    best_alg_2_ix = find(rate_evol == best_rate_alg_2);
    best_beta_alg_2 = beta_range(mod(best_alg_2_ix-1,length(beta_range))+1);
    best_alf_alg_2 = alpha_range(floor((best_alg_2_ix-1)/length(beta_range))+1);
            
    all_rates_all_graphs{mem_ix}{2}{alg_name_count} = {best_rate_alg_2, best_alg_2_ix, best_beta_alg_2 , best_alf_alg_2};

    
end

if (alg_name == 3) % Alg 2: "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks"
    % note that this algorithm is built to minimize the average of (1/n)
    % sum_i f_i for some functions f_i, whose Proximal Operator we use
    % bellow. The algorithm specifies that the proximal operator should be
    % computed for (1/n) f_i, and we do so.
    % For us i are edges and n is the number of edges, and f_e =
    % 0.5*(xi - xj)^2 + 0.5*(delta/dim)*|X|^2, where, we recall, delta/dim is done above 
    
    Y_init = rand(dim, numE);
    Theta_init = rand(dim,1);
    
    L_is = 1*(2 + delta)*ones(numE , 1);
    R = 1; %varying R and L_is is basically the same thing as far as the behaviour of the algorithm goes
    fixing_factor = 3; %this does not seem to make a big difference.
    num_iter = 50000;
    
    L_is_range = 0.01*(2 + delta):0.01*(2 + delta):0.1*(2 + delta);
    rate_evol = [ ];
    for L_is_val =  L_is_range
            L_is = L_is_val*ones(numE , 1);
            [evol, K] = alg_2_Scaman_18_cann_prob(Y_init, Theta_init , @ProxF, @AccGoss, num_iter, numE , Lap_line_G, delta, E1,E2,target,L_is, R, fixing_factor);
            plot([1:K:num_iter*K],evol'); % each iteration corresponds to K gossip steps
            drawnow;
            rate_est = estimate_rate_out_of_plot(evol);
            rate_evol = [rate_evol, rate_est];
            disp(rate_est);
    end
    best_rate_alg_3 = min(rate_evol(rate_evol < 0));
    best_alg_3_ix = find(rate_evol == best_rate_alg_3);
    best_L_is_val_alg_3 = L_is_range(  best_alg_3_ix  );
    
    all_rates_all_graphs{mem_ix}{2}{alg_name_count} = {best_rate_alg_3, best_alg_3_ix, best_L_is_val_alg_3};

end

if (alg_name == 4) % Alg in Table 1: "Distributed Optimization Using the Primal-Dual Method of Multipliers"
   
    X_init = randn(dim, numE, 1);
    U_init = randn(dim, numE, numE);
    
    rho = 0.1; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.1;
    num_iter = 300;
       
    alpha_range = 0.001:0.001:0.01;
    rho_range = 0.000001:0.000001:0.00001;%:0.0000001:0.00001;
    rate_evol = [];
    for alp =  alpha_range
        for rho = rho_range
            evol = PDMM_cann_prob(X_init, U_init, rho / numE, alp, numE, num_iter,  Adj_line_G, @ProxF ,  E1, E2, delta / numE, target);
    
            plot([1:1:num_iter],evol'); 
            drawnow;
            rate_est = estimate_rate_out_of_plot(evol);
            rate_evol = [rate_evol, rate_est];

            disp(rate_est);
        end
    end
    best_rate_alg_4 = min(rate_evol(rate_evol < 0));
    best_alg_4_ix = find(rate_evol == best_rate_alg_4);
    best_rho_alg_4 = rho_range(mod(best_alg_4_ix-1,length(rho_range))+1);
    best_alf_alg_4 = alpha_range(floor((best_alg_4_ix-1)/length(rho_range))+1);

    all_rates_all_graphs{mem_ix}{2}{alg_name_count} = {best_rate_alg_4, best_alg_4_ix, best_rho_alg_4, best_alf_alg_4};

    
end

if (alg_name == 5) % Consensus ADMM of the form sum_e f_e(x_e) subject to x_e = Z_(e,e') and x_e' = Z_(e,e') if (e,e') is in the line graph.
   
    X_init = randn(dim,numE,1);
    U_init = randn(dim,numE,numE);
    
    rho = 0.000001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.1;
    num_iter = 1000;
    
    alpha_range = 0.001:0.003:0.03;
    rho_range = 0.000001:0.000001:0.00001;%:0.0000001:0.00001;
    rate_evol = [];
    for alp =  alpha_range
        for rho = rho_range
            evol = ADMM_edge_Z_edge_cann_prob(X_init, U_init, rho / numE, alp, numE, num_iter,  Adj_line_G, @ProxF ,  E1, E2, delta / numE, target);
            rate_est = estimate_rate_out_of_plot(evol);

            plot([1:1:num_iter*1],evol'); 
            drawnow;
            disp(rate_est);
            
            rate_evol = [rate_evol, rate_est];

            disp(rate_est);
        
        end
    end
    best_rate_alg_5 = min(rate_evol(rate_evol < 0));
    best_alg_5_ix = find(rate_evol == best_rate_alg_5);
    best_rho_alg_5 = rho_range(mod(best_alg_5_ix-1,length(rho_range))+1);
    best_alf_alg_5 = alpha_range(floor((best_alg_5_ix-1)/length(rho_range))+1);

    all_rates_all_graphs{mem_ix}{2}{alg_name_count} = {best_rate_alg_5, best_alg_5_ix, best_rho_alg_5, best_alf_alg_5};
    
    
end


if (alg_name == 6) % Consensus ADMM of the form sum_e f_e(x_e) subject to x_e = x_e' if (e,e') is in the line graph. The difference between this algorithm and the one above (Alg 5) is that here we do not use the consensus variable Z_(e,e') in the augmented lagrangian

    X_init = randn(dim,numE);
    U_init = randn(dim,numEline);
    
    rho = 0.00001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 1;
    num_iter = 1000;
   
    
    alpha_range = 0.01:0.04:0.4;
    rho_range = 0.000001:0.000001:0.00001;%:0.0000001:0.00001;
    rate_evol = [];
    for alp =  alpha_range
        for rho = rho_range
            evol = ADMM_edge_edge_no_Z_cann_prob(X_init, U_init, rho / numE, alp, numE, num_iter,  Adj_line_G, @ProxF ,numEline, E1line, E2line, E1, E2, delta / numE, target);
            
            rate_est = estimate_rate_out_of_plot(evol);

            plot([1:1:num_iter*1],evol'); 
            drawnow;
            disp(rate_est);
            
            rate_evol = [rate_evol, rate_est];

            disp(rate_est);
        
        end
    end
    best_rate_alg_6 = min(rate_evol(rate_evol < 0));
    best_alg_6_ix = find(rate_evol == best_rate_alg_6);
    best_rho_alg_6 = rho_range(mod(best_alg_6_ix-1,length(rho_range))+1);
    best_alf_alg_6 = alpha_range(floor((best_alg_6_ix-1)/length(rho_range))+1);    
    
    all_rates_all_graphs{mem_ix}{2}{alg_name_count} = {best_rate_alg_6, best_alg_6_ix, best_rho_alg_6, best_alf_alg_6};

    
end


if (alg_name == 7) % Consensus ADMM of the form sum_( e = (i,j) \in E) f_e(x_ei,x_ej) subject to x_ei = z_i if i touches edges e in the graph G.
    
    num_iter = 1000;
    num_iter_last_hist = 100;
    
    log_eps = -10;
    verbose = 0;

    %rho = optimizableVariable('r',[0.00001,0.001],'Type','real'); % for ring graph
    rho = optimizableVariable('r', [0.000001,0.00003],'Type','real'); % for K-hop graph
    % for E-R graph
    gamma = optimizableVariable('g',[0.5,2],'Type','real');  % if gamma is larger than 2 ADMM's transition matrix will be unstable
    
    fun = @(x) ADMM_over_relaxed_node_Z_node_cann_prob_time(x.r, x.g,numV,log_eps,0,num_iter,num_iter_last_hist, numE , E1, E2,delta, target);
   
    results = bayesopt(fun,[rho, gamma],'Verbose',0,'AcquisitionFunctionName','expected-improvement-plus');
    T_best = results.MinObjective;
    rho_best = results.XAtMinObjective.r;
    gamma_best = results.XAtMinObjective.g;
   
    dim = numV;
   
    rng(1);
    X_init = 1 + 0.01*randn(2,numE);
    Z_init = 1 + 0.01*randn(dim,1);
    U_init = 1 + 0.01*randn(2,numE); 

    [evol,~] = ADMM_over_relaxed_node_Z_node_cann_prob(X_init, U_init, Z_init, rho_best, gamma_best,dim, numE, num_iter,num_iter_last_hist, @ProxFPair , E1, E2, delta, target);

    figure;
    plot(evol);
    
    Walk_G = diag(sum(Adj_G).^(-1))*Adj_G;
    w_Walk_G = real(eig(full(Walk_G)));
    w_star = max(w_Walk_G(w_Walk_G < 0.9999999));
    w_bar = min(w_Walk_G(w_Walk_G > -0.9999999));
    % if the graph has even-length cycles we use these formulas
    rho_star = 2*sqrt(1 - w_star^2);
    gamma_star = 4*inv(3 - sqrt((2-rho_star)/(2+rho_star)));
    tau_star = gamma_star - 1;
    err_star = log(tau_star.^(1:num_iter));
    hold on;
    plot(err_star);

    
    
    
end
    

end


function T = ADMM_over_relaxed_node_Z_node_cann_prob_time(rho, gamma,numV,log_eps,verbose,num_iter,num_iter_last_hist, numE , E1, E2,delta, target)
    dim = numV;
   
    rng(1);
    X_init = randn(2,numE);
    Z_init = randn(dim,1);
    U_init = randn(2,numE); 

    [evol,~] = ADMM_over_relaxed_node_Z_node_cann_prob(X_init, U_init, Z_init, rho, gamma,dim, numE, num_iter,num_iter_last_hist, @ProxFPair , E1, E2, delta, target);
    if (evol(end) < log_eps)
        T = find(diff(sign(evol - log_eps)) < 0, 1, 'last' );
        
        if (verbose == 1)
            hold on;
            plot(evol);
            plot([T,T],[max(evol),min(evol)]);
            plot([1,num_iter],[log_eps,log_eps]);
            hold off;
        end
    else
        T = num_iter+1;
    end
end


function T = grad_desc_cann_prob_time(alf,numV,log_eps,verbose,num_iter,num_iter_last_hist, Lap_G, numE , delta, target)
    rng(1);
    dim = numV;
    X_init = 1 + 0.01*randn(dim,1);
    [evol, ~] = grad_desc_cann_prob(X_init, alf*numE, num_iter,num_iter_last_hist, Lap_G, numE , delta, target);
    if (evol(end) < log_eps)
        T = find(diff(sign(evol - log_eps)) < 0, 1, 'last' );
        
        if (verbose == 1)
            hold on;
            plot(evol);
            plot([T,T],[max(evol),min(evol)]);
            plot([1,num_iter],[log_eps,log_eps]);
            hold off;
        end
    else
        T = num_iter+1;
    end
end

function test_TA_spec_min()

    numV = 20;
    graph_type = 7;

    [numV,  numE, numEline, Adj_G,Lap_G, Adj_line_G, Lap_line_G, E1, E2, E1line, E2line ] = generate_graph_data(numV, graph_type);

    
    Walk_G = diag(sum(Adj_G).^(-1))*Adj_G;
    w_Walk_G = real(eig(full(Walk_G)));
    w_star = max(w_Walk_G(w_Walk_G < 0.9999999));
    rho_star = 2*sqrt(1 - w_star^2);
    gamma_star = 4*inv(3 - sqrt((2-rho_star)/(2+rho_star)));
    tau_star = gamma_star - 1;
    
    rho = optimizableVariable('r',[0.1,3],'Type','real'); 
    gamma = optimizableVariable('g',[0.5,3],'Type','real'); 
    
    fun = @(x) TA_spec(x.g,x.r,Adj_G,numV,numE,[0,0,1,0]);
   
    results = bayesopt(fun,[rho, gamma],'Verbose',0,'AcquisitionFunctionName','expected-improvement-plus');
    tau_best = results.MinObjective;
    rho_best = results.XAtMinObjective.r;
    gamma_best = results.XAtMinObjective.g;
  
    1 == 1;
    
end

function out = TA_spec(gamma,rho,Adj_G,numV,numE,out_vec)

    Walk_G = diag(sum(Adj_G).^(-1))*Adj_G;
    spec = (eig(full(Walk_G)));

    TA_spec_p = (1 - gamma/2) + (gamma/(2 + rho))*(  spec + sqrt(-1)*sqrt( 1 - (rho/2)^2 - spec.^2)    );
    TA_spec_n = (1 - gamma/2) + (gamma/(2 + rho))*(  spec - sqrt(-1)*sqrt( 1 - (rho/2)^2 - spec.^2)    );

    tau_spec = [TA_spec_p;TA_spec_n];

    tau_spec = abs(tau_spec);
    tau = max(tau_spec(tau_spec < max(tau_spec) - 0.000001)); %ignore the largest one, 
    
    TA_matrix = get_TA_matrix(Adj_G,rho,gamma,numV,numE);
    spec_via_TA_matrix = abs(eig(TA_matrix));
    tau_via_TA_matrix = max(spec_via_TA_matrix(spec_via_TA_matrix < max(spec_via_TA_matrix) - 0.000001));  %ignore the largest one, 
    
    
    out = [];
    if (out_vec(1) == 1)
        out = [out, tau];
    end
    if (out_vec(2) == 1)
        out = [out, tau_spec];
    end
    if (out_vec(3) == 1)
       out = [out, tau_via_TA_matrix];
    end
    if (out_vec(4) == 1)
       out = [out, spec_via_TA_matrix];
    end

end

function TA = get_TA_matrix(Adj_G,rho,gamma,numV,numE)
    
    In = abs(incidence(graph(Adj_G)))';
    S = zeros(2*numE,numV);
    for i = 1:size(In,1)
        l = sort(find(In(i,:)));
        S(2*i-1,l(1)) = 1;
        S(2*i,l(2)) = 1;
    end
    B = S*inv(S'*S)*S';
    A = inv(eye(2*numE) + (1/rho)*kron(eye(numE) , [1,-1;-1,1]));
    TA = eye(2*numE) - gamma*(A + B - 2*B*A);

end

function test_GradConjF()
    while(1)
        dim = 3;
        X = rand(dim,1);
        i = 1;
        E1 = 1;
        E2 = 2;
        target = 1;
        delta = 0.01;

        [GRAD] = GradF(X, i,delta,E1,E2,target);

        [XX] = GradConjF(GRAD, i, delta, E1,E2,target);

        disp(norm(X - XX));
        if (norm(X - XX) > 10^(-5))
            break;
        end
    end
    
end

% this function computes the gradient of conjugate of the i-th function in the objective that we are trying to optimize
% namely, the conjudate gradient of (0.5 * (x_i - x_j)^2 + 0.5*delta*(X - target)^2 )
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
% Prox(N) = argmin_X    (1/numE)*(0.5 * (x_i - x_j)^2 + 0.5*delta*(X - target)^2 ) + 0.5*rho* (X - N)^2
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