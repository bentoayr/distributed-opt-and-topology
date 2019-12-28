%numV = 15;
%graph_type = 7;

%numV = 35;
%graph_type = 5;

numV = 40;
graph_type = 3;





[numV,  numE, numEline, Adj_G,Lap_G, Adj_line_G, Lap_line_G, E1, E2, E1line, E2line ] = generate_graph_data(numV, graph_type);



dim = 2;
D = ones(numV);%rand(numV); % matrix of random distances. Note that only the distances corresponding to pairs that are edges matter. The rest does not matter.
D = (D + D')/2; % This is not really necessary, but makes D more interpertable

delta = 1; % this delta is already rescaled by E
target = 1; % this pushes all of the points to the (1,1,1,1,1,1....,1) region of the space

p = 2;
q = 2;

num_iter = 10000;

f1 = figure(1) ;ax1 = axes ;hold(ax1,'on');
set(gca, 'FontName', 'Times New Roman');
set(gca, 'FontSize', 18);
set(gca, 'defaultLineLineWidth', 2);
set(gcf, 'color', 'w');
box on;

f2 = figure(2) ;ax2 = axes ; hold(ax2,'on');
set(gca, 'FontName', 'Times New Roman');
set(gca, 'FontSize', 18);
set(gca, 'defaultLineLineWidth', 2);
set(gcf, 'color', 'w');
box on;


for alg_name = [0, 3, 4, 5, 6, 7, 8]


if (alg_name == 0)
    
    rng(1);
    X_init = 1+0.01*randn(dim,numV);

    %alp = 0.06; 
    %alp = 0.06;
    alp = 0.96;
    [evol_obj, evol_X] = grad_desc_non_conv_prob(X_init, alp, @compute_objective,@GradF,num_iter, Adj_G, D, numE ,E1,E2, delta, target);

end


if (alg_name == 3)

    rng(1);
    Y_init = 1+0.01*randn(dim*numV, numE);
    Theta_init = 1+0.01*randn(dim,numV,1);
    
    R = 1; %varying R and L_is is basically the same thing as far as the behaviour of the algorithm goes
    fixing_factor = 3; %this does not seem to make a big difference in some of the experiments
    
    %L_isval = 9.51;
    %L_isval = 8.51;
    L_isval = 9.51;
        
    
    L_is = L_isval*ones(numE , 1);
    
    [evol_obj, K, evol_AveX] = alg_2_Scaman_18_non_conv(Y_init, Theta_init , p,q, @ProxF, @AccGoss, @compute_objective, num_iter, numE , Lap_line_G, D, delta, E1,E2,target,L_is, R,     fixing_factor);

end

if (alg_name == 4) % Alg in Table 1: "Distributed Optimization Using the Primal-Dual Method of Multipliers"
    
    rng(1);
    X_init = 1+0.01*randn(dim , numV , numE , 1);
    U_init = 1+0.01*randn(dim,numV,numE,numE);
    
    
    %rho = 0.1;
    %alp = 0.81;
    
    %rho = 0.1;
    %alp = 0.81;
    
    rho = 0.1;
    alp = 0.21;
           
    [evol_obj, evol_AveX] = PDMM_non_conv(p,q,X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G, D, @ProxF , @compute_objective, E1, E2, delta, target);

end

if (alg_name == 5) % Consensus ADMM of the form (1/numE)* sum_e f_e(x_e) subject to x_e = Z_(e,e') and x_e' = Z_(e,e') if (e,e') is in the line graph.

    rng(1);
    X_init = 1+0.01*randn(dim,numV,numE,1);
    U_init = 1+0.01*randn(dim,numV,numE,numE);
    
    %rho = 0.02;
    %alp = 0.61;
    
    %rho = 0.01;
    %alp = 0.01;
    
    rho = 0.09;
    alp = 0.21;
    
    [evol_obj, evol_AveX] = ADMM_edge_Z_edge_non_conv(p,q,X_init, U_init, rho, alp, numE, num_iter,  Adj_line_G, D, @ProxF , @compute_objective, E1, E2, delta, target);
end

if (alg_name == 6) % Consensus ADMM of the form (1/numE)* sum_e f_e(x_e) subject to x_e = x_e' if (e,e') is in the line graph. The difference between this algorithm and the one above (Alg 5) is that here we do not use the consensus variable Z_(e,e') in the augmented lagrangian

    rng(1);
    X_init = 1 + 0.01*randn(dim, numV, numE);
    U_init = 1 + 0.01*randn(dim, numV, numEline);
    
      
    %rho = 0.01;
    %alp = 0.001;
    
    %rho = 0.01;
    %alp = 0.141;
    
    rho = 0.91;
    alp = 0.001;
    
    [evol_obj, evol_AveX] = ADMM_edge_edge_no_Z_non_conv(p,q,X_init, U_init, rho, alp, numE, numEline, num_iter,  Adj_line_G, D, @ProxF , @compute_objective, E1, E2, E1line,E2line,delta, target);
    
end

if (alg_name == 7) % Consensus ADMM of the form (1/numE)* sum_( e = (i,j) \in E) f_e(x_ei,x_ej) subject to x_ei = z_i if i touches edges e in the graph G.

    rng(1);
    X_init = 1 + 0.01*randn(dim,2,numE);
    Z_init = 1 + 0.01*randn(dim,numV);
    U_init = 1 + 0.01*randn(dim,2,numE); 

    
    %rho = 0.96;
    %alp = 1.6;
    
    %rho = 0.01;
    %alp = 1.2;
    
    rho = 0.01;
    alp = 0.6;
    
    [evol_obj, evol_Z] = ADMM_node_Z_node_non_conv(p,q,X_init, U_init, Z_init, rho, alp, numE, num_iter, @ProxFPair , @compute_objective, Adj_G, D,  E1, E2, delta, target);
    
end

if (alg_name == 8) % Consensus over-relaxed ADMM of the form (1/numE)* sum_( e = (i,j) \in E) f_e(x_ei,x_ej) subject to x_ei = z_i if i touches edges e in the graph G.

    rng(1);
    X_init = 1 + 0.01*randn(dim,2,numE);
    Z_init = 1 + 0.01*randn(dim,numV);
    U_init = 1 + 0.01*randn(dim,2,numE); 
    
    Walk_G = diag(sum(Adj_G).^(-1))*Adj_G;
    w_Walk_G = (eig(full(Walk_G)));
    w_star = max(w_Walk_G(w_Walk_G < 0.9999999));
    w_bar = min(w_Walk_G(w_Walk_G > -0.9999999));
    rho_star = 2*sqrt(1 - w_star^2);

    %gamma_star = 4*inv(3 - sqrt((2-rho_star)/(2+rho_star)));
    gamma_star = 2; % this is used when we have a ring with odd number of nodes

    [evol_obj, evol_Z] = ADMM_over_relaxed_node_Z_node_non_conv(p,q,X_init, U_init, Z_init, rho_star, gamma_star, numE, num_iter, @ProxFPair , @compute_objective, Adj_G, D,  E1, E2, delta, target);
    
end

 plot(ax1,evol_obj); 
 plot(ax2,log(abs(evol_obj(end) - evol_obj)));


end

xlabel(ax1,'iteration');
ylabel(ax1,'objective value');
legend(ax1,'GD', 'MSDA', 'PADMM', 'ADMM (All +Cons)', 'ADMM (All -Cons)', 'ADMM (Partial +Cons)', 'ADMM (Partial +Cons + Opt)');
legend(ax1,'Location', 'Best');
legend(ax1,'boxoff');

xlabel(ax2,'iteration');
ylabel(ax2,'objective value');
legend(ax2,'GD', 'MSDA', 'PADMM', 'ADMM (All +Cons)', 'ADMM (All -Cons)', 'ADMM (Partial +Cons)', 'ADMM (Partial +Cons + Opt)');
legend(ax2,'Location', 'Best');
legend(ax2,'boxoff');
        
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