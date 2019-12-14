
% random graph
numV = 30;
G = rand(numV) > 0.5;
G = graph(triu(G,1) + triu(G,1)'); % undirected random E-R graph. This is not a matrix. It is a graph object.
E1 = G.Edges.EndNodes(:,1);
E2 = G.Edges.EndNodes(:,2);

Adj_G = adjacency(G);
Lap_G = laplacian(G);
Inc_G = abs(incidence(G)); % it seems like Matlab orders edges based on the "lexicographical" indices. So edge (1,2) comes before edge (2,3). Also, there is no edge (2,1) in G. Also, it seems like the incidence matlab function puts signs on the elements, even for undirected graphs. that is why we use the abs() outside of it.
Adj_line_G = Inc_G'*Inc_G - 2*eye(G.numedges); % the relation between the line graph and the incidence matrix is well known. see e.g. https://en.wikipedia.org/wiki/Incidence_matrix#Undirected_and_directed_graphs

line_G = graph(Adj_line_G);
Lap_line_G = laplacian(line_G);

% we make a simple choice for the Gossip matrix, it being equal to the Laplacian of our line graph
W = Lap_line_G;

numE = G.numedges;
numEline = line_G.numedges;


dim = 2;



D = ones(numV);%rand(numV); % matrix of random distances. Note that only the distances corresponding to pairs that are edges matter. The rest does not matter.
D = (D + D')/2; % This is not really necessary, but makes D more interpertable

p = 1;
q = 1;

evol = [];

Alg_name = 5;

if (Alg_name == 0)
    
    X = randn(dim,numV);
    alp = 0.1;
    
    for t = 1:num_iter
        for i = 1:numV
            X(:,i) = X(:,i) - alp* GradF(X, i , Adj_G ,D,numE);
        end
        
        err =  log(compute_objective(X,D,p,q,E1,E2));

        evol = [evol, err ];
        plot([1:1:t*1],evol'); 
        %scatter(X(1,:)',X(2,:)'); % we can vizualize the position of the points
        drawnow;
    end

end
    

if (Alg_name == 5)

    X = randn(dim,numV,numE,1);
    U = randn(dim,numV,numE,numE);
    U_old = U;
    
    rho = 0.00001; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.01;
    
    for t = 1:num_iter
        
        X_old = X;
        U_old = U;

        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));  
            X(:,:,e,1) =   ProxF( p,q, mean( -permute(U(:,:,e,Neig_e),[1,2,4,3])  + 0.5*(X_old(:,:,Neig_e,1) + U_old(:,:,Neig_e,e) + permute(U_old(:,:,e,Neig_e),[1,2,4,3]) + repmat(X_old(:,:,e,1),1,1,length(Neig_e),1)) , 3) ,  rho*numE*length(Neig_e)   ,  D, e ,E1, E2  );     
        end
        
        for e = 1:numE
            Neig_e = find(Adj_line_G(e,:));
            U(:,:,e, Neig_e) = U(:,:,e, Neig_e) + alp*(0.5*(    -U_old(:,:,e, Neig_e)  + permute( -X(:,:,Neig_e,1)    -U_old(:,:,Neig_e,e) + repmat(X(:,:,e,1),1,1,length(Neig_e),1) , [1,2, 4, 3])    ));
        end
                
        
        err =  log(compute_objective(X(:,:,1,1),D,p,q,E1,E2));
        
        evol = [evol, err ];
        plot([1:1:t*1],evol'); 
        %scatter(X(1,:,1,1)',X(2,:,1,1)'); % we can vizualize the position of the points
        drawnow;
    end


end

if (Alg_name == 6)

    X = randn(dim,2,numE);
    Z = randn(dim,numV);
    U = randn(dim,2,numE); 
    
    U_old = U;
    rho = 1; % the algorithm should always converge no matter what rho we choose. However, convergence might be really really slow.
    alp = 0.1;
    
    for t = 1:num_iter

        for e = 1:numE
            i = E1(e); j = E2(e);
            [X1,X2] =  ProxFPair(p,q, Z(:,i) - U(:,1,e),Z(:,j) - U(:,2,e)  , rho*numE , D(i,j) );                
            X(:,1,e) = X1; X(:,2,e) = X2;
        end
        
        for i = 1:numV
            e1Neigh = find(E1 == i);
            e2Neigh = find(E2 == i);            
            Z(:,i) = (sum(X(:,1,e1Neigh) + U(:,1,e1Neigh),3) + sum(X(:,2,e2Neigh) + U(:,2,e2Neigh),3)) / (length(e1Neigh) + length(e2Neigh));
        end
        
        for e = 1:numE
            i = E1(e); j = E2(e);
            U(:,1, e) = U(:,1, e) + alp*( X(:,1,e)  - Z(:,i)  ); 
            U(:,2, e) = U(:,2, e) + alp*( X(:,2,e)  - Z(:,j)  ); 
        end
        
        err =  log(compute_objective(Z,D,p,q,E1,E2));
        
        evol = [evol, err ];
        plot([1:1:t*1],evol'); 
        %scatter(Z(1,:)',Z(2,:)'); % we can vizualize the position of the points
        drawnow;
    end
    
end

function test_PO()
    p = 1;
    q = 1;

    rho = 3;

    while (1)
        dim = 2;
        N1 = randn(dim,1);
        N2 = randn(dim,1);
        d = rand;

        [xi, xj] = ProxFPair(p,q,N1,N2,rho,d);

        xit = xi + 0.01*randn(dim,1);
        xjt = xj + 0.01*randn(dim,1);

        val_opt = abs(norm(xi - xj)^p - d^p)^q + (rho/2)*norm(xi - N1)^2 + (rho/2)*norm(xj - N2)^2;
        val_pert = abs(norm(xit - xjt)^p - d^p)^q + (rho/2)*norm(xit - N1)^2 + (rho/2)*norm(xjt - N2)^2;

        if ( sign([-val_opt + val_pert]) ~= 1)
            break;
        end

        disp(-val_opt + val_pert);

    end

end


function obj = compute_objective(X,D,p,q,E1,E2)
    numE = length(E1);
    obj = 0;
    for e = 1:numE
        i = E1(e); j = E2(e);
        d = D(i,j);
        xi = X(:,i);
        xj = X(:,j);
        obj = obj + abs(norm(xi - xj)^p - d^p)^q;
    end
    obj = obj / numE;
end

% this is the gradient with respect to variable xi of the objective
% (1/numE) sum_(i,j)  | |xi - xj|^p  - d^p |^q 
function GradF = GradF(X, i , Adj_G, D,numE)
    GradF = 0;
    xi = X(:,i);
    Neig_i = find(Adj_G(i,:));
    for j = Neig_i
        d = D(i,j);
        xj = X(:,j);
        GradF = GradF + 4*(norm(xi - xj)^2 - d^2)*(xi - xj);
    end
    GradF = GradF/numE;
end


% this is the proximal operator min_xi xj  | |xi - xj|^p  - d^p |^q + 0.5*rho*(X - N)^2
function [x_opt] = ProxF(p,q,N,rho,D,e,E1,E2)

    x_opt = N; % most components will not change
    i = E1(e); j = E2(e);
    d = D(i,j);
    N1 = N(:,i);
    N2 = N(:,j);
    [x1_opt,x2_opt] = ProxFPair(p,q,N1,N2,rho,d);
    x_opt(:,1) = x1_opt; x_opt(:,2) = x2_opt;
   

end

% this is the proximal operator min_xi xj  | |xi - xj|^p  - d^p |^q + 0.5*rho*(xi - Ni)^2 + 0.5*rho*(xj - Nj)^2
function [x1_opt,x2_opt] = ProxFPair(p,q,N1,N2,rho,d)

    N_p_tilde = N1 + N2;
    N_m_tilde = N1 - N2;
    
    x_p_opt = N_p_tilde;
    x_m_opt = mult_dim_prob(p,q,N_m_tilde,rho/2,d);
    
    x1_opt = (x_p_opt + x_m_opt)/2;
    x2_opt = (x_p_opt - x_m_opt)/2;

end


function x_opt = mult_dim_prob(p,q,N,rho,d)

    norm_tilde_N = norm(N/d);
    rho_tilde = rho*d^(2 - p*q);
    x_tilde_tilde = one_dim_prob(p,q,norm_tilde_N,rho_tilde);
    
    x_opt = d*x_tilde_tilde*N/norm(N);

end

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

