function [evol, evol_AveTheta] = alg_1_Scaman_17_non_conv(Y_init, Theta_init , GradConjF, compute_objective, num_iter, numE ,D, Lap_line_G, delta, E1,E2,target,alpha, beta)

    dim  = size( Theta_init , 1);
    numV = size( Theta_init , 2);
    
    W = Lap_line_G;

    % the choice of the following values is according to Scaman et al. 2017, "Optimal algorithms for smooth and strongly convexdistributed optimization in networks"
    kappa_l = beta / alpha;
    specW = sort(eig(W));
    eta_1 = alpha/specW(end);
    gamma = specW(2)/specW(end);
    mu_1 = (sqrt(kappa_l) - sqrt(gamma)) / (sqrt(kappa_l) + sqrt(gamma));
    
    Y = Y_init*real(sqrtm(full(W)));
    Y = reshape(Y , dim , numV , numE);
    X = Y;
    Theta = repmat(  Theta_init  , 1 , 1 , numE );
    
    evol = nan(num_iter,1);
    evol_AveTheta = nan(dim,numV,100);
    
    for t = 1:num_iter
        for e = 1:numE
            i = E1(e); j = E2(e); d = D(i,j);
            Theta(:,:,e) = GradConjF( X(:,:,e), e, d, numV, delta ,target , E1, E2); % we rescale delta by numV so that the two terms in our objective have more or less the same size regardless of the size of the graph that we are deadling with
        end
        Y_old = Y;

        % this could be done more efficiently
        % notice that reshaping operations take almost no time
        Theta = reshape(Theta, dim*numV, numE);
        ThetaW = Theta*W;
        ThetaW = reshape(ThetaW, dim, numV, numE);
        Theta = reshape(Theta,dim,numV,numE);

        Y = X - eta_1*ThetaW;
        X = (1 + mu_1)*Y - mu_1*Y_old;
        
        AveTheta = mean(Theta,3); % this is the mean computed over all of the nodes. We use this mean to check on the performance of the algorithm
        
        err =  compute_objective(AveTheta,D,2,2,E1,E2,delta,target); % here we could have computed the objective of local (edge) estimate that is performing the worse

        evol(t) = err;
        
        if (num_iter - t < 100)
           evol_AveTheta( : , : , 100 - (num_iter - t) ) = AveTheta; 
        end

    %scatter(Theta(1,:,2)',Theta(2,:,2)','.');
    %drawnow;
    end
end