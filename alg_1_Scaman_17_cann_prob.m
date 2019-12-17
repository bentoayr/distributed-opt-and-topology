function evol = alg_1_Scaman_17_cann_prob(Y_init, Theta_init , GradConjF, num_iter, numE , Lap_line_G, delta, E1,E2,target,alpha, beta)

    W = Lap_line_G;

    % the choice of the following values is according to Scaman et al. 2017, "Optimal algorithms for smooth and strongly convexdistributed optimization in networks"
    kappa_l = beta / alpha; 
    specW = sort(eig(W)); 
    eta_1 = alpha/specW(end); 
    gamma = specW(2)/specW(end);
    mu_1 = (sqrt(kappa_l) - sqrt(gamma)) / (sqrt(kappa_l) + sqrt(gamma));
    

    Y = Y_init*real(sqrtm(full(W)));
    X = Y;
    Theta = repmat(Theta_init,1,numE);

    evol = [];
    for t = 1:num_iter
        for e = 1:numE 
            Theta(:,e) = GradConjF( X(:,e) , e , delta, E1,E2,target); 
        end
        Y_old = Y;
        Y = X - eta_1*Theta*W;
        X = (1 + mu_1)*Y - mu_1*Y_old;
        
        evol = [evol, log(norm( Theta(:,:) - target ,1)) ];

    end
end