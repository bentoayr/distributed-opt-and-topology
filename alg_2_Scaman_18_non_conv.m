function [evol, K, evol_AveTheta] = alg_2_Scaman_18_non_conv(Y_init, Theta_init ,p,q, ProxF, AccGoss, compute_objective, num_iter, numE , Lap_line_G, D, delta, E1,E2,target,L_is, R,     fixing_factor)

    dim  = size( Theta_init , 1);
    numV = size( Theta_init , 2);

    W = Lap_line_G;

    % the choice of the following values is according to Scaman et al. 2018, "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks"
    
    specW = sort(eig(W)); 
    gamma = specW(2)/specW(end);
    c1 = (1 - sqrt(gamma))/ (1 + sqrt(gamma));
    c2 = (1 + gamma) / (1 - gamma);
    c3 = 2/ ((1+gamma)*specW(end));
    K = floor(1 / sqrt(gamma));
    
    L_l = norm(L_is , 2)/sqrt(numE);

    eta_3  = ((1 - c1^K)/(1 + c1^K))*(numE*R/L_l);
    sigma = (1/fixing_factor)*(1/eta_3)*(1 + c1^(2*K))/((1 - c1^K)^2); %note that there is a typo in the arxiv paper "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks" in the specificaion of the Alg 2. In the definition of sigma, tau should be eta

    %M = num_iter;
    %eps = 4*R*L_l/num_iter; % %note that there is a typo in the arxiv paper "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks" in the specificaion of the Alg 2. In the definition of T. It should be T = 4 R L_l / eps
    
    W_Acc = AccGoss(eye(numE), W, K,c2,c3);
    
    
    Y = Y_init*real(sqrtm(full(W)));
    Y = reshape(Y , dim , numV , numE);

    Theta = repmat(  Theta_init  , 1 , 1 , numE );
    Theta_old = Theta;
    
    evol = nan(num_iter,1);
    evol_AveTheta = nan(dim,numV,100);

    
    if (norm(full(W_Acc),2)*sigma*eta_3 > 1)
        disp(['Convergene condition not met because ', num2str(norm(full(W_Acc),2)*sigma*eta_3), ' is bigger than 1']);
        %return;
    end
       
    sum_Theta = 0;
    for t = 1:num_iter

        % reshape Theta and Theta_old
        Theta = reshape(Theta, dim*numV, numE);
        Theta_old = reshape(Theta_old, dim*numV, numE);
        
        % appy distributed averaging operator
        ThetaW = AccGoss(2*Theta - Theta_old, W, K,c2,c3);
        % reshape result
        ThetaW = reshape(ThetaW, dim,numV, numE);
        
        % update dual
        Y = Y - sigma*ThetaW;

        % reshape back
        Theta = reshape(Theta, dim,numV, numE);
        %Theta_old = reshape( Theta_old, dim , numV, numE ); % we do not need
        %to reshape Theta_old back because it is going to be overwritten in
        %the next line
        
        Theta_old = Theta;
        Theta_tilde = Theta;
        for e = 1:numE

            N = eta_3*Y(:,:,e) + Theta(:,:,e);
            rho = (1/eta_3);
            
            N_fixed = (  numE*rho*N + (delta/numV)*target  )/(numE*rho + (delta/numV));
            rho_fixed = numE*rho + (delta/numV);
            
            % note that the ProxF, as implemented, does not take into account delta or
            % target. The PO that takes care of delta and target, is very
            % similar to the PO ProxF with a small change in the input N
            % and parameter rho. That is why we use the transformation
            % above.
            
            Theta_tilde(:,:,e) = ProxF(p,q, N_fixed , rho_fixed ,D,e, E1, E2 );
            
            %ProxF(p,q,N,rho,D,e,E1,E2)


        end
        Theta = Theta_tilde;
        
        sum_Theta = sum_Theta + sum(Theta,3)/numE; % the paper asks to compute the average in space and time
        
        
        AveTheta = sum_Theta/t;
        
        err =  compute_objective(AveTheta,D,p,q,E1,E2,delta,target); % here we could have computed the objective of local (edge) estimate that is performing the worse

        evol(t) = err;
        
        if (num_iter - t < 100)
           evol_AveTheta( : , : , 100 - (num_iter - t) ) = AveTheta; 
        end
        
        
        %evol = [evol, err ];
        %plot([1:1:t*1],evol'); 
        %scatter(sum_Theta(1,:)'/t,sum_Theta(2,:)'/t); % we can vizualize the position of the points

        %drawnow;
        
        
    end
    
end