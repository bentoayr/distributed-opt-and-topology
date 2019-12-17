function [evol, K] = alg_2_Scaman_18_cann_prob(Y_init, Theta_init , ProxF, AccGoss, num_iter, numE , Lap_line_G, delta, E1,E2,target,L_is, R)

    W = Lap_line_G;

    % the choice of the following values is according to Scaman et al. 2018, "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks"
    
    specW = sort(eig(W)); 
    gamma = specW(2)/specW(end);
    c1 = (1 - sqrt(gamma))/ (1 + sqrt(gamma));
    c2 = (1 + gamma) / (1 - gamma);
    c3 = 2/ ((1+gamma)*specW(end));
    K = floor(1 / sqrt(gamma));
    
    L_l = norm(L_is,2)/sqrt(numE);

    fixing_factor = 3;

    eta_3  = ((1 - c1^K)/(1 + c1^K))*(numE*R/L_l);
    sigma = (1/fixing_factor)*(1/eta_3)*(1 + c1^(2*K))/((1 - c1^K)^2); %note that there is a typo in the arxiv paper "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks" in the specificaion of the Alg 2. In the definition of sigma, tau should be eta
    sum_Theta = 0;

    %M = num_iter;
    %eps = 4*R*L_l/num_iter; % %note that there is a typo in the arxiv paper "Optimal Algorithms for Non-Smooth Distributed Optimization in Networks" in the specificaion of the Alg 2. In the definition of T. It should be T = 4 R L_l / eps
    
    W_Acc = AccGoss(eye(numE), W, K,c2,c3);
    
    
    Y = Y_init*real(sqrtm(full(W)));
    %X = Y;
    Theta = repmat(Theta_init,1,numE);
    Theta_old = Theta;

    evol = [];
    
    if (norm(full(W_Acc),2)*sigma*eta_3 > 1)
        disp(['Convergene condition not met because ', num2str(norm(full(W_Acc),2)*sigma*eta_3), ' is bigger than 1']);
        %return;
    end
       
    
    for t = 1:num_iter

        Y = Y - sigma*AccGoss(2*Theta - Theta_old, W, K,c2,c3);

        Theta_old = Theta;
        Theta_tilde = Theta;
        for e = 1:numE

            Theta_tilde(:,e) = ProxF( eta_3*Y(:,e) + Theta(:,e) , e, 1/eta_3 , E1, E2, numE, delta, target);
            %Theta_tilde(:,e) = AppProxF( eta_3*Y(:,e) + Theta(:,e) , e, 1/eta_3 , num_iter, eta_3);
            %disp(   norm(   Theta_tilde(:,e) - tmp  )  );

        end
        Theta = Theta_tilde;
        
        sum_Theta = sum_Theta + sum(Theta,2)/numE; % the paper asks to compute the average in space and time
        evol = [evol, log(norm(sum_Theta/t - target)) ];

        
    end
end