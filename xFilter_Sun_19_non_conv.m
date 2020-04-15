% the algorithm is meant to solve (1/M) sum^M_i=1 f_i
% it follows the description in Haoran Sun and Mingyi Hong 2019
% this code assumes the same L-bound on the gradient of each of the
% functions f_i
% the code is implemented as if x_i are scalars, but they can be variables
% of any dimension
% many of the matrices are diagonal matrices with constant elements in the
% diagonal. We treat them as simply constant as often as possible to accelerate calculations
% each node, which is an edge in our problem, has a variable of dimension dim*numV

function [evol_obj, evol_X_aveg, Q, std_X_filter] = xFilter_Sun_19_non_conv(GradFPair,compute_objective,X_xFilter_init, flag,fix_factor_1,fix_factor_2,dim,num_iter,num_iter_last_hist, numE , numV, Lap_line_G, D, delta, E1,E2,target,L)


    evol_obj = nan(num_iter,1);
    evol_X_aveg = nan(dim,numV,num_iter_last_hist);

    % first we define serveral constants according to the paper

    L_matrix = L;  % in this code this is a scalar
   
    K_matrix = L;
       
    Lap_hat = Lap_line_G; % this is obtained by simplifying the expression L^{-1/2} F^T K F L^{-1/2}, where F = A P^{1/2}, where A is a normalized incidence matrix, and P is the degree diagonal matrix
    eiglist = eig(Lap_line_G);
    
    lambda_min_pos_Lap_hat = min(eiglist(eiglist > 10^(-6))); % smallest non-zero eigenvalue
    lambda_min_Lap_hat = min(eiglist);
    lambda_max_Lap_hat = max(eiglist);
    
    Gamma_matrix = sqrt(96*L_matrix/numE); % in this code this is a scalar
    Sigma_matrix = fix_factor_1*sqrt(48 * 96  *K_matrix / (numE * lambda_min_pos_Lap_hat));  % in this code this is a scalar
    
    R = (Gamma_matrix.^(-2))*(Sigma_matrix.^(2))*Lap_hat + eye(numE); %this is obtained by simplifying Gamma^{-2}*F^T \Sigma^2 F + I_M, and noticing tha Gamma and Sigma are diagonal with constant elements in the diagonal
    
    xi_R = (1 + (Gamma_matrix.^(-2))*(Sigma_matrix.^(2))*lambda_min_Lap_hat)/(1 + (Gamma_matrix.^(-2))*(Sigma_matrix.^(2))*lambda_max_Lap_hat);
    
    theta_for_eta = min(1,Gamma_matrix.^2)*xi_R;
    
    eta = (theta_for_eta^2)/(16 + 128*numE*max(1,(Sigma_matrix^2)* (Gamma_matrix.^(-2))*(Sigma_matrix.^(2))*lambda_max_Lap_hat +1));
    
    Q = ceil(-0.25*fix_factor_2*log(eta/4)*sqrt(1/xi_R)); % needs to be integer
    
    theta_R_for_alpha = (1 + (Gamma_matrix.^(-2))*(Sigma_matrix.^(2))*lambda_min_Lap_hat) + (1 + (Gamma_matrix.^(-2))*(Sigma_matrix.^(2))*lambda_max_Lap_hat);
    
    rho_0 = (1 - xi_R)/(1 + xi_R);
    
    tau = 2/theta_R_for_alpha;
    
    % second we write the iterations according to Algorithm 2
    
    X_xFilter = X_xFilter_init; % this is the matrix of all vectors, one per node
    d_xfilter = zeros(dim*numV,numE);
    X_tilde_xFilter = zeros(dim*numV,numE);
    for e = 1:numE
        grad = GradFPair( reshape(X_xFilter(:,e),dim,numV)  ,e,  D(E1(e),E2(e)) ,numV,delta,target, E1,E2);
        grad = reshape(grad, dim*numV,1);
        d_xfilter(:,e) = -(Gamma_matrix^(-2))*(grad);
        X_tilde_xFilter(:,e) = X_xFilter(:,e) -  d_xfilter(:,e);
    end
    
    for t = 1:num_iter
        %disp(t);
        X_xFilter_old = X_xFilter;
        if (flag == 1) %this allows us to turn the filtering step on or off
            % step S2 (filtering)
            
            %d_xfilter = randn(size(d_xfilter));
            u_0_xFilter = X_xFilter;%randn(size(X_xFilter));
            u_1_xFilter = ((eye(numE) - tau*R)*(X_xFilter'))' + tau*d_xfilter;
            u_2_XFilter = u_1_xFilter;
            alpha_xFilter = 2;

            for s = 2:Q
                alpha_xFilter =  1;%4 / (4 - alpha_xFilter*(rho_0^2)); % the suggestion given by the authors doesn't seem to work
                u_2_XFilter = alpha_xFilter* ((eye(numE) - tau*R)*(u_1_xFilter'))' +	 (1 - alpha_xFilter)*u_0_xFilter + tau*alpha_xFilter*d_xfilter;
                u_1_xFilter = u_2_XFilter;
                u_0_xFilter = u_1_xFilter;
                %disp(norm((inv(R)*d_xfilter')' - u_2_XFilter));
            end        
            
            X_xFilter = u_2_XFilter;
        else
            X_xFilter = (inv(R)*d_xfilter')'; % if we use this equation then we are basically doing step S2 in a centralized manner. This should help the algorithm
        end
        
        % step S3 prediction
        X_tilde_xFilter_old = X_tilde_xFilter;
        grad_sum = 0;
        for e = 1:numE
            grad = GradFPair( reshape(X_xFilter(:,e),dim,numV)  ,e,  D(E1(e),E2(e)) ,numV,delta,target, E1,E2);
            grad_sum = grad_sum + grad;
            grad = reshape(grad, dim*numV,1);
            X_tilde_xFilter(:,e) = X_xFilter(:,e) - (Gamma_matrix^(-2))*(grad);
        end
        %disp(norm(grad_sum/numE));
        
        % step S4 tracking
        d_xfilter = d_xfilter + (X_tilde_xFilter - X_xFilter) - (((Gamma_matrix.^(-2))*(Sigma_matrix.^(2))*Lap_hat)*X_xFilter')';
        
        
        % update the history
      
        AveX = mean(X_xFilter,2); % this is the mean computed over all of the nodes. We use this mean to check on the performance of the algorithm
        AveX = reshape(AveX, dim, numV);
        
        obj = compute_objective(AveX,D,2,2,E1,E2,delta,target); % here we could have computed the objective of local (edge) estimate that is performing the worse
        
        evol_obj(t) = obj;
        evol_X_aveg(:,:,t) = AveX;
       
        if (num_iter - t < num_iter_last_hist)
           evol_AveTheta( : , : , num_iter_last_hist - (num_iter - t) ) = AveX; 
        end
        
        
    end
    
    std_X_filter = std(X_xFilter,0,2);
    
end