
test_PO()

function test_PO()
    p = 1;
    q = 1;

    rho = 3;

    while (1)
        dim = 2;
        N1 = randn(dim,1);
        N2 = randn(dim,1);
        d = rand;

        [xi, xj] = pairPO(p,q,N1,N2,rho,d);

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


function [x1_opt,x2_opt] = pairPO(p,q,N1,N2,rho,d)

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

