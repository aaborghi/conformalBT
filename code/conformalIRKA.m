function [Ar,Br,Cr,sigma] = conformalIRKA(A, B, C, r, phi, init, max_iters)
    % Implements the Iterative Rational Krylov Algorithm (IRKA) for MIMO systems
    %
    % Inputs:
    %   A - system matrix (n x n)
    %   B - input matrix (n x m)
    %   C - output matrix (p x n)
    %   r - reduced order of the system
    %   phi - interpolation function
    %   init - inital interpolation points
    %   max_iters - maximum number of iterations
    %
    % Outputs:
    %   Ar - reduced system matrix (r x r)
    %   Br - reduced input matrix (r x m)
    %   Cr - reduced output matrix (p x r)

    % Step 1: Initialize interpolation points (sigma) and tangential directions
    tol = 1e-6;
    [n, m] = size(B);
    p = size(C, 1);
    sigma = init;       % Initial interpolation points (you can modify this)
    new_sigma = zeros(size(sigma));
    L = randn(m, r);    % Initial tangential directions for inputs
    R = randn(p, r);    % Initial tangential directions for outputs

    V = zeros(n, r);
    W = zeros(n, r);
    
    for iter = 1:max_iters
        for i = 1:r
            V(:,i) = (A - sigma(i) * speye(n)) \ (B * L(:,i));
            W(:,i) = ((A - sigma(i) * speye(n))') \ (C' * R(:,i));
        end
        [V, ~] = qr(V, 0);
        [W, ~] = qr(W, 0);
        Ar = (W'*V)\(W' * A * V);
        Br = (W'*V)\(W' * B);
        Cr = C * V;
        
        s_old = sigma;
        [X,s] = eig(Ar);
        s = diag(s);
        for i=1:r
            new_sigma(i) = phi(s(i));
        end
        
        conv_crit = norm(sort(new_sigma)-sort(s_old))/norm(s_old);
        sigma = new_sigma;
        L = (X\Br).';
        R = Cr*X;
        % L = Br'*X;
        % R = Cr*Y;
        fprintf('Iteration %d - Convergence %f \n', iter, conv_crit);
        if conv_crit < tol
            break;
        end
    end
    sigma = new_sigma;
end
