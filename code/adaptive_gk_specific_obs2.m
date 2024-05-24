function [G,GK] = adaptive_gk_specific_obs2( AA, CC, psi, dpsi,tol, I )
% [G,GK] = adaptive_gk_specific_obs2( AA, CC, psi, dpsi,tol, I )
% This function computes the observability Gramian for the wave equation
% numerical example through an adaptive Gauss-Kronrod scheme. 
% INPUTS 
% - AA: is the A matrix of the LTI system
% - CC: output matrix C of the LTI system
% - psi, dpsi: conformal map and its derivative (as function handle)
% - tol: chosen tolerance for the algorithm
% - I: integration interval
% OUTPUT
% - G, GK: Gauss and Gauss-Kronrod approximated integrals. For the Gramian
% choose GK


err_est = inf;
noi = 1;
err_good = 0;


n = size(AA,1);
m = size(CC,1);
G = zeros(n,n);
GK = zeros(n,n);
load GK % load gauss-kronrod weights


while err_est > tol
    
    err_bad = 0;
    nobi = 0;     
    for i = 1:size(I,1) 
        
        Gi = zeros(n,n);
        GKi = zeros(n,n);
        
        % part of the integrated function of the Gramian
        Robs = @(z) (CC/(psi(1i.*z)*speye(n)-AA)) * sqrt(dpsi(1i.*z)); 
        a = I(i,1);
        b = I(i,2);
        middle_matrix = zeros(15*m,n);
        w = zeros(1,15); % needed for vectorization process for the weigths
        for j = 1:15
            t = (b-a)/2*nodes(j)+(a+b)/2;
            % multiplication is faster when matrix is full
            middle_matrix(((j-1)*m+1):(j*m),:) = full(Robs(t./(1-t.^2)));
            w(j) = ((t.^2+1)./(1-t.^2).^2);
        end
        % computing weights as matrix
        weightsG = kron(diag(w_G)*sparse(diag(w(1:7))),speye(m));
        weightsGK = kron(diag(w_GK)*sparse(diag(w)),speye(m));
        % computing the Gauss and Gauss-Kronrod quadrature
        Gi = middle_matrix(1:(7*m),:)' * weightsG * middle_matrix(1:(7*m),:);
        GKi = middle_matrix' * weightsGK * middle_matrix;
        
        Gi = ((b-a)/2)*Gi; 
        GKi = ((b-a)/2)*GKi; 
        % computing the error estimate
        err_est_i = norm(Gi-GKi,'fro')/norm(GKi,'fro');
        
        % if result is bad then refine
        if (size(I,1)*err_est_i < tol-err_good)
            G = G + Gi; 
            GK = GK + GKi;

        else
            noi = noi + 1;
            nobi = nobi+1;
            list_of_bad_int(nobi) = i;
            err_bad = err_bad + err_est_i;
        end
        err_est = err_good + err_bad;      
    end
    Inew = zeros(2*nobi,2);
    for k = 1:nobi
        Inew(2*k-1,:) = [I(list_of_bad_int(k),1),(I(list_of_bad_int(k),1)+I(list_of_bad_int(k),2))/2];
        Inew(2*k,:) = [(I(list_of_bad_int(k),1)+I(list_of_bad_int(k),2))/2,I(list_of_bad_int(k),2)];
    end
    I = Inew;
    err_est
    
end
% toc

end
