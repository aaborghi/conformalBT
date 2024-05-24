%% heat equation H2
clear; clc;
rng(42)

% constructing the heat equation
n = 200;
nx = n;
xa = 0;
xb = 1;
hx = (xb-xa)/(nx+1);
ex = ones(nx,1);
Laplace_x = 1/hx^2*spdiags([ex -2*ex ex], -1:1, nx, nx);
AA = Laplace_x;
CC = zeros(1,nx);
for i = 1:nx
    if i*hx >= 0.1 && i*hx <= 0.4
        CC(i) = hx;
    end
end
BB = zeros(nx,1);
BB(end) = 1/hx^2;
BB = sparse(BB);
CC = sparse(CC);


% define center and radius of disk
c = -1.7e5; 
R = 1.7e5;


% compute Gramians
psiinv = @(x) ((x-c*speye(n))/R+speye(n))/((x-c*speye(n))/R-speye(n));

Alyap = psiinv(AA);
Blyap = sqrt(2*R) * ((c*speye(n)+R*speye(n)-AA) \ (BB));
Clyap = sqrt(2*R) * (((c*speye(n)+R*speye(n)-AA)') \ (CC'));

U = lyapchol(Alyap, Blyap);
L = lyapchol(Alyap', Clyap);
[Z,S,Y] = svd(L*U', 'econ');

% define conformal map
psi = @(x) c+R*(x+1)./(x-1);
dpsi = @(x) 2*R/(x-1).^2;

% balancing the FOM
[V,Sigma,~] = svd((U'*U)); 
Kinv = @(x) (psi(x)/sqrt(dpsi(x)))*speye(n) - AA/sqrt(dpsi(x));
Kb = @(x) V'*(Kinv(x)*V);
Cb = CC*V;
Bb = V'*BB; 

% construct the ROM with conformalBT
rset = 4:2:20;

for i = 1:size(rset,2)
    r = rset(i);

    Z1 = Z(:,1:r);
    Y1 = Y(:,1:r);
    S1 = S(1:r,1:r);  S1half = sqrt(S1);
    
    Wr = L'*Z1/S1half;
    Vr = U'*Y1/S1half;

    Ar = Wr'*AA*Vr;
    Br = Wr'*BB;
    Cr = CC*Vr;

    [H2A(i),~] = H2Anorm(AA,BB,CC,Ar,Br,Cr,psi,dpsi);
    
    % search for supremum
    w = [-logspace(0,5,1000)+1,logspace(0,5,1000)-1];
    partbound = zeros(1,size(w,2));
    for j = 1:1:size(w,2)
        Kbal = Kb(1i*w(j)); 
        Kr = Kbal(1:r,1:r);
        K12 = Kbal(1:r,r+1:end);
        N = Kr\K12;
        partbound(j) = norm((Cb(1:r)*N)' * (Cb(1:r)*N-2*Cb(r+1:end)));
    end
    
    
    bound(i) = Cb(r+1:end)*Sigma(r+1:end, r+1:end)*(Cb(r+1:end)') + ...
        max(partbound)*trace(Sigma(r+1:end, r+1:end));
    
    fprintf('reduced order %i, bound %f, H2A %f\n', r, sqrt(bound(i)), H2A(i));
end

bound = sqrt(bound);

%% Plots
figure()
semilogy(rset, H2A, 'r-x', 'Linewidth', 1.5); hold on
semilogy(rset, bound, 'k--','Linewidth', 1.5); hold off
xlabel('$r$', 'Interpreter','latex');
legend('$\|G-G_r\|_{\mathcal{H}_2}$','error bound','fontsize',20, 'interpreter','latex', 'Location', 'northwest')
