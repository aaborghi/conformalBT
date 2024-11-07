%% schroedinger h2

clear; clc;
rng(42)

% constructing the Schroedinger equation
nx = 1000; %1000
m = 2;
q = 2;
xa = 0;
xb = 1;
nu = 1;
hx = (xb-xa)/(nx+1);
xd = xa+hx:hx:xb-hx;
ex = ones(nx,1);
I = speye(nx);
Laplace_x = 1/hx^2*spdiags([ex -2*ex ex], -1:1, nx, nx);
e1x = I(:,1);
enx = I(:,nx);
O = sparse(nx,nx);
A = nu*Laplace_x;
B = zeros(nx,m);
C = zeros(q,nx);

for i = 1:nx
    if i*hx >= 0.4 && i*hx <= 0.5
        B(i,1) = 1;  
    end
    if i*hx >= 0.5 && i*hx <= 0.6
        B(i,2) = 1; 
    end
    if i*hx >= 0.1 && i*hx <= 0.3
        C(1,i) = hx;
    end
    if i*hx >= 0.7 && i*hx <= 0.9
        C(2,i) = hx;
    end
end
C = sparse(C);
B = sparse(B);
n = size(A,1);
A=-1i*A;


% Computing the ROM with conformalBT on a disk
U = lyapchol(1i*A, B);
L = lyapchol((1i*A)', C');
[Z,S,Y] = svd(L*U', 'econ');

% define conformal map
psi = @(x) -1i*x;
dpsi = @(x) -1i;

% balancing the FOM
[V,Sigma,~] = svd((U'*U)); 
Kinv = @(x) (psi(x)/sqrt(dpsi(x)))*speye(n) - A/sqrt(dpsi(x));
Kb = @(x) V'*(Kinv(x)*V);
Cb = C*V;
Bb = V'*B; 

% construct the ROM with conformalBT
rset = 2:2:20;

for i = 1:size(rset,2)
    r = rset(i)

    Z1 = Z(:,1:r);
    Y1 = Y(:,1:r);
    S1 = S(1:r,1:r);  S1half = sqrt(S1);
    
    Wr = L'*Z1/S1half;
    Vr = U'*Y1/S1half;

    Ar = Wr'*A*Vr;
    Br = Wr'*B;
    Cr = C*Vr;

    [H2A(i),~] = H2Anorm(A,B,C,Ar,Br,Cr,psi,dpsi);
    
    % Running conformalIRKA
    phi = @(z) (conj(z)); 
    init = -500i -1000i*rand(r,1);
    [Ar_,Br_,Cr_,~] = conformalIRKA(A,B,C,r,phi,init,500);

    [H2A_(i),~] = H2Anorm(A,B,C,Ar_,Br_,Cr_,psi,dpsi);

    computing the bound
    w = [-logspace(0,5,1000)+1,logspace(0,5,1000)-1];
    partbound = zeros(1,size(w,2));
    for j = 1:1:size(w,2)
        Kbal = Kb(1i*w(j)); 
        Kr = Kbal(1:r,1:r);
        K12 = Kbal(1:r,r+1:end);
        N = Kr\K12;
        partbound(j) = norm((Cb(:,1:r)*N)' * (Cb(:,1:r)*N-2*Cb(:,r+1:end)));
    end


    bound(i) = trace(Cb(:,r+1:end)*Sigma(r+1:end, r+1:end)*(Cb(:,r+1:end)')) + ...
        max(partbound)*trace(Sigma(r+1:end, r+1:end));

    fprintf('reduced order %i, bound %f, H2A %f\n', r, bound(i), H2A(i));
end


bound = sqrt(bound);

%% Plots

figure()
semilogy(rset, H2A, 'r-x', 'Linewidth', 1.5); hold on
semilogy(rset, H2A_, 'b--o', 'Linewidth', 1.5);
semilogy(rset, bound, 'k--','Linewidth', 1.5); hold off
xlabel('$r$', 'Interpreter','latex');
ylabel('$\mathcal{H}_2(A)$ error', 'Interpreter','latex');
legend('conformalBT','conformalIRKA','$\mathcal{H}_2$ bound','fontsize',20, 'interpreter','latex', 'Location', 'northwest')
ax = gca;
ax.FontSize = 14;