%% heat equation
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


% Running Algorithm conformalBT with disk
r = 10; 
c = -1.7e5; 
R = 1.7e5;

psiinv = @(x) ((x-c*speye(n))/R+speye(n))/((x-c*speye(n))/R-speye(n));
Alyap = psiinv(AA);
Blyap = sqrt(2*R) * ((c*speye(n)+R*speye(n)-AA) \ (BB));
Clyap = sqrt(2*R) * (((c*speye(n)+R*speye(n)-AA)') \ (CC'));

U = lyapchol(Alyap, Blyap);
L = lyapchol(Alyap', Clyap);

[Z,S,Y] = svd(L*U', 'econ');

Z1 = Z(:,1:r);
Y1 = Y(:,1:r);
S1 = S(1:r,1:r);  S1half = sqrt(S1);

Wr = L'*Z1/S1half;
Vr = U'*Y1/S1half;

Ar = Wr'*AA*Vr;
Br = Wr'*BB;
Cr = CC*Vr;


% Running classic BT
Alyap2 = AA;
Blyap2 = BB;
Clyap2 = CC';

U = lyapchol(Alyap2, Blyap2);
L = lyapchol(Alyap2', Clyap2);
[Z,S,Y] = svd(L*U', 'econ');

Z2 = Z(:,1:r);
Y2 = Y(:,1:r);
S2 = S(1:r,1:r);  S2half = sqrt(S2);

Wr = L'*Z2/S2half;
Vr = U'*Y2/S2half;

Ar2 = (Wr'*AA*Vr);
Br2 = (Wr'*BB);
Cr2 = CC*Vr;


% dynamics impulse response
dynamics = @(t,x,A,B) A*x;
options = odeset('RelTol',1e-8,'AbsTol',1e-12);
[t1,x] = ode23(dynamics,linspace(0,1,1000),BB,options,AA,BB);
y1 = CC*x.';

[t2,xr] = ode23(dynamics,linspace(0,1,1000),Br,options,Ar,Br);
yr = Cr*xr.';
error1 = y1-yr;

[t4,xr2] = ode23(dynamics,linspace(0,1,1000),Br2,options,Ar2,Br2);
yr2 = Cr2*xr2.';
error2 = y1-yr2;

%% Plots
figure
subplot(2,1,1)
plot(t1,y1,'r-', 'Linewidth', 1); hold on
plot(t2,yr,'b--', 'Linewidth', 1);
hold off
legend('$y(t)$','$y_r(t)$','fontsize',20, 'interpreter','latex', 'Location', 'northeast')

subplot(2,1,2)
semilogy(t2,abs((error1)./(y1)),'k', 'Linewidth', 1.5); hold on
semilogy(t2,abs((error2)./(y1)),'--', 'Linewidth', 1.5, 'Color',[0.8500 0.3250 0.0980]);
ylim([1e-11,1e-1])
xlabel('time [s]','fontsize',20,'interpreter','latex')
ylabel('$|y(t)-y_r(t)|/|y(t)|$','fontsize',20, 'interpreter','latex')
legend('\texttt{conformalBT}','BT','fontsize',20, 'interpreter','latex', 'Location', 'northeast')
