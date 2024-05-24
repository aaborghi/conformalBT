%% wave equation

clear; clc;
rng(42)

% Constructing the discretized wave equation
nx = 2500; 
n = 2*nx; 
m = 2;
q = 2;
xa = 0;
xb = 1;
nu = 1;
damping = 0;
hx = (xb-xa)/(nx+1);
xd = xa+hx:hx:xb-hx;
ex = ones(nx,1);
I = speye(nx);
Laplace_x = 1/hx^2*spdiags([ex -2*ex ex], -1:1, nx, nx);
e1x = I(:,1);
enx = I(:,nx);
O = sparse(nx,nx);
A = nu*Laplace_x;
AA = [O,I;A,-damping*I];
AA = sparse(AA);
B = zeros(nx,m);
C = zeros(q,nx);
for i = 1:nx
    if i*hx >= 0.1 && i*hx <= 0.2
        B(i,1) = 1; 
    end
    if i*hx >= 0.8 && i*hx <= 0.9
        B(i,2) = 1;
    end
    if i*hx >= 0.3 && i*hx <= 0.5
        C(1,i) = hx; 
    end
    if i*hx >= 0.6 && i*hx <= 0.7
        C(2,i) = hx;
    end
end

BB = [zeros(nx,m);B];
BB = sparse(BB);
CC = [C,zeros(q,nx)];
CC = sparse(CC);
II = speye(n);

% Running Algorithm conformalBT with ellipse
r = 40; % reduced order
R = 1+1e-5; % the more this parameter approaches 1 the more the ellipse is thin
M = 1e4; % multiplication factor for the ellipse
c = 1e-6; % center of the ellipse
r1 = 0.5*(R+inv(R)); % major axis of the ellipse
r2= 0.5*(R-inv(R)); % minor axis of the ellipse
t = linspace(0,2*pi,500);

% conformal map and its derrivative
psi = @(x) c + 0.5*1i*M.*(R.*(x+1)./(x-1)+(x-1)./(R.*(x+1)));
dpsi = @(x) 1i*M.*(-R./((x-1).^2)+1/(R.*(x+1).^2));

tol = 1e-6;

% controllability Gramian
tic;
[~,Xcgk] = adaptive_gk_specific_con2(AA,BB,psi,dpsi,tol,[-1,1]);
Xc = (Xcgk)/(2*pi);
toc

% observability Gramian
tic;
[~,Xogk] = adaptive_gk_specific_obs2(AA,CC,psi,dpsi,tol,[-1,1]);
Xo = (Xogk)/(2*pi);
toc


% conformalBT
U = lrcf(Xc, 1e-12);
L = lrcf(Xo, 1e-12);
[Z,S,Y] = svd(L*U', 'econ');

Z1 = Z(:,1:r);
Y1 = Y(:,1:r);
S1 = S(1:r,1:r);  S1half = sqrt(S1);

Wr = L'*Z1/S1half;
Vr = U'*Y1/S1half;

Ar = (Wr'*AA*Vr);
Br = (Wr'*BB);
Cr = CC*Vr;

eig(Ar)

% Computing systems output trajectories for impulse response
dynamics = @(t,x,A,B) A*x;
options = odeset('RelTol',1e-8,'AbsTol',1e-12);
[t1,x] = ode23(dynamics,linspace(0,5,1000),BB*ones(m,1),options,AA,BB);
y1 = CC*x';
[t2,xr] = ode23(dynamics,linspace(0,5,1000),Br*ones(m,1),options,Ar,Br);
y2 = Cr*xr.';
y3 = y1-y2;

%% Plots
figure()
set(gcf,'position',[100,100,1100,500])
subplot(2,1,1)
plot(t1,real(y1),'r-', 'Linewidth', 3); hold on
plot(t2,real(y2),'b--', 'Linewidth', 3); hold off
ax = gca;
ax.FontSize = 14;
legend('$y(t)$','fontsize',20, 'interpreter','latex', 'Location', 'southeast')
subplot(2,1,2)

forerror = zeros(1,size(t1,1));
for i = 1:size(t1,1)
    froerror(i) = norm(y1(:,i) - y2(:,i),'fro');
end


plot(t1,froerror,'k', 'Linewidth', 1.5);
ax = gca;
ax.FontSize = 14; 
ylabel('$|y(t)-y_r(t)|$','fontsize',20, 'interpreter','latex')
xlabel('time [s]','fontsize',20,'interpreter','latex')
legend('\texttt{conformalBT}','fontsize',20, 'interpreter','latex', 'Location', 'southeast', 'NumColumns',2)

