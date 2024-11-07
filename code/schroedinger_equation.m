%% schroedginer trajectories

clear; clc;
rng(42)

% Constructing the discretized Schroedinger equation
nx = 1000;
m = 2;
q = 2;
xa = 0;
xb = 1;
nu = 1;
hx = (xb-xa)/(nx+1);
ex = ones(nx,1);
Laplace_x = 1/hx^2*spdiags([ex -2*ex ex], -1:1, nx, nx);
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

% Running Algorithm conformalBT
r = 6; %6

U = lyapchol(1i*A, B);
L = lyapchol((1i*A)', C');
[Z,S,Y] = svd(L*U', 'econ');

Z1 = Z(:,1:r);
Y1 = Y(:,1:r);
S1 = S(1:r,1:r);  S1half = sqrt(S1);

Wr = L'*Z1/S1half;
Vr = U'*Y1/S1half;

Ar = (Wr'*A*Vr);
Br = (Wr'*B);
Cr = C*Vr;

eig(Ar)


% Running conformalIRKA
phi = @(z) (conj(z)); 
init = -500i -1000i*rand(r,1);
[Ar_,Br_,Cr_,sigma] = conformalIRKA(A,B,C,r,phi,init,500);

eig(Ar_)


% Computing systems output trajectories
inputu = @(t) exp(-(t-1).^2./0.1)-2*exp(-(t-3).^2./0.01);
dynamics = @(t,x,A,B) A*x+B*ones(m,1)*inputu(t);
options = odeset('RelTol',1e-8,'AbsTol',1e-12); %1e-8
[t1,x] = ode23(dynamics,linspace(0,5,1000),zeros(nx,1),options,A,B);
y1 = C*x.';
[~,xr] = ode23(dynamics,linspace(0,5,1000),zeros(r,1),options,Ar,Br);
y2 = Cr*xr.';
error = y1-y2;
[~,xr_] = ode23(dynamics,linspace(0,5,1000),zeros(r,1),options,Ar_,Br_);
y2_ = Cr_*xr_.';
error_ = y1-y2_;
gaussinput = inputu(t1); 

froerror = zeros(1,size(t1,1));
froerror_ = zeros(1,size(t1,1));
for i = 1:size(t1,1)
    froerror(i) = norm(y1(:,i) - y2(:,i),'fro')/norm(y1(:,i),'fro');
end
for i = 1:size(t1,1)
    froerror_(i) = norm(y1(:,i) - y2_(:,i),'fro')/norm(y1(:,i),'fro');
end


%% Plots
figure()
set(gcf,'position',[100,100,1100,800])
subplot(3,1,1)
plot(t1,real(y1),'r-', 'Linewidth', 3); hold on
plot(t1,real(y2),'b--', 'Linewidth', 3);
plot(t1,imag(y1),'r-.', 'Linewidth', 3);
plot(t1,imag(y2),'b:', 'Linewidth', 3);
title(['\fontsize{14}{0}\selectfont Schr\"odinger equation'],'Interpreter','latex')
ax = gca;
ax.FontSize = 18; 
legend({['\fontsize{13}{0}\selectfont Re$\{y(t)\}$'],['\fontsize{13}{0}\selectfont Re$\{\widehat{y}_r(t)\}$'],['\fontsize{13}{0}\selectfont Im$\{y(t)\}$'],['\fontsize{13}{0}\selectfont Im$\{\widehat{y}_r(t)\}$']},'fontsize',20, 'interpreter','latex', 'Location', 'northwest', 'NumColumns',4)
subplot(3,1,2)
semilogy(t1,froerror,'k', 'Linewidth', 3); hold on
semilogy(t1,froerror_,'r--', 'Linewidth', 3);
ax = gca;
ax.FontSize = 18; 
ylabel(['\fontsize{14}{0}\selectfont $|(y(t)-\hat{y}_r(t))/y(t)|$'], 'interpreter','latex')
legend('conformalBT','Extended IRKA','fontsize',22, 'interpreter','latex', 'Location', 'southeast', 'NumColumns',2)
subplot(3,1,3)
plot(t1,gaussinput,'k:', 'Linewidth', 3);
ylabel(['\fontsize{14}{0}\selectfont $u(t)$'], 'interpreter','latex')
xlabel(['\fontsize{14}{0}\selectfont time [s]'],'interpreter','latex')
ax = gca;
ax.FontSize = 18; 
