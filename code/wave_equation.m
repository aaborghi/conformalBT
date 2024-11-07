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

r = 40; % reduced order
R = 1+1e-5; % the more this parameter approaches 1 the more the ellipse is thin
M = 1e4; % multiplication factor for the ellipse
c = 1e-6; % center of the ellipse
r1 = 0.5*(R+inv(R)); % major axis of the ellipse
r2= 0.5*(R-inv(R)); % minor axis of the ellipse
t = linspace(0,2*pi,500);


psi = @(x) c + 0.5*1i*M.*(R.*(x+1)./(x-1)+(x-1)./(R.*(x+1))); % conformal map and its derrivative
dpsi = @(x) 1i*M.*(-R./((x-1).^2)+1/(R.*(x+1).^2));

tol = 1e-6;


tic;
[~,Xcgk] = adaptive_gk_specific_con2(AA,BB,psi,dpsi,tol,[-1,1]);
Xc = (Xcgk)/(2*pi); % controllability Gramian
toc


tic;
[~,Xogk] = adaptive_gk_specific_obs2(AA,CC,psi,dpsi,tol,[-1,1]);
Xo = (Xogk)/(2*pi); % observability Gramian
toc

%%

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

% conformalIRKA
init = 0.5 + 10i*randn(r,1);
[Ar_,Br_,Cr_,~] = conformalIRKA(AA,BB,CC,r,@phi_j,init,100);
eig(Ar_)


% Computing systems output trajectories for impulse response
dynamics = @(t,x,A,B) A*x;
options = odeset('RelTol',1e-8,'AbsTol',1e-12);
% [t1,x] = ode23(dynamics,linspace(0,5,1000),BB*ones(m,1),options,AA,BB);
% y1 = CC*x';
[t2,xr] = ode23(dynamics,linspace(0,5,1000),Br*ones(m,1),options,Ar,Br);
y2 = Cr*xr.';
error = y1-y2;

[~,xr_] = ode23(dynamics,linspace(0,5,1000),Br_*ones(m,1),options,Ar_,Br_);
y2_ = Cr_*xr_.';
error_ = y1-y2_;


%% Plots
figure()
set(gcf,'position',[100,100,1100,500])
subplot(3,1,1)
plot(t1,y1(1,:),'r-', 'Linewidth', 3); hold on
plot(t2,y2(1,:),'b--', 'Linewidth', 3);
plot(t2,y2_(1,:),'k:', 'Linewidth', 3); hold off
ax = gca;
ax.FontSize = 14;
legend('$y(t)$','\texttt{conformalBT}','\texttt{conformalIRKA}', 'interpreter','latex', 'Location', 'southeast')

subplot(3,1,2)
plot(t1,y1(2,:),'r-', 'Linewidth', 3); hold on
plot(t2,y2(2,:),'b--', 'Linewidth', 3);
plot(t2,y2_(2,:),'k:', 'Linewidth', 3); hold off
ax = gca;
ax.FontSize = 14;

subplot(3,1,3)

froerror = zeros(1,size(t1,1));
for i = 1:size(t1,1)
    froerror(i) = norm(y1(:,i) - y2(:,i),'fro');
end
froerror_ = zeros(1,size(t1,1));
for i = 1:size(t1,1)
    froerror_(i) = norm(y1(:,i) - y2_(:,i),'fro');
end


plot(t1,froerror,'k', 'Linewidth', 1.5); hold on
plot(t1,froerror_,'r--', 'Linewidth', 1.5); hold off
ax = gca;
ax.FontSize = 14; 
ylabel('$|y(t)-y_r(t)|$','fontsize',20, 'interpreter','latex')
xlabel('time [s]','fontsize',20,'interpreter','latex')
legend('\texttt{conformalBT}','\texttt{conformalIRKA}', 'interpreter','latex', 'Location', 'southeast', 'NumColumns',2)



%% Function for the computation of the interpolation points
% This function is equal to \phi used to compute the interpolation points 
% for systems with poles inside a Bernstein ellipse
function snew = phi_j(s)
            R = 1+1e-5; % the more this parameter approaches 1 the more the ellipse is thin
            M = 1e4; % multiplication factor for the ellipse
            c = -5e-3; % center of the ellipse
            true_sqrt = sqrt((-1i*(s-c)/M)^2-1);
            z1 = R/((-1i*(s-c)/M+true_sqrt)/R);
            z1 = c+1i*0.5*M*conj((z1+1/z1));
            z2 = R/((-1i*(s-c)/M-true_sqrt)/R);
            z2 = c+1i*0.5*M*conj((z2+1/z2));
            % The switching of the sign is due to numerical reasons
            % Here we are choosing the positive root of the inverse
            % Joukowski transform
            if real(-1i*(s-c)/M) < 0 
                snew = z2;
            else
                snew = z1;
            end
end
