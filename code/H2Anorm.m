function [abserror,relerror] = H2Anorm(A,B,C,Ar,Br,Cr,psicom,dpsi)
% [abserror,relerror] = H2Anorm(A,B,C,Ar,Br,Cr,psicom,dpsi)
% computes the H2Abar error norm 
% 
% INPUTS
% A,B,C = full order model system matrices
% Ar,Br,Cr = reduced order model system matrices
% psicom = conformal map
% dpsi = derivative of the conformal map
% 
% OUTPUTS
% abserror = absolute H2(\bar A^c) error norm
% relerror = relative H2(\bar A^c) error norm

n = length(A);
r = length(Ar);
fom = @(z) (C*((psicom(1i.*z)*speye(n)-A)\B)) .* sqrt(dpsi(1i.*z));
rom = @(z) (Cr*((psicom(1i.*z)*speye(r)-Ar)\Br)) .* sqrt(dpsi(1i.*z));
funerror = @(z) trace((fom(z)-rom(z))*(fom(z)-rom(z))');
funfom = @(z) trace(fom(z)*(fom(z)'));
funrom = @(z) trace(rom(z)*(rom(z)'));
abserror = sqrt((1/(2*pi))*integral(funerror,-Inf,Inf,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true));
relerror = abserror/sqrt((1/(2*pi))*integral(funfom,-Inf,Inf,'RelTol',1e-8,'AbsTol',1e-12,'ArrayValued',true));
end