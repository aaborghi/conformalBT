function L = lrcf(X,trunc_tol)

[UX,DX] = eig(0.5*(X+X'));
Xtemp.L = UX;
Xtemp.D = DX;
Xa = truncation(Xtemp,trunc_tol);
L = (Xa.L*sqrt(Xa.D))';