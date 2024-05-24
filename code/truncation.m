function T = truncation(S,trunctol)
[Q,R] = qr(S.L,0);
tmp = R*S.D*R';
tmp = 0.5*(tmp+tmp'); % important to ensure *exact* symmetry, may break down otherwise ! 
[U,D] = eig(tmp);
d = diag(D);
[~,y] = sort(abs(d),'descend');
U = U(:,y);
D = diag(d(y));
index = find(diag(D)/D(1,1) > trunctol,1,'last');
T.L = Q*U(:,1:index);
T.D = D(1:index,1:index);
end