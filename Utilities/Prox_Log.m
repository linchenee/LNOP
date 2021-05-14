function [X] = Prox_Log(A, lambda, epsilon)
% The proximal operator of the tensor logarithmic norm, solving the following optimization problem:
% min_{X}  0.5*||X-A||_F^2 + lambda*||X||_L
% ---------------------------------------------
% Input:
%  A:        n1*n2*n3 tensor
%  lambda:   penalty parameter (lambda>0)
%  epsilon:  constant in the definition of tensor logarithmic norm (epsilon>0)
% ---------------------------------------------
% Output:
%  X:        n1*n2*n3 tensor
% ---------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%

Af = fft(A, [], 3);
n3 = size(A, 3);
half = round(n3/2);
even = (mod(n3,2) == 0);

[U, S, V] = svd(Af(:,:,1),'econ');
S1 = Shink_Log(diag(S), lambda, epsilon);
Xf(:,:,1) = U * diag(S1) * V';

for i = 2 : half 
 [U, S, V] = svd(Af(:,:,i),'econ');
 S1 = Shink_Log(diag(S), lambda, epsilon);
 Xf(:,:,i) = U * diag(S1) * V';
 Xf(:,:,n3-i+2) = conj(Xf(:,:,i));
end

if even
 [U, S, V] = svd(Af(:,:,half+1),'econ');
 S1 = Shink_Log(diag(S), lambda, epsilon);
 Xf(:,:,half+1) = U * diag(S1) * V';
end

X = ifft(Xf, [], 3);

end
 

function [x] = Shink_Log(s, lambda, epsilon)
% Shrinkage of singular values
% ---------------------------------------------
% Input:
%  s:        n*1 vector
%  lambda:   penalty parameter (lambda>0)
%  epsilon:  constant in the definition of tensor logarithmic norm (epsilon>0)
% ---------------------------------------------
% Output:
%  x:        n1*1 vector
% ---------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%

n = length(s);
x = zeros(n,1);
delta = (s + epsilon).^2 - 4 * lambda;

for i = 1 : n
 if delta(i) > 0
  temp = sqrt(delta(i));
  if temp > epsilon - s(i) && ...
     0.125 * (temp - s(i) - epsilon).^2 + lambda * log((s(i) + temp + epsilon) / 2 / epsilon) < 0.5 * s(i).^2
    x(i) = 0.5 * (temp + s(i) - epsilon);
   end
  end
end
 
end
