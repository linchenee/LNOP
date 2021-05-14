function [X] = LNOP(Y, Omega, opts)
% LNOP algorithm for tensor recovery, solving the following optimization problem:
% min_{X,N}  lambda * ||X||_L, s.t., X + N = Y_omega, ||N_omega||_p <= indicator (0<p<=1 or p=2).
% ---------------------------------------------
% Input:
%  Y:              n1*n2*n3 tensor
%  Omega:          locations of the known entries in the tensor
%  opts.epsilon:   constant in the definition of tensor logarithmic norm (epsilon>0)
%  opts.lambda:    scaling factor in the objective function, used for the flexible shrinkage of singular values (lambda>0)
%  opts.p:         constant in the Lp-ball projection scheme (0<p<=1 or p=2)
%  opts.indicator: tolerance parameter to control the Lp-norm of the fitting error (indicator>0)
%  opts.iter:      iteration number
%  opts.mu:        penalty parameter in the ADMM (mu>0)
%  opts.mu_max:    maximum of the penalty parameter (mu_max>0)
%  opts.rho:       update rate of the penalty parameter (rho>=1)
%  opts.tol:       termination tolerance
% ---------------------------------------------
% Output:
%  X:              n1*n2*n3 recovered tensor
% ---------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%

if isfield(opts,'p');          p         = opts.p;          end
if isfield(opts,'epsilon');    epsilon   = opts.epsilon;    end
if isfield(opts,'lambda');     lambda    = opts.lambda;     end
if isfield(opts,'indicator');  indicator = opts.indicator;  end
if isfield(opts,'iter');       iter      = opts.iter;       end
if isfield(opts,'mu');         mu        = opts.mu;         end
if isfield(opts,'mu_max');     mu_max    = opts.mu_max;     end
if isfield(opts,'rho');        rho       = opts.rho;        end
if isfield(opts,'tol');        tol       = opts.tol;        end

[n1,n2,n3] = size(Y);
omega = zeros(n1,n2,n3);
omega(Omega) = 1;
Y_Omega = Y(Omega);    % vector
Y_omega = Y .* omega;  % tensor

%% Initialization 
X = ones(n1,n2,n3) .* ~omega + Y_omega;  % X=randn(n1,n2,n3);  
N = zeros(n1,n2,n3);  % auxiliary variable
Z = zeros(n1,n2,n3);  % Lagrange multiplier

%% Iteration
for i = 1 : iter 
 %% Update X 
 X = Prox_Log(Y_omega - N + Z / mu, lambda / mu, epsilon);
 
 %% Update N 
 N = Z / mu - X;
 N(Omega) = Proj_Lp(Y_Omega + N(Omega), p, indicator);
     
 %% Update Z   
 Z = Z + mu .* (Y_omega - X - N);

%  %% Stop Criterion   
%  stopCriterion = norm(Y_omega(:) - X(:) - N(:)) / norm(Y_omega(:));
%  if stopCriterion < tol
%   fprintf('The iteration number of LNOP is %d\n', i);
%   break;
%  end

 mu = min(rho * mu, mu_max);
end

end
