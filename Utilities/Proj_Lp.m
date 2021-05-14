function [x] = Proj_Lp(y, p, indicator)
% Lp-ball projection scheme, solving the following optimization problem:
% min_{x}  0.5*||x-y||_F^2, s.t., ||x||_p <= indicator.
% ---------------------------------------------
% Input:
%  y:         n*1 vector
%  p:         constant in the Lp-ball projection scheme (0<p<=1 or p=2)
%  indicator: tolerance parameter to control the Lp-norm of the fitting error (indicator>0)
% ---------------------------------------------
% Output:
%  x:         n*1 vector
% ---------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%

if norm(y,p) <= indicator
 x = y; 
 
elseif p > 0 && p < 1
 tol = 1e-8;     % this one can be tuned
 num_outer = 8;  % this one can be tuned
 num_inner = 2;  % this one can be tuned
 abs_y = abs(y);   
 bisect_left = 0;  
 bisect_right = max(abs_y)^(2 - p) / ((2 - p)^(2 - p) * (2 - 2 * p)^(p - 1));
 temp = (2 - p) * (2 - 2 * p)^((p - 1) / (2 - p));
 
 %% The outer iteration is to find the Lagrange multiplier 'theta' by the bisection method.
 for iter_outer = 1 : num_outer
  theta = (bisect_left + bisect_right) / (2 + 8 * (iter_outer == 1));
%   theta = (bisect_left + bisect_right) / 2;

  %% The inner iteration is to find the larger root of the equation 'x + theta * p * x^(p-1) = |y|' by the Newton (or biscetion) method.
  subset = abs_y >  temp * theta^(1 / (2 - p));
  y_subset = abs_y(subset);
  x_subset = y_subset;  % initialization of the larger root of the equation
  for iter_inner = 1 : num_inner
   Temp = theta * p * x_subset.^(p - 2);
   x_subset = x_subset - (x_subset .* (Temp + 1) - y_subset) ./ ((p - 1) * Temp + 1);
  end

  norm_p = norm(x_subset, p);
  if  abs(norm_p - indicator) / length(abs_y) / indicator < tol
   break;
  elseif norm_p < indicator  
   bisect_right = theta;  % thete needs to be smaller in the next iteration.
  else   
   bisect_left = theta;   % thete needs to be larger in the next iteration.
  end
 end
 x = zeros(size(y)); 
 x(subset) = x_subset;
 x = x .* sign(y); 
 
 elseif p == 1
  tol = 1e-4;  % this one can be tuned
  abs_y = abs(y);
  theta = max(abs_y) / 2;  % initialization of the Lagrange multiplier 'theta' 
  
  %% The iteration is to find the Lagrange multiplier 'theta' by the Newton (or biscetion) method.
  while 1
   subset = abs_y(abs_y >= theta); 
   temp = length(subset);
   if (abs(-theta * temp + sum(subset) - indicator) <= tol)
     break;
   end
   theta = (sum(subset) - indicator) / temp;
  end
  x = (max(0, abs_y - theta)) .* sign(y);
  
 elseif p == 2
  x = (indicator / norm(y,p)) .* y; 
 end
 
end
