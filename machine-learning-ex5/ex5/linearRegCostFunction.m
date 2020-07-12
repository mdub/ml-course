function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% "X" (Mx2) = inputs, plus bias term
% "y" (Mx1) = outputs
% "theta" (2x1) = model parameters
% "lambda" (1x1) = regularisation parameter

% compute:
%   "J" (1x1)
%   "grad" (2x1) = gradients for theta

% "h" (Mx1) = hypotheses
h = X * theta;

% "theta_reg" is "theta" with the first term zeroed
theta_reg = theta;
theta_reg(1,1) = 0;

% "J" is sum of squared errors (divided by 2*m), plus regularisation
J = (
        sum((h - y) .^ 2) +
        sum(theta_reg .^ 2) * lambda
    ) / m / 2;

% "grad" is the derivative
grad = (
        sum((h - y) .* X)' + 
        theta_reg * lambda
    ) / m;

% =========================================================================

grad = grad(:);

end
