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

% "J" is sum of squared errors (divided by 2m)
J = (
        sum((h - y) .^ 2) +
        sum(theta(2:end) .^ 2) / lambda
    ) / m / 2;

% =========================================================================

grad = grad(:);

end
