function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Forward propagate through the layers
% "a1" (5000x401) = first layer
a1 = addOnes(X);
% "a2" (5000x26) = hidden layer
a2 = addOnes(sigmoid(a1 * Theta1'));
% "a3" (5000x10) = final (output) layer
a3 = sigmoid(a2 * Theta2');

% y is a 5000x1 matrix, mapping: example -> label
% to compute J efficiently we want a 5000x10 matrix mapping: example -> (col == y)
% "actual" (5000x10) = logic array of output labels
actual = eye(num_labels)(y,:);

% now calculate the diff between actual and predicted
predicted = a3;
cost_parts = -actual .* log(predicted) - (1 - actual) .* log(1 - predicted);
base_cost = sum(sum(cost_parts)) / m;

% regularise theta, apart from the bias terms
theta_reg = [ Theta1(:,2:end)(:); Theta2(:,2:end)(:) ];
regularisation = sum(theta_reg .^ 2) * lambda / 2 / m;

J = base_cost + regularisation;

% -------------------------------------------------------------

% "d3" (5000x10) = delta between output layer and actual
d3 = a3 - actual;       

% "g2" (5000x26) = sigmoid gradient of a2
g2 = a2 .* (1 - a2); 
% "d2" (5000x25) = element-wise gradient
d2 = ((d3 * Theta2) .* g2)(:,2:end);

% "Theta1_grad" (25x401) = gradient for Theta1
Theta1_grad = d2' * a1 / m;

% "Theta2_grad" (10x26) = gradient for Theta2
Theta2_grad = d3' * a2 / m;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
