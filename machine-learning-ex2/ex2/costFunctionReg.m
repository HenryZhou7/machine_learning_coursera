function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta); % number of features

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Cost Function
sigma1 = 0;
sigma2 = 0;

for i = 1:m,
  temp_h = sigmoid(theta' * X(i, :)');
  sigma1 = sigma1 + (-y(i) * log(temp_h) - (1 - y(i)) * log(1 - temp_h));
end

for j = 2:n,
  sigma2 = sigma2 + theta(j) ^ 2;
end

J = sigma1 / m + lambda * sigma2 / (2 * m);



% Gradient
% Calculate the gradient with respect to theta0
sigma = 0;
for i = 1:m,
  temp_h = sigmoid(theta' * X(i, :)');
  sigma = sigma + (temp_h - y(i)) * X(i, 1);
end
grad(1) = sigma / m;

% Calculate the rest of the vector
for j = 2:n,
  sigma = 0;
  
  for i = 1:m,
    temp_h = sigmoid(theta' * X(i, :)');
    sigma = sigma + (temp_h - y(i)) * X(i, j);
  end
  
  grad(j) = sigma / m + lambda * theta(j) / m;

end  
  






% =============================================================

end
