function [test_targets, w] = LS(train_patterns, train_targets, test_patterns, weights)

% Classify using the least-squares algorithm
% Inputs:
% 	train_patterns  - Train patterns
%	train_targets	- Train targets
%   test_patterns   - Test patterns
%	Weights	- Weighted for weighted least squares (Optional)
%
% Outputs
%	test_targets	- Predicted targets
%	w			    - Decision surface parameters
%
%See also Perceptron, LMS
%
%Example:
% load clouds
% t = LMS(patterns, targets, patterns, []);
% disp(mean(t == targets))

[Dim, Nf]       = size(train_patterns);
Dim             = Dim + 1;
train_patterns(Dim,:) = ones(1,Nf);
test_patterns(Dim,:)  = ones(1,size(test_patterns,2));

%Weighted LS or not?
if (nargin < 4)
    weights = ones(1, Nf);
else
    switch length(weights),
    case Nf + 1,
        %Ada boost form
        weights = weights(1:Nf);
    case Nf,
        %Do nothing
    otherwise
        weights = ones(1, Nf);
    end
end
 X=(train_patterns .* (ones(Dim,1)*weights)) * train_patterns';
 epsilon=1e-8;
w               = ( inv(X+epsilon*eye(Dim))*(train_patterns .* (ones(Dim,1)*weights)) * train_targets')';
test_targets    = w * test_patterns;

%If there are only two classes, collapse the targets to classes {1,2}
if (length(unique(train_targets)) == 2)
   test_targets = (test_targets > mean(unique(train_targets)));
end