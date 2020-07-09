function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.1;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
% if true to disable training, only to perserve pretty indentation :
if false
    C_vect = [0.01,0.03,0.1,0.3,1,3,10,30];
    sigma_vect = [0.01,0.03,0.1,0.3,1,3,10,30];
    x1 = [1 2 1]; x2 = [0 4 -1];

    err_matrix = zeros(length(C_vect), length(sigma_vect));


    for i = 1:length(C_vect)
        for j = 1:length(sigma_vect)

            model= svmTrain(X, y, C_vect(i), @(x1, x2) gaussianKernel(x1, x2, sigma_vect(j))); 
            predictions = svmPredict(model, Xval);

            err_matrix(i,j) = mean(double(predictions ~= yval));
        endfor
    endfor

    minimum = min(min(err_matrix));
    [x,y]=find(err_matrix==minimum);
    disp(C_vect(x));
    disp(sigma_vect(y));
end
% =========================================================================

end
