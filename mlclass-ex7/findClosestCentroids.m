function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

examples = size(X,1);
for i = 1:examples
    distance = inf;
    % i
    for j = 1:K
        d = sum((centroids(j,:)-X(i,:)).^2);
        if distance > d
            distance = d;
            idx(i) = j;
        end
    end
end

% =============================================================

end


% [Part 1 of EX7 - findClosestCentroids](https://class.coursera.org/ml-004/forum/thread?thread_id=3056)
% Bruno Azevedo Costa· 5 days ago 
% Hey Louise, I was getting into the same result. 
% I'm not sure if you have already done it right, 
% but the issue with my code was that I was computing || X(i) - centroids(j) || ^2,
% while I should have used X(i,:) - centroids(j,:) so that it computes over all features! 

% octave:16> magic(3)
% ans =

%    8   1   6
%    3   5   7
%    4   9   2

% octave:17> magic(3)(1)
% ans =  8
% octave:18> magic(3)(1,:)
% ans =

%    8   1   6

% octave:19> 