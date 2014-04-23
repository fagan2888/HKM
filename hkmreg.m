% Feb 25th Implementing HKM regression as function
function [fhat] = hkmreg(X,Y,B,h)

% Data size
[N,d] = size(X);

% Transform data
normX = sqrt(sum(X.^2,2));
S = X./repmat(normX,1,d);
U = Y(:)./normX;
B = B(:);


% Kernel definition
% ----------------
% Tuning parameters
r = 1;

% Following HKM, using integral function
K = @(p1,h1) 1/((2*pi)^d) * ...
    integral(@(x) cos(p1.*x).*x.^(d-1) .* (1-x.^r*h1.^r),0,1/h1,'ArrayValued',true);


% Numerator stuff
% ------------------------

% Kernel function
G = @(t) (t>=0) .* (t<=1);
% Smoothing parameter
g = .45;


% Denominator stuff
% -----------------------


% MAIN LOOP
denominator = zeros(N,1);

% Evaluation points in numerator: p = S'b-U
p = S*B - U;

% Evaluate kernel at each point in the S grid
numerator = K(p,h);

for s = 1:N, % Counts data points
    % Evaluation point at the denominator
    q = 1/g^2 * (1 - S*S(s,:)');
    % Evaluate denominator at all points
    denominator(s) = 1/N * sum(G(q));
    
end

% Normalize the denominator
cg = sum(denominator); % Normalization constant
denominator = denominator./cg;

% Compute fhat(b)
fhat = 1/N * sum(numerator ./ denominator );

end



