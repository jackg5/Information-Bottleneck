%% Entropy
% Computes the entropy of a distribution in bits.
%
% Inputs:
% * Px = probability distribution of variable X (vector)
% * gamma (optional) = Renyi parameter (default is 1)
%
% Outputs:
% * H = Renyi-entropy of Px in bits

function H = entropy(Px)
    % Throw away non-zero components of X because 0log0 := 0
    X = nonzeros(Px);
    logs = log2(X);
    H = -dot(logs,X);
end