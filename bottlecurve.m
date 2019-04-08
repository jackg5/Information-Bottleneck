%% Plot Generalized Bottleneck
% Computes the bottleneck curve for the given distribution for the
% functional given by
%
% L = H(T) - alpha*H(T|X) - beta*I(T;Y)

% Inputs:
% * Pxy = Joint distribution of X and Y for which to compute the
% bottleneck. Must be given as a matrix of size |X| x |Y|.
% * alpha (optional) = tradeoff parameter for the conditional entropy given
% by H(T|X). Must be in [0,Inf[. Default is 1.
% * delta (optional) = for a fixed beta's Hga value and a partition's Hga
% value, this beta will be optimized if |beta_Hga - partition_Hga| < delta.
% Must be positive and non-zero. Default is 1 / (4N).
% * epsilon (optional) = the convergence value for the bottleneck function.
% Must be positive and non-zero. Default is 10^-8.
% * display (optional) = parameter which chooses which information planes
% to display in figures. Options are (a) "ib" which displays Tishby's
% information plane, (b) "dib" which displays Strouse's Deterministic
% Information Plane, (c) "gib" which automatically determines which plane
% to display, (d) "all" which shows the ib, dib, and generalized ib
% planes, or (e) "none" which displays nothing. Default is "all".
% * betas = a vector of beta values for which to compute the
% plane points, thereby ignoring N. If empty, set betas to 
% [0, 0.25, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, ...
%            60, 70, 80, 90, 100, 200, 500, Inf];
%
% Outputs:
% * Hga = H_gamma(T) - alpha*H(T|X), the generalized information (General
% X-axis).
% * Ht = H(T), the shannon entropy values for each beta (DIB plane X-axis).
% * Ixt = I(X;T), the mutual information values for each beta (IB plane
% X-axis).
% * Iyt = I(T;Y), the mutual information values of the output which is
% common to all information planes.
% * Bs = beta values that were found which partition the curve into N
% points.
% * Qt = A matrix of size |T| x |Bs|. Every column of this matrix is the
% distribution q(T) for a beta in Bs.
function [Hga,Ht,Ixt,Iyt,Bs,Qt] = bottlecurve( Pxy,...
                                            alpha,...
                                            epsilon,...
                                            display,...
                                            betas, ...
                                            maxIterations)
    % Set defaults for 2nd parameter
    if nargin < 2
        % Use H_gamma(T) - H(T|X)
        alpha = 1;
    end
    % Set defaults for 3rd parameter
    if nargin < 3
        epsilon = 10^-8;
    end
    % Set defaults for 4th parameter
    if nargin < 4
        display = "all";
    end
    % Set defaults for the 5th parameter
    if nargin < 5
        betas = [0, 0.25, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, ...
            60, 70, 80, 90, 100, 200, 500, Inf];
    end
    % Set defaults for the 6th parameter
    if nargin < 6
        maxIterations = 30;
    end
    
    % Sort all beta values inputted so they are in order.
    Bs = sort(betas);
    
    % Validate all parameters to ensure they are valid.
    validate(alpha,epsilon,display,betas);
    
    % Get the distribution of X so we can compute upper limits on
    % horizontal axes.
    Px = makeDistribution(sum(Pxy,2));
    
    % Get the upper limits for the ib and dib plane, given by H(X), as well
    % as the upper limit for the gib plane, given by H_gamma(X).
    Hx = entropy(Px);
    
    % Compute the mutual information between X and Y, which is the upper
    % limit of the vertical axis on all planes.
    Pygx = makeDistribution(Pxy ./ Px, 2);
    Ixy = mi(Pygx,Px);

    minHga = getCurvePoints(Pxy, [0], alpha, epsilon, 50);
    % Compute the positions on the curve for each of the planes, where
    % Hs is entropies H(T), Hgas is the generalized entropy
    % H_gamma(T) - alpha H(T|X), IbXs is the mutual information I(X;T),
    % and IbYs is the mutual information I(T;Y).
    [Hga,Ht,Ixt,Iyt,Qt] = getCurvePoints(Pxy, Bs, alpha,...
                                         epsilon, maxIterations);
    % If we are not to display anything, don't do any more processing and
    % exit the function now.
    if strcmp(display,'none')
        return;
    end
    % Set boolean flags based on the 'display' string to choose which
    % curves to display.
    displayAll = strcmp(display,'all');
    displayIB = strcmp(display,'ib') || displayAll;
    displayDIB = strcmp(display,'dib') || displayAll;
    gibSelected = strcmp(display,'gib') || displayAll;
    % By default, we do not display the third plane. This is handled in the
    % conditions on alpha and gamma later.
    displayGIB = false;
    % The name that is to be shown in the legend depends on whether this
    % was the IB or DIB
    curveName = bottleneckType(alpha);
    
    % In the case of the GIB plane, we need to choose which plane to
    % display - the IB, the DIB, or the third plane.
    if gibSelected
        if strcmp(curveName, 'IB')
            displayIB = true;
        elseif strcmp(curveName, 'DIB')
            displayDIB = true;
        else
            displayGIB = true;
        end
    end
    Bstring = mat2str(Bs);
    Bstring = [' Betas = ' Bstring];
    curveName = [curveName Bstring];
    
    % Handle the case where we have to display the IB plane
    if displayIB
        % Create a figure for the IB plane
        ibf = figure;
        % Plot the IB curve for these betas
        plot(Ixt,Iyt);
        % Put in the titles, legend, and axes labels
        xlabel('I(X;T)');
        ylabel('I(T;Y)');
        title(sprintf('IB Plane for |X|=%d and |Y|=%d',...
            size(Pxy,1),size(Pxy,2)));
        % Plot the box within which the curve should lie, which is given by
        % H(X) and I(X;Y)
        hold on;
        plot([0,Hx],[Ixy,Ixy]); % I(X;Y) horizontal line
        plot([Hx,Hx],[0,Ixy]); % H(X) vertical line
        hold off;
        % Plot the legend in the bottom right corner with no background or
        % outline
        legend({curveName,'I(X;Y)','H(X)'},'Location','Southeast');
        legend('boxoff');
    end
    
    % Now handle the case where we have to display the DIB plane
    if displayDIB
        % Create a figure for the DIB plane
        dibf = figure;
        % Plot the DIB curve for these betas
        plot(Ht,Iyt);
        % Put in the titles, legend, and axes labels
        xlabel('H(T)');
        ylabel('I(T;Y)');
        title(sprintf('DIB Plane for |X|=%d and |Y|=%d',...
            size(Pxy,1),size(Pxy,2)));
        % Plot the box within which the curve should lie, which is given by
        % H(X) and I(X;Y)
        hold on;
        plot([0,Hx],[Ixy,Ixy]); % I(X;Y) horizontal line
        plot([Hx,Hx],[0,Ixy]); % H(X) vertical line
        hold off;
        % Plot the legend in the bottom right corner with no background or
        % outline
        legend({curveName,'I(X;Y)','H(X)'},'Location','Southeast');
        legend('boxoff');
    end
    
   
end

%% Validation
% Validate all inputs to the public function, bottlecurve().
function validate(alpha, epsilon, display, betas)
    % Ensure all numerical inputs are valid.
    assert(alpha >= 0 && alpha < Inf, ...
        "BottleCurve: alpha must be in [0,Inf[");
    assert(epsilon > 0, "BottleCurve: epsilon must be positive non-zero.");
    
    % Set display string
    displayString = string(display);
    % Ensure display string is one of the allowable options
    validDisplay = strcmp(displayString,"ib");
    validDisplay = validDisplay || strcmp(displayString,"dib");
    validDisplay = validDisplay || strcmp(displayString,"all");
    validDisplay = validDisplay || strcmp(displayString,"none");
    % Validate the display string
    assert( validDisplay, ...
        strcat("BottleCurve: valid inputs for display are ",...
                "'ib'",...
                ", 'dib'",...
                ", 'all'",...
                ", or 'none'."));
            
    % Validate the vector of betas by checking that all values inputted are
    % positive.
    assert( sum(betas >= 0) == length(betas),...
        "BottleCurve: betas must all be non-negative values.");
end

%% Find Generalized Curve Points
% Given some beta values and parameters for the bottleneck, this will
% compute the entropy H(T), generalized renyi-alpha information
% H_gamma(T) - alpha*H(T|X), and mutual informations I(X;T) and I(T;Y) for
% all the beta values.
%
% This is done by iterating the bottleneck functional multiple times and
% taking distribution q(t|x) which minimizes the L-functional
% L = H(T) - alpha*H(T|X) - beta*I(T;Y)
%
% Every time a lower L value is found, a loop cycle resets. If no lower L
% value is found within a maximum number of iterations, the loop cycle ends
% and the bottleneck points associated with that lowest L value are added
% to the output lists.
%
% Inputs:
% * Pxy = a joint distribution of X and Y
% * Bs = beta values to optimize
% * alpha = parameter in front of H(T|X) in the L-functional
% * epsilon = convergence parameter for the bottleneck function
% * maxIterations = number of iterations where L must not become smaller
% before we consider the functional converged for a fixed beta.
%
% Outputs:
% * Hgas = generalized entropy values, H_gamma(T) - alpha*H(T|X)
% * Hs = entropy values for T, H(T)
% * IbXs = mutual information I(X;T)
% * IbYs = mutual information I(T;Y)
% * Qts = distributions q(T) in a matrix
function [Hgas,Hs,IbXs,IbYs,Qts] = getCurvePoints(Pxy, Bs, alpha, ...
                                              epsilon, maxIterations)
    debug = true;
    numBetas = length(Bs);
    
    % Initialize the output vectors
    Hgas = zeros(1,numBetas);
    Hs = zeros(1,numBetas);
    IbXs = zeros(1,numBetas);
    IbYs = zeros(1,numBetas);
    Qts = zeros(size(Pxy,1), numBetas);
    
    % Initialize a waitbar to display output to the user
    bar = waitbar(0,'','Name','Compute Curve Points');
    % Counter to keep track of how many iterations have gone, for the
    % waitbar to update properly. Initialize to zero since we add 1 to it
    % immediately in the loop.
    betaIndex = 0;
    
    % Loop over each beta and compute the output values for each one
    for beta = Bs
        % Update the index counter for the waitbar
       betaIndex = betaIndex + 1;
       % Update the waitbar itself
       waitbar(betaIndex / numBetas, bar, ...
           sprintf('Computing bottleneck for beta = %.2f\n (%d of %d)',...
                beta, betaIndex, numBetas));
            
        if debug
            fprintf('-> Searching for optimal L for beta = %.3f\n',beta);
        end
        % Compute an initial bottleneck position for this beta
        [~,Qt,L,Ixt,Iyt,Hgt,Htgx] = bottleneck(Pxy, beta, alpha, ...
                                               epsilon, debug);
        Hga = Hgt - alpha*Htgx;
        
        % Flag to keep looking if the max number of iterations is not
        % reached
        optimalLFound = false;
        % Skip the while loop if beta = 0 or beta = Inf, since these are
        % deterministic cases where we know we cannot improve the results.
        if beta == 0 || beta == Inf
            optimalLFound = true;
        end
        
        % Count how many times this beta value converges to a point in the
        % information plane that is suboptimal compared to the previous 
        % beta value.
        
        % Continue to run the bottleneck until we find an optimal L. Of
        % course, we can skip the case where beta = 0 or beta = Inf
        
        % Can comment this out if you don't care how many iterations. (will
        % be okay for testing for an approximate bottleneck curve)
        
        while ~optimalLFound
            % Compute a new L value and new outputs.
            [~, newQt, newL, newIxt, newIyt, newHga] = ...
                optimalbottle(Pxy, alpha, beta, ...
                              maxIterations, epsilon);
            
            % If a new minimum L was found upon optimizing, update the
            % outputs.
            if newL < L
                % Update the "optimal" output variables
                Qt = newQt;
                L = newL;
                Ixt = newIxt;
                Iyt = newIyt;
                Hga = newHga;
            end
%      
            if alpha == 0 && L > (Hgas(max(betaIndex-1,1)) - Bs(max(betaIndex-1,1))*IbYs(max(betaIndex-1,1)) + epsilon)
                fprintf('This L is greater than previous L (%.8f > %.8f)! Try again.\n',...
                    L, (Hgas(max(betaIndex-1,1)) - Bs(max(betaIndex-1,1))*IbYs(max(betaIndex-1,1))));
            % Otherwise, this is the optimal position
            else
                optimalLFound = true;
            end

        end
        
        % Compute the H(T) - alpha*H(T|X) of this optimal
        % distribution
        Hgas(betaIndex) = Hga;
        
        % Compute the entropy of this optimal distribution
        Hs(betaIndex) = entropy(Qt);
        
        % Insert the mutual information values into their respective output
        % variables
        IbXs(betaIndex) = Ixt;
        IbYs(betaIndex) = Iyt;
        
        % Insert the distribution into its output variable location
        Qts(:,betaIndex) = Qt;
    end
    
    % Close the waitbar
    delete(bar);
end

%% Determine bottleneck type
% This finds the bottleneck name based on the alpha and gamma values.
%
% Inputs:
% * alpha = parameter in front of H(T|X)
%
% Outputs:
% * name = name of the bottleneck. This is 'IB', 'DIB', or 'other'
function name = bottleneckType(alpha)

        % In this case when alpha is 0, we have H(T) - beta I(T;Y),
        % which is the DIB
        if alpha == 0
            name = 'DIB';
        % If alpha is 1, we have I(X;T) - beta I(T;Y), which is the IB
        elseif alpha == 1
            name = 'IB';
        else
            name = 'Other';
        end
    % Anything else means we clearly do not have an IB or DIB situation
    % and are in the Generalized bottleneck scenario.
end