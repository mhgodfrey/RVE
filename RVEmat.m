function Entropy = RVEmat(VE, fs, fc, W, tau)
%
% function Entropy = RVEmat(VirtualTimecourses, fs, fc, W, tau)
%
% Compute the Rank Vector Entropy as in Robinson 2012 Frontiers Comp Neuro
%
% Changed to be able to take full set of virtual sensor timecourses at once 
% INPUT AS ONE ROW PER SENSOR
%
% Inputs: VE - data vector/matrix
%         fs - sampling rate of data
%         fc - low pass frequency
%         W - Window length
%         tau - entropy decay constant
%
% Ouput - entropy vector
%
% Notes: data should already be bandpassed to a broadband signal. 
% Eg. Stephen used 4-150 Hz in his paper with a 600 Hz sampling rate
%     a W of 5 and a tau of 0.6 (s). 
%
% Probably works better on continuous rather than trial data - takes a while for the algorithm to "warm-up"
% and ends with some zeros at the end
%
% SM 2013 based on Matt Brookes code

%% Lookup table
Nsymbols = factorial(W);
Lookup = perms(1:W);

power10s = zeros(1,W);
for i = 1:W
   power10s(i) = 10.^(i-1);
end
power10s = power10s';

Lookup_nums = Lookup * power10s; %This creates a numeric representation for each member of the lookup table

%% Precompute
Xi = ceil(fs/(2*fc)); % lag
Wkl = (W-1)*Xi; % window length
last_sample = size(VE,2) - (Wkl+1);
const = 1/log(Nsymbols); % for entropy calculation

F = ones(size(VE,1),Nsymbols); % Predefine
Entropy = zeros(size(VE));

% For leaky integrator
alpha = exp(-1/(tau*fs));

%% CALCULATE ENTROPY
for loop = 1:last_sample 
    %find the values in the sub window
    Wk = VE(:,loop:Xi:loop +Wkl);
    
    %compute the rank vector
    [Y,rankvec] = sort(Wk,2);
    clear Y
    
    % turn vectors into lookup numbers
    rankvec_nums = rankvec*power10s;
    
    F = F.*alpha; % make the integrator leak :-)
    
    thisbin = zeros(size(VE,1),1); % predefine
    for n = 1:length(rankvec_nums)
        thisbin(n) = find(rankvec_nums(n) == Lookup_nums);  % Find
        F(n,thisbin(n)) = F(n,thisbin(n)) + 1;
    end
    
    % Compute the probabilities based on frequencies
    P = F./repmat(sum(F,2),1,size(F,2));
    logP = log(P);
    
    % Compute the Entropy    
    Entropy(:,loop) = const * sum((-P).*logP,2);
end