function AllEntropy = RVEmat_fast(cfg)
%
% function AllEntropy = RVEmat_fast(cfg);
%
% Compute the Rank Vector Entropy as in Robinson 2012 Frontiers Comp Neuro
%
% Computes RVE of different chunks of the signal in parallel. Chunks
% overlap to remove 'warming up' effect
%
% Inputs: cfg.Data - data vector/matrix (Nchannels x Nsamples)
%         cfg.SampleFreq - sampling rate of data, fs
%         cfg.LowpassFreq - low pass frequency, fc
%         cfg.SlidingWindow - Window length, W
%         cfg.TimeConstant - entropy decay constant, tau
%         cfg.NumWorkers- number of workers available in parallel pool
%         cfg.Overlap - defines overlap of blocks done in parallel: use at
%                       least 6
%
% Ouput - entropy matrix (channels x time)
%
% Notes: data should already be bandpassed to a broadband signal.
% Eg. Stephen used 4-150 Hz in his paper with a 600 Hz sampling rate
%     a W of 5 and a tau of 0.6 (s).
%
% Probably works better on continuous rather than trial data - takes a 
% while for the algorithm to "warm-up" and ends with some zeros at the end
%
% SM 2013 based on Matt Brookes code
% Updated by Megan Godfrey 2018

warning off
%% Define variables
VE = cfg.Data;
fs = cfg.SampleFreq;
fc = cfg.LowpassFreq;
W = cfg.SlidingWindow;
tau = cfg.TimeConstant;
N = cfg.NumWorkers;
threshold = 10^-(cfg.Overlap);

%% Precompute
% Lookup table
Nsymbols = factorial(W);
Lookup = perms(1:W);

power10s = zeros(1,W);
for i = 1:W
    power10s(i) = 10.^(i-1);
end
power10s = power10s';

Lookup_nums = Lookup * power10s; %This creates a numeric representation for each member of the lookup table

Xi = ceil(fs/(2*fc)); % lag
Wkl = (W-1)*Xi; % window length
const = 1/log(Nsymbols); % for entropy calculation

%% Setting blocks for parallel processing
L = RVE_FindWindowLength(tau,threshold,fs);
Ltot = size(VE,2);
Lsec = ceil(Ltot/N);
L1 = Lsec + L/2;
Li = Lsec + L;

Block = zeros(N,2);
Block(1,:) = [1,L1];
for nb1 = 2:N-1;
    Block(nb1,:) = [(nb1-1)*Lsec-L/2+1,(nb1*Lsec+L/2)];
end
Block(N,:) = [Block(N-1,2)-L+1,Ltot];

if min(Block(:,1))<0 || max(Block(:,2))>Ltot
    error('Signal is too short to be cut into that many sections. Consider using fewer workers or alternatively reduce the Overlap or Time Constant')
end

%% Parallel pool
poolobj = gcp('nocreate');
if size(poolobj,1)==0
    parpool(N)
elseif poolobj.NumWorkers<N
    delete(gcp)
    parpool(N)
end

EntropyCell = cell(N);
parfor nb = 1:N
    VEblock = VE(:,Block(nb,1):Block(nb,2));
    
    last_sample = size(VEblock,2) - (Wkl+1);
    
    %% Predefine
    F = ones(size(VEblock,1),Nsymbols);
    Entropy = zeros(size(VEblock));
    
    % For leaky integrator
    alpha = exp(-1/(tau*fs));
    
    %% CALCULATE ENTROPY
    for loop = 1:last_sample
        %find the values in the sub window
        Wk = VEblock(:,loop:Xi:loop +Wkl);
        
        %compute the rank vector
        [Y,rankvec] = sort(Wk,2);
        
        % turn vectors into lookup numbers
        rankvec_nums = rankvec*power10s;
        
        F = F.*alpha; % make the integrator leak :-)
        
        thisbin = zeros(size(VEblock,1),1); % predefine
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
    EntropyCell{nb} = Entropy;
end
AllEntropy = zeros(size(VE));
for nc = N:-1:1
    AllEntropy(:,Block(nc,1):Block(nc,2)-(Wkl+1)) = EntropyCell{nc}(:,1:end-(Wkl+1));
end

end

function Lmax = RVE_FindWindowLength(tau,threshold,fs)
% WindowLength = RVE_FindWindowLength(tau,threshold,Fs)
%
% For finding the time window (in seconds) that has a significant effect on
% the RVE calculation at any given time point.
% This is dependent on:
% tau = time constant (usually ~0.3s)
% threshold = gives cutoff point for acceptable loss of information going
%             into calculaton
% Fs = sampling frequency

L = 1e6;
x = 1:L; % Time window before time point considered
alpha = exp(-1/(tau*fs)); % Leaky integrator constant
maxFl = x.*alpha.^x; % Maximum possible effect a given time point can have

Lmax = L-length(maxFl(maxFl<threshold));
end
