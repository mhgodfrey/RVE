# RVE
Alterations to the original rank-vector-entropy Matlab code (Robinson 2012)

RVEmat.m takes data from all channels at once (input as Nchannels x Nsamples) and gives 
RVE timecourses in a matrix with the same dimensions.

RVEmat_fast.m does the same but implements parallel processing to speed up the calculation.
