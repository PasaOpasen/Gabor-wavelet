function [Wt]=GaborWavelet(omega,t,Gabor_coef)
    Wt = 0.3251520240633*sqrt(omega)*exp(-0.5*Gabor_coef*t*t*(omega*0.187390625129278)^2+1.0i*omega*t);
