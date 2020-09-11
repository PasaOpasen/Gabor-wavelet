function [Ft] = Fourier_signal(AA,om,ut,t0,TT)
        for i=1:AA            
            for j=1:TT
               f_t(j) = exp(-1.0i*om(i)*t0(j));
            end;
            f_ft = times(f_t,ut);
            Ft(i)=0.5*(f_ft(1)*(t0(2)-t0(1))+f_ft(TT)*(t0(TT)-t0(TT-1)));
            for k=2:TT-1;
               Ft(i)=Ft(i)+f_ft(k)*0.5*(t0(k+1)-t0(k-1));
            end;
        end;
        
        