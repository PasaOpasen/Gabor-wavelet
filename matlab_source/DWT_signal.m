function [Wab] = DWT_signal(ut,a,b,t0,AA,BB,TT,omega0,Gabor_coef)

h_step=t0(2)-t0(1);

    for j=1:AA
        for i=1:BB
            for k=1:TT

                t_cur=(t0(k)-b(i))/a(j);               
                psi_t(k) = conj(GaborWavelet(omega0,t_cur,Gabor_coef));

            end;
            f_psi = times(psi_t,ut);
            Wab(i,j)=0.5*(f_psi(1)+f_psi(TT));
            for k=2:TT-1;
                Wab(i,j)=Wab(i,j)+f_psi(k);
            end;
        end;
        Wab(:,j)=Wab(:,j)*h_step/sqrt(a(j));
    end;  
