%рисует картинку по сетке, максимум в одном из узлов сетки
%считать сюда файл data

point_x = 448;
f=180;
if f <= 60
    f_u = 5*f*1000;
else
    f_u = 2*f*1000;
end
fs = 10000000;
filt_ord = 10; f_nyquist = fs/2; Wn = f_u/f_nyquist;
[B,A] = butter(filt_ord,Wn,'low');
data_f2 = filtfilt(B,A,data);


for i=1:14000
    t1(i)=1.0e-07*(i-1);
end;


AA = 50; BB = 50; %размеры сетки
TT1 =200; %с какого момента времени начинать анализ 
TT = 800;  % по какой момент вести анализ
Gabor_coef = 0.5;



data_f(1:14000)= 1; 
fre=180000; %частота
Ncycles=5; Wab0=zeros(AA,BB);
omega0=(2*pi*fre*0.002)/3121; %безразмерная частота
om_i(1:2)=[.8 1.2]*omega0;


% Assingment of elements in arrays
for j=1:TT; 
    t0(j) = t1(j)*1560500; 
end;
for j=1:BB; 
    b(j)=t0(TT1+1) + (j-1)*(t0(TT)-t0(TT1+1))/(BB-1); 
end;
for j=1:AA; 
    om0(j)=om_i(1) + (j-1)*(om_i(2)-om_i(1))/(AA-1); 
end;
for j=1:AA; 
    a(j)=omega0/om0(j); 
end;
for j=1:BB; 
    aa(j,1:AA)=a(1:AA); 
    om(j,1:AA)=om0(1:AA);
end;
for j=1:AA; 
    bb(1:BB,j)=b(1:BB); 
end;

for i=1:AA
    for j=1:AA
        om_original(i,j)=(om(i,j)*3121)/(2*pi*0.002)/1000; %frequency in kHz
    end;
end;

for i=1:BB
    for j=1:BB
        bb_original(i,j)=(bb(i,j))/1560500*10000;  % time in ms
    end;
end;

 for j=1:(TT-TT1); data_f(j) = data_f2(TT1+j); end;


Wab0 = DWT_signal(data_f(1:(TT-TT1)),a,b,t0(TT1+1:TT),AA,BB,(TT-TT1),Ncycles,omega0,Gabor_coef);
Wab=abs(Wab0);

[Y,I]=max(Wab);
for i=1:BB
    if Y(i)==max(Y)
        jj=i;%jj - номер максимума по времени (т.е. время прихода несущей частоты)
    end;
end;
ii=I(jj); %ii - номер максимума по частоте (т.е. несущая частота)
Fr=om(ii,jj); %несущая частота в безразмерных
Fr_time=bb(ii,jj); %время прихода несущей частоты в безразмерных
Fr_original=(Fr*3121)/(2*pi*0.002);
Fr_time_original=(Fr_time)/1560500*10000;

% Graphic output in new windows
h=figure(1);
    load('SunsetColors','mycmap');
    contourf(bb_original,om_original,Wab,20,'EdgeColor','none');
	set(h,'Colormap',mycmap); colorbar;
    ylabel('\omega, kHz','fontsize',14,'color','k');
    xlabel('время, t, мс','fontsize',14,'color','k');
    title('PWAS16, частота = 150 кГц','fontsize',14,'color','k');
   % set(gca,'XTick',[t1(1)*10000:(t1(TT+1)*10000)/10:t1(TT+1)*10000]);
   % set(gca,'XTick',[0, 100, 200]);
   % set(gca, 'FontSize', 14);
    
   % figure(1); plot(t1(1:TT),data_f(1:TT)); xlim=(1:t1(TT));

% figure(3);
%     h=surface(bb,om,Wab,'EdgeColor','none');
%     colorbar; colormap(mycmap);