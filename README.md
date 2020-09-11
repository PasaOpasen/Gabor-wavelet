# Gabor wavelet

Python realization of wavelet transform with Gabor-kernel (from matlab)

Тут хранятся скрипты для выполнения прямого и обратного вейвлет-преобразования Габора + визуализаций.


## Прямое преобразование

Целью этой части работы была перезапись [кода с матлаба](https://github.com/PasaOpasen/Gabor-wavelet/tree/master/matlab_source) и его ускорение, так как очень часто требуется использовать прямое преобразование на большой сетке.

Тут имеется две функции. Первая - это сама функция вейвлета Габора:

```python
import numpy as np
import math
import cmath

def GaborWavelet(omega, t, Gabor_coef):
    return 0.3251520240633*math.sqrt(omega)*cmath.exp(complex(-0.5*Gabor_coef*(t*omega*0.187390625129278)**2, omega*t))
```

Здесь `omega` и `Gabot_coef` - некоторые параметры, а `t` - время (аргумент), для которого надо посчитать функцию вейвлета.

Вторая функция - это само прямое преобразование:

```python
def DWT_signal(ut, a, b, t0, AA, BB, TT, omega0, Gabor_coef):

    h_step=t0[1]-t0[0]
    
    psi_t = np.empty(TT, dtype = np.complex128)
    Wab = np.empty((BB, AA), dtype = np.complex128)
    
    for j in range(AA):
        for i in range(BB):
            for k in range(TT):

                t_cur=(t0[k]-b[i])/a[j]               
                psi_t[k] = GaborWavelet(omega0, t_cur, Gabor_coef).conjugate()

            f_psi = psi_t * ut
            
                 
            Wab[i,j] =  0.5*(f_psi[0]+f_psi[-1]) + np.sum(f_psi[2:TT-1])
        
    Wab = np.multiply(Wab, h_step/np.sqrt(a))
    
    return Wab
```

`ut` - значения сигнала, `a` - значения по второй оси, `b` - значения по первой оси, `t0` - массив времен, `AA, BB, TT`- длины первых трех массивов (в Matlab и Fortran их принято передавать как аргументы).




