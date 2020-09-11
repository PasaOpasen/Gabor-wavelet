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


Поработав над кодом, я добился очень существенного ускорения в сравнении с [исходным вариантом](https://github.com/PasaOpasen/Gabor-wavelet/blob/master/translation_only.py). Это было сделано с помощью комбинации трёх подходов:

1. оптимизации самого алгоритма, чтобы избавиться от повторных вычислений и множественных обращений к одним и тем же участкам памяти

2. векторизации вычислений, чтобы больше кода выполнялось с помощью низкоуровневых функций

3. jit-компиляции и возможной параллелизации

Эти подходы применялись как к вейвлету Габора, так и к прямому преобразованию, причём в разных пропорциях. В итоге получилось несколько версии этого алгоритма с разными характеристиками. Далее представлены результаты их сравнения.

### Benchmarks

Сделаем проверку скорости на среднем объёме данных:

```python
a = np.arange(1,30,0.5) # 58 
b = np.arange(0,50,0.5) # 100

t = np.arange(0,101) # 101
ut = np.sin(2*math.pi/50 * t) + 100*np.cos(0.4*t)/(t*t+1)

omega = 1

Gabor_coef = 8

```

Проверка делается с помощью инструкций `%timeit` IPython:

```python
%timeit translation_only.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit light_vectorization.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit strong_vectorization.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit numba_strong.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit numba_just.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit numba_vectorization.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
%timeit numba_vec_parallel.DWT_signal(ut, a, b, t, a.size, b.size, t.size, omega, Gabor_coef)
```

Каждый модуль здесь - это отдельная версия алгоритма.

![1](https://github.com/PasaOpasen/Gabor-wavelet/blob/master/images/W(a%2Cb).png)

Итак, для этой размерности данных получаем результаты:

| Version   |      Time      | 
|:----------:|:-------------:|
| translation_only |  7.24 s ± 142 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)|
| light_vectorization |    372 ms ± 8.83 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)   |
| strong_vectorization | 368 ms ± 8.54 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) |
| numba_strong |  219 ms ± 5.17 ms per loop (mean ± std. dev. of 7 runs, 1 loop each) |
| numba_just |    36.3 ms ± 914 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)   |
| numba_vectorization | 35.1 ms ± 350 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)|
| numba_vec_parallel |  8.33 ms ± 85.2 µs per loop (mean ± std. dev. of 7 runs, 100 loops each) |


Как видно, удалось добиться ускорения примерно в 7.24s/35ms = 200 раз, не делая никакого распараллеливания. Параллельная версия работает ещё в 4 раза быстрее на моём 8-ядерном компьютере. 



## Обратное преобразование

Для обратного преобразования нужно иметь исходную матрицу `W(a,b)`, вдобавок желательно посмотреть её тепловую карту, чтобы убедиться, что заданная сетка `a x b` покрывает все возвышения (иногда эту сетку придётся сдвигать либо делать больше). Получить значения сигнала (обратного преобразования) для массиво времен `t` можно командой

```
from Reverse import St_array

s = St_array(t, Wab, a, b, omega, Gabor_coef)

```

![1](https://github.com/PasaOpasen/Gabor-wavelet/blob/master/images/W(a%2Cb)%20from%20sin(2pi%20div%2050%20t).png)

![1](https://github.com/PasaOpasen/Gabor-wavelet/blob/master/images/sin(2pi%20div%2050%20t).png)

![1](https://github.com/PasaOpasen/Gabor-wavelet/blob/master/images/W(a%2Cb)%20from%20sin(2pi%20div%2050%20t)%2Bsin(2pi%20div%20100%20t).png)

![1](https://github.com/PasaOpasen/Gabor-wavelet/blob/master/images/sin(2pi%20div%2050%20t)%2Bsin(2pi%20div%20100%20t).png)

![1](https://github.com/PasaOpasen/Gabor-wavelet/blob/master/images/W(a%2Cb)%20from%20sin(2pi%20div%2050%20t)%20%2B%204sin(2pi%20div%2010%20t).png)

![1](https://github.com/PasaOpasen/Gabor-wavelet/blob/master/images/sin(2pi%20div%2050%20t)%20%2B%204sin(2pi%20div%2010%20t).png)
