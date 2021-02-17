import numpy as np

def m_fun(f):
    return(1125 * np.log(1 + (f/700)))

def m_inv_fun(m):
    return(700 * (np.exp(m / 1125) - 1))

def round_zero(x):
    if (x < 0 or x > 1e100):
        return(0)
    return(x)

def mel_filer(FFT, min_freq, max_freq, nfilters, nfft):
           
    def f_fun(x):
        return(math.floor(x))
           
    def H_fun(m,k):
        f_m_1 = f_freqs[m-1]
        f_m = f_freqs[m]
        f_p_1 = f_freqs[m+1]
           
        if k < f_m_1 or k > f_p_1:
           return(0)
        if k <= f_m:
            return((k - f_m_1) / (f_m - f_m_1))
        else:
            return((f_p_1 - k)/(f_p_1 - f_m))


    energies = np.empty(nfilters)
    
    mel_freqs = np.linspace(m_fun(min_freq), m_fun(max_freq), nfilters+2)
    h_freqs = list(map(m_inv_fun, mel_freqs))
    f_freqs = list(map(f_fun, h_freqs))

    for m in range(1,nfilters-1):
        tot_energy = 0
        for k in range(max_freq - 1):
            amp = FFT[k]
            tot_energy += H_fun(m,k) * amp
        energies[m-1] = round_zero(tot_energy /(h_freqs[m] - h_freqs[m-1]))
    return(energies)


