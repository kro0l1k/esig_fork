import numpy as np
# import iisignature
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import itertools
import random
from scipy.integrate import odeint  # (if you wish to use an ODE solver)
import warnings
from statsmodels.graphics.tsaplots import plot_acf
from fbm import FBM  
from sklearn.decomposition import FastICA
from scipy.signal import lfilter   # compiled C


np.set_printoptions(precision=2)
# from signax_works_perf import Optimizer, ContrastCalculator  
class SignalGenerator:
    """
    Class for   generating synthetic signals.
    
    Also includes function for graduual mixing of channels. 
    """
    @staticmethod
    def sample_s(d, n, ts_type, arma_scale = None, ou_scale = None):
        if ts_type == 'iid':
            for k in range(d):
                if k == 0:
                    # x = np.random.gamma(1, 1, n)
                    # x = np.random.gamma(1.4, 1, n)
                    x =  np.random.gumbel(0, 3, n)

                elif k == 1:
                    x =  np.random.gumbel(0, 3, n)

                elif k == 2:
                    x =  np.random.gamma(d, 3, n)

                else:
                    x = np.random.gumbel(0, 3 * k, n)
                x = x - np.mean(x)
                X = x if k == 0 else np.vstack((X, x))
                
            S = X.T
        elif ts_type == 'OU':
            # get paremeters to generate OU process. theta= 1, ..., 1 * theta_scale
            if ou_scale is None:
                theta = np.ones(d)
            else:
                theta = np.ones(d) * ou_scale
            mu = np.zeros(d)
            x0 = np.zeros(d)
            print(f" when sampling ou: theta: {theta}, mu: {mu}, x0: {x0}")
            S = SignalGenerator.sample_ou_vectorized(n, theta, mu, sigma=1.0, dt=1.0, x0=x0)
        elif ts_type == 'ARMA':
            # if d != 3:
            #     raise ValueError("ARMA process is only implemented for d=3")
            
            p = 10
            a = np.random.rand(d, p) 
            b = np.random.rand(d, p)
            
            # clip the coefficients to be in [-0.1, 0.1]
            a = np.clip(a, -1/p, 1/p)
            b = np.clip(b, -1/p, 1/p)
            if arma_scale is None:
                arma_scale = 1
            a *= arma_scale
            # b *= arma_scale
            
            S = SignalGenerator.sample_ARMA(n, d, a, b)
        
        elif ts_type == 'MA':
            S_no_stack = SignalGenerator.sample_s(d, n, 'iid')
            # create a moving average process from iid noise
            S = np.zeros((n, d))
            for k in range(d):
                # create a moving average process with a window of 5
                S[:, k] = np.convolve(S_no_stack[:, k], np.ones(5)/5, mode='same')
                S[:, k] += np.random.normal(0, 0.1, n)
                S[:, k] -= np.mean(S[:, k])  # Center the signal around zero
        elif ts_type == 'gumbelMA':
            S_no_stack = np.random.gumbel(0, 1, (n, d))
            S_no_stack *= 10
            print(f" means and stds of S_no_stack: {np.mean(S_no_stack, axis=0)}, {np.std(S_no_stack, axis=0)}")

            # create a moving average process from Gumbel noise
            S = np.zeros((n, d))
            for k in range(d):
                # create a moving average process with a window of 5
                S[:, k] = np.convolve(S_no_stack[:, k], np.ones(5)/5, mode='same')
                S[:, k] += np.random.normal(0, 0.1, n)
                S[:, k] -= np.mean(S[:, k])
        elif ts_type == '15per':
            t = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
            S1 = np.sin(np.array(t) * 5 * np.pi / 15)
            S2 = [0,0,5,0,0,-5,0,0,5,0,0,-5,0,0,5]
            S3 = [-1,0,2,0,-1,0,1,0,-1,0,2,0,-1,0,1]
            S = np.array([S1, S2, S3]).T
            
            nr_of_reps = n // S.shape[0]
            S = np.tile(S, (nr_of_reps, 1))
            S = np.float64(S)  # Convert to float64 for consistency
            gaussian_noise = np.random.normal(0, 0.1, S.shape)
            # S += gaussian_noise
            print(" shape of S:", S.shape)
        else:
            raise NotImplementedError("time series type not implemented")
        return S # Return shape (n, d)


    def sample_ou_vectorized(n, theta, mu, sigma=1.0, dt=1.0, x0=None, *,
                        dtype=np.float32, rng=None, memmap_file=None):
        """
        Fast AR(1) solver using SciPy's lfilter (C-backend, O(n) but no Python loop).
        """
        rng = np.random.default_rng(rng)
        theta   = np.asarray(theta,  dtype=dtype)
        mu      = np.asarray(mu,     dtype=dtype)
        sigma   = np.broadcast_to(sigma, theta.shape).astype(dtype)

        α       = np.exp(-theta * dt, dtype=dtype)
        β       = mu * (1 - α)
        var     = sigma**2 * (1 - α**2) / (2*theta)
        var[theta < 1e-10] = sigma[theta < 1e-10]**2 * dt             # small-θ fix

        # allocate output (RAM or on-disk)
        S = (np.memmap(memmap_file, mode='w+', dtype=dtype, shape=(n, len(theta)))
            if memmap_file else np.empty((n, len(theta)), dtype=dtype))

        S[0] = np.zeros_like(mu, dtype=dtype) if x0 is None else np.asarray(x0, dtype=dtype)
        # ε    = rng.normal(0, np.sqrt(var*dt, dtype=dtype), size=(n-1), len(theta))).astype(dtype)
        ε    = rng.gumbel(1, 1, size=(n-1, len(theta))).astype(dtype)  # Gumbel noise
        ε    = (ε - np.mean(ε, axis=0)) * np.sqrt(var) * dt   
        
        for j in range(len(theta)):                                     # still fast, d ≪ n
            # lfilter solves: S_t = α S_{t-1} + ε_t + β   (intercept via adding β to ε)
            S[1:, j], _ = lfilter([1.0], [1.0, -α[j]],
                                ε[:, j] + β[j], zi=[S[0, j] - β[j]])

        return S
    
    def sample_ARMA(n, d, a, b):
        # generate ARMA(p,p) process with d dimensions
        # a and b are [d x p] matrices of coefficients
        p = a.shape[1]  # number of AR coefficients
        print(" p = ", p)
        scale = 1
        # Gaussian is not great for third-order expected signatures, so we use Gumbel
        e = np.random.gumbel(0, 1, (n + p, d))  
        # Center the noise around zero
        e -= np.mean(e, axis=0)  
        e *= scale  # can scale up if needed
        y = np.zeros((n + p, d))

        # generate ARMA process
        for i in range(p, n + p):
            past_y = y[i-p:i][::-1]  # shape (p, d)
            past_e = e[i-p:i][::-1]  # shape (p, d)
            
            ar_term = np.sum(past_y * a.T, axis=0)  # element-wise mult then sum over p
            ma_term = np.sum(past_e * b.T, axis=0)  # element-wise mult then sum over p
            y[i] = ar_term + ma_term + e[i]
        
        # remove the first p samples
        y = y[p:]
        return y

    def confound_pure_signal(self, S, conf_type, conf_strength):
        """
        Takes a (n,d) shaped signal S, with independent channels, and returns
        an (n,d) shaped signal S_conf in which the channels have been corrupted acc.
        to the conf_type. conf_strength [0,1] regulates the level of corruption.
        conf_strenght = 0 <- returns the original signal
        conf_strenght = 1 <- the channels are very corrupted. 
        conf_type can be one of the following:
        - "common_corruptor": a common 1-dim time series is sampled and mixed with every channel.
        - "talking_pairs": the first channel is mixed with the second, the third with the fourth, etc.
        - "two_groups": the channels are split into two groups. the first group is mixed with itself, the second group is mixed with itself.
        - "multiplicative": a random normal is sampled and multiplied with every channel.
        """
        n,d = S.shape
        
        if conf_strength > 1:
            print("conf strength should be <=1. using conf_strength = 1")
            conf_strength = 1
        if conf_strength < 0:
            print("conf_strength should be >=0. using conf_strength = 0")
            conf_strength = 0
            
        def mix_with_common_corruptor(S, corruptor, conf_strength):
            """
            Mixes each channel of S with the common corruptor.
            """
            
            source_means = np.mean(S, axis=0)
            source_stds = np.std(S, axis=0)
            S -= source_means
            S /= source_stds
            
            corruptor = (corruptor - np.mean(corruptor)) / np.std(corruptor)  # standardize the corruptor
            
            for k in range(d):
                k_th_channel = S[:, k]
                k_th_channel = (conf_strength) * corruptor[:, 0] + (1 - conf_strength) * k_th_channel
                S[:, k] = k_th_channel
            # restore the original means and stds
            S *= source_stds * np.sqrt(2)
            S += source_means
            return S

        if conf_type == "common_corruptor_gaussian":
            # sample a 1-dim time series of length n. then linearly mix every channel with it.
            corruptor = np.random.rand(n, 1)
            # corruptor = self.sample_ARMA(1, ())
            S = mix_with_common_corruptor(S, corruptor, conf_strength)     
                
        elif conf_type == "common_corruptor_OU":
            # sample a 1-dim OU process of length n. then linearly mix every channel with it.
            # conf_strength = 0.01 * conf_strength
            theta = np.ones(d) * 0.001 * conf_strength
            OU_1N = SignalGenerator.sample_ou_vectorized(n, theta, mu=np.zeros(1), sigma=1.0, dt=1.0)

            S = mix_with_common_corruptor(S, OU_1N, conf_strength)

        elif conf_type == "common_corruptor_gammaMA":
            # sample a 1-dim gammaMA process of length n. then linearly mix every channel with it.
            p = 10
            a = np.random.rand(1, p) 
            b = np.random.rand(1, p)
            
            # clip the coefficients to be in [-0.1, 0.1]
            a = np.clip(a, -1/p, 1/p)
            b = np.clip(b, -1/p, 1/p)
            
            gammaMA_1N =  SignalGenerator.sample_ARMA(n, 1, a, b)

            S = mix_with_common_corruptor(S, gammaMA_1N, conf_strength)
            
        elif conf_type == "common_corruptor_fbm":
            
            H = 0.2          # Hurst exponent
            T = n/10.0           # time horizon
            f = FBM(n=n, hurst=H, length=T, method='daviesharte')  # fast + exact for grid
            x = f.fbm()       # array of length n+1
            x = x[:n]        # truncate to length n
            # make it (n, 1) shaped
            x = x.reshape(-1, 1)

            S = mix_with_common_corruptor(S, x, conf_strength)

        elif conf_type == "talking_pairs":
            # we confound it with 0.001 * confstrenght because the mixing is so strong
            # conf_strength = 0.4 * conf_strength
            if conf_strength == 0.5:
                print(" we shouldnt use conf_strenght = 0.5 for this corruption type as the neighbouring channels will look exactly the same")
                print("changing conf strenght tp 0.6")
                conf_strength = 0.6

            # confound the first with the second, the third with the fourth, etc.
            for k in range(0,d - 1, 2): # can only do till d-1 so that we can use the k+1 channel 
                # print(k, " in talking pairs") 
                
                k_th_channel = (conf_strength) * S[:, k+1] + (1 - conf_strength) * S[:, k]
                next_channel = (conf_strength) * S[:, k] + (1 - conf_strength) * S[:, k+1]
                
                S[:,k] = k_th_channel
                S[:,k+1] = next_channel
                
        elif conf_type == "two_groups":
            # conf_strength = 0.001 * conf_strength
            # split the channels into two groups. mix the channels within the groups.
            if conf_strength == 1:
                print("conf strenght should be <1. using conf_strength = 0.9")
                conf_strength = 0.9
            
            av1 = np.mean(S[:, :d//2], axis=1)
            av2 = np.mean(S[:, d//2:], axis=1)
            
            # plot the two averages            
            for k in range(d//2):
                S[:, k] = (conf_strength) * av1 + (1 - conf_strength) * S[:, k]
                
            for k in range(d//2, d):
                S[:, k] = (conf_strength) * av2 + (1 - conf_strength) * S[:, k]

        elif conf_type == "multiplicative":
            # generate a random normal and modulate the signal with it.
            noise = np.random.normal(1, 3 * conf_strength, n)
            ones = np.ones_like(noise)

            for k in range(d):
                S[:, k] = S[:, k] * (noise)
        
        elif conf_type == "outliers":
            # introduce outliers in the signal
            num_outliers = int(conf_strength * n)
            outlier_indices = np.random.choice(n, size=num_outliers, replace=False)
            for k in range(d):
                S[outlier_indices, k] += np.random.normal(0, 5, size=num_outliers)

        else:
            # corruption not implemented.
            print(" tried to find confounding type: ", conf_type)
            raise NotImplementedError("confounding type not implemented.")
        
        return S
    
def M_IplusE(W, A):
    d = A.shape[0]
    WA = W @ A
    Id= np.eye(d)
    if WA.shape[0] != d or WA.shape[1] != d:
        raise ValueError("W @ A must be a square matrix of shape (d, d)")
    cost = -np.abs(WA)
    r, c = linear_sum_assignment(cost)
    M = np.zeros_like(WA)
    for i, j in zip(r, c):
        M[i, j] = WA[i, j]
    # Convert JAX arrays to NumPy arrays for better Metal compatibility
    M_np = np.array(M)
    # Use NumPy's matrix inversion instead of JAX's
    if np.linalg.cond(M_np) > 1e3:
        warnings.warn("M_inv nearly singular")
    M_inv = np.linalg.inv(M_np)
    WA_np = np.array(WA)
    E = M_inv @ WA_np - Id
    
    return M, E

def get_rel_err(I_x, A_inv, M, E):
    abs_err = np.linalg.norm(I_x - M @ A_inv, 'fro')
    rel_err = abs_err / np.linalg.norm(M @ A_inv, 'fro')
    return rel_err

def main():
    n = 100_000
    d = 3
    ts_type = 'OU'  # type of time series to sample: 'OU'
    corruption_type = 'common_corruptor_fbm'  # type of corruption to apply
    # ts_type = 'iid'
    # ts_type = 'gammaMA'
    
    index_to_plot = 1000  # Show first 1000 samples
    signal_gen = SignalGenerator()

    S_original = signal_gen.sample_s(d, n, ts_type, arma_scale=1e-5)

    # Create slightly confounded signal (lower corruption)
    S_slight = signal_gen.sample_s(d, n, ts_type, arma_scale=1e-2)
    # S_slight = S_original.copy()
    # S_slight = signal_gen.confound_pure_signal(S_slight, corruption_type, 0.3)

    # Create more confounded signal (higher corruption)
    S_heavy = signal_gen.sample_s(d, n, ts_type, arma_scale=1.0)
    # S_heavy = signal_gen.confound_pure_signal(S_heavy, corruption_type, 0.7)

    # Create the three subplot figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    ### test fastica 
    A = np.array([[1, 2, 0],
                  [0, 1, 0.5],
                  [0, 0, 1]])
    A_inv = np.linalg.inv(A) 
    
    ica    = FastICA(n_components=d, max_iter=100_000, random_state=0)

    X_original = S_original @ A.T
    S_fastica = ica.fit_transform(X_original)   # X shape: (n_samples, n_features)
    W_fastica  = ica.components_
    
    M_d = W_fastica @ A
    print(" uncorrupted M_d:\n", M_d)
    pure_rel_err = get_rel_err(W_fastica, A_inv, *M_IplusE(W_fastica, A))


    X_slight = S_slight @ A.T
    S_slight_fastica = ica.fit_transform(X_slight)   # X shape: (n_samples, n_features)
    W_slight_fastica  = ica.components_
    
    M_slight_d = W_slight_fastica @ A
    slight_rel_err = get_rel_err(W_slight_fastica, A_inv, *M_IplusE(W_slight_fastica, A))
    print(" slightly corrupted M_d:\n", M_slight_d)

    X_heavy = S_heavy @ A.T
    # X_heavy = signal_gen.confound_pure_signal(X_heavy, corruption_type, 0.7)
    S_heavy_fastica = ica.fit_transform(X_heavy)   # X shape: (n_samples, n_features)
    W_heavy_fastica  = ica.components_

    M_heavy_d = W_heavy_fastica @ A
    heavy_rel_err = get_rel_err(W_heavy_fastica, A_inv, *M_IplusE(W_heavy_fastica, A))
    print(" heavily corrupted M_d:\n", M_heavy_d)

    # Plot original signal (left)
    for i in range(d):
        axes[0].plot(S_original[:index_to_plot, i], label=f'Channel {i+1}')
    axes[0].set_title(f'Original {ts_type} Signal')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    # add a box displaying the relative error for the pure
    axes[0].text(0.5, 0.9, f'Relative Error: {pure_rel_err:.4f}')
    axes[0].grid(True, alpha=0.3)
    
    # Plot slightly confounded signal (middle)
    for i in range(d):
        axes[1].plot(S_slight[:index_to_plot, i], label=f'Channel {i+1}')
    axes[1].set_title(f'Slightly Confounded Signal (0.3, "{corruption_type}")')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend()
    axes[1].text(0.5, 0.9, f'Relative Error: {slight_rel_err:.4f}')
    axes[1].grid(True, alpha=0.3)
    
    # Plot heavily confounded signal (right)
    for i in range(d):
        axes[2].plot(S_heavy[:index_to_plot, i], label=f'Channel {i+1}')
    axes[2].set_title(f'More Confounded Signal (0.7, "{corruption_type}")')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Amplitude')
    axes[2].legend()
    axes[2].text(0.5, 0.9, f'Relative Error: {heavy_rel_err:.4f}')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return


if __name__ == "__main__":
    main()
