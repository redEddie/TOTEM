# ============================================================
# 2) Fourier fit on TRAIN (K=15) -> alpha/beta
#    Fourier(t) = α0 + Σ α_k sin(2πkt/s) + Σ β_k cos(2πkt/s)
# ============================================================
def fit_fourier_coeffs(y: np.ndarray, K: int = 20, s: float = 24.0):
    y = np.asarray(y, dtype=float).reshape(-1)
    T = len(y)
    t = np.arange(T, dtype=float)

    X = np.ones((T, 1 + 2*K), dtype=float)
    for k in range(1, K+1):
        X[:, k]     = np.sin(2*np.pi*k*t/s)
        X[:, K + k] = np.cos(2*np.pi*k*t/s)

    theta, *_ = np.linalg.lstsq(X, y, rcond=None)

    alpha0 = float(theta[0])
    alpha  = theta[1:1+K].copy()
    beta   = theta[1+K:1+2*K].copy()
    return alpha0, alpha, beta

def fourier_predict(T: int, alpha0: float, alpha: np.ndarray, beta: np.ndarray, s: float = 24.0):
    K = len(alpha)
    t = np.arange(T, dtype=float)
    yhat = alpha0 * np.ones(T, dtype=float)
    for k in range(1, K+1):
        yhat += alpha[k-1] * np.sin(2*np.pi*k*t/s)
        yhat += beta[k-1]  * np.cos(2*np.pi*k*t/s)
    return yhat

alpha0, alpha, beta = fit_fourier_coeffs(df_train[TARGET_COL].values, K=FOURIER_K, s=FOURIER_S)
print(f"[INFO] Fourier fitted on train | K={FOURIER_K}, s={FOURIER_S}")
print(f"[INFO] alpha0={alpha0:.6f} | alpha.shape={alpha.shape} | beta.shape={beta.shape}")

# 전체 길이에 대해 Fourier feature 생성 (train 기준 t=0..)
fourier_all = fourier_predict(T=len(df), alpha0=alpha0, alpha=alpha, beta=beta, s=FOURIER_S)

df_feat = df.copy()
df_feat["fourier"] = fourier_all
EXOG_COLS = ["fourier"] + EXOG_COLS_BASE