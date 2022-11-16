import os
import pickle5
import numpy as np
import pandas as pd
from scipy import optimize
from calculator import Greek
import datetime

path = r"D:\work\CTA\data\data"


# 0. read in data
def read_pkl(path, file):
    pkl = open(os.path.join(path, file), "rb")
    df = pickle5.load(pkl)
    df.reset_index(inplace=True, drop=df.index.name in df.columns)
    return df


info = read_pkl(path, "etf_option_info_.pkl")
etf = read_pkl(path, "etf_price_standard_.pkl")
opt = read_pkl(path, "etf_option_price_standard_.pkl")
rf = pd.read_csv(os.path.join(path, "cn10ybond_yield_.csv"))

asset, start_date, end_date = "CN.SSE.510050", "2016-01-01", "2022-03-04"


# 1. data processing
def get_quote(df, col, asset, start, end):
    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    date = [i for i in list(df.columns)[2:] if start <= i <= end]
    df = df[(df["level_0"].apply(lambda x: asset in x)) &
            (df["level_1"].apply(lambda x: x in col))][["level_0", "level_1"] + date].copy()
    return df


def get_opt(info, etf, opt, rf, asset="CN.SSE.510050", start="2017-03-19", end="2021-03-19"):
    col = ["Symbol", "OptType", "StrikePrice", "BeginDate", "EndDate", "IsAdj"]
    info = info[(info["Symbol"].apply(lambda x: asset in x)) &
                (info["BeginDate"] <= end) & (info["EndDate"] >= start)][col].reset_index(drop=True).copy()
    info["BeginDate"] = pd.to_datetime(info["BeginDate"], format="%Y-%m-%d")
    info["EndDate"] = pd.to_datetime(info["EndDate"], format="%Y-%m-%d")
    col = ["Open", "Close", "High", "Low", "Volume"]
    etf = get_quote(etf, col, asset, start, end)
    etf = pd.DataFrame(etf[list(etf.columns)[2:]].values.T, columns=col, index=list(etf.columns)[2:]).reset_index()
    col = ["OPEN", "CLOSE", "HIGH", "LOW", "SETTLE", "VOLUME"]
    opt = get_quote(opt, col, asset, start, end)
    opt.set_index(["level_0", "level_1"], inplace=True)
    opt = opt.unstack()
    opt.columns = opt.columns.swaplevel()
    opt = opt.stack().reset_index()
    rf["index"] = pd.to_datetime(rf["index"], format="%Y年%m月%d日")
    rf = rf[(rf["index"] >= start) & (rf["index"] <= end)]
    opt.rename(columns={"level_0": "Symbol", "level_1": "index"}, inplace=True)
    opt = pd.merge(opt, info, how="left", on="Symbol")
    opt = opt[~opt["IsAdj"]]
    opt = pd.merge(opt, etf[["index", "Close"]].rename(columns={"Close": "UnlyPrice"}), how="left", on="index")
    opt = pd.merge(opt, rf[["index", "Close"]].rename(columns={"Close": "rf"}), how="left", on="index")
    opt["rf"] = 0.01 * opt["rf"]
    return info, etf, opt, rf


info, etf, opt, rf = get_opt(info, etf, opt, rf, asset, start_date, end_date)


# 2. generate beta with rolling regression
def get_atm(temp):
    df = temp.copy()
    idx = np.argmin(abs(df["UnlyPrice"] - df["StrikePrice"]))
    df = df.iloc[idx, :]
    T = (df["EndDate"] - df["index"]).days / 365
    F = df["UnlyPrice"] * np.exp(df["rf"] * T)
    vix = Greek.vix(df["UnlyPrice"], df["StrikePrice"], T, df["rf"], df["SETTLE"], 1 * (df["OptType"] == "Call"))
    df["ln_f"] = np.log(F)
    df["ln_sigma"] = np.nan if vix < 0.01 or vix >= 1 else np.log(vix)
    return df


atm_opt = opt.groupby("index").apply(lambda x: get_atm(x)).reset_index(drop=True)


def ols(Y, X):  # perform OLS, already check with statsmodels.api.OLS
    temp = np.c_[Y, X]
    temp = temp[~np.array(np.isnan(temp).any(axis=1)).reshape(len(temp))]
    Y, X = temp[:, 0], temp[:, 1:]
    W = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(Y)  # W = ([X^TX]^-1)X^TY
    Y_ = np.dot(X, W)  # Y_hat = XW
    E = Y-Y_  # residual = Y - Y_hat
    return W, Y_, E


def rolling_ols(Y, X, window=252):
    W = np.repeat(np.nan, (window-1)*X.shape[1]).reshape((window-1), X.shape[1])
    n = len(Y)
    for i in range(window, n+1, 1):
        temp_W, temp_Y, temp_E = ols(Y[i-window:i, :], X[i-window:i, :])
        W = np.concatenate((W, temp_W.T), axis=0)
    return W


atm_opt["c"] = 1
atm_opt[["alpha", "beta"]] = np.array(rolling_ols(np.mat(atm_opt[["ln_sigma"]]), np.mat(atm_opt[["c", "ln_f"]])))
atm_opt[["alpha", "beta"]] = atm_opt[["alpha", "beta"]].fillna(method="ffill")
atm_opt["beta"] = atm_opt["beta"] + 1
atm_opt["ln_sigma"] = atm_opt["ln_sigma"].fillna(method="ffill")
opt = pd.merge(opt, atm_opt[["index", "ln_f", "ln_sigma", "beta"]], how="left", on="index")
opt.rename(columns={"beta": "beta_ols"}, inplace=True)
opt["beta_1"] = 1  # Geometric Brownian Motion


# 3. estimate 1: calibration error
def mse(args, VIX, S, K, T, rf, alpha, beta):  # optimize function: minimize MSE(vix, sabr)
    res, n = 0, len(S)
    for i in range(n):
        res += (VIX[i] - Greek.sabr(S[i], K[i], T[i], rf[i], alpha, beta, args)) ** 2
    return res / n


def dmse(args, VIX, S, K, T, rf, alpha, beta):  # derivative of optimize function
    dmse_v, dmse_r, n = 0, 0, len(S)
    for i in range(n):
        temp = VIX[i] - Greek.sabr(S[i], K[i], T[i], rf[i], alpha, beta, args)
        dmse_v -= temp * Greek.dsabr_v(S[i], K[i], T[i], rf[i], alpha, beta, args)
        dmse_r -= temp * Greek.dsabr_r(S[i], K[i], T[i], rf[i], alpha, beta, args)
    res = np.array([dmse_v, dmse_r])
    return 2 * res / n


def opt_calib(VIX, S, K, T, rf, alpha, beta):
    args_ = (VIX, S, K, T, rf, alpha, beta, )
    x0_, bound_ = np.array([0.5, -0.5]), ((0, None), (-1, 1))
    res = optimize.minimize(mse, x0_, args=args_, method="Powell", bounds=bound_, jac=dmse,
                            tol=1e-16, options={"maxiter": 100})
    return res


def sim_calib(temp, beta):
    df = temp[["Symbol", "index", "SETTLE", "StrikePrice", "UnlyPrice", "rf", "ln_f", "ln_sigma", beta]].copy()
    df.columns = ["Symbol", "index", "mkt", "K", "S", "rf", "ln_f", "ln_sigma", "beta"]
    df["ExePrice"] = (temp["HIGH"] + temp["LOW"] + temp["CLOSE"] + temp["OPEN"]) / 4
    df["T"] = (temp["EndDate"] - temp["index"]).apply(lambda x: x.days / 365)
    df["CorP"] = 1 * (temp["OptType"] == "Call")
    df["vix"] = df.apply(lambda x: Greek.vix(x["S"], x["K"], x["T"], x["rf"], x["mkt"], x["CorP"]), axis=1)
    df = df[(df["vix"] >= 0.01) & (df["vix"] < 1)]  # 0. df 为空
    if len(df) == 0:
        return pd.DataFrame()
    VIX, S, K, T, rf = df["vix"].values, df["S"].values, df["K"].values, df["T"].values, df["rf"].values
    beta = df["beta"].iloc[0]
    alpha = np.exp(df["ln_sigma"].iloc[0] + (1 - beta) * df["ln_f"].iloc[0])
    df["alpha"] = alpha
    sim = opt_calib(VIX, S, K, T, rf, alpha, beta)  # 1. sabr返回nan值; 2. sim.x返回离谱值
    df["sabr"] = df.apply(lambda x: Greek.sabr(x["S"], x["K"], x["T"], x["rf"], alpha, beta, sim.x), axis=1)
    return df


# 4. estimate 2: hedging error
def hedge_error(args, MKT, MKT_1, CorP, weight, S_1, K, T_1, rf_1, ln_f_1, ln_sigma_1, dF, dVIX, idx=None):
    beta, args = args[0], args[1:]
    alpha = np.exp(ln_sigma_1[0] + (1 - beta) * ln_f_1[0])
    w_sum = weight.sum()
    res, n = 0, len(MKT)
    idx = list(range(n)) if idx is None else idx
    for i in idx:
        res += ((MKT_1[i] + Greek.sabr_delta(S_1[i], K[i], T_1[i], rf_1[i], alpha, beta, args, CorP[i]) * dF[i] +
                 Greek.sabr_vega(S_1[i], K[i], T_1[i], rf_1[i], alpha, beta, args) * dVIX[i]) / MKT[i] - 1) ** 2\
               * weight[i]
    return res


def opt_hedge(MKT, MKT_1, CorP, weight, S_1, K, T_1, rf_1, ln_f_1, ln_sigma_1, dF, dVIX, idx=None):
    args_ = (MKT, MKT_1, CorP, weight, S_1, K, T_1, rf_1, ln_f_1, ln_sigma_1, dF, dVIX, idx, )
    x0_, bound_ = np.array([0.5, 0.5, -0.5]), ((0, 1), (0, None), (-1, 1))
    res = optimize.minimize(hedge_error, x0_, args=args_, method="Powell", bounds=bound_,
                            tol=1e-16, options={"maxiter": 100})
    return res


def sim_hedge(temp):
    df = temp[["Symbol", "index", "SETTLE", "StrikePrice", "UnlyPrice", "rf",
               "SETTLE_1", "UnlyPrice_1", "rf_1", "ln_f_1", "ln_sigma_1"]].copy()
    df.columns = ["Symbol", "index", "mkt", "K", "S", "rf", "mkt_1", "S_1", "rf_1", "ln_f_1", "ln_sigma_1"]
    df["ExePrice"] = (temp["HIGH"] + temp["LOW"] + temp["CLOSE"] + temp["OPEN"]) / 4
    df["T"] = (temp["EndDate"] - temp["index"]).apply(lambda x: x.days / 365)
    df["T_1"] = df["T"] + 1 / 365
    df["CorP"] = 1 * (temp["OptType"] == "Call")
    df["weight"] = 1 / (1 + (df["S"] * np.exp(df["rf"] * df["T"]) - df["K"]) ** 2)
    df["vix"] = df.apply(lambda x: Greek.vix(x["S"], x["K"], x["T"], x["rf"], x["mkt"], x["CorP"]), axis=1)
    df["vix_1"] = df.apply(lambda x: Greek.vix(x["S_1"], x["K"], x["T_1"], x["rf_1"], x["mkt_1"], x["CorP"]), axis=1)
    df = df[(df["vix"] >= 0.01) & (df["vix"] < 1) & (df["vix_1"] >= 0.01) & (df["vix_1"] < 1)]
    df["dis"] = abs(df["S"] - df["K"])
    df.sort_values(by="dis", inplace=True)
    n = len(df)
    if n == 0:
        return pd.DataFrame()
    idx = list(range(min(2, n)))
    MKT, MKT_1, CorP, weight = df["mkt"].values, df["mkt_1"].values, df["CorP"].values, df["weight"].values
    S_1, K, T_1, rf_1 = df["S_1"].values, df["K"].values, df["T_1"].values, df["rf_1"].values
    ln_f_1, ln_sigma_1 = df["ln_f_1"].values, df["ln_sigma_1"].values
    dF = (df["S"] * np.exp(df["rf"] * df["T"]) - df["S_1"] * np.exp(df["rf_1"] * df["T_1"])).values
    dVIX = (df["vix"] - df["vix_1"]).values
    sim = opt_hedge(MKT, MKT_1, CorP, weight, S_1, K, T_1, rf_1, ln_f_1, ln_sigma_1, dF, dVIX, idx)
    beta = sim.x[0]
    alpha = np.exp(ln_sigma_1[0] + (1 - beta) * ln_f_1[0])
    df["sabr"] = df.apply(lambda x: Greek.sabr(x["S"], x["K"], x["T"], x["rf"], alpha, beta, sim.x[1:]), axis=1)
    return df


opt[["SETTLE_1", "UnlyPrice_1", "rf_1", "ln_f_1", "ln_sigma_1"]] =\
    opt.groupby("Symbol")[["SETTLE", "UnlyPrice", "rf", "ln_f", "ln_sigma"]].shift(1)
opt["ExePrice"] = opt[["HIGH", "LOW", "CLOSE", "OPEN"]].mean(axis=1)
# path = r"D:\work\CTA\strategy\VolArb_O_SABR_0702\data"
# info.to_csv(os.path.join(path, "info.csv"), index=False)
# etf.to_csv(os.path.join(path, "etf.csv"), index=False)
# opt.to_csv(os.path.join(path, "opt.csv"), index=False)
# rf.to_csv(os.path.join(path, "rf.csv"), index=False)
# df_1 = opt.groupby("index").apply(lambda x: sim_calib(x, "beta_1"))
# df_1.to_csv(os.path.join(path, "beta_1.csv"), index=False)
# df_2 = opt.groupby("index").apply(lambda x: sim_calib(x, "beta_ols"))
# df_2.to_csv(os.path.join(path, "beta_ols.csv"), index=False)
# df_3 = opt.groupby("index").apply(lambda x: sim_hedge(x))
# df_3.to_csv(os.path.join(path, "beta_hedge.csv"), index=False)
# temp = opt[(opt["index"] == "2019-02-11") & (opt["EndDate"] == "2019-03-27")].sort_values(by="StrikePrice").copy()
