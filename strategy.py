import os
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from calculator import Greek

path = r"D:\work\CTA\strategy\VolArb_O_SABR_0702\data"

opt = pd.read_csv(os.path.join(path, "opt.csv"))
df_1 = pd.read_csv(os.path.join(path, "beta_1.csv"))
df_2 = pd.read_csv(os.path.join(path, "beta_ols.csv"))
df_3 = pd.read_csv(os.path.join(path, "beta_hedge.csv"))


def get_signal_1(temp, td_quote, position=1, threshold=0, curr=[], curr_pos={}, days=30, delta_range=(0.1, 0.9), dis=1):
    res = []
    df = temp.copy()
    index = df["index"].iloc[0]
    # Filter 1. maturity & delta
    df["delta"] = df.apply(lambda x: Greek.delta(x["S"], x["K"], x["T"], x["rf"], x["sabr"], x["CorP"]), axis=1)
    df["T"] = (df["T"] * 365).apply(round)
    df = df[(df["T"] >= days) & (abs(df["delta"]) > delta_range[0]) & (abs(df["delta"]) < delta_range[1])]
    if len(df) == 0:  # current position meet the maturity limitation, cover
        for i in curr:
            res.append({"index": index, "Symbol": i, "Position": -1 * curr_pos[i], "Signal": -1,
                        "ExePrice": td_quote[td_quote["Symbol"] == i]["ExePrice"].iloc[0]})
        curr, curr_pos = [], {}
        return res, curr, curr_pos
    date = df["T"].min()
    df = df[df["T"] == date]
    df["T"] = df["T"] / 365
    # Filter 2. rest arbitrage space from current position
    if len(curr) > 0:
        curr_df = df[df["Symbol"].apply(lambda x: x in curr)].copy()
        curr_df["dis"] = curr_df["vix"] - curr_df["sabr"]
        threshold = curr_df["dis"].sum()
        if threshold <= 0 or len(curr_df[curr_df["CorP"] == 1]) == 0 or len(curr_df[curr_df["CorP"] == 0]) == 0:
            for i in curr:  # no more arbitrage space, cover current position
                res.append({"index": index, "Symbol": i, "Position": -1 * curr_pos[i], "Signal": -1,
                            "ExePrice": td_quote[td_quote["Symbol"] == i]["ExePrice"].iloc[0]})
            threshold, curr, curr_pos = 0, [], {}
    # Filter 3. arbitrage space
    df["dis"] = df["vix"] - df["sabr"]
    if len(df[df["CorP"] == 1]) > 0 and len(df[df["CorP"] == 0]) > 0:# and df["dis"].max() < dis:
        call = df[df["CorP"] == 1]["Symbol"].iloc[np.argmax(df[df["CorP"] == 1]["dis"])]
        put = df[df["CorP"] == 0]["Symbol"].iloc[np.argmax(df[df["CorP"] == 0]["dis"])]
        new = [call, put]
        new_df = df[df["Symbol"].apply(lambda x: x in new)].copy()
        new_df.sort_values(by="dis", inplace=True)  # trade the larger arbitrage one and hedge the other one
        new = new_df["Symbol"].to_list()
        arb_space = new_df["dis"].sum()
        if arb_space > threshold:  # new combo has a larger arbitrage space, switch the position
            for i in curr:  # cover current position
                res.append({"index": index, "Symbol": i, "Position": -1 * curr_pos[i], "Signal": -1,
                            "ExePrice": td_quote[td_quote["Symbol"] == i]["ExePrice"].iloc[0]})
            threshold, curr, curr_pos = 0, [], {}
            delta_hedge = abs(new_df["delta"].iloc[0] / new_df["delta"].iloc[1])
            new_pos = {new[0]: -position, new[1]: -position * delta_hedge}
            for i in new:  # open new position
                res.append({"index": index, "Symbol": i, "Position": new_pos[i], "Signal": 1,
                            "ExePrice": td_quote[td_quote["Symbol"] == i]["ExePrice"].iloc[0]})
        else:
            new, new_pos = curr, curr_pos
    else:
        new, new_pos = curr, curr_pos
    return res, new, new_pos


def get_position(temp, opt, position=1, threshold=0, days=30, delta_range=(0.1, 10)):
    df = temp.sort_values(by="index").copy()
    quote = opt[["index", "Symbol", "ExePrice"]].sort_values(by="index").copy()
    quote["ExePrice"] = quote.groupby("Symbol")["ExePrice"].shift(-1)  # execute order tomorrow
    date = df["index"].unique()
    n = len(date)
    res, curr, curr_pos = [], [], {}
    for i in range(n):
        td = date[i]
        td_opt = df[df["index"] == td].copy()
        td_quote = quote[quote["index"] == td].copy()
        td_dict, curr, curr_pos = get_signal_1(td_opt, td_quote, position, threshold, curr, curr_pos, days, delta_range)
        res += td_dict
    close_pos = []
    m = len(res)
    for i in range(m-1, -1, -1):
        if res[i]["Signal"] == 1:
            close_pos.append({"index": date[-1], "Symbol": res[i]["Symbol"], "Position": -1 * res[i]["Position"],
                              "Signal": -1, "ExePrice": quote[(quote["Symbol"] == res[i]["Symbol"]) &
                                                              (quote["index"] == date[-1])]["ExePrice"].iloc[0]})
        else:  # cover all the open positions at the end of the period
            break
    return pd.DataFrame(res+close_pos)


def backtest(pos, opt):
    quote = opt[["index", "Symbol", "ExePrice"]].sort_values(by="index").copy()
    quote["ExePrice"] = quote.groupby("Symbol")["ExePrice"].shift(-1)
    quote = quote[(quote["index"] >= pos["index"].iloc[0]) & (quote["index"] <= pos["index"].iloc[-1])]
    date = quote["index"].unique()
    n = len(date)
    curr, curr_pos, last_price = [], {}, 0
    profit, position, price = [], [], []
    for i in range(n):
        td_quote = quote[quote["index"] == date[i]]
        td_profit = 0
        for j in curr:
            td_price = td_quote[quote["Symbol"] == j]["ExePrice"].iloc[0]
            td_profit += curr_pos[j] * td_price
        td_profit = 0 if last_price == 0 else (td_profit - last_price) / abs(last_price)
        profit.append(td_profit)

        td_pos = pos[pos["index"] == date[i]]
        m = len(td_pos)
        for j in range(m):
            if td_pos["Signal"].iloc[j] == -1:
                curr.remove(td_pos["Symbol"].iloc[j])
                curr_pos.pop(td_pos["Symbol"].iloc[j])
            else:
                curr.append(td_pos["Symbol"].iloc[j])
                curr_pos[td_pos["Symbol"].iloc[j]] = td_pos["Position"].iloc[j]

        last_price = 0
        for j in curr:
            td_price = td_quote[quote["Symbol"] == j]["ExePrice"].iloc[0]
            last_price += curr_pos[j] * td_price
        position.append([i for i in curr])
        price.append(last_price)
    profit = pd.DataFrame({"index": date, "profit": profit, "position": position, "price": price})
    profit["index"] = pd.to_datetime(profit["index"], format="%Y-%m-%d")
    profit["cum_profit"] = (profit["profit"] + 1).cumprod()
    return profit


def statistics(profit, figure=False, title="Back Test"):
    stat = {}
    stat["tot_ret"] = (profit["cum_profit"].iloc[-1] / profit["cum_profit"].iloc[0] - 1)
    stat["ann_ret"] = (stat["tot_ret"] + 1) ** (365 / (profit["index"].iloc[-1] - profit["index"].iloc[0]).days) - 1
    stat["ann_std"] = profit["profit"].std() * np.sqrt(252)
    stat["Sharpe"] = stat["ann_ret"] / stat["ann_std"]
    stat["MDD"] = np.max(np.maximum.accumulate(profit["cum_profit"].values) - profit["cum_profit"].values)
    if figure:
        plt.figure(figsize=(12, 6))
        plt.plot(profit["index"], profit["cum_profit"])
        plt.title(title)
        plt.show()
    return stat


def grid_search(res, temp, para, para_name, idx, n):
    if idx == n:
        res.append(temp.copy())
        return
    for i in para[para_name[idx]]:
        temp[para_name[idx]] = i
        grid_search(res, temp, para, para_name, idx+1, n)
        temp.pop(para_name[idx])
    return


def optimize(df, opt, para, target="Sharpe", max=True):
    pair = []
    grid_search(pair, {}, para, list(para.keys()), 0, len(para))
    res, best_para, best_res = [], {}, -np.inf if max else np.inf
    for p in pair:
        pos = get_position(df, opt, **p)
        profit = backtest(pos, opt)
        stat = statistics(profit)
        res.append((p, stat))
        if (max and stat[target] > best_res) or (not max and stat[target] < best_res):
            best_res = stat[target]
            best_para = p
    return best_para, best_res, res


start, end = "2017-03-01", "2021-03-01"
# best_para, best_res, res = optimize(df_1[(df_1["index"] >= start) & (df_1["index"] <= end)], opt,
#                                     {"days": [5, 10, 21, 42, 63],
#                                      "delta_range": [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]})
# best_para, best_res, res = optimize(df_1[(df_1["index"] >= start) & (df_1["index"] <= end)], opt,
#                                     {"days": [10], "delta_range": [(0.1, 0.9)], "threshold": np.linspace(0, 0.8, 9)})
pos = get_position(df_1[(df_1["index"] >= start) & (df_1["index"] <= end)], opt,
                   **{"position": 1, "threshold": 0, "days": 10, "delta_range": (0.1, 0.9)})
profit = backtest(pos, opt)
stat = statistics(profit, True, "Beta = 1")

# best_para, best_res, res = optimize(df_2[(df_2["index"] >= start) & (df_2["index"] <= end)], opt,
#                                     {"days": [5, 10, 21, 42, 63],
#                                      "delta_range": [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]})
# best_para, best_res, res = optimize(df_2[(df_2["index"] >= start) & (df_2["index"] <= end)], opt,
#                                     {"days": [10], "delta_range": [(0.1, 0.9)], "threshold": np.linspace(0, 0.8, 9)})
pos = get_position(df_2[(df_2["index"] >= start) & (df_2["index"] <= end)], opt,
                   **{"position": 1, "threshold": 0.4, "days": 10, "delta_range": (0.1, 0.9)})
profit = backtest(pos, opt)
stat = statistics(profit, True, "Beta = OLS")

# best_para, best_res, res = optimize(df_3[(df_3["index"] >= start) & (df_3["index"] <= end)], opt,
#                                     {"days": [5, 10, 21, 42, 63],
#                                      "delta_range": [(0.1, 0.9), (0.2, 0.8), (0.3, 0.7)]})
# best_para, best_res, res = optimize(df_3[(df_3["index"] >= start) & (df_3["index"] <= end)], opt,
#                                     {"days": [63], "delta_range": [(0.1, 0.9)], "threshold": np.linspace(0, 0.8, 9)})
pos = get_position(df_3[(df_3["index"] >= start) & (df_3["index"] <= end)], opt,
                   **{"position": 1, "threshold": 0.3, "days": 63, "delta_range": (0.1, 0.9)})
profit = backtest(pos, opt)
stat = statistics(profit, True, "Beta = Hedging")
