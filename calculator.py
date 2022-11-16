import numpy as np
from scipy.stats import norm
from calcbsimpvol import calcbsimpvol as cal_vix


class Greek:  # option greeks calculator
    def __init__(self):
        self.author = "Joey Zheng"

    @staticmethod
    def bsm(S, K, T, rf, sigma, CorP=1):
        sign = (-1) ** (not CorP)
        temp = sigma * np.sqrt(T)
        d1 = sign * ((np.log(S / K) + (rf + 0.5 * sigma * sigma) * T) / temp)
        d2 = sign * (d1 - temp)
        return sign * (S * norm.cdf(d1) - K * np.exp(-rf * T) * norm.cdf(d2))

    @staticmethod
    def delta(S, K, T, rf, sigma, CorP=1):
        d1 = (np.log(S / K) + (rf + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - (1 - CorP)

    @staticmethod
    def vega(S, K, T, rf, sigma):
        temp = np.sqrt(T)
        d1 = (np.log(S / K) + (rf + 0.5 * sigma * sigma) * T) / (sigma * temp)
        return S * norm.pdf(d1) * temp

    @staticmethod
    def vix_n(S, K, T, rf, mkt, CorP=1, tol=1e-8, max_itr=100, sigma=0.2, dw=1e-4, up=10):
        price = Greek.bsm(S, K, T, rf, sigma, CorP)
        pre, count = sigma, 0
        while not (abs(mkt - price) < tol or count >= max_itr):
            sigma += (mkt - price) / Greek.vega(S, K, T, rf, sigma)
            sigma = max(min(sigma, up), dw)
            if abs(sigma - pre) < tol:
                break
            price = Greek.bsm(S, K, T, rf, sigma, CorP)
            pre = sigma
            count += 1
        return sigma

    @staticmethod
    def vix_b(S, K, T, rf, mkt, CorP=1, tol=1e-8, max_itr=100, sigma=0.2, dw=1e-4, up=10):
        price = Greek.bsm(S, K, T, rf, sigma, CorP)
        pre, count = sigma, 0
        while not (abs(mkt - price) < tol or count >= max_itr):
            if mkt > price:
                dw = sigma
                sigma = (sigma + up) / 2
            else:
                up = sigma
                sigma = (sigma + dw) / 2
            if abs(sigma - pre) < tol:
                break
            price = Greek.bsm(S, K, T, rf, sigma, CorP)
            pre = sigma
            count += 1
        return sigma

    @staticmethod
    def vix(S, K, T, rf, mkt, CorP=1):
        return cal_vix(dict(cp=np.array(CorP), P=np.array([mkt]), S=np.array([S]), K=np.array([K]),
                            tau=np.array([T]), r=np.array(rf), q=np.array([0])))[0][0]

    @staticmethod
    def sabr(S, K, T, rf, alpha, beta, args):
        vega, rho = args
        F = S * np.exp(rf * T)
        z = (vega / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
        X = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        a = ((((1 - beta) * alpha) ** 2 / (24 * (F * K) ** (1 - beta)) +
              rho * beta * vega * alpha / (4 * (F * K) ** ((1 - beta) / 2)) +
              (2 - 3 * rho ** 2) * vega ** 2 / 24) * T + 1) * alpha
        b = ((F * K) ** ((1 - beta) / 2)) * (
                    1 + ((1 - beta) * np.log(F / K)) ** 2 / 24 + ((1 - beta) * np.log(F / K)) ** 4 / 1920)
        return a / b * z / X

    @staticmethod
    def sabr_atm(S, T, rf, alpha, beta, args):
        vega, rho = args
        F = S * np.exp(rf * T)
        a = ((((1 - beta) * alpha) ** 2 / (24 * F ** (2 - 2 * beta)) +
              rho * beta * vega * alpha / (4 * F ** (1 - beta)) +
              (2 - 3 * rho ** 2) * vega ** 2 / 24) * T + 1) * alpha
        b = F ** (1 - beta)
        return a / b

    @staticmethod
    def dsabr_v(S, K, T, rf, alpha, beta, args):
        vega, rho = args
        F = S * np.exp(rf * T)
        z = (vega / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
        X = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        a = ((((1 - beta) * alpha) ** 2 / (24 * (F * K) ** (1 - beta)) +
              rho * beta * vega * alpha / (4 * (F * K) ** ((1 - beta) / 2)) +
              (2 - 3 * rho ** 2) * vega ** 2 / 24) * T + 1) * alpha
        b = ((F * K) ** ((1 - beta) / 2)) * (
                    1 + ((1 - beta) * np.log(F / K)) ** 2 / 24 + ((1 - beta) * np.log(F / K)) ** 4 / 1920)
        da_v = alpha * T * (rho * beta * alpha / (4 * (F * K) ** ((1 - beta) / 2)) +
                            (2 - 3 * rho ** 2) * vega / 12)
        dz_v = (F * K) ** ((1 - beta) / 2) * np.log(F / K) / alpha
        return da_v / b * z / X + a / b * (dz_v / X - z * dz_v / X**2)

    @staticmethod
    def dsabr_r(S, K, T, rf, alpha, beta, args):
        vega, rho = args
        F = S * np.exp(rf * T)
        z = (vega / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
        X = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        a = ((((1 - beta) * alpha) ** 2 / (24 * (F * K) ** (1 - beta)) +
              rho * beta * vega * alpha / (4 * (F * K) ** ((1 - beta) / 2)) +
              (2 - 3 * rho ** 2) * vega ** 2 / 24) * T + 1) * alpha
        b = ((F * K) ** ((1 - beta) / 2)) * (
                    1 + ((1 - beta) * np.log(F / K)) ** 2 / 24 + ((1 - beta) * np.log(F / K)) ** 4 / 1920)
        da_r = alpha * T * (vega * beta * alpha / (4 * (F * K) ** ((1 - beta) / 2)) - rho / 4 * vega ** 2)
        dX_r = 1 / (1 - rho) -\
               (z / np.sqrt(1 - 2 * rho * z + z ** 2) - 1) / (np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho)
        return da_r / b * z / X - a / b * z * dX_r / X ** 2

    @staticmethod
    def sabr_delta(S, K, T, rf, alpha, beta, args, CorP=1):
        vega, rho = args
        F = S * np.exp(rf * T)
        z = (vega / alpha) * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
        X = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))
        a = ((((1 - beta) * alpha) ** 2 / (24 * (F * K) ** (1 - beta)) +
              rho * beta * vega * alpha / (4 * (F * K) ** ((1 - beta) / 2)) +
              (2 - 3 * rho ** 2) * vega ** 2 / 24) * T + 1) * alpha
        b = ((F * K) ** ((1 - beta) / 2)) * (
                    1 + ((1 - beta) * np.log(F / K)) ** 2 / 24 + ((1 - beta) * np.log(F / K)) ** 4 / 1920)
        sigma = a / b * z / X
        da_f = alpha * T * (((1 - beta) * alpha) ** 2 / 24 +
                            rho * beta * vega * alpha / 8 * (F * K) ** ((1 - beta) / 2)) *\
               (beta - 1) * (F * K) ** (beta - 2)
        db_f = (1 - beta) / 2 * ((F * K) ** (-(1 + beta) / 2)) * (1 + ((1 - beta) * np.log(F / K)) ** 2 / 24 +
                                                                  ((1 - beta) * np.log(F / K)) ** 4 / 1920) +\
               ((F * K) ** ((1 - beta) / 2)) * ((1 - beta) ** 2 / 24 + (1 - beta) ** 4 / 960 * (np.log(F / K)) ** 2) *\
               2 * np.log(F / K) / F
        dz_f = vega / alpha * (((F * K) ** ((1 - beta) / 2)) / F + (1 - beta) / 2 * ((F * K) ** (-(1 + beta) / 2)) *
                               K * np.log(F / K))
        dsigma_f = (da_f / b - a * db_f / b ** 2) * z / X + (dz_f / X - z * dz_f / X ** 2) * a / b
        return Greek.delta(S, K, T, rf, sigma, CorP) + Greek.vega(S, K, T, rf, sigma) * dsigma_f

    @staticmethod
    def sabr_vega(S, K, T, rf, alpha, beta, args):
        sigma = Greek.sabr(S, K, T, rf, alpha, beta, args)
        return Greek.vega(S, K, T, rf, sigma) * sigma / Greek.sabr_atm(S, T, rf, alpha, beta, args)
