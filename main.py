import numpy as np
import pandas as pd
import yfinance as yf
from river import tree, metrics, drift
from sklearn.metrics import precision_score
from datetime import datetime
import pickle
import os

# =========================================
# CONFIG
# =========================================
STOCKS = {
    "HDFCBANK.NS": 0,
    "ICICIBANK.NS": 1,
    "SBIN.NS": 2,
    "AXISBANK.NS": 3,
    "KOTAKBANK.NS": 4
}

MODEL_FILE = "dailym.pkl"
LOG_FILE = "run_log.txt"

# =========================================
# LOAD / SAVE MODEL
# =========================================
def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            print("Loaded existing model")
            return pickle.load(f)
    else:
        print("Creating new model")
        return tree.HoeffdingAdaptiveTreeClassifier(grace_period=30)


def save_model(model):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)


# =========================================
# FETCH DATA
# =========================================
def fetch_data(ticker, interval="1d", period="1y"):
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=True, progress=False)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    df = df.apply(pd.to_numeric, errors='coerce')
    df.columns = [col if not isinstance(col, tuple) else col[0] for col in df.columns]
    df = df.squeeze().dropna()

    return df


def fetch_niftybank(interval="1d", period="1y"):
    nifty = yf.download("^NSEBANK", interval=interval, period=period,
                        auto_adjust=True, progress=False)

    nifty = nifty[['Close']]
    nifty = nifty.apply(pd.to_numeric, errors='coerce')
    nifty.columns = ['Close_nifty']
    nifty = nifty.squeeze().dropna()

    return nifty


# =========================================
# FEATURES
# =========================================
def compute_features(df, nifty_df, stock_id):
    df = df.join(nifty_df, how='inner')

    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['ret_21d'] = np.log(df['Close'] / df['Close'].shift(21))

    df['nifty_ret_21d'] = np.log(df['Close_nifty'] / df['Close_nifty'].shift(21))
    df['rel_strength'] = df['ret_21d'] - df['nifty_ret_21d']

    ema_9 = df['Close'].ewm(span=9, adjust=False).mean()
    ema_50 = df['Close'].ewm(span=50, adjust=False).mean()
    df['ema_ratio'] = ema_9 / ema_50

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    rs = gain.rolling(14).mean() / (loss.rolling(14).mean() + 1e-9)
    df['rsi'] = (100 - (100 / (1 + rs))).clip(30, 70)

    df['vol_z'] = (df['Volume'] - df['Volume'].rolling(20).mean()) / (
        df['Volume'].rolling(20).std() + 1e-9
    )

    df['volatility_20d'] = df['log_return'].rolling(20).std()
    df['stock_id'] = stock_id

    return df.dropna()


# =========================================
# BUILD DATASET
# =========================================
def build_dataset():
    nifty_df = fetch_niftybank()
    all_data = []

    for ticker, stock_id in STOCKS.items():
        df = fetch_data(ticker)
        df = compute_features(df, nifty_df, stock_id)
        all_data.append(df)

    return pd.concat(all_data).sort_index()


# =========================================
# TARGET
# =========================================
def prepare_classification(df):
    df['target'] = (df['log_return'] > 0).astype(int)
    X = df.drop(columns=['log_return', 'target'])
    y = df['target']
    return X, y


# =========================================
# SIGNAL (SELL EDGE)
# =========================================
def signal_A(prob_up):
    return 0 if prob_up < 0.3 else None


# =========================================
# MAIN PIPELINE
# =========================================
def run():

    print(f"\nRun started: {datetime.now()}")

    model = load_model()
    drift_detector = drift.ADWIN()
    metric = metrics.Accuracy()

    df = build_dataset()
    X, y = prepare_classification(df)

    trades = 0
    wins = 0
    pnl = 0

    for x, target in zip(X.to_dict("records"), y):

        proba = model.predict_proba_one(x)
        prob_up = proba.get(1, 0.5)

        signal = signal_A(prob_up)

        if signal is not None:
            trades += 1

            if signal == 0:
                trade_pnl = 1 if target == 0 else -1
                pnl += trade_pnl

                if trade_pnl > 0:
                    wins += 1

        pred = 1 if prob_up > 0.5 else 0
        metric.update(target, pred)

        drift_detector.update(int(pred != target))

        if drift_detector.drift_detected:
            print("⚠️ Drift detected!")

        model.learn_one(x, target)

    save_model(model)

    win_rate = wins / trades if trades > 0 else 0

    result = {
        "time": str(datetime.now()),
        "accuracy": metric.get(),
        "trades": trades,
        "win_rate": win_rate,
        "pnl": pnl
    }

    print(result)

    with open(LOG_FILE, "a") as f:
        f.write(str(result) + "\n")


if __name__ == "__main__":
    run()