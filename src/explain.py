
import shap, numpy as np, pandas as pd
from model import build_model

X_ts = np.load("data/timeseries.npy")[:100]
X_static = pd.read_csv("data/static.csv").values[:100]

model = build_model(X_ts.shape[1:], (X_static.shape[1],))
explainer = shap.Explainer(model, [X_ts, X_static])
_ = explainer([X_ts, X_static])
print("SHAP explanation generated successfully")
