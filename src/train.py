
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from model import build_model

X_ts = np.load("data/timeseries.npy")
X_static = pd.read_csv("data/static.csv").values
y = pd.read_csv("data/labels.csv").values.flatten()

Xts_tr, Xts_te, Xs_tr, Xs_te, y_tr, y_te = train_test_split(
    X_ts, X_static, y, test_size=0.2, stratify=y, random_state=42
)

classes = np.unique(y_tr)
weights = compute_class_weight("balanced", classes=classes, y=y_tr)
class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

model = build_model(Xts_tr.shape[1:], (Xs_tr.shape[1],))
model.fit(
    [Xts_tr, Xs_tr], y_tr,
    epochs=25,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weight,
    verbose=1
)

model.evaluate([Xts_te, Xs_te], y_te, verbose=1)
