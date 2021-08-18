
from sklearn.datasets import make_classification
from collections import Counter
from imblearn.over_sampling import SMOTE
import pandas as pd

X, y = make_classification(
    n_classes=2,
    class_sep=2,
    weights=[0.9, 0.1],
    n_informative=3,
    n_redundant=1,
    flip_y=0,
    n_features=20,
    n_clusters_per_class=1,
    n_samples=1000, random_state=10)

print(Counter(y))
smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(X, y)
print(Counter(y_smo))
