# main_tkinter.py
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


# --------- Load dataset and prepare model (same logic as prototype) ----------
# Ensure data.csv (Wisconsin dataset) is in the same folder


def prepare_model():
    data = pd.read_csv('data.csv')
    data = data.drop(['id', 'Unnamed: 32'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
    X = data.drop('diagnosis', axis=1).values
    y = data['diagnosis'].values


    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    # For the GUI we will train on full dataset using a few selected features
    # (In practice you would run WOA to pick features; here we pick commonly strong features)
    selected_idx = [0, 2, 3, 7, 12] # example indices; change if you ran WOA
    clf = SVC(probability=True)
    clf.fit(X_scaled[:, selected_idx], y)


    return scaler, clf, selected_idx


scaler, clf, sel_idx = prepare_model()


# ---------- Tkinter GUI ----------
root = tk.Tk()
root.title('Breast Cancer Predictor (Simple)')
root.geometry('540x560')


frame = ttk.Frame(root, padding=12)
frame.pack(fill='both', expand=True)


# feature names (from dataset) - trimmed for GUI
feature_names = [
'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
'compactness_mean','concavity_mean','concave_points_mean','symmetry_mean','fractal_dimension_mean',
# (full dataset has 30 features, we show only first 10 for ease)
]


entries = []
for i,name in enumerate(feature_names[:10]):
    lbl = ttk.Label(frame, text=f"{i+1}. {name}")
    lbl.grid(row=i, column=0, sticky='w', pady=6)
    ent = ttk.Entry(frame, width=20)
    ent.grid(row=i, column=1, pady=6)
    entries.append(ent)

def on_predict():
    try:
        vals = []
        # collect values; if field empty -> treat as 0
        for e in entries:
            t = e.get().strip()
            vals.append(float(t) if t != '' else 0.0)


        # make a full-length sample vector of length 30 (fill missing with zeros)
        sample = np.zeros(30)
        sample[:len(vals)] = np.array(vals)


        sample_scaled = scaler.transform([sample])
        pred = clf.predict(sample_scaled[:, sel_idx])[0]
        prob = clf.predict_proba(sample_scaled[:, sel_idx])[0]


        lab = 'Malignant' if pred == 1 else 'Benign'
        confidence = max(prob)
        messagebox.showinfo('Prediction', f'Result: {lab}\nConfidence: {confidence:.2f}')
    except Exception as ex:
        messagebox.showerror('Error', str(ex))


btn = ttk.Button(frame, text='Predict', command=on_predict)
btn.grid(row=12, column=0, columnspan=2, pady=18)


root.mainloop()

