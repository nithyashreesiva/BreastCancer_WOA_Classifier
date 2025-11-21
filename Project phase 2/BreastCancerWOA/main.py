import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# ----------------------------------------------------
# STEP 1: LOAD DATASET
# ----------------------------------------------------
data = pd.read_csv("data.csv")
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# Convert diagnosis to numeric
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})

X = data.drop('diagnosis', axis=1).values
y = data['diagnosis'].values


# ----------------------------------------------------
# STEP 2: PREPROCESS
# ----------------------------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)


# ----------------------------------------------------
# STEP 3: FITNESS FUNCTION (ACCURACY)
# ----------------------------------------------------
def fitness_function(features):
    selected_features = [i for i in range(len(features)) if features[i] == 1]

    if len(selected_features) == 0:
        return 0  # no features selected = bad solution

    clf = SVC()
    clf.fit(X_train[:, selected_features], y_train)
    y_pred = clf.predict(X_test[:, selected_features])
    return accuracy_score(y_test, y_pred)


# ----------------------------------------------------
# STEP 4: WOA IMPLEMENTATION
# ----------------------------------------------------
def WOA(population_size=10, iterations=20):

    dim = X_train.shape[1]  # 30 features
    whales = np.random.randint(0, 2, (population_size, dim))
    best_whale = whales[0].copy()
    best_score = fitness_function(best_whale)

    for t in range(iterations):
        a = 2 - t * (2 / iterations)

        for i in range(population_size):
            r = np.random.random()
            A = 2 * a * r - a
            C = 2 * r

            if np.random.random() < 0.5:
                # encircling prey
                D = abs(C * best_whale - whales[i])
                new_position = best_whale - A * D
            else:
                # search for prey
                rand_whale = whales[np.random.randint(0, population_size)]
                D = abs(C * rand_whale - whales[i])
                new_position = rand_whale - A * D

            # convert to 0/1 feature selection
            new_position = np.where(new_position > 0.5, 1, 0)

            new_score = fitness_function(new_position)

            if new_score > best_score:
                best_score = new_score
                best_whale = new_position.copy()

        print(f"Iteration {t+1}/{iterations}  --> Best Accuracy: {best_score:.4f}")

    return best_whale, best_score


# ----------------------------------------------------
# STEP 5: RUN WOA
# ----------------------------------------------------
best_features, best_accuracy = WOA()

print("\nBest accuracy:", best_accuracy)
print("Selected features index:", np.where(best_features == 1)[0])
