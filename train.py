import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import skops.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

if __name__ == "__main__":
    drug_df = pd.read_csv('Data/drug200.csv')
    drug_df = drug_df.sample(frac=1) # Randomizing the data
    drug_df.head()

    X = drug_df.drop('Drug', axis=1).values
    y = drug_df.Drug.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=32)

    cat_col = [1, 2, 3]
    num_col = [0, 4]

    transform = ColumnTransformer(
        [
            ("encoder", OrdinalEncoder(), cat_col),
            ("num_imputer", SimpleImputer(strategy='median'), num_col),
            ("num_scaler", StandardScaler(), num_col),
        ]
    )

    pipe = Pipeline(
        steps=[
            ("preprocessin", transform),
            ("model", RandomForestClassifier(n_estimators=100, random_state=32)),
        ]
    )

    pipe.fit(X_train, y_train)

    predictions = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average='macro')

    with open("Results/metrics.txt", "w") as outfile:
        outfile.write(f"\nAccuracy = {str(round(accuracy, 2) * 100)} %, F1 Score = {str(round(f1, 2))}.")


    cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot()

    plt.savefig("Results/model_results.png", dpi=120)

    sio.dump(pipe, "Model/drug_pipeline.skops")

    unknown_types = sio.get_untrusted_types(file="./Model/drug_pipeline.skops")
    sio.load("Model/drug_pipeline.skops", trusted=unknown_types)
