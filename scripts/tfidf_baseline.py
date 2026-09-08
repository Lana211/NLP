import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

DATA = Path("data/raw/bayan_feedback.csv")


def load_split(split_name):
    texts, labels = [], []
    with DATA.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split_name:
                texts.append(row["text"])
                labels.append(row["topic"])
    return texts, labels


def main():
    train_texts, train_labels = load_split("train")
    test_texts, test_labels = load_split("test")

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LinearSVC()
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_test)

    macro_f1 = f1_score(test_labels, preds, average="macro")
    print(f"TF-IDF + LinearSVC baseline macro-F1 (test split): {macro_f1:.4f}")


if __name__ == "__main__":
    main()
