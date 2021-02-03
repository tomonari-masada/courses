import numpy as np
import pandas as pd
from sudachipy import tokenizer, dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import optuna


def parse_sudachi(body_text):
    tokenizer_obj = dictionary.Dictionary().create()
    mode = tokenizer.Tokenizer.SplitMode.C
    tokens = tokenizer_obj.tokenize(body_text, mode)
    words = []
    for t in tokens:
        pos = t.part_of_speech()[0]
        if pos in ['名詞', '動詞', '形容詞', '形状詞']:
            words.append(t.dictionary_form())
    return ' '.join(words)


# ファイルを読む
# df3.to_csv('table_with_morph.csv') で作ったCSVファイル
df = pd.read_csv('table_with_morph.csv')
#y = (df['入学年'] <= 1968) * 2 - 1
y = (df['1運動には'] == '一般学生として参加') * 2 - 1
ratio = (y == 1).sum() / len(y)
ratio = max(ratio, 1.0 - ratio)
print(f'# majority class ratio {ratio:.5f}')


# テキストデータはできるだけ拾うことにしてみる
# （定型的な回答になりがちな設問は除く）
corpus = []
for i in range(df.shape[0]):
    entries = []
    for j in range(1, df.loc[i].shape[0]):
        #print(j, df.columns[j], df.iloc[i,j])
        entry = str(df.iloc[i, j])
        if entry == 'nan':
            continue
        if j in [14, 16, 17, 18, 21, 38, 61, 62, 63, 76, 80, 81, 82, 83, 84]:
            entries.append(entry)
    line = parse_sudachi(' '.join(entries))
    corpus.append(line)
print(f'# number of documents {len(corpus)}')


# optuna用の関数
def objective(trial):
    min_df = trial.suggest_int("min_df", 1, 5)
    max_df = trial.suggest_uniform("max_df", 0.05, 1.0)
    C = trial.suggest_loguniform("C", 0.001, 1000.0)
    intercept_scaling = trial.suggest_uniform("intercept_scaling", 1.0, 100.0)
    
    # TF-IDFの計算
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    X = vectorizer.fit_transform(corpus).toarray()
    vocab = np.array(vectorizer.get_feature_names())

    clf = LinearSVC(C = C,
                    intercept_scaling = intercept_scaling,
                    max_iter = 10000,
                    )

    errors = []
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits)
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, clf.predict(X_train))
        penalty = (1.0 - train_acc) * 10000.0
        errors.append(penalty + 1.0 - accuracy_score(y_test, clf.predict(X_test)))
    return np.array(errors).mean()


study = optuna.create_study()
study.optimize(objective, n_trials = 1000)

print(study.best_params)
print(1.0 - study.best_value)

params = study.best_params

# TF-IDFの計算
vectorizer = TfidfVectorizer(min_df=params["min_df"], max_df=params["max_df"])
X = vectorizer.fit_transform(corpus).toarray()
vocab = np.array(vectorizer.get_feature_names())
print(f'# shape {X.shape}', flush=True)

clf = LinearSVC(C = params["C"],
                intercept_scaling = params["intercept_scaling"],
                max_iter = 10000,
                )
clf.fit(X, y)
coef = clf.coef_.squeeze()
ind = np.argsort(- coef)
print(f'class +1 top 20 words {",".join(list(vocab[ind[:20]]))}')
ind = np.argsort(coef)
print(f'class -1 top 20 words {",".join(list(vocab[ind[:20]]))}')
