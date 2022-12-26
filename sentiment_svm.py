import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


df = pd.DataFrame({'text': ['This movie was amazing!',
                            'I hated this movie',
                            'The plot was confusing',
                            'I was bored during the movie',
                            'I loved the acting in this film'],
                   'sentiment': ['positive', 'negative', 'neutral', 'negative', 'positive']})

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sentiment'], test_size=0.2)

model = Pipeline([
    ('vectorizer', TfidfVectorizer()),
    ('classifier', SVC())
])

param_grid = {
    'vectorizer__ngram_range': [(1, 1), (1, 2)],
    'vectorizer__max_df': [0.5, 1.0],
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['linear', 'rbf']
}
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

model = grid_search.best_estimator_
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print('Accuracy:', accuracy)

new_text = 'This movie was terrible!'
prediction = model.predict([new_text])
print('Prediction:', prediction)
