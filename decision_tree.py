from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_iris()
x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

model = DecisionTreeClassifier(criterion="gini")
model.fit(x_train, y_train)
# model_entropy = DecisionTreeClassifier(criterion="entropy")
# model_entropy.fit(x_train, y_train)

y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize = (12,8))
plot_tree(model,
          feature_names=data.feature_names,
          class_names = data.target_names,
          filled = True)
plt.show()