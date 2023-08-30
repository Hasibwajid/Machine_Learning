import pickle
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
warnings.filterwarnings('ignore')
iris = datasets.load_iris()

'''Model to predict wheather the flower is virginica iris or not '''


x = iris.data.astype('int') #features in data
y = iris.target.astype('int')#lables in target

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,random_state=1)
classifier = LogisticRegression()

classifier.fit(X_train,Y_train)

#
# print(classifier.predict([[23,45,66,77]]))

with open('model_pickle','wb') as file:
    pickle.dump(classifier , file)




#0 - Iris-Setosa
#1 - Iris-Versicolour
#2 - Iris-Virginica
