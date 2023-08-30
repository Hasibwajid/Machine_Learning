from flask import Flask,render_template,request
import pickle
import numpy as np
# app = Flask(__name__,template_folder='templates')
app = Flask(__name__)

model = pickle.load(open("model_pickle", "rb"))
@app.route('/')
def hello_world():
    # if(request.method== 'POST'):
    #     sepal_L = request.form['sapleL']
    #     sepal_W = request.form['sapleW']
    #     petal_L = request.form['petalL']
    #     petal_W = request.form['petalW']
    #     y_pred = [[sepal_L,sepal_W,petal_L, petal_W]]
    #     prediction_val = model.predict(y_pred)
    #     if prediction_val <= 0.5:
    #         return render_template('iris.html',pred='Iris-Setosa')
    #     elif prediction_val < 1.5:
    #         return render_template('iris.html',pred='Iris-Versicolour')
    #     else:
    #         return render_template('iris.html',pred='Iris-Virginica')
    return  render_template('iris.html')





@app.route('/predict',methods=['POST','GET'])
def predict():
    print(request.form)
    if (request.method == 'POST'):
        sepal_L = request.form['sapleL']
        sepal_W = request.form['sapleW']
        petal_L = request.form['petalL']
        petal_W = request.form['petalW']
        y_pred = [[sepal_L, sepal_W, petal_L, petal_W]]
        prediction = model.predict(y_pred)
        if prediction <= 0.5:
            return render_template('iris.html', pred='This is Iris-Setosa flower .')
        elif prediction < 1.5:
            return render_template('iris.html', pred='This is Iris-Versicolour flower .')
        else:
            return render_template('iris.html', pred='This is Iris-Virginica flower .')



if __name__ =='__main__':
    app.run(debug=True)


# 0 - Iris-Setosa
# 1 - Iris-Versicolour
# 2 - Iris-Virginica
