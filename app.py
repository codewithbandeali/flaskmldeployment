from flask import Flask, render_template, request
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField
from wtforms.validators import DataRequired
import pickle
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'

class InputForm(FlaskForm):
    cement = FloatField('Cement', validators=[DataRequired()])
    slag = FloatField('Slag', validators=[DataRequired()])
    flyash = FloatField('Fly Ash', validators=[DataRequired()])
    water = FloatField('Water', validators=[DataRequired()])
    superplasticizer = FloatField('Superplasticizer', validators=[DataRequired()])
    coarseaggregate = FloatField('Coarse Aggregate', validators=[DataRequired()])
    fineaggregate = FloatField('Fine Aggregate', validators=[DataRequired()])
    age = FloatField('Age', validators=[DataRequired()])
    submit = SubmitField('Predict Strength')

@app.route('/', methods=['GET', 'POST'])
def index():
    form = InputForm()
    result = None
    if form.validate_on_submit():
        data = np.array([[form.cement.data, form.slag.data, form.flyash.data, form.water.data,
                          form.superplasticizer.data, form.coarseaggregate.data, form.fineaggregate.data,
                          form.age.data]])
        try:
            print("Loading model...")
            with open('clf.pkl', 'rb') as file:
                model = pickle.load(file)
            print("Model loaded successfully.")
            print(f"Predicting with data: {data}")
            prediction = model.predict(data)
            result = f'Predicted Strength: {prediction[0]:.2f}'
            print(f"Prediction result: {result}")
        except Exception as e:
            result = f'Error: {str(e)}'
            print(result)
    return render_template('index.html', form=form, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
