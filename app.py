from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the model and data
pipe = joblib.load('pipe.joblib')
df = joblib.load('df.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get form data
        company = request.form['company']
        type_ = request.form['type']
        ram = int(request.form['ram'])
        weight = float(request.form['weight'])
        touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
        ips = 1 if request.form['ips'] == 'Yes' else 0
        screen_size = float(request.form['screen_size'])
        resolution = request.form['resolution']
        cpu = request.form['cpu']
        hdd = int(request.form['hdd'])
        ssd = int(request.form['ssd'])
        gpu = request.form['gpu']
        os = request.form['os']

        # Process resolution and calculate ppi
        X_res = int(resolution.split('x')[0])
        Y_res = int(resolution.split('x')[1])
        ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

        # Prepare the query array
        query = np.array([company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
        query = query.reshape(1, -1)

        # Make the prediction
        try:
            predicted_price = int(np.exp(pipe.predict(query)[0]))
            result = f"The predicted price of this configuration is ${predicted_price}"
        except Exception as e:
            result = f"Error in prediction: {e}"

        return render_template('index.html', result=result, df=df)

    # Initial GET request, render the form
    return render_template('index.html', df=df)

if __name__ == '__main__':
    app.run(debug=True)
