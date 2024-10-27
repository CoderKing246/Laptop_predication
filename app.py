from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html', companies=df['Company'].unique(), 
                           types=df['TypeName'].unique(),
                           cpus=df['Cpu brand'].unique(),
                           gpus=df['Gpu brand'].unique(),
                           os_list=df['os'].unique())

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    company = request.form.get('company')
    type = request.form.get('type')
    ram = int(request.form.get('ram'))
    weight = float(request.form.get('weight'))
    touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
    ips = 1 if request.form.get('ips') == 'Yes' else 0
    screen_size = float(request.form.get('screen_size'))
    resolution = request.form.get('resolution')
    cpu = request.form.get('cpu')
    hdd = int(request.form.get('hdd'))
    ssd = int(request.form.get('ssd'))
    gpu = request.form.get('gpu')
    os = request.form.get('os')

    # Calculate PPI
    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res ** 2 + Y_res ** 2) ** 0.5) / screen_size

    # Prepare query
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, 12)

    # Predict price
    predicted_price = int(np.exp(pipe.predict(query)[0]))

    return render_template('result.html', price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
