from flask import Flask, render_template, request
import joblib
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names, but .* was fitted with feature names", category=UserWarning)

app = Flask(__name__)

min_values = [32.4, 18.7, 24.7, 14.3, 22.2, 9.9, 8.1, 15.7, 9.9, 85.9, 72.6, 57.9, 64.0, 78.8, 46.3, 22.4, 19.6, 29.0, 28.4, 16.4, 13.0]
max_values = [47.4, 34.7, 38.0, 27.5, 35.66, 16.7, 13.3, 24.3, 17.2, 134.8, 118.7, 113.2, 121.1, 128.3, 75.7, 42.4, 32.5, 49.0, 47.7, 29.3, 19.6]
mid_values = [round((min_val + max_val) / 2, 1) for min_val, max_val in zip(min_values, max_values)]
labels=['Biacromial diameter','Biiliac diameter (pelvic breadth)','Bitrochanteric diameter','Chest depth','Chest diameter','Elbow diameter','Wrist diameter',
        'Knee diameter','Ankle diameter','Shoulder girth','Chest girth','Waist girth','Navel (abdominal) girth','Hip girth','Thigh girth','Bicep girth','Forearm girth',
        'Knee girth','Calf maximum girth','Ankle minimum girth','Wrist minimum girth']

weightmodel=joblib.load('weight_rfr_fs.pkl')
heightmodel=joblib.load('height_rfr_fs.pkl')
gendermodel=joblib.load('gender_knn.pkl')

def predw(inp):
    pred_weight=weightmodel.predict(inp)
    return round(pred_weight[0], 2)

def predh(inp):
    pred_height=heightmodel.predict(inp)
    return round(pred_height[0], 2)

def predg(inp):
    pred_gender=gendermodel.predict(inp)
    if pred_gender[0]==1:
        gender='Male'
    else:
        gender="Female"
    return gender

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        values = [float(request.form[f'value_{i}']) for i in range(len(min_values))]
        
        selected_values = []
        for i in range(len(values)):
            selected_values.append((values[i]))
            
        input_vector=np.array(selected_values).reshape(1,-1)
        
        gender = predg(input_vector)
        height = predh(input_vector)
        weight = predw(input_vector)
        
        return render_template('index.html', num_values=len(min_values), min_values=min_values, max_values=max_values, selected_values=selected_values, gender=gender, height=height, weight=weight, labels=labels,mid_values=mid_values)
    else:
        return render_template('index.html', num_values=len(min_values), min_values=min_values, max_values=max_values, labels=labels,mid_values=mid_values)
    
if __name__ == '__main__':
    app.run(debug=True)

