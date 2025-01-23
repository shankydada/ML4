from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

# Initialize Flask app with dynamic template folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Base directory of the project
TEMPLATE_DIR = os.path.join(BASE_DIR, "template")      # Template folder path
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")       # Model file path

# Initialize Flask app
app = Flask(__name__,template_folder=TEMPLATE_DIR)

# Load the trained model (complete pipeline)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "advanced_model.pkl")
model = joblib.load(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form
        inputs = {
            'AttendanceRate': float(request.form['attendanceRate']),
            'StudyHoursPerWeek': int(request.form['studyHours']),
            'FamilyIncome': int(request.form['familyIncome']),
            'ParentEducationLevel': int(request.form['parentEducationLevel']),
            'SchoolRating': int(request.form['schoolRating']),
            'ParticipationInExtracurriculars': float(request.form['extracurricularParticipation']),
            'HomeworkCompletionRate': float(request.form['homeworkCompletion']),
            'ClassroomBehaviorScore': int(request.form['classroomBehavior']),
            'HoursOfSleepPerNight': float(request.form['sleepHours'])
        }

        # Prepare input data for prediction
        input_data = pd.DataFrame([inputs])

        # Predict final grades using the loaded pipeline
        prediction = model.predict(input_data)[0]
        prediction = round(prediction, 2)

        return render_template('predict.html', prediction=prediction)
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return render_template('error.html', error_message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
