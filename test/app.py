from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app, resources={r"/predict_logistic": {"origins": "http://localhost:3000"},
                     r"/predict_rf": {"origins": "http://localhost:3000"}})

logistic_model_filename = 'best_model_lr.pkl'  
logistic_model = joblib.load(logistic_model_filename)

X_train_logistic_filename = 'X_train_for_models.csv'  
X_train_logistic = pd.read_csv(X_train_logistic_filename)

imputer_logistic = SimpleImputer(strategy='mean')
scaler_logistic = StandardScaler()

imputer_logistic.fit(X_train_logistic)
scaler_logistic.fit(X_train_logistic)

rf_model_filename = 'best_model_rf.pkl' 
rf_model = joblib.load(rf_model_filename)

X_train_rf_filename = 'X_train_for_models.csv'  
X_train_rf = pd.read_csv(X_train_rf_filename)

imputer_rf = SimpleImputer(strategy='mean')
scaler_rf = StandardScaler()

imputer_rf.fit(X_train_rf)
scaler_rf.fit(X_train_rf)


@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/predict_logistic', methods=['POST'])
def predict_logistic():
    print('Received logistic regression prediction request')  

    
    user_input = request.get_json()
    print('User input for logistic regression:', user_input) 

    user_input_df = pd.DataFrame([user_input])

    user_input_df = user_input_df.reindex(columns=X_train_logistic.columns, fill_value=0)

    user_input_df = user_input_df.fillna(0)

    relevant_features = ['sport', 'phone', 'mini', 'men', 'women', 'outdoor',
                     'color', 'light', 'fit', 'cloth', 'decor', 'set', 'waterproof', 
                     'eye', 'new', 'pro', 'accessori', 'watch', 'cover', 'home', 'led',
                     'holder', 'accessories', 'sports',
                     'camping', 'phones', 'health', 'technology', 'beauty', 'electronic']

    user_input_selected = user_input_df[relevant_features]

    user_input_selected = user_input_selected[X_train_logistic.columns]

    user_input_scaled = scaler_logistic.transform(user_input_selected)

    if user_input_scaled.shape[1] != X_train_logistic.shape[1]:
        user_input_scaled = user_input_scaled[:, :X_train_logistic.shape[1]]

    prediction = logistic_model.predict(user_input_scaled)

    return jsonify({'prediction_logistic': str(prediction[0])})

@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    print('Received Random Forest prediction request')  # Add this line

    user_input = request.get_json()
    print('User input for Random Forest:', user_input) 

    user_input_df = pd.DataFrame([user_input])

    user_input_df = user_input_df.reindex(columns=X_train_rf.columns, fill_value=0)

    user_input_df = user_input_df.fillna(0)

    relevant_features_rf = ['sport', 'phone', 'mini', 'men', 'women', 'outdoor',
                            'color', 'light', 'fit', 'cloth', 'decor', 'set', 'waterproof',
                            'eye', 'new', 'pro', 'accessori', 'watch', 'cover', 'home', 'led',
                            'holder', 'accessories', 'sports',
                            'camping', 'phones', 'health', 'technology', 'beauty', 'electronic']

    user_input_selected_rf = user_input_df[relevant_features_rf]

    user_input_selected_rf = user_input_selected_rf[X_train_rf.columns]

    user_input_scaled_rf = scaler_rf.transform(user_input_selected_rf)

    if user_input_scaled_rf.shape[1] != X_train_rf.shape[1]:
        
        user_input_scaled_rf = user_input_scaled_rf[:, :X_train_rf.shape[1]]

    prediction_rf = rf_model.predict(user_input_scaled_rf)

    return jsonify({'prediction_rf': str(prediction_rf[0])})


if __name__ == '__main__':
    app.run(debug=True)
