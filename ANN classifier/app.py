{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "ce4599d7-1e4e-440b-9ff0-c3207ca1c6e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import tensorflow as tf\n",
    "from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder\n",
    "import datetime\n",
    "import pickle\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "1683d274-12be-4840-a0e7-883e775e0c70",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.\n"
     ]
    }
   ],
   "source": [
    "# Load the trained model\n",
    "model = tf.keras.models.load_model('model.h5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "6046e2c2-0d5d-4366-9661-2201a5d29c35",
   "metadata": {},
   "outputs": [],
   "source": [
    "# load the encoder and scaler\n",
    "\n",
    "# Load the encoders and scaler\n",
    "with open('label-encode_gender.pkl', 'rb') as file:\n",
    "    label_encoder_gender = pickle.load(file)\n",
    "\n",
    "with open('onehot_encode_geo.pkl', 'rb') as file:\n",
    "    onehot_encoder_geo = pickle.load(file)\n",
    "\n",
    "with open('scaler.pkl', 'rb') as file:\n",
    "    scaler = pickle.load(file)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "88f0b193-ecfb-4269-a033-78bef178215c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DeltaGenerator()"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## streamlit app\n",
    "st.title('Customer Churn Prediction')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c281e9f2-6a3a-4f19-a3b4-0c585a3f1853",
   "metadata": {},
   "outputs": [],
   "source": [
    "# User input\n",
    "geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])\n",
    "gender = st.selectbox('Gender', label_encoder_gender.classes_)\n",
    "age = st.slider('Age', 18, 92)\n",
    "balance = st.number_input('Balance')\n",
    "credit_score = st.number_input('Credit Score')\n",
    "estimated_salary = st.number_input('Estimated Salary')\n",
    "tenure = st.slider('Tenure', 0, 10)\n",
    "num_of_products = st.slider('Number of Products', 1, 4)\n",
    "has_cr_card = st.selectbox('Has Credit Card', [0, 1])\n",
    "is_active_member = st.selectbox('Is Active Member', [0, 1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "133f3ec6-c903-495d-bf9d-d9c4d21fdde4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare the input data\n",
    "input_data = pd.DataFrame({\n",
    "    'CreditScore': [credit_score],\n",
    "    'Gender': [label_encoder_gender.transform([gender])[0]],\n",
    "    'Age': [age],\n",
    "    'Tenure': [tenure],\n",
    "    'Balance': [balance],\n",
    "    'NumOfProducts': [num_of_products],\n",
    "    'HasCrCard': [has_cr_card],\n",
    "    'IsActiveMember': [is_active_member],\n",
    "    'EstimatedSalary': [estimated_salary]\n",
    "})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "0e44ed11-e70b-47af-ab85-5b502307bb2e",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\SACHIN\\AppData\\Local\\Programs\\Python\\Python312\\Lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but OneHotEncoder was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    }
   ],
   "source": [
    "# One-hot encode 'Geography'\n",
    "geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()\n",
    "geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))\n",
    "\n",
    "# Combine one-hot encoded columns with input data\n",
    "input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "9c28a489-0c62-47a6-ab16-72e540210482",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Scale the input data\n",
    "input_data_scaled = scaler.transform(input_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "68160341-3d0f-4411-8df5-fc1acef88255",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 394ms/step\n"
     ]
    }
   ],
   "source": [
    "# Predict churn\n",
    "prediction = model.predict(input_data_scaled)\n",
    "prediction_proba = prediction[0][0]\n",
    "\n",
    "st.write(f'Churn Probability: {prediction_proba:.2f}')\n",
    "\n",
    "if prediction_proba > 0.5:\n",
    "    st.write('The customer is likely to churn.')\n",
    "else:\n",
    "    st.write('The customer is not likely to churn.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc85b08f-ce96-4c8b-946a-e6672f3b131b",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
