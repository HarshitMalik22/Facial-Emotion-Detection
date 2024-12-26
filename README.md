Facial Emotion Detection System
A comprehensive project for recognizing facial emotions using machine learning and deep learning techniques. This project processes facial images, trains a deep learning model, and implements a real-time emotion recognition system using webcam integration.
🧠 Objective
To classify facial expressions into one of the following emotions:
•	Angry 😡
•	Disgust 🤢
•	Fear 😨
•	Happy 😄
•	Sad 😢
•	Surprise 😲
•	Neutral 😐
________________________________________
🚀 Features
•	Data Preprocessing: Conversion of raw image data into a structured CSV format.
•	Deep Learning Model: Trained using convolutional neural networks (CNNs) with TensorFlow and Keras.
•	Pre-Trained Model: Includes saved weights and models for direct inference.
•	Real-Time Detection: Webcam integration for real-time emotion recognition.
•	Visualization: Training performance metrics and model architecture.
________________________________________
📂 Files Overview
File	Description
emotion.ipynb	Jupyter notebook with the full pipeline: preprocessing, model training, and evaluation.
fer2013_preprocessed.csv	Preprocessed dataset derived from the FER-2013 dataset.
Facial_Expression_Detection_System.hf5	Trained model in HDF5 format, ready for deployment.
fer.weights.h5	Weights of the trained model.
haarcascade_frontalface_default.xml	Haar Cascade file for face detection in real-time webcam tests.
Webcam_test.py	Python script to test the emotion detection system using a webcam.
best_model.png	Visualization of the best-performing model architecture.
Facial Expression Recognition.json	JSON configuration for model architecture and parameters.
________________________________________
📊 Model Performance
Metric	Value
Training Accuracy	XX%
Test Accuracy	XX%
F1 Score	XX
________________________________________
🛠️ Setup
Prerequisites
Install the required libraries:
pip install numpy pandas tensorflow matplotlib seaborn scikit-learn opencv-python opencv-python-headless
Steps to Run
1.	Clone the Repository:
2.	git clone https://github.com/HarshitMalik22/Facial-Emotion-Detection.git
3.	cd Facial-Emotion-Detection
4.	Run the Notebook:
o	Open emotion.ipynb to preprocess the dataset, train the model, and evaluate it.
5.	Real-Time Testing:
o	Ensure a webcam is connected to your system.
o	Run the Webcam_test.py script: 
o	python Webcam_test.py
6.	Deploy or Extend:
o	Use the saved model (Facial_Expression_Detection_System.hf5) and weights (fer.weights.h5) for deployment or further experiments.
________________________________________
📂 Project Workflow
1. Data Preprocessing
•	Convert images to grayscale.
•	Resize images to 48x48 pixels.
•	Save flattened pixel data into fer2013_preprocessed.csv.
2. Model Training
•	Train a CNN model on the FER-2013 dataset using TensorFlow/Keras.
•	Save the best model (Facial_Expression_Detection_System.hf5) and weights.
3. Real-Time Detection
•	Use Haar Cascades for face detection.
•	Load the pre-trained model to classify emotions in live webcam feed.
________________________________________
🖥️ Technologies Used
•	Languages: Python
•	Libraries: TensorFlow, Keras, NumPy, Pandas, OpenCV, Matplotlib, Seaborn
•	Tools: Jupyter Notebook, Haar Cascades
________________________________________
📂 Folder Structure
Facial-Emotion-Detection/
│
├── best_model.png                           # Model architecture visualization
├── emotion.ipynb                            # Training and evaluation pipeline
├── Facial Expression Recognition.json       # Model configuration
├── Facial_Expression_Detection_System.hf5   # Saved trained model
├── fer.weights.h5                           # Model weights
├── fer2013_preprocessed.csv                 # Preprocessed dataset
├── haarcascade_frontalface_default.xml      # Haar Cascade file for face detection
├── Webcam_test.py                           # Script for real-time detection
├── README.md                                # Project documentation
└── requirements.txt                         # List of required dependencies
________________________________________
📜 Usage Example
Real-Time Emotion Detection
1.	Ensure the pre-trained model files are in the directory.
2.	Run the Webcam_test.py script: 
3.	python Webcam_test.py
4.	Observe real-time emotion predictions on your webcam feed.
________________________________________
👨‍💻 Contributor
•	Harshit Malik
GitHub: HarshitMalik22
________________________________________
🌟 Acknowledgments
•	FER-2013 Dataset: Kaggle
•	TensorFlow and Keras Documentation
•	Haar Cascades for Face Detection by OpenCV
________________________________________
