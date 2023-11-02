import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load the KDD Cup '99 dataset (you may need to download it first)
data = pd.read_csv('kddcup99.csv')

# Define column names for the dataset
columns = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
           "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted",
           "num_root", "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
           "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
           "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
           "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
           "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
           "dst_host_srv_rerror_rate", "activity"]

data.columns = columns

# Define categorical features for label encoding
categorical_features = ["protocol_type", "service", "flag"]

# Apply label encoding to categorical features
label_encoder = LabelEncoder()
for feature in categorical_features:
    data[feature] = label_encoder.fit_transform(data[feature])

# Map target variable 'activity' to binary: normal or attack
attack_types = ['neptune', 'back', 'teardrop', 'smurf', 'pod', 'land', 'apache2', 'imap', 'ftp_write']
data['activity'] = data['activity'].apply(lambda x: 'attack' if x in attack_types else 'normal')

# Define features (X) and target variable (y)
X = data.drop('activity', axis=1)
y = data['activity']

# Encode target variable 'activity' as binary classes
y = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create sequences from the data for LSTM input
sequence_length = 10  # Define a sequence length for capturing temporal dependencies
X_train_seq = []
X_test_seq = []
y_train_seq = []  # Add this to store corresponding labels for training sequences

# Create sequences from the training data
for i in range(len(X_train) - sequence_length + 1):
    X_train_seq.append(X_train[i:i+sequence_length])
    y_train_seq.append(y_train[i+sequence_length-1])  # Append the label corresponding to the last element of the sequence

X_train_seq = np.array(X_train_seq)
y_train_seq = np.array(y_train_seq)

# Create sequences from the testing data
for i in range(len(X_test) - sequence_length + 1):
    X_test_seq.append(X_test[i:i+sequence_length])
X_test_seq = np.array(X_test_seq)

# Create a deep learning model with LSTM layers for binary classification
model = Sequential()
model.add(LSTM(128, input_shape=(sequence_length, X_train_seq.shape[2]), return_sequences=True))
model.add(Dropout(0.5))  # Add a dropout layer
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))  # Add a dropout layer
model.add(LSTM(32))
model.add(Dropout(0.5))  # Add a dropout layer
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

noise_factor = 0.1  # Adjust the noise factor as needed
X_train_seq_noisy = X_train_seq + noise_factor * np.random.normal(0, 1, X_train_seq.shape)

# Train the model with noisy input data
model.fit(X_train_seq_noisy, y_train_seq, epochs=10, batch_size=64)

# Evaluate the model
y_pred = model.predict(X_test_seq)
y_pred = (y_pred > 0.5)  # Convert predictions to binary values (0 or 1)

accuracy = accuracy_score(y_test[sequence_length - 1:], y_pred)
report = classification_report(y_test[sequence_length - 1:], y_pred)

print("Accuracy:", accuracy)
print(report)