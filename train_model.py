import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import joblib

class GestureModelTrainer:
    def __init__(self, dataset_path='gesture_dataset/gestures.csv'):
        self.dataset_path = dataset_path
        self.model = None
        self.label_encoder = None
        self.history = None
        
    def load_data(self):
        if len(X) == 0:
            raise ValueError("Dataset is empty. Collect data first.")

        """Load and preprocess the dataset"""
        print("Loading dataset...")
        data = pd.read_csv(self.dataset_path)
        
        # Separate features and labels
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        
        print(f"Total samples: {len(X)}")
        print(f"Unique gestures: {len(np.unique(y))}")
        print(f"Gestures: {np.unique(y)}")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape, num_classes):
        """Build CNN-style model for gesture classification"""
        model = keras.Sequential([
            # Input layer
            layers.Input(shape=(input_shape,)),
            
            # Reshape for 1D convolution (treating landmarks as sequence)
            layers.Reshape((21, 3)),  # 21 landmarks, 3 coordinates each
            
            # Convolutional layers
            layers.Conv1D(64, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, 3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            
            # Dense layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            # Output layer
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def train(self, epochs=100, batch_size=32):
        """Train the model"""
        # Load data
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        self.model = self.build_model(X_train.shape[1], num_classes)
        
        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Summary:")
        self.model.summary()
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train
        print("\nTraining model...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        print("\nEvaluating model...")
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Test Accuracy: {test_acc * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_acc
    
    def save_model(self, model_path='models/gesture_model.h5', 
                   encoder_path='models/label_encoder.pkl'):
        """Save trained model and label encoder"""
        import os
        os.makedirs('models', exist_ok=True)
        
        print("\nSaving model...")
        self.model.save(model_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"Model saved to {model_path}")
        print(f"Label encoder saved to {encoder_path}")
    
    def plot_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy plot
        ax1.plot(self.history.history['accuracy'], label='Train')
        ax1.plot(self.history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss plot
        ax2.plot(self.history.history['loss'], label='Train')
        ax2.plot(self.history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        print("Training history plot saved to models/training_history.png")
        plt.show()

def main():
    print("=" * 60)
    print("VIRTUAL KEYBOARD - CNN MODEL TRAINING")
    print("=" * 60)
    
    trainer = GestureModelTrainer()
    
    # Train model
    accuracy = trainer.train(epochs=100, batch_size=32)
    
    # Save model
    trainer.save_model()
    
    # Plot results
    trainer.plot_history()
    
    print("\n" + "=" * 60)
    print(f"Training Complete! Final Accuracy: {accuracy * 100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()