# CNN Classification on MNIST

This project utilizes **CNN (Convolutional Neural Network)** for multi-class classification on the MNIST handwritten digit dataset. This document provides a brief overview of the **Structure of the Repository, Steps to Reproduce the Results, Model Architecture, Hyperparameter Tuning Methods, Results, and Techniques to Prevent Overfitting**.

---
## 1. Structure of the Repository
```
MNIST_CNN
┌── data
├── logs
├── saves
├── environment.yml
├── README.md
├── mnist_cnn.py
├── mnist_cnn_optuna.py
├── model.py
└── utils.py
```

- **`data/`**: Folder that contains the MNIST dataset.  
- **`logs/`**: Folder that contains log files.  
- **`saves/`**: Folder that contains saved models and training record figures.  
- **`environment.yml`**: A Conda environment file describing the Python packages and versions needed to reproduce the environment.  
- **`README.md`**: The README file, providing instructions and information about the project.  
- **`mnist_cnn.py`**: The primary script for training and testing the CNN model on the MNIST dataset.  
- **`mnist_cnn_optuna.py`**: A script integrating Optuna for hyperparameter tuning on the CNN model.  
- **`model.py`**: The script defining the CNN model architecture.  
- **`utils.py`**: Utility scripts, such as custom data loaders, logging functions, or other helper routines.

---  
## 2. Steps to Reproduce the Results

1. **Download the repository**  
2. **Create and activate the Conda environment**  
   ```bash
   conda env create -f environment.yml
   conda activate your_env_name
   ```
3. **Run the CNN training script**  
   ```bash
   python mnist_cnn.py
   ```
   - This script automatically downloads the MNIST dataset and stores it in the `data/` folder, trains the model with early stopping, and saves training logs in the `logs/` folder.  
   - The trained model and related training curves are stored in the `saves/` folder, which includes the following files:
     - `accuracies.png`: Training and validation accuracy plot.  
     - `errors.png`: Training and validation error plot.  
     - `loss_train_and_val.png`: Training and validation loss curve.  
     - `final_state_dict.pth`: Final model state dictionary.  
     - `model_last.pt`: The last saved model checkpoint.  

4. **Run Optuna hyperparameter search**  
   ```bash
   python mnist_cnn_optuna.py
   ```
   - This script performs multiple trials to search for the best hyperparameters, outputs the optimal results, and generates related visualizations (e.g., `optimization_history.png`).

## 3. Model Architecture

This project uses a **CNN** model to process the 2D MNIST images, each of size *28 × 28* (one color channel). The network structure is organized as follows:

1. **Input Layer**: The MNIST images, shaped as `(batch_size, 1, 28, 28)`, are fed into the network.
2. **Convolutional and Pooling Layers**:  
   - **Conv1**: A 2D convolution layer with *16* output channels, a *5 × 5* kernel, and stride *1*. The output is passed through a *2 × 2* max-pooling operation.  
   - **Conv2**: Another 2D convolution layer with *32* output channels, again using a *5 × 5* kernel (stride *1*), followed by a *2 × 2* max-pooling.  
3. **Dropout2D**: A spatial dropout (`drop1`) is applied to the 2D feature maps to help prevent overfitting.  
4. **Flatten**: The 2D feature maps are reshaped into a single vector `(batch_size, 32 × 7 × 7)` to prepare for fully connected layers.  
5. **Fully Connected Layers**:  
   - **FC1**: A linear layer with a hidden size of *96*. After applying a ReLU activation, another dropout (`drop2`) is introduced.  
   - **FC2**: A final linear layer mapping from the hidden size *96* to the *10* output classes (digits *0*–*9*).  
6. **Log-Softmax Output**: In the forward pass, the output of `FC2` is passed through a `log_softmax` function (with dimension `dim = 1`). During training, we pair this with a negative log-likelihood loss function (`NLLLoss`), using the integer digit labels. 

A simplified flow diagram is illustrated below:
```
(Input) --> Conv1 -> MaxPool2d -> Conv2 -> MaxPool2d -> Dropout2d
          --> Flatten -> FC1 -> Dropout -> FC2 -> log_softmax -> NLLLoss
```

---
## 4. Hyperparameter Tuning Methods

### 1. **Script Arguments (`argparse`)**  
   - **Learning rate** (`--lr`): Can be manually specified (default: `0.001`).
   - **Weight decay** (`--weight-decay`): Default is `1.96e-06`.
   - **Batch size** (`--batch-size`): Default is `64`.  

### 2. **Optuna Hyperparameter Optimization**  
   - The script `mnist_cnn_optuna.py` contains the `objective()` function, which automatically searches for optimal hyperparameters, including:
     - Optimizer (Adam / SGD / RMSprop )  
     - Learning rate (`1e-5` to `1e-1`, log scale)
     - Weight decay (`1e-6` to `1e-2`, log scale) 
     - Batch size (`64` to `256`, step size of 64)
     - dropout_rate (`0` to `0.5`, step size of 0.1)  
     - fc2_input_dim (`32` to `128`, step size of 32)  
     - dropout_rate2 (`0` to `0.3`, step size of 0.1) 

Optuna performs multiple trials to explore different hyperparameter configurations, returning the **best validation accuracy** along with the corresponding hyperparameters. Additionally, it generates various visualizations, including the **optimization history plot, contour plot, parallel coordinate plot, and parameter importance plot**.

---

## 5. Results

With the default configuration (`Optimizer:adam, dropout_rate=0.2, fc2_input_dim=96, dropout_rate2=0.1, lr=0.001, batch_size=64, weight_decay=1.96e-06`), after training for 30 epochs (with early stopping enabled), the model generally achieves high accuracy on the MNIST dataset. The training results (which may vary slightly due to random seed or environment differences) are as follows:

- **Training set**: Accuracy ~ 99.9542%, Error ~ 0.0458%  
- **Validation set**: Accuracy ~ 98.9250%, Error ~ 1.0750%  
- **Test set**: Accuracy ~ 99.0800%, Error ~ 0.9200%  

The log file in `logs/record.txt` contains details of each epoch, including loss, accuracy, error rates, and time-related statistics such as **Time, Time total, and Time remain**.  

Training process visualizations can be found in `saves/YYYYMMDD_HHMMSS_f/`, including the following files:  
- `accuracies.png`: Accuracy curves  
- `errors.png`: Error curves  
- `loss_train_and_val.png`: Training and validation loss curves  
- `final_state_dict.pth`: The final model state dictionary  
- `model_last.pt`: The last saved model checkpoint  

---

## 6. Techniques to Prevent Overfitting

1. **Dropout**: Applied dropout of 0.2 after the convolutional layers and 0.1 after the fully connected layer to randomly deactivate some neurons, helping to reduce overfitting.
3. **Early Stopping**: If the validation loss does not improve for `patience` consecutive epochs, training stops early (default `patience=10`).  
4. **Optimization of Hidden Layer Dimensions**: Properly tuning `hidden_size`, along with L2 regularization and batch size adjustments, helps mitigate overfitting.  

