# cifar-dataset-duel

👾 **CIFAR-10 vs CIFAR-100 – Dataset Showdown**

This project compares the **CIFAR-10** and **CIFAR-100** databases with respect to complexity and training tendencies using a simple Convolutional Neural Network (CNN) to observe:

- 📈 **Training vs validation accuracy trends among the two datasets**
- ⚔️ **Impact of dataset complexity on network performance and accuracy**
- 💡 **Effect of regularisation using Dropout on overfitting**

---

## 🔬 Key Results

_(Results without regularisation)_

| Dataset   | Training Accuracy | Validation Accuracy |
| --------- | ----------------- | ------------------- |
| CIFAR-10  | ~97%              | ~71%                |
| CIFAR-100 | ~92%              | ~35%                |

_(Results after adding a 0.5 dropout layer)_

| Dataset   | Training Accuracy | Validation Accuracy |
| --------- | ----------------- | ------------------- |
| CIFAR-10  | ~90%              | ~72%                |
| CIFAR-100 | ~66%              | ~39%                |

---

## 🛠️ Tech Stack Used

- Python 3.11 + TensorFlow 2.18 Using CUDA
- Keras API
- CNN Layers
- Dropout Function
- Matplotlib for visualisation

## What I've Learned

- Architecture of CNNs and various layers involved.
- Overfitting can lead to false promises while observing the training metrics and disappoint in deployment phase.
- Regularisation of a model is important to prevent overfitting.
- Complexity of datasets has a big implication on a neural network's performance.
- Each dataset is unique and thus, there is no existence of a one-size-fits-all model.
- MatPlotLib is fun to use as a visualisation tool.

## Try This Out Yourself

- 1. Clone the repo

git clone <repo-url>
cd cifar-dataset-duel

- 2. Install Dependencies

pip install tensorflow matplotlib

- 3. Run The Script

python main.py

## What is included in the Repo

- 'c10-init.py' and 'c100-init.py' are tests scripts I used to first experiment with the datasets. Not really useful for the project but included nonetheless.

- 'mplttest.py' is another test script I used to understand matplotlib's 'pyplot' fucntion.

- 'main.py' is the _actual script_ to be run.

-'c10.png' and 'c100.png' will be the two plots created _after_ the main script is run. My plots have been uploaded.
