# Fine-Tuning DenseNet-201 using FastAI

This repository demonstrates how to fine-tune a **DenseNet-201** deep learning model for image classification using the **FastAI** library. The core implementation is contained within the `finalModel.ipynb` Jupyter Notebook.

## 📖 Project Overview

The primary goal of this project is to showcase transfer learning by adapting a pre-trained DenseNet-201 model (originally trained on ImageNet) to a new dataset. FastAI is utilized to streamline the data loading, model creation, and training loop.

### Key Features
*   **Model Architecture**: DenseNet-201
*   **Framework**: FastAI (PyTorch backend)
*   **Technique**: Transfer Learning (Fine-Tuning)

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.6+
*   Jupyter Notebook or JupyterLab

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/arupa444/Fine-Tuning-Densenet201-using-FastAI.git
    cd Fine-Tuning-Densenet201-using-FastAI
    ```

2.  **Install dependencies**
    It is recommended to use a virtual environment. Install standard data science libraries and FastAI:
    ```bash
    pip install fastai jupyterlab
    ```
    *(Note: FastAI will automatically install the compatible version of PyTorch)*

## 💻 Usage

1.  Launch Jupyter Lab:
    ```bash
    jupyter lab
    ```
2.  Open `finalModel.ipynb`.
3.  Run the notebook cells sequentially to observe the data loading, model setup, and training process.

## 📊 Notebook Highlights (`finalModel.ipynb`)

The notebook covers the following critical steps:
*   **DataBlock API**: Flexible data loading and augmentation pipelines.
*   **Vision Learner**: Initializing `densenet201` with pretrained weights.
*   **lr_find()**: FastAI's utility to find the optimal learning rate.
*   **fine_tune()**: The one-cycle training policy for efficient transfer learning.
*   **Interpretation**: Using `ClassificationInterpretation` to view confusion matrices and top losses.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📝 License

This project is open-source.
