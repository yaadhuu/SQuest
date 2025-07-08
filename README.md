# SQuest: Smart Summarizer & Question Answerer



SQuest (Smart Quest) is an intelligent command-line tool designed for efficient text summarization and interactive question-answering. It harnesses the power of state-of-the-art Natural Language Processing (NLP) models from Hugging Face Transformers. Users can quickly condense large texts into customizable summaries and then query those summaries for specific information, making information extraction intuitive and fast.

Whether you're looking to grasp the core ideas of a lengthy article or get quick answers from a summarized document, SQuest is built to streamline your workflow.

## ✨ Features

* **Intelligent Text Summarization:** Condenses long texts into concise summaries using the powerful `facebook/bart-large-cnn` model, specifically fine-tuned for summarization.
* **Customizable Summary Lengths:** Choose between 'short', 'medium', or 'long' summaries to fit your needs, with adjustable control over output length.
* **Interactive Question Answering:** Ask follow-up questions directly about the generated summary using `deepset/roberta-base-squad2`, a robust model for extractive QA capable of identifying when an answer isn't present.
* **GPU Accelerated (CUDA Support):** Designed for optimal performance on NVIDIA GPUs using CUDA, significantly speeding up model inference. Automatically falls back to CPU if no compatible GPU is detected.
* **User-Friendly Interface:** Simple command-line prompts guide you through text input, summary length selection, and interactive question-asking.
* **Built with LangChain:** Leverages the LangChain framework for seamless integration and chaining of NLP models, offering a modular and scalable architecture.



## 🛠️ Technologies Used

* **Python 3.8+**
* **Hugging Face Transformers:** For state-of-the-art NLP models.
* **LangChain:** For building robust NLP applications.
* **PyTorch:** The underlying deep learning framework.
* **CUDA:** For GPU acceleration on NVIDIA hardware.

## 📦 Setup and Installation

Follow these steps to get SQuest up and running on your local machine.

### Prerequisites

* **Python 3.8+** (Using `conda` or `venv` for environment management is highly recommended)
* **Git**

---

### **1. GPU Setup (CUDA - Recommended for Performance)**

If you have an NVIDIA GPU, configuring CUDA and CuDNN is vital for optimal performance. This allows SQuest to leverage your GPU for significantly faster processing.

* **Check CUDA Compatibility:**
    * Verify your NVIDIA GPU's CUDA compute capability. You can find this on NVIDIA's developer website.
    * Based on your GPU, determine the compatible CUDA Toolkit version.

* **Install NVIDIA GPU Driver:**
    * Ensure you have the latest stable NVIDIA GPU drivers installed for your operating system. Download them from the [NVIDIA Driver Downloads](https://www.nvidia.com/drivers) page.

* **Install CUDA Toolkit:**
    * Download and install the appropriate CUDA Toolkit version from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive). **Select the version that matches your GPU driver and PyTorch's compatibility matrix.**
    * **Crucial Step:** Follow NVIDIA's official installation instructions precisely. This typically involves correctly setting environment variables like `PATH` and `CUDA_HOME`.

### **2. Project Setup**

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR_USERNAME/YOUR_REPO].git
    cd SQuest
    ```
    *Replace `[YOUR_USERNAME/YOUR_REPO]` with the actual URL of your GitHub repository.*

2.  **Create a Virtual Environment (Highly Recommended):**
    This isolates your project's dependencies from your system's Python packages.

    Using `conda` (if you have Anaconda/Miniconda installed):
    ```bash
    conda create -n squest_env python=3.9
    conda activate squest_env
    ```
    Or using `venv`:
    ```bash
    python -m venv squest_env
    # On Windows
    .\squest_env\Scripts\activate
    # On macOS/Linux
    source squest_env/bin/activate
    ```

3.  **Install Dependencies:**

    **First, install PyTorch with CUDA support.** Visit the [PyTorch website](https://pytorch.org/get-started/locally/) to get the precise command for your operating system, package manager (pip), and **CUDA version**. For example, if you installed CUDA 11.8:

    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
    *(Adjust `cu118` to match your CUDA Toolkit version, e.g., `cu121` for CUDA 12.1).*

    **Next, install the remaining libraries:**

    ```bash
    pip install transformers langchain_huggingface
    ```

    *Alternatively, you can create a `requirements.txt` file (recommended practice for better dependency management) with the following content:*

    **`requirements.txt` content:**
    ```
    transformers
    langchain_huggingface
    # torch is installed separately to ensure CUDA compatibility
    ```
    *Then, you would run:*
    ```bash
    # 1. Install specific torch+CUDA version (e.g., for CUDA 11.8)
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    # 2. Install the rest from requirements.txt
    pip install -r requirements.txt
    ```

    *SQuest automatically detects if a GPU (CUDA) is available and utilizes it for faster processing. If not, it gracefully falls back to using the CPU.*


