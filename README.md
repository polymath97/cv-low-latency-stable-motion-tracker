# Beyblade Detection App

This project uses OpenCV, YOLO, and Streamlit to detect beyblades in video frames and analyze their movements.

## Project Structure

- `app.py`: The main application script.
- `model.pt`: The YOLO model file.
- `requirements.txt`: The file listing all the Python dependencies.

## Requirements

- Python 3.10 or higher
- pip (Python package installer)
- Docker (optional)

## Setup and Run

### Using Virtualenv

1. **Create and Activate Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

2. **Install Dependencies**

    ```bash
    Salin kode
    pip install -r requirements.txt

3. **Run the App**

    ```bash
    Salin kode
    streamlit run app.py

### Using Anaconda/Miniconda

1. **Create and Activate Conda Environment**

    ```bash
    Salin kode
    conda create --name beyblade-detection python=3.8
    conda activate beyblade-detection

2. **Install Dependencies**

    ```bash
    Salin kode
    pip install -r requirements.txt

3. **Run the App**

    ```bash
    Salin kode
    streamlit run app.py

### Using Docker (Optional)

1. **Run the App (Make sure docker desktop is running)**

    ```bash
    Salin kode
    sh run_app.sh

