# cv_pfe_project

This project implements a computer vision pipeline for object detection, tracking, depth estimation, scene analysis, and speech synthesis. It processes video inputs to detect and track objects, estimate their depth, analyze scenes, and generate audio descriptions.
Project Structure

Setup

Clone the repository:
git clone https://github.com/MOHAMEDmakhloufi/https://github.com/MOHAMEDmakhloufi/cv_pfe_project.git.git

cd cv_pfe_project


Create a conda environment.


Install dependencies:
pip install -r requirements.txt


Configure the project:

Edit configs/main_config.yaml to set paths and parameters.
Place model weights and input videos in data/external/.



Usage
Run the main pipeline:
python src/main.py

Dependencies

Python 3.8+
Libraries listed in requirements.txt (e.g., PyTorch, OpenCV, gTTS, YOLO, MiDaS)