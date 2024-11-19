# demo
The Python script developed for gaze behavior analysis implements multiple machine learning algorithms to process and analyze eye-tracking data recorded during a drone piloting task. The script takes as input a CSV file containing temporal gaze coordinates (t, gx, gy) and performs comprehensive analysis through various computational methods. It employs DBSCAN clustering to identify fixation patterns, K-means clustering to determine main areas of attention, and Isolation Forest for detecting anomalous gaze behaviors. Additionally, it utilizes Principal Component Analysis (PCA) for dimensionality reduction and pattern identification, and Support Vector Machine (SVM) for classifying different types of eye movements. The results are visualized through six distinct plots: Fixation Clusters showing areas of concentrated gaze, Attention Areas displaying the main regions of interest, a Velocity Profile illustrating the speed of eye movements over time, Detected Anomalies highlighting unusual gaze patterns, PCA Pattern Analysis revealing underlying viewing strategies, and Movement Classification categorizing different types of eye movements. This comprehensive analysis provides insights into the participant's visual attention patterns, scanning strategies, and overall gaze behavior during the drone operation task.

# Requirements

To work with this project, you will need the following tools:

1. **A code editor**: It is recommended to use **Visual Studio Code** for easier code editing and to take advantage of its extensions and debugging tools. You can download Visual Studio Code from its [official website](https://code.visualstudio.com/).

2. **Python 3**: This project is developed in **Python 3**. Make sure you have the latest version of Python 3 installed on your system. You can check if Python is already installed by running the following command in your terminal or command prompt:

  ```bash
  python --version
  ```
  If you need to install Python, you can download it from the [official Python website](https://www.python.org/).

3. **Recommended Extensions for Visual Studio Code**:

  - **Python**: For syntax highlighting, debugging, and running Python scripts.
  - **Pylance**: For improved autocompletion and error detection.

4. **Install the Required Packages**: To install the required dependencies for this project, use the following command to install the necessary Python packages:
  ```bash
  pip3 install numpy scikit-learn pandas matplotlib
  ```
  **numpy**: A package for numerical computing in Python, providing support for arrays and matrices, and mathematical functions to operate on them.
  
  **scikit-learn**: A machine learning library that includes tools for classification, regression, clustering, and more.
  
  **pandas**: A powerful data manipulation and analysis library that provides data structures like DataFrame to handle structured data.
  
  **matplotlib**: A plotting library for creating static, animated, and interactive visualizations in Python.
