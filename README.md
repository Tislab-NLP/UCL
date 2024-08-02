# UCL: Mitigating Implicit Bias in Toxic Speech Detection via Unbiased Contrastive Learning

This repository contains the code for the UCL (Unbiased and Fair Toxic Speech Detection) project, which is intended for submission to the ICDE 2025 conference.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [System Requirements](#system-requirements)
- [License](#license)

## Introduction
UCL is a state-of-the-art toxic speech detection system that addresses both explicit and implicit biases in text data. The system aims to improve the accuracy and fairness of toxic speech detection by employing advanced debiasing techniques.

## Features
- **Bias Mitigation**: Reduces both explicit and implicit biases in toxic speech detection.
- **High Accuracy**: Utilizes cutting-edge algorithms to achieve high accuracy in identifying toxic content.
- **Fairness**: Ensures equitable treatment of different demographic groups on social platforms.
- **Scalability**: Designed to handle large-scale datasets and real-time detection.

## Installation
To install the UCL project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Tislab-NLP/UCL.git
    cd UCL
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
**Train the model**:

    ```bash
    python main.py
    ```
    
## System Requirements
- Python 3.8 or higher
- pip
- Additional requirements are listed in `requirements.txt`

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
