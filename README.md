# Acceleration-Based User Authentication using Neural Networks

## Overview
This repository contains the MATLAB code and coursework report for our final year AI and Machine Learning project at the University of Plymouth. The project focuses on acceleration-based user authentication using machine learning techniques, particularly a Feedforward Neural Network (FFMLP).

## Project Details
- **Module Code**: PUSL3123
- **Coursework Title**: AI & ML Final Coursework Report
- **University**: University of Plymouth
- **Deadline**: December 15, 2024
- **Group**: 83

## Team Members
- **Osadi Kiriella** (10899590)
- **Chathuruni De Silva** (10899700)
- **Khashanie Barua** (10899167)
- **Athurugirige M Amasha** (10899282)
- **Kavithma S Samarawickrama** (10899192)

## Project Description
This project investigates the viability of using acceleration-based features for user authentication. We analyze intra-user and inter-user variances and train a Feedforward Neural Network (FFMLP) to classify users based on motion data. Feature engineering techniques like Principal Component Analysis (PCA) and variance-based selection were used to optimize performance.

### Key Components
1. **Variance Analysis**: Evaluates user-specific feature patterns.
2. **Neural Network Model**: Implements FFMLP using MATLAB.
3. **Optimization Techniques**: Includes PCA and classifier comparisons (Random Forest, SVM, Naive Bayes, etc.).
4. **Performance Evaluation**: Uses accuracy, precision, recall, F1-score, and confusion matrices.

## Results
- **FFMLP Accuracy**: 92.59% (testing phase)
- **Best Classifiers**: Random Forest & Naive Bayes (98.15% accuracy)
- **Optimized Feature Selection**: PCA improved efficiency while maintaining predictive power.

## Repository Structure
```
├── MATLAB_Code/         # Contains MATLAB scripts for training and evaluation
├── Report/              # Coursework report in PDF format
├── README.md            # Project documentation
```

## Installation & Usage
### Prerequisites
- MATLAB (with Neural Network Toolbox)
- Required dataset (acceleration-based motion data from 10 users)

### Running the Model
1. Clone the repository:
   ```bash
   git clone https://github.com/kavithmasamadinie/AI-ML-Course_work.git
   ```
2. Open MATLAB and navigate to the `MATLAB_Code/` directory.
3. Run the main script:
   ```matlab
   run('main_script.m')
   ```
4. Review the output metrics and model performance.

## References
This project is based on research in biometric authentication, feature extraction, and deep learning for motion-based user verification. See the report for full references.

## Contributors
Each member contributed to different tasks, including data preprocessing, model training, report writing, and optimization techniques.

## License
This project is for academic purposes only and follows the university's academic integrity policies.

## Acknowledgments
We thank Dr. Neamah AI-Naffakh for guidance and feedback throughout this coursework.

For further details, please refer to the [coursework report](Report/Group_83_PUSL3123_Final_Report.pdf).
