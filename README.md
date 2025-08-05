# Mushroom Classification Project

## Overview
This project implements a comprehensive machine learning analysis for mushroom classification to determine whether a mushroom is edible or poisonous. The analysis includes multiple classification algorithms, feature engineering, and performance evaluation.

## Dataset
The project uses the **Mushroom Classification Dataset** containing 8,124 samples with 23 features describing various characteristics of mushrooms:

### Features
- **class**: Target variable (edible 'e' or poisonous 'p')
- **cap-shape**: Shape of the cap
- **cap-surface**: Surface texture of the cap
- **cap-color**: Color of the cap
- **bruises**: Whether the mushroom bruises
- **odor**: Odor of the mushroom
- **gill-attachment**: How gills attach to the stalk
- **gill-spacing**: Spacing of the gills
- **gill-size**: Size of the gills
- **gill-color**: Color of the gills
- **stalk-shape**: Shape of the stalk
- **stalk-root**: Root of the stalk
- **stalk-surface-above-ring**: Surface above the ring
- **stalk-surface-below-ring**: Surface below the ring
- **stalk-color-above-ring**: Color above the ring
- **stalk-color-below-ring**: Color below the ring
- **veil-type**: Type of veil
- **veil-color**: Color of the veil
- **ring-number**: Number of rings
- **ring-type**: Type of ring
- **spore-print-color**: Color of spore print
- **population**: Population abundance
- **habitat**: Habitat where mushroom grows

### Dataset Statistics
- **Total samples**: 8,124
- **Edible mushrooms**: 4,208 (51.8%)
- **Poisonous mushrooms**: 3,916 (48.2%)
- **Features**: 23 (22 features + 1 target)
- **Missing values**: None

## Methodology

### Data Preprocessing
1. **Label Encoding**: Converted categorical variables to numerical format
2. **Data Standardization**: Applied StandardScaler for feature scaling
3. **Train-Test Split**: 80% training, 20% testing with random state for reproducibility

### Classification Algorithms
The project implements and compares 7 different classification algorithms:

1. **Decision Tree Classifier**
2. **K-Nearest Neighbors (KNN)**
3. **Random Forest Classifier**
4. **XGBoost**
5. **Naive Bayes (Gaussian)**
6. **Support Vector Classification (SVC)**
7. **Logistic Regression**

### Model Evaluation
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the ROC curve
- **Cross-Validation**: 5-fold cross-validation for robust evaluation
- **Confusion Matrix**: Visual representation of predictions vs actual

### Hyperparameter Optimization
Implemented GridSearchCV for:
- Random Forest parameters
- Support Vector Machine parameters
- Gradient Boosting parameters

## Results

### Model Performance Comparison

| Method | Accuracy | Precision | Recall | F1 | ROC AUC |
|--------|----------|-----------|--------|----|---------|
| Decision Tree | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| KNN | 0.9969 | 0.9948 | 0.9987 | 0.9968 | 0.9970 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Naive Bayes | 0.9157 | 0.9005 | 0.9250 | 0.9126 | 0.9161 |
| SVC | 0.9902 | 0.9974 | 0.9819 | 0.9896 | 0.9898 |
| Logistic Regression | 0.9520 | 0.9578 | 0.9405 | 0.9491 | 0.9515 |

### Key Findings
- **Best Performers**: Decision Tree, Random Forest, and XGBoost achieved perfect accuracy (100%)
- **Consistent Performance**: Most models achieved >95% accuracy
- **Feature Importance**: Analysis reveals which mushroom characteristics are most critical for classification
- **Robust Results**: Cross-validation confirms model stability

## Files Description

- `Mushroom Classification.ipynb`: Main Jupyter notebook containing the complete analysis
- `mushrooms.csv`: Dataset file with mushroom characteristics
- `README.md`: This documentation file

## Setup and Installation

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Running the Project
1. Clone this repository
2. Ensure all required packages are installed
3. Open `Mushroom Classification.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells to reproduce the analysis

### Required Python Packages
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

## Project Structure
```
├── Mushroom Classification.ipynb    # Main analysis notebook
├── mushrooms.csv                    # Dataset
├── README.md                       # Project documentation
└── Mushroom Classification.pdf     # PDF version of notebook
```

## Key Insights

1. **High Accuracy**: The dataset allows for very accurate classification, with multiple models achieving near-perfect performance
2. **Feature Engineering**: Label encoding and standardization improved model performance
3. **Model Diversity**: Different algorithms provide similar high performance, suggesting the problem is well-defined
4. **Practical Application**: This classification system could be valuable for mushroom identification in real-world scenarios

## Future Enhancements

- **Feature Selection**: Implement feature selection techniques to identify the most important characteristics
- **Deep Learning**: Explore neural network approaches
- **Real-time Classification**: Develop a web application for real-time mushroom classification
- **Additional Datasets**: Incorporate more mushroom species and characteristics

## Author
**Brian Nyakeri Barongo** - MSc Data Mining Student

## License
This project is for educational purposes as part of MSc Data Mining coursework.

---

*Note: This classification system is for educational purposes only. Always consult with experts before consuming any wild mushrooms.* 