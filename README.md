Collecting workspace informationBelow is a suggested content for your README file:

```markdown
# Human Activity Recognition (HAR) System

This project implements a Human Activity Recognition system that leverages smartphone sensor data to classify human activities. The system is built incrementally over three phases:

- **Phase 1: Binary Classification** – Distinguish "Walking" from "Not Walking".  
- **Phase 2: Ternary Classification** – Recognize three activities (e.g., Walking, Sitting, Standing).  
- **Phase 3: Multi-Class Classification** – Classify all six activities: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING.

## Data Sources

- **Dataset:** The UCI HAR Dataset is located in the [`data/UCI HAR Dataset`](data/UCI%20HAR%20Dataset) folder.
  - For details on features, refer to [`data/UCI HAR Dataset/features.txt`](data/UCI%20HAR%20Dataset/features.txt).
  - For activity labels, see [`data/UCI HAR Dataset/activity_labels.txt`](data/UCI%20HAR%20Dataset/activity_labels.txt).
  - Dataset description is provided in [`data/UCI HAR Dataset/README.txt`](data/UCI%20HAR%20Dataset/README.txt).

## Project Structure

- **data/** – Contains the UCI HAR Dataset files.
- **outputs/** – Stores metrics and saved models for each classification phase.
- **src/** – Includes Jupyter notebooks with code for each phase:
  - [`har_binary_classification.ipynb`](src/har_binary_classification.ipynb)
  - [`har_ternary_classification.ipynb`](src/har_ternary_classification.ipynb)
  - [`har_multiclass_classification.ipynb`](src/har_multiclass_classification.ipynb)
- **requirements.txt** – Lists the project dependencies.
- **.gitignore** – Specifies files and directories to be excluded from version control.

## Environment Setup

1. **Python Version:** Python 3.x  
2. **Dependencies:**  
   Install the required libraries using:
   ```sh
   pip install -r requirements.txt
   ```
   Main libraries include:
   - Pandas, NumPy, SciPy for data handling and signal processing  
   - Matplotlib, Seaborn, and Plotly for visualization  
   - TensorFlow/Keras (or PyTorch) for model building

## Execution Instructions

1. **Prepare the Dataset:**  
   - Download and unzip the UCI HAR Dataset into the data folder.
   
2. **Run the Notebooks:**  
   - Open the desired phase notebook in Visual Studio Code or Jupyter Notebook:
     - Binary classification: har_binary_classification.ipynb  
     - Ternary classification: har_ternary_classification.ipynb  
     - Multi-class classification: har_multiclass_classification.ipynb
   - Execute all cells to preprocess the data, train the models, and evaluate performance.

3. **Review Outputs:**  
   - Model files and evaluation metrics are stored in the outputs directory.

## Project Phases Overview

### Phase 1: Binary Classification
- **Goal:** Distinguish Walking vs. Not Walking.
- **Highlights:**  
  - Filtering the dataset to isolate walking samples.
  - Normalization and segmentation of sensor data.
  - Training a CNN model with binary cross-entropy loss.
- **Outputs:**  
  - Metrics: har_binary_classifier_metrics.json  
  - Model: har_binary_classifier.h5

### Phase 2: Ternary Classification
- **Goal:** Recognize three activities (e.g., Walking, Sitting, Standing).
- **Highlights:**  
  - Expanding the data subset for three classes.
  - Preprocessing and segmentation similar to Phase 1.
  - Building a CNN model with a softmax output.
- **Outputs:**  
  - Metrics: har_ternary_classifier_metrics.json  
  - Model: har_ternary_classifier.h5

### Phase 3: Multi-Class Classification
- **Goal:** Recognize all six activities.
- **Highlights:**  
  - Using the full dataset for comprehensive training.
  - Implementing a hybrid CNN-LSTM model.
  - Evaluating using per-class and overall performance metrics.
- **Outputs:**  
  - Metrics: Saved in the outputs directory (e.g., `har_multiclass_classifier_metrics.json`)
  - Model: Saved as `har_multiclass_classifier.h5`

## License

This project is intended for educational and research purposes. Please consult the dataset's README ([`data/UCI HAR Dataset/README.txt`](data/UCI%20HAR%20Dataset/README.txt)) for licensing details.

## Acknowledgements

Thanks to the contributors of the UCI HAR Dataset and the research community for providing the foundation for this project.
```