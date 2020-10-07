# disaster_response_pipeline_project
 

### Table of Contents

1. [Project Summary and File Description](#summary)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Acknowledgements](#summary)


    
## Project Summary and File Description<a name="summary"></a>

This Project is part of Data Science Nanodegree Program by Udacity. The project purpuse is to build a Natural Language Processing (NLP) model to categorize messages for disaster recovery. The model is trained on categorized messages to predict unseen messages.

The project is implimented in steps to achieve the final goal, following are the files description:
1. ETL Pipeline Preparation.ipynb: to preprocess the data files recieved to train the model, this file is then used to prepare the preprocessing modular code.
2. ML Pipeline Preparation.ipynb: the file where the model is built, trained, tested. this files contains three tests for three different model for optimization.
3. data folder: containd the datafiles, database and the process_data.py file, which the modular code for preprocessing.
4. models folder: contains the modular code for building the best model (train_classifier.py) and the model (classifier.pkl)
5. app folder: contains the web app files: run.py master.html go.html



## Installation <a name="installation"></a>
- This project is developed using Python 3

- Libraries used:

    `Pandas, scikit-learn, re, SQLalchemy for data processing and machine learning`

    `NLTK for Natural Language Process Libraries`

    `matplotlib for visualizations`
    
    `Pickle for saving and loading the model`
    
    `plotly and flask for web app and web viz`
    

- run the following commans in command line within the project directory: 
   a.Save preprocessed data: 
      python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   b.train the model:
      python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   c.run the following command within the directory where run.py is located:
      python run.py
   d.Go to http://0.0.0.0:3001/
    
## Results<a name="results"></a>

*AdaboostClassifier was used with the following best parameters based on grid search {'moclf__estimator__learning_rate': 1, 'moclf__estimator__n_estimators': 100}

*Link to GitHub repository: https://github.com/Lamasheg/disaster_response_pipeline_project.git


## Acknowledgements<a name="licensing"></a>
Credits go to udacity and Figureeight for providing the opportunity to practice our skills with real life data!
