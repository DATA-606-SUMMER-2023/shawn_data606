# Src Folder

This folder is unfortunately rather unorganized, so the files are listed out here.

## Streamlit App

1_Data.py and the pages folder make up the streamlit app.

You can access the streamlit app by running `streamlit run .\src\1_Data.py` in you venv on the parent folder. The Streamlit app represents most of the work and is the most functional script. The streamlit app is capable of Loading data, EDA, training/testing, and visualizing model prediction.

However, note that you will need to provide the path to the _subjects_small_ folder within the data folder since the entire dataset was too larget o inclde in the github. Metadata and EDA has been included for the entire dataset, which can be viewed in the _Exploration_ section of the app after choosing _subjects_ as the subject path in the first section of the app.

## mp.py

A messy script used to process the raw asf/amc files using multiprocessing. It's a script meant to be run on its own, but should not be run now. I left it just in case.

## parsing.py, plotting.py, posing.py, modeling.py

All of these scripts are modules used in notebooks and the app.
- Parsing is for parsing asf/amc files as well
- Plotting is for visualizing prediction in plotly
- Posing is for applying frame data to the rest pose of the subject (each frame is a delta applied to a skeleton at rest)
- Modeling contains sklearn models for Linear, Polynomial, and Random Forest regressors

## preliminary_eda.ipynb, eda.ipynb
Notebooks showing both preliminary EDA and EDA. Due to time/size constraints, much of the EDA in the notebooks applies to only the first 3 subjects of the dataset.

## workflow.ipynb, ml.ipynb
The notebooks are not very well commented but demonstrate some of the process for training the data.

Since the streamlit app does all this, I do not recommend looking deeply into these notebooks, they were partially used as chicken scratch, but I kept them in the projects just in case.

