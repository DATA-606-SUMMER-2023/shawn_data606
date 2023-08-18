# UMBC Data Science Capstone Project

__Regression Modeling Upper Body Pose Estimation for Virtual Reality HMD and Controllers__

- **Author:** Shawn Oppermann
- **Term:** Summer 2023

## Folder Descriptions

### data

Contains data files for the CMU motion capture dataset. Only EDA and metadata have been included in the full _subjects_ folder. However, a small subset has been included under _subjects\_small_ including the first 3 subjects found on the website.

### docs

Contains EDA description, project proposal, report, and presentation. Also contains the images used in documents/presentations.

### models

Contains the models trained on the dataset. Only the models with a small enough filesize for git were included.

### src

Contains all code, notebooks, and the streamlit app used to both train and present models. Refer to the video for a full tour on how to use the tool.

In order to run the streamlit app, set up a python virtual environment in this main directory (not src), install packages using requirements.txt, and run `streamlit run .\src\1_Data.py `

