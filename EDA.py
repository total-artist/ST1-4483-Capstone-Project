"""
The First Step in software development is the Exploratory Data Analysis
What is EDA?
Exploratory DAta Analysis is a critical process of performing investigations on data to discover patterns,
spot anomalies and check assumptions with the help of summarized stats and graphical representations [2].
"""

#Here we import required packages for Exploratory Data Analysis
#The "import os" imports the os module which provides a way to interact with the operating system, thus allowing us to read/write/handle directories.
import os

import matplotlib
#The "import pandas" command import the pandas library, pandas is used in data manipulation in Python
import pandas as pd

#The "import numpy" command imports the numpy library, which is used in numeric computation.
import numpy as np

#The "import matoplotlib.pylot" command imports matplotlib.pyplot library  for plotting and creating visualizations Python.
import matplotlib.pyplot as plt
import pip

#The "import seabor" command is used to import seaborn which is a library also for data visualization.
import seaborn as sns

#The "missingno" command imports the missingno library, which is used for visualizing missing data.
import missingno as msno

#The "import plotly.graph_objects" command imports the import plotly.graph_objects module, this module is used in creating interactive plots in Python
import plotly.graph_objects as go

#The "import plotly.express" command imports the plotly.express module which is used for creating interactive visualizations.
import plotly.express as px
#matplotlib inline
import warnings
warnings.filterwarnings('ignore')

#Here we import and read the dataset (train.csv) for the Titanic topic
#this data set will be used in training the algorithm.
df = pd.read_csv("train.csv")

#Here we are instructing the dataset to show us the first 5 rows.
df.head() #display the first 5 rows
print(df.head())


#Here we are instructing the dataset to show us the last 5 rows.
df.tail() #display the last 5 rows
print("n",df.tail())

#The shape method is applied to the dataframe to show the number of rows and columns
#rows are the attributes and the columns are the samples.
df.shape
print("\n","Rows by Columns",df.shape)

#The columns attribute is used to retrieve the names of the attributes/columns.
df.columns
print("\n","Index of the columns:", "\n", df.columns)

#The nunique() method is used to count the unique values for each column.
df.nunique()
print("\n", "Here we count the number of unique values of the dataset:", "\n", df.nunique())

#The info() method prints out complete information about the dataframe.
print("\n", "Here we print out info about the dataset:")
df.info()


#Here we will be displaying data related to attributes in their own bar graphs
#The variable "fig" is used to create a new figure object with the plt.figure() function. the figure size is 20 and 19
fig = plt.figure(figsize = (15,9))

#The "ax" variable creates an instance of the current axes of figure using the get current axes (gca()) function.
ax=fig.gca()

#This line is used to create a histrogram of each column in the train.csv dataframe using the hist() function. the bin size is 15
#What is a histogram? Histograms are used to provide a visual interpretation of numerical data.
df.hist(ax=ax,bins =30)

#This line shows the histograms
plt.show()

#The lecture on the 6th April 2023 said we can install this code and run it, so I will be doing it here.
#!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
#Finished the installation.

#I will be using this pandas_profiling library and the Profile Report code to generate an interactive HTML report using the
#pandas_profiling library to perform EDA on the dataframe.
#Here we import ProfileReport class from the pandas_profiling library
from pandas_profiling import ProfileReport

#Here I use the variable "profile" to create a ProfileReport object from the dataframe.
profile = ProfileReport(df,title="Titanic Survival EDA", html={'style':{'full_width':True}})

#The profile.to_notebook_iframe function displays the report as an interactive HTML page.
profile.to_notebook_iframe()

#The profile.to_file() function saves the HTML page
profile.to_file("titanic_report.html")

#The import webbrowser command imports the webrowser module and the web webbrowser.open_new_tab() function allows us to
#open the titanic_report.html
import webbrowser
webbrowser.open_new_tab("titanic_report.html")

