# Welcome To Machine Learning For Predicting Device Failure!

This is a small repo that describes a machine learning model for predicting device failure in a remote location.

## Getting Started
Here are a few steps in order to run the pipeline for yourself.

### requirements
Please make sure you have the following installed on your computer:
- Python
- virtualenv

### clone repo

``` $ git clone https://github.com/JoshZastrow/failure-detection.git```

### create environment

Navigate to inside the project repo

```$ cd failure-detection```

Create a virtual environment named "venv"

```$ virtualenv venv```

Startup the environment

```$ source venv/bin/activate```

Install the required packages into this environemnt

```$ pip install -r requirements.txt```

### display pipeline

```$ python3 pipeline.py show```

### run training pipeline

```$ python3 pipeline.py run```

### viewing notebook
You can open up the html version of the notebook by clicking on index.html and opening it in your web browser.
Alternatively you can spin up a jupter server and open the ipnyb directly.

```$ jupyter lab```


### serving predictions
If you have your own dataset that you would like to create predictions for, set the mode argument of the pipeline to serve and provide the filepath to your dataset. The pipeline only works with .csv's.

```$ python3 pipeline.py run --mode serve --fpath ./data/<table_name>.csv```

For more information on the available arguments, run

```$ python3 pipeline.py run --help```


