Use python 3.7 to practice Tensorflow 2.x.
===  
Dependence:
-------
-Python 3.7.10  
-tensorflow 2.1.0  
-tensorflow-gpu 2.1.0  
-numpy 1.19.5

You can follow as the command to create the conda virtual environment to practice TF 2.x.
-------
$: conda create --name tf2_gpu python=3.7  
$: conda activate tf2_gpu  
$: conda install tensorflow-gpu==2.1.0  
$: pip install numpy==1.19.5  
$: conda install -c conda-forge tensorflow-hub  


In "08.Depth_example_on_Functional_API.py" file, we msut to dwonload a [Multi_Digit_Mnist](https://www.kaggle.com/dataset/eb9594e5b728b2eb74ff8d5e57a9b74634330bfa79d9195d6ebdc7745b9802c3) dataset to practice how to use functional API method with two or more outputs.
