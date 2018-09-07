# em_boundary_prediction
Simple boundary prediction for neurodata with NN

needs the packages:
- https://github.com/inferno-pytorch/inferno (branch: super-dev)
- https://github.com/inferno-pytorch/neurofire (branch: mad)

also we need to add the PYTHONPATHs:

```
export PYTHONPATH="/path/to/inferno:$PYTHONPATH"
export PYTHONPATH="/path/to/neurofire:$PYTHONPATH"
```

For training, we need to run train.py with arguments:

1: project_folder
2: config_name (name of the folder the configs are in)
3(optional): config_folder the folder which the configs are in, needed for starting with sBatch default="./configs/" 
4(optional): max_train_iters number of training iterations, default=100000
