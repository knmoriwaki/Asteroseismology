# Asteroseismology

Estimate the inclination angle from the spectrum

## Requirement

- Python 3.8+

Install the following libraries with `pip`.
- torch==1.12.0
- torchvision==0.13.0
- torchinfo
- torchhk
- torchbnn
- pandas
- seaborn
- sklearn
- scikit-learn
- tqdm
- matplotlib

## How to Run

- Put your training and validation data (e.g., 0000000.0.data, Combinations.txt) at `./training_data` and your test data at `./test_data`

- Create a normalization parameter file at `param` with two columns corresponding `xmin` and `dx`. 
The input and target data will be normalized as `d = ( d - xmin ) / dx`. 
The first `n_feature` rows corresponds to the input data and the last `len(output_id)` rows to the output. `dx` should be greater than 0.
If you use `nllloss` as the loss function, then the resulting pdf will cover the range \[xmin, xmin + dx\].

- Then run training and test code by 
```
./sub.sh
```

- You can change the parameters in run.sh. For the output_id, you can choose multiple ids (column number) by, e.g., `./sub.sh --output_id 1 2 3 --isTrain`.


- After the training, use plot.ipynb to check the model performance. 

Loss function:  
![loss](figures/loss.png) 

Validation output:  
![test](figures/val.png) 


You can check the model structure in the output file at `./tmp`


- Input shape: (batch_size, seq_length, n_feature)

- Output shape: (batch_size, output_dim)


Loss functions:

- NLLLoss: set output_dim > 1. This parameter is not used for the other loss functions.

## References


## Known Issues


