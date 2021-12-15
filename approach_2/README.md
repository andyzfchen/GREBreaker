To train the model with default training set ('scs_traing.csv'), learning rate equal to 2e-5, and save it as 'model.torch', run the following sample command:

`python run.py --train --lr 2e-5 --model 'model.torch'`

To evaluate the model saved as 'model.torch' with the test set 'sat_testing.csv', run the following sample command:

`python run.py --test --model 'model.torch' --test_data 'sat_testing.csv'`
