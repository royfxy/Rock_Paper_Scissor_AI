# ✊✌️✋ Rock_Paper_Scissor_AI
A computer-human Rock Paper Scissor game

# Data
To capture the data, use the following code:
```python
python main.py --run data_cap --gesture <rock|paper|scissor>
```

# Train
To train the gesture prediction model, use the following code:
```python
python main.py --run train [--model_name <name_of_model>]
```
Using `--k_fold` will run k-fold cross validation on the model.
```python
python main.py --run train [--model_name <name_of_model>] --k_fold <number_of_folds>
```

# Play the game
To play the game, use the following code:
```python
python main.py [--run game] [--model_name <name_of_model>]
```