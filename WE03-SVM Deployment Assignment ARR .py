import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import pickle

# Uncomment the following snippet of code to debug problems with finding the .pkl file path
# This snippet of code will exit the program and print the current working directory.
#import os
#exit(os.getcwd())

ownerprediction_model = pickle.load(open('svm_lin_model_example01.pkl', "rb"))

print("\n*****************************************************")
print("*Predicting the Ownership of RidingMowers*")
print("*****************************************************\n")
Income = float(input("Enter the Income: "))
Lot_Size = float(input("Enter the Lot_size: "))
df = pd.DataFrame({'Income': [Income],'Lot_Size': [Lot_Size]})
result = ownerprediction_model.predict(df)
probability = ownerprediction_model.predict_proba(df)
treatment = ('NoOwner', 'owner')
print(f"\nAiswarya's prediction model indicates probability of ownership at {probability[0][1]:.4f}, therefore it's indicated that the person should be the {treatment[result[0]]} of RidingMowers.\n")
