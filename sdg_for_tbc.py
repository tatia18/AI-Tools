#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 00:59:50 2024

@author: tatiatsiklauri
"""

print("\n\nGenerate Synthetic Data!!!", "\n\n")
file_name = input("Enter the name of the csv file: \n")


path_of_a_file = "/Users/tatiatsiklauri/Desktop/NBG/SDG/databases/" + file_name


!pip install sklearn

# Open Data
import pandas as pd
real_data = pd.read_csv(path_of_a_file)

print("Your data is ready.\nThis is head of your data:\n")
pd.set_option('display.max_columns', None)
print(real_data.head())

# Find Categorical Variables
categories = real_data.select_dtypes(include=[object])
discrete_cols = list(categories)
for cat in real_data.columns:
    # print(len(real_data[cat].unique()))
    if cat not in discrete_cols and len(real_data[cat].unique()) < 10:
        discrete_cols.append(cat)
        # print(discrete_cols)
if discrete_cols == []:
    print("\nThere are no categorical variables.")
else:
    print(f"\nThese are categorical variables \n{discrete_cols}")
    response = input("\nContinue?: (y/n)  ")
    if response in ["no", "n"]:
        sys.exit()


# Customization
response_drop = input("Do you want to drop categoricals? (y/n)\n")
response_split = input("Do you want to split the data into train-test? (y/n)\n")
response_nas = input("Do you want to drop NAs? (y/n)\n")
#Method
method = input("Which method do you want to use for data generation? (Copula or CTGAN)\n")
method = method.lower()
if method == "ctgan":
    epoch_user = input("Enter the number of epochs: ")
    epoch_user = int(epoch_user)
#Evaluation
response_evaluation = input("See the evaluation?: (y/n):  ")

if discrete_cols != []: 
    # Drop Categoricals
    if response_drop in ["yes", "y"]:
        real_train = real_train.drop(columns = discrete_cols)
        print("Categorical varaibles dropped.\n")

# Divide into Train-Test
if response_split in ["yes", "y"]:
    from sklearn.model_selection import train_test_split
    real_train, real_test = train_test_split(real_data, test_size=0.3)
    print("Data is splitted into train-test.\n")
else:
    real_train=real_data
    print("Entire data is used as train.\n")


# Drop NAs
# response = input("\nDo you want to drop NAs? (y/n):  ")
if response_nas in ["yes", "y"]:
    real_train = real_train.dropna(axis=0)
    print("NAs dropped.\n")


#save
real_train.to_csv(f"/Users/tatiatsiklauri/Desktop/result/{file_name[:-4]}_real_train_{method}", index=False)
print(f"\ntrain data is saved as '{file_name[:-4]}_real_train_{method}'")

#Metadada
print("##############################################")
print("Here is your Metadata:")
intermediate_metadata = dict()
for col in list(real_train.columns):
    if col not in discrete_cols:
        intermediate_metadata[col] = {"sdtype": "numerical"}
    elif col in discrete_cols:
        intermediate_metadata[col] = {"sdtype": "categorical"}
metadata = dict()
metadata["columns"] = intermediate_metadata
print(metadata)
print("##############################################")

####################
print(f"\n\nNow generate data with {method.upper()}")
# import time
# time.sleep(3) # Sleep for 3 seconds

if method == "copula":
    # Copula
    # %pip install copulas
    from copulas.multivariate import GaussianMultivariate
    model = GaussianMultivariate()
    print("\n\nFitting the data with copula model...")
    # time.sleep(3) # Sleep for 3 seconds
    model.fit(real_train)
    print("\n\nModel Fitted!!!")
    
    # Create synthetic Data
    print("\n\nSampling synthetic data...")
    # time.sleep(3) # Sleep for 3 seconds
    synthetic_data = model.sample(len(real_train))
    print("\n\nSYNTHETIC DATA IS READY!!!\n")
elif method == "ctgan":
    # CTGAN
    # %pip install ctgan
    from ctgan import CTGAN
    ctgan = CTGAN(epochs=epoch_user, verbose=True)
    print("\n\nFiltting the data with GAN model...")
    ctgan.fit(real_train, discrete_cols)
    print("\n\nModel Fitted!!!")

    # Create synthetic Data
    print("\n\nSampling synthetic data...")
    # time.sleep(3) # Sleep for 3 seconds
    synthetic_data = ctgan.sample(len(real_train))
    print("\n\nSYNTHETIC DATA IS READY!!!\n")

#save
synthetic_data.to_csv(f"/Users/tatiatsiklauri/Desktop/result/{file_name[:-4]}_synthetic_data_{method}", index=False)
print(f"synthetic data is saved as '{file_name[:-4]}_synthetic_data_{method}'")

if response_evaluation in ["no", "n"]:
    sys.exit()



# Plot Graphs
# %pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

def plot_num(x):
    sns.kdeplot(real_train[x],shade = True, label = "real")
    sns.kdeplot(synthetic_data[x], shade = True, label = "synthetic")
    plt.legend()

def plot_cat(x):
    sns.histplot(real_train[x], label = "real")
    sns.histplot(synthetic_data[x], label = "synthetic")
    plt.legend()
    

def plot_cum(x):
    sns.ecdfplot(real_train[x], label = "real")
    sns.ecdfplot(synthetic_data[x], label = "synthetic")
    plt.legend()


for var in real_train.columns:
    if var in discrete_cols:
        plt.figure(list(real_train.columns).index(var))
        plot_cat(var)
    elif var not in discrete_cols:
        plt.figure(list(real_train.columns).index(var))
        plot_num(var)
        # print(var)

# Correlation Matrix
import seaborn as sn
plt.figure(100)
s = sn.heatmap(real_train.corr(), annot=True)
s.set(xlabel = "real data")
plt.show()
plt.figure(101)
s = sn.heatmap(synthetic_data.corr(), annot=True)
s.set(xlabel = "synthetic data")
plt.show()
print("Correlation Matrixes are ready!!!\n\n")



#################################
print("##############################################")
print("Running Quality Report...\n")
from sdmetrics.reports.single_table import QualityReport
report_q = QualityReport()
report_q.generate(real_train, synthetic_data, metadata)
table_column_shapes= report_q.get_details(property_name='Column Shapes')
table_column_pair_trends = report_q.get_details(property_name='Column Pair Trends')
print("\nTables of quality metrics are ready ('table_column_shapes' & 'table_columns_pair_trends'\n")
print("##############################################")

print("##############################################")
print("Running Diagnostic Report...\n")
from sdmetrics.reports.single_table import DiagnosticReport
report_d = DiagnosticReport()
report_d.generate(real_train, synthetic_data, metadata)
table_coverage =report_d.get_details(property_name='Coverage')
table_synthesis = report_d.get_details(property_name='Synthesis')
table_boundaries = report_d.get_details(property_name='Boundaries')
print("\nTables of diagnostic metrics are ready ('table_coverage' & 'table_synthesis' & 'table_boundaries')\n")
print("##############################################")



print("THE JOB IS DONE :*")





