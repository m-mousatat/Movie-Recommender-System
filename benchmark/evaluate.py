
import csv
import pandas as pd
import numpy as np


import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise import Dataset
from surprise.model_selection import cross_validate, train_test_split , GridSearchCV
from surprise import accuracy
from surprise import SVD
from surprise import dump



def evaluation1(number):
  # Specify the CSV file path
  file_path = f"evaluate/user_movie_prediction_matrix{number}.csv"

  # Initialize an empty list to store the loaded data
  user_movie_prediction_matrix = []

  # Open the file in read mode
  with open(file_path, mode='r', newline='') as file:
      reader = csv.reader(file)

      # Iterate through each row in the CSV file
      for row in reader:
          # Convert each element in the row to the appropriate data type (e.g., int or float)
          # Append the row to the loaded_data list
          loaded_row = [float(cell) for cell in row]
          user_movie_prediction_matrix.append(loaded_row)
  
  path_to_datasets= "data/raw/ml-100k/"
  # Load ratings data
  ratings_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
  ratings_test = pd.read_csv(f'{path_to_datasets}u{number}.test', sep='\t', names=ratings_cols, encoding='latin-1')
  true_result=[]
  predicted_result=[]
  count =0
  for index, row in ratings_test.iterrows():
    true_result.append(row['rating'])
    prediction = user_movie_prediction_matrix[row['user_id']][row['movie_id']]
    if prediction == 0:
      count +=1
      predicted_result.append(3)
    else:
      predicted_result.append(prediction)
  # Convert the lists to NumPy arrays
  true_result = np.array(true_result)
  predicted_result = np.array(predicted_result)

  # Calculate the Mean Squared Error (MSE)
  mse = np.mean((true_result - predicted_result) ** 2)
  print(f"Mean Squared Error: {mse} of dataset {number}")
  return mse
def evaluation2(dataset_number):
  path_to_datasets= "data/raw/ml-100k/"
  # Load user data
  users_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
  users = pd.read_csv(f'{path_to_datasets}u.user', sep='|', names=users_cols, encoding='latin-1')

  # Load ratings data
  ratings_cols = ['user_id', 'movie_id' ,'rating', 'unix_timestamp']
  ratings = pd.read_csv(f'{path_to_datasets}u{dataset_number}.test', sep='\t', names=ratings_cols, encoding='latin-1')

  # Load movies data
  movies_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL'] + ['genre_' + str(i) for i in range(19)]
  movies = pd.read_csv(f'{path_to_datasets}u.item', sep='|', names=movies_cols, encoding='latin-1', usecols=range(24))

  df = ratings.merge(movies, left_on='movie_id', right_on='movie_id', how='left')

  df = df[['user_id', 'title', 'rating']]

  
  reader = Reader(rating_scale=(1,5))
  data = Dataset.load_from_df(df, reader)
  train, test = train_test_split(data, test_size=0.99)
  model_filename = f"models/surprise_svd_model{dataset_number}"

  loaded_model = dump.load(model_filename)[1]
  predictions = loaded_model.test(test)
  RMSE = accuracy.rmse(predictions)
  print(f"of dataset {dataset_number}")
  return RMSE
if __name__ == "__main__":
  arguments = 2
  
  print("############## Method 1 Evaluation ##############")
  average1 = 0
  for i in range(1,arguments+1):
    average1+=evaluation1(i)
  average1 = average1 / arguments
  print(f"Average Mean Squered Error: {average1} of all datasets")
  print("\n\n")
  print("############## Method 2 Evaluation ##############")
  average2 = 0
  for i in range(1,6):
    average2+=evaluation2(i)
  average2 = average2 / 5
  print(f"Average Mean Squered Error: {average2} of all datasets")

