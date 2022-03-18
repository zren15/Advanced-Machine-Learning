import numpy as np
import pandas as pd
from scipy import sparse

def proc_col(col):
    """Encodes a pandas column with values between 0 and n-1.
 
    where n = number of unique values
    """
    uniq = col.unique()
    name2idx = {o:i for i,o in enumerate(uniq)}
    return name2idx, np.array([name2idx[x] for x in col]), len(uniq)

def encode_data(df):
    """Encodes rating data with continous user and movie ids using 
    the helpful fast.ai function from above.
    
    Arguments:
      train_csv: a csv file with columns user_id,movie_id,rating 
    
    Returns:
      df: a dataframe with the encode data
      num_users
      num_movies
      
    """
    ### BEGIN SOLUTION
    df['userId'] = proc_col(df['userId'])[1]
    df['movieId'] = proc_col(df['movieId'])[1]
    num_users = proc_col(df['userId'])[2]
    num_movies = proc_col(df['movieId'])[2]
    ### END SOLUTION
    return df, num_users, num_movies

def encode_new_data(df_val, df_train):
    """ Encodes df_val with the same encoding as df_train.
    Returns:
    df_val: dataframe with the same encoding as df_train
    """
    ### BEGIN SOLUTION
    user = proc_col(df_train['userId'])[0]
    movie = proc_col(df_train['movieId'])[0]
    dup_userid = set(df_train['userId'])&set(df_val['userId'])
    dup_movieid = set(df_train['movieId'])&set(df_val['movieId'])
    df_val = df_val.iloc[np.where(df_val['userId'].isin(list(dup_userid)))]
    df_val = df_val.iloc[np.where(df_val['movieId'].isin(list(dup_movieid)))]
    df_val['userId']= np.array([user[i] for i in df_val['userId']])
    df_val['movieId']= np.array([movie[i] for i in df_val['movieId']])
    ### END SOLUTION
    return df_val

def create_embedings(n, K):
    """ Create a numpy random matrix of shape n, K
    
    The random matrix should be initialized with uniform values in (0, 6/K)
    Arguments:
    
    Inputs:
    n: number of items/users
    K: number of factors in the embeding 
    
    Returns:
    emb: numpy array of shape (n, num_factors)
    """
    np.random.seed(3)
    emb = 6*np.random.random((n, K)) / K
    return emb


def df2matrix(df, nrows, ncols, column_name="rating"):
    """ Returns a sparse matrix constructed from a dataframe
    
    This code assumes the df has columns: MovieID,UserID,Rating
    """
    values = df[column_name].values
    ind_movie = df['movieId'].values
    ind_user = df['userId'].values
    return sparse.csc_matrix((values,(ind_user, ind_movie)),shape=(nrows, ncols))

def sparse_multiply(df, emb_user, emb_movie):
    """ This function returns U*V^T element wise multi by R as a sparse matrix.
    
    It avoids creating the dense matrix U*V^T
    """
    
    df["Prediction"] = np.sum(emb_user[df["userId"].values]*emb_movie[df["movieId"].values], axis=1)
    return df2matrix(df, emb_user.shape[0], emb_movie.shape[0], column_name="Prediction")

def cost(df, emb_user, emb_movie):
    """ Computes mean square error
    
    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]
    
    Arguments:
      df: dataframe with all data or a subset of the data
      emb_user: embedings for users
      emb_movie: embedings for movies
      
    Returns:
      error(float): this is the MSE
    """
    ### BEGIN SOLUTION
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0], column_name="rating")
    Y_hat = sparse_multiply(df,emb_user, emb_movie).toarray()
    N = Y.nnz
    Y = Y.toarray()
    error = np.power(Y-Y_hat, 2).sum()/N
    ### END SOLUTION
    return error

def finite_difference(df, emb_user, emb_movie, ind_u=None, ind_m=None, k=None):
    """ Computes finite difference on MSE(U, V).
    
    This function is used for testing the gradient function. 
    """
    e = 0.000000001
    c1 = cost(df, emb_user, emb_movie)
    K = emb_user.shape[1]
    x = np.zeros_like(emb_user)
    y = np.zeros_like(emb_movie)
    if ind_u is not None:
        x[ind_u][k] = e
    else:
        y[ind_m][k] = e
    c2 = cost(df, emb_user + x, emb_movie + y)
    return (c2 - c1)/e

def gradient(df, Y, emb_user, emb_movie):
    """ Computes the gradient.
    
    First compute prediction. Prediction for user i and movie j is
    emb_user[i]*emb_movie[j]
    
    Arguments:
      df: dataframe with all data or a subset of the data
      Y: sparse representation of df
      emb_user: embedings for users
      emb_movie: embedings for movies
      
    Returns:
      d_emb_user
      d_emb_movie
    """
    ### BEGIN SOLUTION
    Y_hat = sparse_multiply(df, emb_user, emb_movie)
    diff = Y - Y_hat
    d_emb_user = -2/Y.nnz * diff * emb_movie
    d_emb_movie = -2/Y.nnz * diff.transpose() * emb_user
    ### END SOLUTION
    return d_emb_user, d_emb_movie

# you can use a for loop to iterate through gradient descent
def gradient_descent(df, emb_user, emb_movie, iterations=100, learning_rate=0.01, df_val=None):
    """ Computes gradient descent with momentum (0.9) for a number of iterations.
    
    Prints training cost and validation cost (if df_val is not None) every 50 iterations.
    
    Returns:
    emb_user: the trained user embedding
    emb_movie: the trained movie embedding
    """
    Y = df2matrix(df, emb_user.shape[0], emb_movie.shape[0])
    ### BEGIN SOLUTION
    momentum = 0.9 
    gradient_user_0, gradient_movie_0 = 0,0 
    for i in range(iterations):   
        gradient_user, gradient_movie = gradient(df, Y, emb_user, emb_movie)
        gradient_user_0 = momentum * gradient_user_0 + (1 - momentum) * gradient_user
        gradient_movie_0 = momentum * gradient_movie_0 + (1 - momentum) * gradient_movie
        emb_user = emb_user - learning_rate * gradient_user_0
        emb_movie = emb_movie - learning_rate * gradient_movie_0
        if i%50 == 0:
            print("training cost:", cost(df, emb_user, emb_movie))
            if df_val != None:
                print("validation cost:", cost(df_val, emb_user, emb_movie))
    ### END SOLUTION
    return emb_user, emb_movie

