from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
import warnings
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices

###########################################################################################################################
class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.
    
    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any], 
                 normalize_numeric: bool = False) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.
        normalize_numeric : bool, default=False
            Whether to normalize numeric columns after mapping.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column
        self.normalize_numeric = normalize_numeric
        self.column_means = {}
        self.column_stds = {}

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - computes statistics needed for normalization if enabled.
        """
        if self.normalize_numeric:
            # Store means and standard deviations for numeric columns
            numeric_cols = X.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                self.column_means[col] = X[col].mean()
                self.column_stds[col] = X[col].std()
                if self.column_stds[col] == 0:
                    self.column_stds[col] = 1  # Prevent division by zero
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.
        If normalize_numeric is True, also normalize numeric columns.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'
        warnings.filterwarnings('ignore', message='.*downcasting.*')

        # Check for mapping issues
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        
        # Apply normalization if enabled
        if self.normalize_numeric:
            for col in self.column_means:
                if col in X_.columns:
                    X_[col] = (X_[col] - self.column_means[col]) / self.column_stds[col]
        
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        """
        self.fit(X, y)
        result: pd.DataFrame = self.transform(X)
        return result
        
###########################################################################################################################
class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, X, y=None):
      return self

  def transform(self, X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    target_cols = [self.target_column] if isinstance(self.target_column, str) else self.target_column

    missing_cols = set(target_cols) - set(X.columns)
    assert not missing_cols, f"CustomOHETransformer.transform unknown column {missing_cols}"

    X_encoded = pd.get_dummies(
        X,
        columns=target_cols,
        dummy_na=self.dummy_na,
        drop_first=self.drop_first
    )
    
    # Convert boolean columns to float
    bool_cols = X_encoded.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X_encoded[col] = X_encoded[col].astype(float)
        
    return X_encoded

###########################################################################################################################
class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

    #your code below
    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
      assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
      print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
      return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
      assert isinstance(X, pd.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
      X_ = X.copy()
      missing_cols = set(self.column_list) - set(X.columns)

      if self.action == 'drop':
          if missing_cols:
              warnings.warn(f'{self.__class__.__name__}.transform dropping unknown columns: {missing_cols}')
          X_.drop(columns=self.column_list, errors='ignore', inplace=True)
      else:  # action == 'keep'
          assert not missing_cols, f'{self.__class__.__name__}.transform unknown columns to keep: {missing_cols}'
          X_ = X_[self.column_list]

      return X_

###########################################################################################################################
class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies 3-sigma clipping to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It clips values in the target column to be within three standard
    deviations from the mean.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply 3-sigma clipping on.

    Attributes
    ----------
    high_wall : Optional[float]
        The upper bound for clipping, computed as mean + 3 * standard deviation.
    low_wall : Optional[float]
        The lower bound for clipping, computed as mean - 3 * standard deviation.
    """
    def __init__(self, target_column: str):
        self.target_column = target_column
        self.high_wall = None
        self.low_wall = None

    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame), f'Expected DataFrame but got {type(X)} instead.'
        assert self.target_column in X.columns, f'Unknown column {self.target_column}'
        assert pd.api.types.is_numeric_dtype(X[self.target_column]), \
            f'Expected numeric type in column {self.target_column}'

        mean = X[self.target_column].mean()
        std = X[self.target_column].std()

        self.low_wall = mean - 3 * std
        self.high_wall = mean + 3 * std

        return self

    def transform(self, X: pd.DataFrame):
        assert self.low_wall is not None and self.high_wall is not None, "Sigma3Transformer.fit has not been called."
        X_copy = X.copy()
        X_copy[self.target_column] = X_copy[self.target_column].clip(self.low_wall, self.high_wall)
        return X_copy

###########################################################################################################################
class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that applies Tukey's fences (inner or outer) to a specified column in a pandas DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in a scikit-learn pipeline.
    It clips values in the target column based on Tukey's inner or outer fences.

    Parameters
    ----------
    target_column : Hashable
        The name of the column to apply Tukey's fences on.
    fence : Literal['inner', 'outer'], default='outer'
        Determines whether to use the inner fence (1.5 * IQR) or the outer fence (3.0 * IQR).

    Attributes
    ----------
    inner_low : Optional[float]
        The lower bound for clipping using the inner fence (Q1 - 1.5 * IQR).
    outer_low : Optional[float]
        The lower bound for clipping using the outer fence (Q1 - 3.0 * IQR).
    inner_high : Optional[float]
        The upper bound for clipping using the inner fence (Q3 + 1.5 * IQR).
    outer_high : Optional[float]
        The upper bound for clipping using the outer fence (Q3 + 3.0 * IQR).

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'values': [10, 15, 14, 20, 100, 5, 7]})
    >>> tukey_transformer = CustomTukeyTransformer(target_column='values', fence='inner')
    >>> transformed_df = tukey_transformer.fit_transform(df)
    >>> transformed_df
    """
    def __init__(self, target_column: str, fence: Literal['inner', 'outer'] = 'outer'):
        self.target_column = target_column
        self.fence = fence
        self.inner_low: Optional[float] = None
        self.outer_low: Optional[float] = None
        self.inner_high: Optional[float] = None
        self.outer_high: Optional[float] = None

    def fit(self, X: pd.DataFrame, y=None):
        assert isinstance(X, pd.DataFrame), f'Expected DataFrame but got {type(X)} instead.'
        assert self.target_column in X.columns, f'Unknown column {self.target_column}'
        assert pd.api.types.is_numeric_dtype(X[self.target_column]), \
            f'Expected numeric dtype in column {self.target_column}'

        q1 = X[self.target_column].quantile(0.25)
        q3 = X[self.target_column].quantile(0.75)
        iqr = q3 - q1

        self.inner_low = q1 - 1.5 * iqr
        self.inner_high = q3 + 1.5 * iqr
        self.outer_low = q1 - 3.0 * iqr
        self.outer_high = q3 + 3.0 * iqr

        return self

    def transform(self, X: pd.DataFrame):
        assert self.inner_low is not None, "TukeyTransformer.fit has not been called."

        X_copy = X.copy()
        if self.fence == 'inner':
            low = self.inner_low
            high = self.inner_high
        else:
            low = self.outer_low
            high = self.outer_high

        X_copy[self.target_column] = X_copy[self.target_column].clip(lower=low, upper=high)
        return X_copy
        
###########################################################################################################################
class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  """Applies robust scaling to a specified column in a pandas DataFrame.
    This transformer calculates the interquartile range (IQR) and median
    during the `fit` method and then uses these values to scale the
    target column in the `transform` method.

    Parameters
    ----------
    column : str
        The name of the column to be scaled.

    Attributes
    ----------
    target_column : str
        The name of the column to be scaled.
    iqr : float
        The interquartile range of the target column.
    med : float
        The median of the target column.
  """
  def __init__(self, column):
    self.target_column = column
    self.iqr = None
    self.med = None

  def fit(self, X, y=None):
    """
    Compute the interquartile range (IQR) and median of the target column.

    Parameters
    ----------
    X : pandas.DataFrame
        The input DataFrame.
        y : Ignored
        Not used, present here for API consistency by convention.

    Returns
    -------
    self : CustomRobustTransformer
        The fitted CustomRobustTransformer instance.
    """
    self.iqr = X[self.target_column].quantile(0.75) - X[self.target_column].quantile(0.25)
    self.med = X[self.target_column].median()
    return self

  def transform(self, X):
    """
    Apply robust scaling to the target column of the input DataFrame.

    Parameters
    ----------
    X : pandas.DataFrame
        The input DataFrame.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the target column scaled using robust scaling.
    """
    if self.iqr is None or self.med is None:
      raise NotFittedError("This CustomRobustTransformer instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")

    if self.target_column not in X.columns:  # Complete the if statement
        raise ValueError(f"Column '{self.target_column}' not found in input DataFrame.")

    X_robust = X.copy()
    X_robust[self.target_column] = (X[self.target_column] - self.med) / self.iqr
    return X_robust
      
###########################################################################################################################
class CustomKNNTransformer(BaseEstimator, TransformerMixin):
  """Imputes missing values using KNN.

  This transformer wraps the KNNImputer from scikit-learn and hard-codes
  add_indicator to be False. It also ensures that the input and output
  are pandas DataFrames.

  Parameters
  ----------
  n_neighbors : int, default=5
      Number of neighboring samples to use for imputation.
  weights : {'uniform', 'distance'}, default='uniform'
      Weight function used in prediction. Possible values:
      "uniform" : uniform weights. All points in each neighborhood
      are weighted equally.
      "distance" : weight points by the inverse of their distance.
      in this case, closer neighbors of a query point will have a
      greater influence than neighbors which are further away.
  """
  #your code below
  def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        from sklearn.impute import KNNImputer
        self.imputer = KNNImputer(
            n_neighbors=n_neighbors,
            weights=weights,
            add_indicator=False
        )

  def fit(self, X, y=None):
    """Fit the imputer on X.

    Parameters
    ----------
    X : pandas DataFrame
      Input data.
    y : Ignored
      Not used, present for API consistency.

    Returns
    -------
    self : object
      Fitted transformer.
    """
    assert isinstance(X, pd.DataFrame), f'Expected DataFrame but got {type(X)}'
    self.imputer.fit(X)
    self.columns_ = X.columns
    return self

  def transform(self, X):
    """Impute missing values in X.

    Parameters
    ----------
    X : pandas DataFrame
      The input data to complete.

    Returns
    -------
    pandas DataFrame
    Copy of X with imputed values.
    """
    assert isinstance(X, pd.DataFrame), f'Expected DataFrame but got {type(X)}'
    imputed_array = self.imputer.transform(X)
    imputed_df = pd.DataFrame(imputed_array, columns=self.columns_, index=X.index)
    return imputed_df

  def fit_transform(self, X, y=None):
    """Fit to data, then transform it.

    Parameters
    ----------
    X : pandas DataFrame
      Input data.
    y : Ignored
      Not used, present for API consistency.

    Returns
    -------
    pandas DataFrame
    Copy of X with imputed values.
    """
    return self.fit(X, y).transform(X)

###########################################################################################################################
class CustomTargetTransformer(BaseEstimator, TransformerMixin):
    """
    A target encoder that applies smoothing and returns np.nan for unseen categories.

    Parameters:
    -----------
    col: name of column to encode.
    smoothing : float, default=10.0
        Smoothing factor. Higher values give more weight to the global mean.
    """

    def __init__(self, col: str, smoothing: float =10.0):
        self.col = col
        self.smoothing = smoothing
        self.global_mean_ = None
        self.encoding_dict_ = None

    def fit(self, X, y):
        """
        Fit the target encoder using training data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.fit expected Dataframe but got {type(X)} instead.'
        assert self.col in X, f'{self.__class__.__name__}.fit column not in X: {self.col}. Actual columns: {X.columns}'
        assert isinstance(y, Iterable), f'{self.__class__.__name__}.fit expected Iterable but got {type(y)} instead.'
        assert len(X) == len(y), f'{self.__class__.__name__}.fit X and y must be same length but got {len(X)} and {len(y)} instead.'

        #Create new df with just col and target - enables use of pandas methods below
        X_ = X[[self.col]]
        target = self.col+'_target_'
        X_[target] = y

        # Calculate global mean
        self.global_mean_ = X_[target].mean()

        # Get counts and means
        counts = X_[self.col].value_counts().to_dict()    #dictionary of unique values in the column col and their counts
        means = X_[target].groupby(X_[self.col]).mean().to_dict() #dictionary of unique values in the column col and their means

        # Calculate smoothed means
        smoothed_means = {}
        for category in counts.keys():
            n = counts[category]
            category_mean = means[category]
            # Apply smoothing formula: (n * cat_mean + m * global_mean) / (n + m)
            smoothed_mean = (n * category_mean + self.smoothing * self.global_mean_) / (n + self.smoothing)
            smoothed_means[category] = smoothed_mean

        self.encoding_dict_ = smoothed_means

        return self

    def transform(self, X):
        """
        Transform the data using the fitted target encoder.
        Unseen categories will be encoded as np.nan.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input data to transform.
        """

        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.encoding_dict_, f'{self.__class__.__name__}.transform not fitted'

        X_ = X.copy()

        # Map categories to smoothed means, naturally producing np.nan for unseen categories, i.e.,
        # when map tries to look up a value in the dictionary and doesn't find the key, it automatically returns np.nan. That is what we want.
        X_[self.col] = X_[self.col].map(self.encoding_dict_)

        return X_

    def fit_transform(self, X, y):
        """
        Fit the target encoder and transform the input data.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data features.
        y : array-like of shape (n_samples,)
            Target values.
        """
        return self.fit(X, y).transform(X)

###########################################################################################################################
def find_random_state(
    features_df: pd.DataFrame,
    labels: Iterable,
    transformer: TransformerMixin,
    n: int = 200
                  ) -> Tuple[int, List[float]]:
    """
    Finds an optimal random state for train-test splitting based on F1-score stability.

    This function iterates through `n` different random states when splitting the data,
    applies a transformation pipeline, and trains a K-Nearest Neighbors classifier.
    It calculates the ratio of test F1-score to train F1-score and selects the random
    state where this ratio is closest to the mean.

    Parameters
    ----------
    features_df : pd.DataFrame
        The feature dataset.
    labels : Union[pd.Series, List]
        The corresponding labels for classification (can be a pandas Series or a Python list).
    transformer : TransformerMixin
        A scikit-learn compatible transformer for preprocessing.
    n : int, default=200
        The number of random states to evaluate.

    Returns
    -------
    rs_value : int
        The optimal random state where the F1-score ratio is closest to the mean.
    Var : List[float]
        A list containing the F1-score ratios for each evaluated random state.

    Notes
    -----
    - If the train F1-score is below 0.1, that iteration is skipped.
    - A higher F1-score ratio (closer to 1) indicates better train-test consistency.
    """

    model = KNeighborsClassifier(n_neighbors=5)
    Var: List[float] = []  # Collect test_f1/train_f1 ratios

    for i in range(n):
        train_X, test_X, train_y, test_y = train_test_split(
            features_df, labels, test_size=0.2, shuffle=True,
            random_state=i, stratify=labels  # Works with both lists and pd.Series
        )

        # Apply transformation pipeline
        transform_train_X = transformer.fit_transform(train_X, train_y)
        transform_test_X = transformer.transform(test_X)

        # Train model and make predictions
        model.fit(transform_train_X, train_y)
        train_pred = model.predict(transform_train_X)
        test_pred = model.predict(transform_test_X)

        train_f1 = f1_score(train_y, train_pred)

        if train_f1 < 0.1:
            continue  # Skip if train_f1 is too low

        test_f1 = f1_score(test_y, test_pred)
        f1_ratio = test_f1 / train_f1  # Ratio of test to train F1-score

        Var.append(f1_ratio)

    mean_f1_ratio: float = np.mean(Var)
    rs_value: int = np.abs(np.array(Var) - mean_f1_ratio).argmin()  # Index of value closest to mean

    return rs_value, Var
      
###########################################################################################################################
titanic_transformer = Pipeline(steps=[
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('joined', CustomOHETransformer(target_column='Joined')),
    ('fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('normalize', CustomMappingTransformer('Age', {}, normalize_numeric=True)),
], verbose=True)

#now invoke it
#transformed_df = titanic_transformer.fit_transform(titanic_features)

###########################################################################################################################
customer_transformer = Pipeline(steps=[
    #add drop step below
    ('drop', CustomDropColumnsTransformer(['ID'], 'drop')),
    ('time spent', CustomTukeyTransformer('Time Spent', 'inner')),
    ], verbose=True)

#now invoke it
#transformed_df = customer_transformer.fit_transform(customer_features)

