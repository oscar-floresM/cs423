# Heart Disease Dataset Pipeline Design Guide

## Dataset Overview
- **Columns:** ['age', 'gender', 'height', 'weight', 'smoke', 'alco', 'active', 'cardio']
- **Target Variable:** cardio (0 = no heart disease, 1 = heart disease)
- **Categorical Features:** gender (1 = male, 2 = female)
- **Binary Features:** smoke, alco, active, cardio (0 = no, 1 = yes)
- **Numerical Features:** age, height, weight

## Step-by-Step Design Choices

### 1. Gender Mapping (`map_gender`)
* **Transformer:** `CustomMappingTransformer('gender', {1: 0, 2: 1})`
* **Design Choice:** Binary encoding of gender with female as 1 and male as 0
* **Rationale:** 
  * Converts original encoding (1=male, 2=female) to standard binary format
  * Simple categorical mapping that preserves the binary nature of the feature without increasing dimensionality
  * Maintains consistency with common ML practices where female is encoded as 1

### 2. Outlier Treatment for Age (`tukey_age`)
* **Transformer:** `CustomTukeyTransformer(target_column='age', fence='outer')`
* **Design Choice:** Tukey method with outer fence for identifying extreme outliers
* **Rationale:**
  * Outer fence (Q1-3×IQR, Q3+3×IQR) identifies only the most extreme outliers
  * Age in medical datasets may have legitimate outliers (very young or elderly patients) that should be preserved unless extreme
  * Preserves the natural age distribution while handling data entry errors

### 3. Outlier Treatment for Height (`tukey_height`)
* **Transformer:** `CustomTukeyTransformer(target_column='height', fence='outer')`
* **Design Choice:** Tukey method with outer fence for identifying extreme outliers
* **Rationale:**
  * Height has natural human variation that should be preserved
  * Outer fence maintains most of the original distribution while identifying measurement errors
  * Medical datasets often contain height outliers due to unit conversion errors (cm vs inches)

### 4. Outlier Treatment for Weight (`tukey_weight`)
* **Transformer:** `CustomTukeyTransformer(target_column='weight', fence='outer')`
* **Design Choice:** Tukey method with outer fence for identifying extreme outliers
* **Rationale:**
  * Weight has high natural variability in medical populations
  * Outer fence preserves legitimate weight variations while handling extreme measurement errors
  * Important for cardiovascular risk assessment where weight extremes may be clinically relevant

### 5. Age Scaling (`scale_age`)
* **Transformer:** `CustomRobustTransformer(target_column='age')`
* **Design Choice:** Robust scaling for Age feature
* **Rationale:**
  * Robust to outliers compared to standard scaling
  * Uses median and interquartile range instead of mean and standard deviation
  * Appropriate for age which may not follow normal distribution in medical datasets
  * Prevents age-related bias in distance-based algorithms

### 6. Height Scaling (`scale_height`)
* **Transformer:** `CustomRobustTransformer(target_column='height')`
* **Design Choice:** Robust scaling for Height feature
* **Rationale:**
  * Height measurements may have different units or scales in medical records
  * Robust scaling reduces influence of remaining outliers after Tukey treatment
  * Essential for KNN imputation to work effectively with mixed-scale features

### 7. Weight Scaling (`scale_weight`)
* **Transformer:** `CustomRobustTransformer(target_column='weight')`
* **Design Choice:** Robust scaling for Weight feature
* **Rationale:**
  * Weight has high variability and potentially skewed distribution
  * Robust scaling maintains relative relationships while normalizing scale
  * Critical for cardiovascular models where weight relationships are non-linear

### 8. Imputation (`impute`)
* **Transformer:** `CustomKNNTransformer(n_neighbors=5)`
* **Design Choice:** KNN imputation with 5 neighbors
* **Rationale:**
  * Uses relationships between features to estimate missing values
  * k=5 balances between too few neighbors (overfitting) and too many (underfitting)
  * Particularly appropriate for medical data where features are often correlated
  * More accurate than simple mean/median imputation for cardiovascular risk factors

## Pipeline Execution Order Rationale

1. **Categorical encoding first** to prepare gender for subsequent numerical operations
2. **Outlier treatment before scaling** to prevent outliers from affecting scaling parameters
3. **Scaling before imputation** so that distance metrics in KNN aren't skewed by unscaled features
4. **Imputation last** to fill missing values using all preprocessed features

## Features Not Requiring Preprocessing

The following binary features are already optimally encoded and require no transformation:
- **smoke** (0 = no, 1 = yes)
- **alco** (0 = no, 1 = yes) 
- **active** (0 = no, 1 = yes)
- **cardio** (0 = no, 1 = yes) - Target variable

## Performance Considerations

* **RobustScaler instead of StandardScaler** due to presence of outliers in medical data
* **KNN imputation instead of simple imputation** to preserve relationships between cardiovascular risk factors
* **Outer fence for outlier detection** to maintain clinical relevance of extreme values
* **No target encoding needed** as binary features are already optimally encoded

## Medical Domain Considerations

* **Age scaling critical** for cardiovascular risk models where age is a primary factor
* **Height/weight preprocessing essential** for BMI-related cardiovascular risk assessment
* **Gender encoding maintains interpretability** for sex-based cardiovascular risk differences
* **Binary lifestyle factors** (smoke, alco, active) preserved in clinically meaningful format