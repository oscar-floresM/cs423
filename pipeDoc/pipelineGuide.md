# Step-by-Step Design Choices for Student Performance Pipeline

## 1. Gender Mapping (`map_gender`)

- **Transformer:** CustomMappingTransformer('gender', {'male': 0, 'female': 1})
- **Design Choice:** Binary encoding of gender with female as 1 and male as 0
- **Rationale:** Simplifies gender feature into numeric form, reducing dimensionality while preserving information

## 2. Parental Education Mapping (`map_parent_edu`)

- **Transformer:** CustomMappingTransformer('parental level of education', {
  "some high school": 0, "high school": 1, "some college": 2, 
  "associate's degree": 3, "bachelor's degree": 4, "master's degree": 5})
- **Design Choice:** Ordinal encoding of parental education based on increasing education levels
- **Rationale:** Preserves the natural ordering of education levels, potentially impacting student performance

## 3. Lunch Mapping (`map_lunch`)

- **Transformer:** CustomMappingTransformer('lunch', {'standard': 0, 'free/reduced': 1})
- **Design Choice:** Binary encoding of lunch type
- **Rationale:** Free/reduced lunch may correlate with socioeconomic status; encoding allows the model to capture this factor

## 4. Target Encoding for Race/Ethnicity (`target_race`)

- **Transformer:** CustomTargetTransformer(col='race/ethnicity', smoothing=10)
- **Design Choice:** Target encoding with smoothing factor of 10
- **Rationale:**
  - Converts nominal categorical variable into numeric representation based on its relationship with the target
  - Smoothing prevents overfitting by balancing category means with global mean

## 5. Outlier Treatment for Math Score (`tukey_math`)

- **Transformer:** CustomTukeyTransformer(target_column='math score', fence='outer')
- **Design Choice:** Tukey method using outer fences to detect extreme outliers
- **Rationale:** Protects model from extreme math score outliers that could distort model performance

## 6. Outlier Treatment for Reading Score (`tukey_reading`)

- **Transformer:** CustomTukeyTransformer(target_column='reading score', fence='outer')
- **Design Choice:** Tukey method using outer fences for outlier detection
- **Rationale:** Same rationale as math score; removes extreme values that could bias learning

## 7. Scaling for Math Score (`scale_math`)

- **Transformer:** CustomRobustTransformer(column='math score')
- **Design Choice:** Robust scaling using median and interquartile range
- **Rationale:** Minimizes impact of outliers while normalizing scale for math scores

## 8. Scaling for Reading Score (`scale_reading`)

- **Transformer:** CustomRobustTransformer(column='reading score')
- **Design Choice:** Robust scaling using median and interquartile range
- **Rationale:** Same as math score scaling, ensuring both numeric features are robustly scaled

## 9. Imputation (`impute`)

- **Transformer:** CustomKNNTransformer(n_neighbors=5)
- **Design Choice:** KNN imputation using 5 nearest neighbors
- **Rationale:**
  - Fills missing values by leveraging relationships between samples
  - k=5 balances between stability and responsiveness to data structure

---

# Pipeline Execution Order Rationale

- Categorical mapping happens first to convert all categorical features into numeric form.
- Target encoding is applied after mapping to transform nominal columns based on target relationships.
- Outlier handling is applied before scaling to prevent extreme values from influencing scaling.
- Robust scaling prepares numeric columns for modeling while handling any remaining outliers.
- Imputation is applied last, after scaling, so KNN distance calculations are not distorted by unscaled features.

---

# Performance Considerations

- Used `RobustScaler` because test scores may contain outliers or non-normal distributions.
- `KNN imputation` captures complex relationships between features when filling missing values.
- `Target encoding` with smoothing prevents overfitting on rare race/ethnicity categories.
- Pipeline is modular and can easily be adapted for different machine learning models or expanded features.