import pandas as pd

def split_feature_types(X: pd.DataFrame):
    X = X.copy()
    cat_cols = []
    continuous_cols = []

    for col in X.columns:
        col_data = X[col]

        if pd.api.types.is_datetime64_any_dtype(col_data):
            X[col] = col_data.astype(str).fillna('missing')
            cat_cols.append(col)

        elif (
            pd.api.types.is_object_dtype(col_data)
            or pd.api.types.is_categorical_dtype(col_data)
            or pd.api.types.is_bool_dtype(col_data)
        ):
            X[col] = col_data.astype(str).fillna('missing')
            cat_cols.append(col)

        else:
            non_null_unique = pd.Series(col_data.dropna().unique())
            if len(non_null_unique) <= 2:
                unique_set = set(non_null_unique.tolist())
                if unique_set.issubset({0, 1}) or unique_set.issubset({0.0, 1.0}) or unique_set.issubset({'0', '1'}):
                    X[col] = col_data.fillna(-1).astype(str)
                    cat_cols.append(col)
                else:
                    continuous_cols.append(col)
            else:
                continuous_cols.append(col)

    return X, cat_cols, continuous_cols