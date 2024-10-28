import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

test_df = pd.read_csv('C:/Users/sanjana/Downloads/clothing_items_new_500.csv')
train_df = pd.read_csv("C:\\Users\\sanjana\\Downloads\\clothing_items_new2_1000.csv")
valid_df = pd.read_csv("C:\\Users\\sanjana\\Downloads\\clothing_items.csv")


input_cols=train_df.columns.tolist()[0:-1]
target_col='price'

train_inputs=train_df[input_cols]
train_target=train_df[target_col]

valid_inputs=valid_df[input_cols]
valid_target=valid_df[target_col]

test_inputs=test_df[input_cols]
test_target=test_df[target_col]

encoder=OneHotEncoder(categories='auto',sparse_output=False)
con_df_1=pd.concat([train_df,valid_df],ignore_index=True)
full_df=pd.concat([test_df,con_df_1],ignore_index=True)

encoder.fit(full_df[input_cols])
encoded_cols=encoder.get_feature_names_out(input_cols).tolist()

train_inputs[encoded_cols]=pd.DataFrame(encoder.transform(train_inputs[input_cols]))
valid_inputs[encoded_cols]=pd.DataFrame(encoder.transform(valid_inputs[input_cols]))
test_inputs[encoded_cols]=pd.DataFrame(encoder.transform(test_inputs[input_cols]))

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_inputs[encoded_cols],train_df[target_col])


pickle.dump(model,open("model.pkl","wb"))
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
