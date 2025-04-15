import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from pymongo import MongoClient
from train import booking

load_dotenv()
uri = os.getenv("url")

client = MongoClient(uri)

db = client["fatllamaclone"]

items_collection = db['items']
bookings_collection = db['bookings']

items_data = list(items_collection.find({}, {"_id": 0}))
df_items = pd.DataFrame(items_data)

bookings_data = list(bookings_collection.find({}, {"_id": 0}))
df_bookings = pd.DataFrame(bookings_data)

bookings_df = pd.merge(df_items, df_bookings, on='item_id', how = 'inner')
bookings_df['rental_duration'] = (pd.to_datetime(bookings_df['return_end']) - pd.to_datetime(bookings_df['rental_start'])).dt.total_seconds() / 3600
bookings_df['price_per_hour'] = bookings_df['price_per_day']/24
bookings_df['expected_payment'] = bookings_df['price_per_day'] * bookings_df['rental_days']
bookings_df['payment_ratio'] = bookings_df['final_payment'] / bookings_df['expected_payment']
bookings_df['payment_ratio'] = bookings_df['payment_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)


if __name__ == "__main__":
    churnn = pd.read_csv('Notebooks\\IPYNB\\CHRN.csv')
    print(churnn)

    df_Churn = pd.merge(churnn, booking(bookings_df), on='user_id', how='inner')
    df_Churn = df_Churn.drop(columns={'last_booking_date','first_booking_date'})

    X = df_Churn.drop(columns=["user_id", "churned"])
    final_model = "xgb_churn_model.pkl"
    y= final_model.predict(X)
    print(X,y)