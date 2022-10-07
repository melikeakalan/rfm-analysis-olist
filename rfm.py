###################################################
# PROJECT: Customer Segmentation with RFM
###################################################

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import datetime as dt
import matplotlib.pyplot as plt
import squarify

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import seaborn as sns
from helpers.data_prep import *
from helpers.eda import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)

import warnings
warnings.filterwarnings("ignore")


#############################################
# Project Tasks
#############################################

#############################################
# Task 1: Gathering Data from DataBase and Data Understanding
#############################################

# Step 1: Gathering Data from DataBase( PostgreSQL )
user = "******"
password = "******"
host = "localhost"
port = "5432"
database = "olist_db"

connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"
con = create_engine(connection_string)

df_customers = pd.read_sql('select * from customers', con)
df_geolocation = pd.read_sql('select * from geolocation', con)
df_orders = pd.read_sql('select * from orders', con)
df_order_items = pd.read_sql('select * from order_items', con)
df_order_payments = pd.read_sql('select * from order_payments', con)
df_order_reviews = pd.read_sql('select * from order_reviews', con)
df_products = pd.read_sql('select * from products', con)
df_sellers = pd.read_sql('select * from sellers', con)
df_translations = pd.read_sql('select * from product_translation', con)


# Step 2: Understanding Data
df_customers.head()

df_name_list = ['df_customers', 'df_geolocation', 'df_orders', 'df_order_items', 'df_order_payments',
               'df_order_reviews', 'df_products', 'df_sellers', 'df_translations']

for df in df_name_list:
    check_df_names(df)

# Step 3: Missing Values Imputations

# DF with NA values:
# df_orders, df_order_reviews ve df_products

# for df_orders
df_orders['order_approved'].fillna(df_orders['order_purchase'], inplace=True)
df_orders['order_delivered_carrier'].fillna(df_orders['order_approved'], inplace=True)
df_orders['order_delivered_customer'].fillna(df_orders['order_estimated_delivery'], inplace=True)

# for df_order_reviews
df_order_reviews = df_order_reviews[~(df_order_reviews['review_title'].isnull())].reset_index(drop=True)
df_order_reviews['review_comment'] = df_order_reviews['review_comment'].replace(np.nan, 'None')

# for df_products
df_products.isnull().sum()
df_products_na_list = ['product_category', 'product_name_length', 'product_desc_length', 'product_photos_qty',
                       'product_weight_grams', 'product_length_cm', 'product_height_cm', 'product_width_cm']

for col in df_products_na_list:
    df_products = df_products[~(df_products[col].isnull())]

# Controlling of NA Values
df_orders.isnull().sum()
df_order_reviews.isnull().sum()
df_products.isnull().sum()


# Step 4: Merging All Dataframes
df_merge = pd.merge(df_orders, df_order_payments, on='order_id')
df_merge = pd.merge(df_merge, df_customers, on='customer_id')
df_merge = pd.merge(df_merge, df_order_items, on='order_id')
df_merge = pd.merge(df_merge, df_sellers, on='seller_id')
df_merge = pd.merge(df_merge, df_order_reviews, on='order_id')
df_merge = pd.merge(df_merge, df_products, on='product_id')

df_translations.columns = ['product_category', 'category_translation']
df_merge = pd.merge(df_merge, df_translations, on='product_category')

df_merge.head()
df_merge.info()
df_merge.shape   # 13801,40
df_merge.isnull().sum()

# Step 5: Deleting Duplicated Rows

# Showing Duplicated Rows
df_merge[df_merge.duplicated(subset={'order_id',
                                     'customer_id',
                                     'order_purchase',
                                     'order_delivered_customer'}, keep='first')].head(30)
# Deleting Duplicated Rows
# duplicate olan satırları sil ama first rows'larını elimde tut
df_merge = df_merge.drop_duplicates(subset={'order_id',
                                            'customer_id',
                                            'order_purchase',
                                            'order_delivered_customer'}, keep='first')
df_merge.shape


# Step 6: Converting to datetime

df_merge['order_purchase'] = pd.to_datetime(df_merge['order_purchase'],
                                            infer_datetime_format=True,
                                            errors='ignore')

str(df_merge["order_purchase"].dtype)
df_merge["order_purchase"].head()


# Step 7: Finding unique prodcuts
df_merge['product_category'].nunique()  # 66

# Step 8: Finding 10 most existing unique products and showing them in a bar-plots
df_merge.groupby(['product_category']).agg({'product_id': 'count'}).sort_values(by=['product_id'],
                                                                                ascending=False).head(10)

# Bar-plot
plt.figure(figsize=(6, 10))
top10_sold_products = df_merge.groupby('product_category')['product_id'].count().sort_values(ascending=False).head(10)
sns.barplot(x=top10_sold_products.index, y=top10_sold_products.values)
plt.xticks(rotation=80)
plt.xlabel('Product category')
plt.title('Top 10 products')
plt.show()



#############################################
# Task 2: Calculating RFM Metrics
#############################################

# MONETARY
df_merge['TOTAL_SALES_QUANTITY'] = df_merge['payment_value'] * df_merge['payment_installments']

df_monetary = df_merge.groupby(['customer_unique_id'],
                               group_keys=False,
                               as_index=False).agg({'TOTAL_SALES_QUANTITY': 'sum'}).reset_index(drop=True)
df_monetary.head()

df_monetary.rename(columns={'TOTAL_SALES_QUANTITY': 'monetary'}, inplace=True)
df_monetary.head()


# FREQUENCY
df_frequency = df_merge.groupby(['customer_unique_id'],
                                group_keys=False,
                                as_index=False).agg({'order_id': 'count'}).reset_index(drop=True)

df_frequency.rename(columns={'order_id': 'frequency'}, inplace=True)
df_frequency.head()

# Merging df_monetary and df_freq tables
df = pd.merge(left=df_monetary,
              right=df_frequency,
              on='customer_unique_id',
              how='inner')
df.head()

# RECENCY
df_merge['order_purchase'].max()      # 2018-08-29

analysis_date = dt.datetime(2018, 8, 30)

df_merge['DAYS'] = (analysis_date - df_merge['order_purchase']).dt.days

df_recency = df_merge.groupby(['customer_unique_id'],
                              group_keys=False,
                              as_index=False).agg({'DAYS': 'min'}).reset_index(drop=True)

df_recency.rename(columns={'DAYS': 'recency'}, inplace=True)
df_recency.head()

rfm = pd.merge(left=df,
               right=df_recency,
               on='customer_unique_id',
               how='inner')
rfm.head()
rfm.shape

# Visualize metrics
plt.figure(figsize=(12, 10))
plt.subplot(3, 1, 1); sns.distplot(rfm['recency'])
plt.subplot(3, 1, 2); sns.distplot(rfm['frequency'])
plt.subplot(3, 1, 3); sns.distplot(rfm['monetary'])
plt.show()


#############################################
# Task 3: Segmentation of Customers by RFM Metrics with K-Means
#############################################

# Step 1: Data Standardization
rfm_scaler = rfm[['monetary', 'frequency', 'recency']]
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(rfm_scaler)
model_df = pd.DataFrame(model_scaling, columns=rfm_scaler.columns)
model_df.head()

# Step 2: Finding optimal number of clusters
kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()

# Step 3: Building K-Means Model
k_means = KMeans(n_clusters=5, random_state=42).fit(model_df)
segments = k_means.labels_
segments

# Step 4: Looking at frequency of segments
pd.DataFrame(segments).value_counts()
rfm['segment'] = segments
rfm.head()

# Step 5: Statistical examination of each segment
final_merge = df_merge[['customer_unique_id',
                        'order_id', 'order_status', 'order_purchase', 'payment_type', 'payment_installments',
                        'payment_value']]
final_merge.head()

final_df = pd.merge(left=final_merge,
                    right=rfm,
                    on='customer_unique_id',
                    how='inner')
final_df.head()

final_df.groupby('segment').agg({'payment_installments': ['median', 'min', 'max'],
                                 'payment_value': ['median', 'min', 'max'],
                                 'monetary': ['median', 'min', 'max'],
                                 'frequency': ['median', 'min', 'max'],
                                 'recency': ['median', 'min', 'max']})

rfmStats = rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "median", "count", "std"])
rfmStats

newIndex=["segment_0","segment_1","segment_2","segment_3","segment_4"]
rfmStats.index=newIndex


# Visualize Segments
plt.figure(figsize=(10, 6))
squarify.plot(sizes=rfmStats["recency"]["count"], label=rfmStats.index,
              color=["cornsilk", "pink","royalblue", "red", "yellow"], alpha=.4)
plt.suptitle("Treemap: Number of Customers", fontsize=20)