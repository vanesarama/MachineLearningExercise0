import pandas as pd
import matplotlib.pyplot as plt 


veri1 = pd.read_csv("melbourne_airbnb_listings.csv")
veri2 = pd.read_csv("spambase_features.csv")

numeric_attributes = veri1.select_dtypes(include=['float64','int64'])

numeric_attributes.hist(bins=10, figsize=(15,10), color= 'skyblue', edgecolor = 'black')
plt.tight_layout()
plt.show()






