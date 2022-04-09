from pandas import read_csv
from matplotlib import pyplot
from sklearn.decomposition import PCA

#Path of the CSV dataset
path = r"pima-indians-diabetes.csv"

#Set title for the data columns
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

#Read CSV dataset
data = read_csv(path, names=names)

#Print first 5 rows of the data set
print(data[0:5])

#Copy the values of data to an array 
array = data.values

#Seperate array into input and output compoents
X = array[:,0:8]
y = array[:,8]

#Set the number of variance we want from PCA
pca = PCA(n_components=4)

#Run the PCA in the input data
fit = pca.fit(X)

#Show the calculated variance values
print("Explained Variance: ",fit.explained_variance_ratio_)


print(fit.components_)
