
#Importação do dataset
#==============================================================================
from sklearn import datasets
iris = datasets.load_iris()
digitos = datasets.load_digits()


#Visualização inicial do dataset
#==============================================================================
print(digitos.data)


#
#==============================================================================
digitos.target

