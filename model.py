import numpy as np
import pandas as pd

traindata = pd.read_csv('train.csv', header = 0)
testdata = pd.read_csv('test.csv', header = 0)

Y = traindata['Survived'];
Y = Y.values.reshape(Y.shape[0],1)
Y = Y.T

traindata.drop('Survived', 1 , inplace = True)

data = traindata.append(testdata)
data.reset_index(inplace=True)
data.drop('index', inplace=True, axis=1)

#feature engineering

data['Title'] = data['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())

Title_Dictionary = {"Capt":       "Officer",
                        "Col":        "Officer",
                        "Major":      "Officer",
                        "Jonkheer":   "Royalty",
                        "Don":        "Royalty",
                        "Sir" :       "Royalty",
                        "Dr":         "Officer",
                        "Rev":        "Officer",
                        "the Countess":"Royalty",
                        "Dona":       "Royalty",
                        "Mme":        "Mrs",
                        "Mlle":       "Miss",
                        "Ms":         "Mrs",
                        "Mr" :        "Mr",
                        "Mrs" :       "Mrs",
                        "Miss" :      "Miss",
                        "Master" :    "Master",
                        "Lady" :      "Royalty"}

data['Title'] = data.Title.map(Title_Dictionary)

data['Fare'].fillna(data['Fare'].median(), inplace = True);

grouped_train = data.head(891).groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()

grouped_test = data.iloc[891:].groupby(['Sex','Pclass','Title'])
grouped_median_test = grouped_test.median()

def fillAges(row, grouped_median):
    if row['Sex']=='female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 1, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 1, 'Mrs']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['female', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['female', 1, 'Royalty']['Age']

    elif row['Sex']=='female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 2, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 2, 'Mrs']['Age']

    elif row['Sex']=='female' and row['Pclass'] == 3:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 3, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 3, 'Mrs']['Age']

    elif row['Sex']=='male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 1, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 1, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['male', 1, 'Royalty']['Age']

    elif row['Sex']=='male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 2, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 2, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 2, 'Officer']['Age']

    elif row['Sex']=='male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 3, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 3, 'Mr']['Age']

data.head(891).Age = data.head(891).apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age'])
                                                  else r['Age'], axis=1)

data.iloc[891:].Age = data.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age'])
                                                  else r['Age'], axis=1)

data['Title'].replace(['Officer','Royalty','Mrs','Mr','Miss','Master'],[1,2,3,4,5,6], inplace = True)

data.drop('Name', axis = 1, inplace = True)
data.drop('Cabin', axis = 1 , inplace = True)
data.drop('Ticket', axis = 1, inplace =  True)
data['Embarked'].fillna('S', inplace = True);

data['Sex'].replace(['male','female'],[1,2], inplace = True)
data['Embarked'].replace(['S','C','Q'],[1,2,3], inplace = True)

train = data.head(891)
test = data.iloc[891:]

Xtrain = train
Ytrain = Y

def sigmoid(x):
	sigx = 1/(1+np.exp(-x))
	return sigx

##initialisation
n1 = 32 #units in layer 1
n2 = 32 # units in layer 2
n3 = 32 # units in layer 3
n4 = 8
m = Xtrain.shape[1]  #no. of training examples
ar = 0.2 #learning rate

#weights and biases

w1 = np.random.randn(n1,Xtrain.shape[0])*0.01
b1 = np.zeros((n1,1))
w2 = np.random.randn(n2,n1)*0.01
b2 = np.zeros((n2,1))
w3 = np.random.randn(n3,n2)*0.01
b3 = np.zeros((n3,1))
w4 = np.random.randn(n4,n3)*0.01
b4 = np.zeros((n4,1))
w5 = np.random.randn(1,n4)*0.01
b5 = 0
##iteration

z1 = np.dot(w1,Xtrain) + b1
a1 = sigmoid(z1)

z2 = np.dot(w2,a1)+b2
a2 = sigmoid(z2)

z3 = np.dot(w3,a2) + b3
a3 = sigmoid(z3)

z4 = np.dot(w4,a3) + b4
a4 = sigmoid(z4)

z5 = np.dot(w5,a4) + b5
a5 = sigmoid(z5)

l = np.sum(-Ytrain*np.log(a5) - (1-Ytrain)*np.log(1-a5))

iterations = 10000

for i in range(iterations):

	dz5 = a5-Ytrain
	dw5 = (1/m)*(np.dot(dz5,(a4.T)))
	db5 = (1/m)*np.sum(dz5, axis = 1, keepdims = True)

	dz4 = np.dot((w5.T),dz5) * (a4*(1-a4))
	dw4 = (1/m)*(np.dot(dz4,(a3.T)))
	db4 = (1/m)*np.sum(dz4, axis = 1, keepdims = True)

	dz3 = np.dot((w4.T),dz4) * (a3*(1-a3))
	dw3 = (1/m)*(np.dot(dz3,(a2.T)))
	db3 = (1/m)*np.sum(dz3, axis = 1, keepdims = True)

	dz2 = np.dot((w3.T),dz3) * (a2*(1-a2))
	dw2 = (1/m)*(np.dot(dz2,(a1.T)))
	db2 = (1/m)*np.sum(dz2, axis = 1, keepdims = True)

	dz1 = np.dot((w2.T),dz2) * (a1*(1-a1))
	dw1 = (1/m)*(np.dot(dz1,(Xtrain.T)))
	db1 = (1/m)*np.sum(dz1, axis = 1, keepdims = True)

	w5 = w5 - ar*dw5
	w4 = w4 - ar*dw4
	w3 = w3 - ar*dw3
	w2 = w2 - ar*dw2
	w1 = w1 - ar*dw1

	b5 = b5 - ar*db5
	b4 = b4 - ar*db4
	b3 = b3 - ar*db3
	b2 = b2 - ar*db2
	b1 = b1 - ar*db1

	z1 = np.dot(w1,Xtrain) + b1
	a1 = sigmoid(z1)

	z2 = np.dot(w2,a1)+b2
	a2 = sigmoid(z2)

	z3 = np.dot(w3,a2) + b3
	a3 = sigmoid(z3)

	z4 = np.dot(w4,a3) + b4
	a4 = sigmoid(z4)

	z5 = np.dot(w5,a4) + b5
	a5 = sigmoid(z5)

	yh = np.zeros((a5.shape[0],a5.shape[1]))

	l = np.sum(-Ytrain*np.log(a5) - (1-Ytrain)*np.log(1-a5))

	#
	#print("error = ", l)

for j in range(a5.shape[1]):
		if a5[0,j] > 0.5:
			yh[0,j] = 1
		else:
			yh[0,j] = 0

loss = np.zeros((Ytrain.shape[0],Ytrain.shape[1]))

loss = yh-Ytrain

print("loss train =", np.sum(np.abs(loss)))
print("error train = ", l)

#prediction

Xfinal = test.T

z1 = np.dot(w1,Xfinal) + b1
a1 = sigmoid(z1)
z2 = np.dot(w2,a1)+b2
a2 = sigmoid(z2)

z3 = np.dot(w3,a2) + b3
a3 = sigmoid(z3)

z4 = np.dot(w4,a3) + b4
a4 = sigmoid(z4)

z5 = np.dot(w5,a4) + b5
a5 = sigmoid(z5)


yfinal = np.zeros((a5.shape[0],a5.shape[1]))
for j in range(a5.shape[1]):
		if a5[0,j] > 0.5:
			yfinal[0,j] = 1
		else:
			yfinal[0,j] = 0

print(yfinal)
