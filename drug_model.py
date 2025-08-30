#Importing necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
#Dataset reading, showcase and basic cleaning
df= pd.read_csv('/home/sulav/Desktop/python/plt/drug200.csv')
print(df.keys())
df.dropna(inplace=True)
new_df=df.copy()

#Replacing strings by int for ease in model creation
bp_chol_assigning = {'LOW':-1,'NORMAL':0,'HIGH':1}
new_df['BP']=new_df['BP'].apply(lambda x: bp_chol_assigning.get(x,x))
new_df['Cholesterol']=new_df['Cholesterol'].apply(lambda x: bp_chol_assigning.get(x,x))

sex_assigning= {'M':1,'F':0}
new_df['Sex']=new_df['Sex'].apply(lambda x: sex_assigning.get(x,x))

#Model Creation
features=['BP','Cholesterol','Na_to_K']
target='Drug'

X=new_df[features]
y=new_df[target]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=67)

model=KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)

#Model Accuracy and test on new input
y_pred=model.predict(X_test)
acc=accuracy_score(y_test,y_pred)*100
print(f'\nThe accuracy of this model is : {acc:.3f}%')
new_patient_data = {
	'BP': [-1],           # Low(-1), Normal(0), High(1)
	'Cholesterol': [0],  # Low(-1), Normal(0), High(1)
	'Na_to_K': [11.349]
}

new_patient_df = pd.DataFrame(new_patient_data, columns=features)
predicted_drug = model.predict(new_patient_df)
print(predicted_drug)
