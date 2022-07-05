import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense

df = pd.read_csv('Date_Fruit_Datasets.csv',  delimiter=",")

#getting all values between 0 and 1
all_max_values = []
all_classes = df["Class"].unique()
num_classes = df["Class"].unique().size

for item in df.columns:
    if item != "Class":
        if df[item].max() > 0:
            all_max_values.append(df[item].max())
            df[item] = df[item].div(df[item].max())
        else:
            all_max_values.append(df[item].min())
            df[item] = df[item].div(df[item].min())


#dropping the column Class to have the "x" datas in shape
df_rmv_classes = df.drop(['Class'], axis=1)

#minMaxScaler implementation
#scaler = MinMaxScaler(feature_range=(0, 1))
#df_rmv_classes = scaler.fit_transform(df_rmv_classes)

print(df_rmv_classes)

df_classes=df['Class']
#transforming the strings of the classes to numbers to be able to process it later
transform_numbers = range(num_classes)
df_classes.replace(all_classes,transform_numbers,inplace=True)



#check the distribution of the fruits(Datteln) --> the distribution in this dataset is unequal
class_dist = df_classes.value_counts()
print(class_dist)


#optimze trainingsData --> find the best random_state to train in a equal distribution
rnd_count = 0
all_seeds = {}
for item in range(500):
    x_train, x_test, t_train, t_test = train_test_split(df_rmv_classes, df_classes, test_size=0.3, random_state=rnd_count)
    diff_sum = 0
    total_sum = 0
    for idx,cval in enumerate(t_train.value_counts()):
        diff_sum = cval - class_dist[idx]
        total_sum = abs(total_sum) + abs(diff_sum)
    all_seeds[rnd_count] = total_sum
    rnd_count += 1

print(all_seeds)
print(min(all_seeds, key=all_seeds.get))

print("test: ")
print(df_rmv_classes)
print(df_classes)

#doing the train test split with the best random_state
best_rnd_state = min(all_seeds, key=all_seeds.get)
x_train,x_test,t_train,t_test=train_test_split(df_rmv_classes,df_classes,test_size=0.3,random_state=best_rnd_state)


#one hot vectors
t_train = tf.keras.utils.to_categorical(t_train, num_classes)
t_test = tf.keras.utils.to_categorical(t_test, num_classes)

model = tf.keras.models.Sequential()

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=16, activation='relu'))
model.add(Dense(units=12, activation='relu'))
model.add(Dense(units=num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

batch_size=16
training_epochs = 100
training_history = model.fit(
    x_train, # input
    t_train, # output
    batch_size=batch_size,
    verbose=1, # Suppress chatty output; use Tensorboard instead
    epochs=training_epochs,
    validation_data=(x_test, t_test),
    callbacks=[tensorboard_callback],
)

print(model.metrics_names, model.evaluate(x_test,t_test))
model.summary()

model.save('saved_model/project1')