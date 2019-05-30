#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import keras  
from keras.datasets import mnist 
from keras.models import Model 
from keras.layers import Dense, Input
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten 
from keras import backend as k 


# In[3]:


#loading the data
(x_train, y_train), (x_test, y_test) = mnist.load_data() 


# In[4]:


img_rows, img_cols=28, 28 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1) 
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) 
inpx = (img_rows, img_cols, 1)
print(inpx)

x_train = x_train.astype('float32') 
x_test = x_test.astype('float32') 
x_train /= 255
x_test /= 255


# In[5]:


y_train = keras.utils.to_categorical(y_train)
print(y_train)
y_test = keras.utils.to_categorical(y_test) 


# In[6]:


inpx = Input(shape=inpx) 
print(inpx)
layer1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(inpx) 
layer2 = Conv2D(64, (3, 3), activation='relu')(layer1) 
layer3 = MaxPooling2D(pool_size=(3, 3))(layer2) 
layer4 = Dropout(0.5)(layer3) 
layer5 = Flatten()(layer4) 
layer6 = Dense(250, activation='sigmoid')(layer5) 
layer7 = Dense(10, activation='softmax')(layer6) 


# In[7]:


model = Model([inpx], layer7) 
model.compile(optimizer=keras.optimizers.Adam(), 
loss=keras.losses.categorical_crossentropy, 
metrics=['accuracy']) 


model.fit(x_train, y_train, epochs=5, batch_size=200)


# In[8]:


score = model.evaluate(x_test, y_test, verbose=0) 
print('loss=', score[0]) 
print('accuracy=', score[1]) 


# In[9]:


y_pred=model.predict(x_test)


# In[10]:


y_pred=np.argmax(y_pred,axis=1)


# In[11]:


print(y_pred[1])


# In[16]:


import matplotlib.pyplot as plt
x_test=x_test.reshape(x_test.shape[0],28,28)
plt.imshow(x_test[1])
plt.show()


# In[17]:


model.save('project.h5')


# In[ ]:


get_ipython().system('tar -zcvf project.tgz project.h5')


# In[ ]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# In[ ]:


wml_credentials={
    "url": "https://eu-gb.ml.cloud.ibm.com",
  "access_key": "vtWdt-Lm7hPiS4X2IRCW6YI-J20HdlXvzj3EWgNnfox6",
    "username": "6e8e2977-19b0-4810-aa45-8c134f080049",
    "password": "7875be87-db08-4d4c-9c10-b9b8e8017d77",
  "instance_id": "94daaf32-f93d-4a4d-a53c-e418651fb936"
  
}


# In[ ]:


client=WatsonMachineLearningAPIClient(wml_credentials)


# In[ ]:


metadata={
    client.repository.ModelMetaNames.NAME:"keras",
    client.repository.ModelMetaNames.FRAMEWORK_LIBRARIES:[{'name':'keras','version':'2.1.3'}],
    client.repository.ModelMetaNames.FRAMEWORK_VERSION:"1.5",
    client.repository.ModelMetaNames.FRAMEWORK_NAME:"tensorflow"
}


# In[ ]:


model_details = client.repository.store_model( model="project.tgz", meta_props=metadata )


# In[ ]:


model_id = model_details["metadata"]["guid"]
model_deployment_details = client.deployments.create( artifact_uid=model_id, name="deployment" )


# In[ ]:


scoring_endpoint=client.deployments.get_scoring_url(model_deployment_details)


# In[ ]:


scoring_endpoint

