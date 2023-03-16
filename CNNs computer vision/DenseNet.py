
------------------DenseNet--------------
from tensorflow.keras import Model
from tensorflow.keras.applications.densenet  import DenseNet169

preTrainedModelDenseNet169 = DenseNet169(input_shape =( ImageSize, ImageSize, 3), include_top = False, 
weights = None)

preTrainedModelDenseNet169.load_weights("../input/densenet-keras/DenseNet-BC-169-32-no-top.h5")

for layer in preTrainedModelDenseNet169.layers:
    layer.trainable = False  
    
preTrainedModelDenseNet169.summary()



#DenseNet169 Model
x=Flatten()(preTrainedModelDenseNet169.output)

#Fully Connection Layers
# FC1
x=Dense(1024, activation="relu")(x)
x=BatchNormalization()(x)
x=Dense(512, activation="relu")(x)

#Dropout to avoid overfitting effect
x=Dropout(0.2)(x)

# FC2
x=Dense(256, activation="relu")(x)
x=Dense(128, activation="relu")(x)


#output layer
x=Dense(4,activation="sigmoid")(x)


modelDenesNet=Model(preTrainedModelDenseNet169.input,x)
modelDenesNet.summary()


------------------VGG19--------------
from tensorflow.keras.applications.vgg19  import VGG19
from tensorflow.keras import Model

preTrainedModelVGG19 = VGG19(input_shape =( ImageSize, ImageSize, 3), include_top = False)
# preTrainedModelVGG19.load_weights("../input/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")

for layer in preTrainedModelVGG19.layers:
    layer.trainable = False  
preTrainedModelVGG19.summary()

#mobileNet Model
x=Flatten()(preTrainedModelVGG19.output)

#Fully Connection Layers
# FC1
x=Dense(1024, activation="relu")(x)
x=Dense(512, activation="relu")(x)
#Dropout to avoid overfitting effect
x=Dropout(0.2)(x)
# FC2
x=Dense(256, activation="relu")(x)
x=Dense(128, activation="relu")(x)


#output layer
x=Dense(4,activation="sigmoid")(x)


ModelVGG19=Model(preTrainedModelVGG19.input,x)
ModelVGG19.summary()







