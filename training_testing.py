import tensorflow as tf
import poly_reg_model
import numpy as np

my_model=poly_reg_model.poly_reg
SZ_YX = (488,248)  #shape of input IVOCT images
NUM_NEIGHBOURS = 1
PAD_SIZE=76
CROP_PIECE=50 # number of lumen line fragments 
CROP_LENGTH=int((SZ_YX[1]+PAD_SIZE*2)/CROP_PIECE)
DEG=2 # order of polynomial 
channels = 1 + NUM_NEIGHBOURS * 2
DTYPE = np.float32

model = my_model((SZ_YX[0],SZ_YX[1]+PAD_SIZE*2,channels),(CROP_PIECE,DEG+1))
model.summary()


loss_basic_func=tf.keras.losses.MeanSquaredError()
x_range=tf.range(CROP_LENGTH,dtype=DTYPE)
tf_lrn_rate = tf.Variable(np.asarray(0, dtype=DTYPE), name="lrn_rate", trainable=False)
opt_func = tf.keras.optimizers.Adam(learning_rate=tf_lrn_rate)

@tf.function
def polynomial_fit(predictions):
    predictions_all=[]
    predictions=tf.squeeze(predictions,[1])
    for j in range (predictions.shape[0]):
        predictions_position=[]
        for k in range(predictions.shape[1]):
            prediction_piece=predictions[j,k,:]
            prediction_piece=tf.split(prediction_piece,prediction_piece.shape[0])
            y_range=tf.math.polyval(prediction_piece,x_range)
            predictions_position.append(y_range)
            predictions_position_image=tf.stack(predictions_position,0)
        predictions_all.append(predictions_position_image)
    return tf.stack(predictions_all,0)

@tf.function    
def loss_func(labels,predictions):
    predictions_polynomial=polynomial_fit(predictions)
    return (loss_basic_func(labels,predictions_polynomial))

@tf.function
def _train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt_func.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def _pred_step(images):
    predictions = model(images, training=False)
    predictions=polynomial_fit(predictions)
    return predictions
