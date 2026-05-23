#import sys
#sys.path.insert(1,'/kaggle/input/fourier-decomposition-method')
from fdm import fdm
# from livelossplot import PlotLossesKeras
from keras import layers , models
import keras    
from keras import regularizers
import tensorflow as tf
# from keras.models import Model
# from keras.layers import Input, Flatten, Dense
from kymatio.keras import Scattering2D
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Model
from scipy import stats
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.metrics import roc_curve, auc

def get_rhythm(sf,sec,container,dataset_idx = 0):
    channels = 19 if dataset_idx == 0 else 16
    data = []
    print('--- Building Rhythms ---')
    for i in range(container.shape[0]):
        gamma,beta,alpha,theta,delta = [],[],[],[],[]
        for j in range(channels):
            curr = fdm(container[i,j,:].reshape(sf*sec,1), fs = sf , fc = np.array([4,8,12,35]), data_type='columns',plot_subbands= False)
            gamma.append(curr[:,0])
            beta.append(curr[:,1])
            alpha.append(curr[:,2])
            theta.append(curr[:,3])
            delta.append(curr[:,4])
        data.append(np.array([gamma,beta,alpha,theta,delta]))
    print('--- Done Making Rhythms ---')
    return np.array(data)

def train_test_splitter(data,labels):
    print('--- Train Test Splitting ---')
    x_train,x_test,y_train,y_test = train_test_split(np.array(data),np.array(labels),test_size=0.2,random_state=42)
    y_train = y_train.astype('double')
    y_test = y_test.astype('double')

#     print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
#     print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
    print('--- Done Splitting ---')
    return x_train,x_test,y_train,y_test


def DenseNet(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.relu,kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4),bias_regularizer=regularizers.L2(1e-4),activity_regularizer=regularizers.L2(1e-5))(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        #Creates a sequence of numbers that begins at start and extends by increments of delta up to but not including limit.
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

class MyLayer(layers.Layer):
    def call(self, x):
        return tf.reshape(x,[-1,x.shape[1],x.shape[2]*x.shape[3]])
    
class concatLayer(layers.Layer):
    def call(self,features1,features2,features3,features4,features5):
        return tf.concat([features1,features2,features3,features4,features5],axis=1)


def build_model(dataset_idx,sf,sec,head_size=256,num_heads=4,ff_dim=4,num_transformer_blocks=1,mlp_units=[128],mlp_dropout=0.4,dropout=0.25):
    num_channels = 19 if dataset_idx == 0 else 16
    inputs = layers.Input(shape=(5,num_channels,sf*sec))
    
    # encoded_patches = PatchEncoder(25, 128)(patches)
    x = None
    y1 = Scattering2D(J=2, L=4)(inputs[:,0,:,:])
    y1 = MyLayer()(y1)
    y1 = PatchEncoder(y1.shape[1], 128)(y1)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(y1, head_size, num_heads, ff_dim,  dropout)
    representation1 = layers.LayerNormalization(epsilon=1e-6)(x)
    representation1 = layers.Flatten()(representation1)
    features1 = DenseNet(representation1, hidden_units=[1024,128], dropout_rate=0.5)

    y2 = Scattering2D(J=2, L=4)(inputs[:,1,:,:])
    y2 = MyLayer()(y2)
    y2 = PatchEncoder(y2.shape[1], 128)(y2)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(y2, head_size, num_heads, ff_dim,  dropout)
    representation2 = layers.LayerNormalization(epsilon=1e-6)(x)
    representation2 = layers.Flatten()(representation2)
    features2 = DenseNet(representation2, hidden_units=[1024,128], dropout_rate=0.5)

    y3 = Scattering2D(J=2, L=4)(inputs[:,2,:,:])
    y3 = MyLayer()(y3)
    y3 = PatchEncoder(y3.shape[1], 128)(y3)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(y3, head_size, num_heads, ff_dim,  dropout)
    representation3 = layers.LayerNormalization(epsilon=1e-6)(x)
    representation3 = layers.Flatten()(representation3)
    features3 = DenseNet(representation3, hidden_units=[1024,128], dropout_rate=0.5)

    y4 = Scattering2D(J=2, L=4)(inputs[:,3,:,:])
    y4 = MyLayer()(y4)
    y4 = PatchEncoder(y4.shape[1], 128)(y4)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(y4, head_size, num_heads, ff_dim,  dropout)
    representation4 = layers.LayerNormalization(epsilon=1e-6)(x)
    representation4 = layers.Flatten()(representation4)
    features4 = DenseNet(representation4, hidden_units=[1024,128], dropout_rate=0.5)

    y5 = Scattering2D(J=2, L=4)(inputs[:,4,:,:])
    y5 = MyLayer()(y5)
    y5 = PatchEncoder(y5.shape[1], 128)(y5)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(y5, head_size, num_heads, ff_dim, dropout)
    representation5 = layers.LayerNormalization(epsilon=1e-6)(x)
    representation5 = layers.Flatten()(representation5)
    features5 = DenseNet(representation5, hidden_units=[1024,128], dropout_rate=0.65)

#     features = tf.concat([features1,features2,features3,features4,features5],axis=1)
    features = concatLayer()(features1,features2,features3,features4,features5)
    # features5 = DenseNet(features, hidden_units=[128,64], dropout_rate=0)
    logits = layers.Dense(1, activation = 'sigmoid')(features)
   
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def get_model(dataset_idx,sf,sec):
    
    model = build_model(
        dataset_idx,sf,sec,
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=1,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        metrics=["binary_accuracy"],
    )
#     model.summary()
    return model

def training_plots(history,sec):
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(history.history['binary_accuracy'])
    plt.plot(history.history['val_binary_accuracy'])
    plt.legend(['accuracy','val_accuracy'])
    plt.title('Accuracy Plot')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss','val_loss'])
    plt.title('Loss Plot')
    plt.savefig('./images/window-'+str(sec)+'-training.png')

def get_cm(y_test,x_test,model,sec):
    cm = confusion_matrix(y_test, model.predict(x_test).round())
    print(f"Results for time window {sec} here")
    print(f"Confusion Matrix Array : {cm}")
    print(f"Acuracy : {(cm[0][0]+cm[1][1])/(cm[0][1]+cm[0][0]+cm[1][0]+cm[1][1])*100}")
    print(f"Precision : {(cm[1][1]/(cm[1][1]+cm[0][1]))*100}")
    print(f"Recall : {(cm[1][1]/(cm[1][1]+cm[1][0]))*100}")
    print(f"Specificity : {cm[0][0]/(cm[0][0]+cm[1][0])}")
    print(f"ROC-AUC Score : {roc_auc_score(y_test,  model.predict(x_test))}")
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
#     plt.show()
    plt.savefig('./images/window-'+str(sec)+'-CM.png')

def kruskal_test(healthy,sz):
    return stats.kruskal(healthy,sz)


def get_boxplot(model,x_test,y_test,sec):

# Get the output of the second-to-last layer
    penultimate_layer_model = Model(inputs=model.input, outputs=model.layers[-2].output)
# Now you can use this model to get the activations
    penultimate_output = penultimate_layer_model.predict(x_test)
    weights, biases = model.layers[-1].get_weights()
    raw_logits = np.dot(penultimate_output, weights) + biases
    sz = []
    healthy = []
    for i in range(len(y_test)):
        if y_test[i] == 0:
            healthy.append(raw_logits[i][0])
        else :
            sz.append(raw_logits[i][0])
            H, p = kruskal_test(healthy, sz)
    print(f"Time Window {sec} KW-H statistic: {H:.3f}, p-value: {p:.3e}")
            
    #print(f" Time Window {sec} KW- H-test : {kruskal_test(healthy,sz)}")
           
    fig, ax = plt.subplots(figsize = (9,8))
    box1 = ax.boxplot([sz,healthy],positions = [1,0],widths=0.6, patch_artist=True, labels = ['SZ','Healthy'],sym='.')
    # plt.xlabel('Output without activation')
    plt.ylabel('Deep Feature')
    plt.title('Output Box Plot')
#     plt.show()
    plt.savefig('./images/window-'+str(sec)+'-boxplot.png')
    print(f' Mean and Variance of BoxPlot : {np.array(sz).mean(), np.array(sz).std(), np.array(healthy).mean(), np.array(healthy).std()}')

    
def tsne_plot(model,x_test,y_test,sec):
    # Create intermediate layer model
    intermediate_layer_model = keras.Model(inputs=model.input,
                                           outputs=model.layers[-2].output)

    # Get intermediate layer output for x_train
    intermediate_output = intermediate_layer_model.predict(x_test)

    # Initialize t-SNE model
    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10)

    # Fit and transform the intermediate output using t-SNE
#     print(intermediate_output)
    X_tsne = tsne.fit_transform(np.array(intermediate_output))

    # Plot the t-SNE embeddings using Plotly Express
    fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=y_test)
    fig.update_layout(
        title="t-SNE visualization of Intermediate Layer Embeddings",
        xaxis_title="t-SNE Component 1",
        yaxis_title="t-SNE Component 2",
    )
    fig.show()
#     fig.write_image("./images/fig-window-"+str(sec)+".png")

def roc_plot(model,x_test,y_test,sec):
    if hasattr(model, "decision_function"):
        y_score = model.decision_function(x_test)
    else:
        y_score = model.predict(x_test)

    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
#     plt.show()
    plt.savefig('./images/window-'+str(sec)+'-roc.png')
 
 
def start_training(x_train,x_test,y_train,y_test,dataset_idx,sec):
    print('--- Building Model ---')
    model = get_model(dataset_idx,250 if dataset_idx==0 else 128,sec)
    print('--- Model Procurred ---')
    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]#,PlotLossesKeras()]
    print('--- Starting Training ---')
    history = model.fit(
        x_train,
        y_train,
        epochs=150,
        batch_size=8,
        callbacks=callbacks,
        validation_split = 0.2
    )

    perf = model.evaluate(x_test,y_test)

    print(f" For Time-Window {sec} Test Accuracy : {perf[1]}, Test Loss : {perf[0]}" )

    training_plots(history,sec)

    get_cm(y_test,x_test,model,sec)

    get_boxplot(model,x_test,y_test,sec)
    
    tsne_plot(model,x_test,y_test,sec)
    
    roc_plot(model,x_test,y_test,sec)
    
    return model

