import numpy as np
import wandb

class MLP:
    def __init__(self, input_size, output_size,regression, multilabel = False, hidden_layers=1, neurons_per_layer=[16], activation='relu',
                optimizer='sgd', learning_rate=0.01, batch_size=32, epochs=100):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.weights,self.biases = self.InitializeWeightsAndBiases()
        self.regression = regression
        self.multilabel = multilabel

    def InitializeWeightsAndBiases(self):
        weights = []
        biases = []
        weights.append(np.random.randn(self.input_size, self.neurons_per_layer[0]) * np.sqrt(2 / self.input_size))
        biases.append(np.random.randn(1, self.neurons_per_layer[0]) * 0.01)
        # Hidden layers
        for i in range(1, self.hidden_layers):
            weights.append(np.random.randn(self.neurons_per_layer[i - 1], self.neurons_per_layer[i]) * np.sqrt(2 / self.neurons_per_layer[i-1]))
            biases.append(np.random.randn(1, self.neurons_per_layer[i]) * 0.01)
        # Last hidden layer to output layer
        weights.append(np.random.randn(self.neurons_per_layer[-1], self.output_size) * np.sqrt(2 / self.neurons_per_layer[-1]))
        biases.append(np.random.randn(1, self.output_size) * 0.01)
        return weights,biases

    def AlterLearningRate(self, lr):
        self.learning_rate = lr

    def AlterActivationFunction(self, activation):
        self.activation = activation

    def AlterOptimizer(self, optimizer,batch_size=32):
        self.optimizer = optimizer
        if(optimizer == 'mbgd'):
            self.batch_size = batch_size

    def get_params(self):
        return {
            'learning_rate': self.learning_rate,
            'activation': self.activation,
            'optimizer': self.optimizer,
            'hidden_layers': self.hidden_layers,
            'neurons_per_layer': self.neurons_per_layer,
            'batch_size': self.batch_size,
            'epochs': self.epochs
        }

    def f(self, Z, activation):
        if activation == 'relu':
            return np.maximum(0, Z)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation == 'tanh':
            return np.tanh(Z)
        else:
            return Z

    def softmax(self,Z):
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1,keepdims=True)

    def ForwardPropagation(self, X):
        A = X
        activations = [A]
        z_values = [X]
        ran = len(self.weights)-1

        for i in range(ran):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A = self.f(Z, self.activation)
            activations.append(A)
            z_values.append(Z)
        Z = np.dot(A, self.weights[len(self.weights)-1]) + self.biases[len(self.weights)-1]
        if self.regression == False:
            A = self.softmax(Z)
            activations.append(A)
            z_values.append(Z)
        elif self.regression == True and self.multilabel == False:
            activations.append(Z)
            z_values.append(Z)
        elif self.regression == True and self.multilabel == True:
            A = self.f(Z, 'sigmoid')
            activations.append(A)
            z_values.append(Z)
        self.z_values = z_values
        return activations

    def reg_metrics(self,y_pred,y):
        n = len(y)

        mse = np.sum((y_pred-y)**2)/n

        rmse = np.sqrt(mse)

        mean_y = np.mean(y)
        r2 = 1 - (np.sum((y - mean_y) ** 2) / np.sum((y - y_pred) ** 2))
        
        return mse,rmse,r2

    def log_accuracy(self,y_train_pred,y_train,y_val_pred,y_val):
        accuracy1 = 0
        accuracy2 = 0
        for i in range(len(y_train_pred)):
            accuracy1 += (y_train_pred[i] == y_train[i])
        accuracy1 = accuracy1/len(y_train) * 100
        for i in range(len(y_val_pred)):
            accuracy2 += (y_val_pred[i] == y_val[i])
        accuracy2 = accuracy2/len(y_val) * 100
        if self.wb:
            wandb.log({"Accuracy_Train":accuracy1, "Accuracy_Val":accuracy2})
        return accuracy2
        
    def BackPropagation(self, activations, Y):
        m = Y.shape[0]
        dZ =  -(Y-activations[-1])
        gradients_W = []
        gradients_b = []
        for i in range(len(self.weights)-1,-1,-1):
            dW = np.dot(activations[i].T, dZ) / m
            db = np.sum(dZ, axis=0, keepdims=True) / m

            gradients_W.append(dW)
            gradients_b.append(db)

            f_dash = self.f_dash(self.z_values[i],activations[i],self.activation)
            dZ = np.dot(dZ, self.weights[i].T) * f_dash
        return gradients_W[::-1], gradients_b[::-1]

    def f_dash(self,Z,activations,activation):
        if activation == 'relu':
            return self.ReLuDerivative(Z)
        elif activation == 'sigmoid':
            return self.SigmoidDerivative(activations)
        elif activation == 'tanh':
            return self.TanhDerivative(activations)
        else:
            return 1

    def ReLuDerivative(self,x):
        return np.where(x > 0, 1, 0)

    def SigmoidDerivative(self,x):
        return x * (1-x)

    def TanhDerivative(self,x):
        return 1 - x**2

    def UpdateWeightsAndBiases(self, gradients_w, gradients_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]

    def LossFunction(self, pred, Y):
        if self.regression == False or self.multilabel == True:
            m = Y.shape[0]
            cross_entropy = -np.log(pred[range(m), Y.argmax(axis=1)])
            loss = np.sum(cross_entropy) / m
        else:
            loss = np.mean((pred - Y) ** 2)

        return loss

    def fit(self, X_train, Y_train, X_val=None, Y_val=None, early_stopping=True, patience=7, wb = False):
        self.wb=wb
        best_val_loss = float('inf')
        patience_counter = 0

        self.val_losses = []
        for epoch in range(self.epochs):
            self.Train(X_train,Y_train)
            if early_stopping and X_val is not None and Y_val is not None:
                predictions_val = self.predict(X_val)
                predictions_train = self.predict(X_train)
                validation_loss = self.LossFunction(predictions_val, Y_val)
                training_loss = self.LossFunction(predictions_train, Y_train)
                if validation_loss < best_val_loss:
                    best_val_loss = validation_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                if self.wb==True:
                    if self.regression:
                        # print(validation_loss)
                        wandb.log({"epoch": epoch + 1, "MSE": validation_loss})
                    else:
                        wandb.log({"epoch": epoch + 1, "train_loss": training_loss, "val_loss":validation_loss})
            if early_stopping and X_val is not None and Y_val is not None:
                self.val_losses.append(validation_loss)
        return self.val_losses
    def predict(self, X):
        activations = self.ForwardPropagation(X)
        return activations[-1]

    def predictMultilabel(self,X):
        last = self.predict(X)
        last = np.where(last>0.5,1,0)
        return last

    def Train(self,X,Y):
        if self.optimizer == 'sgd':
            self.SGD(X,Y)
        elif self.optimizer == 'bgd':
            self.BGD(X,Y)
        elif self.optimizer == 'mbgd':
            self.MBGD(X,Y,self.batch_size)

    def SGD(self,X,Y):
        for i in range(X.shape[0]):
            X_single = X[i:i+1, :] 
            Y_single = Y[i:i+1, :] 

            activations = self.ForwardPropagation(X_single)
            gradients_w,gradients_b = self.BackPropagation(activations, Y_single)
            self.UpdateWeightsAndBiases(gradients_w,gradients_b)

    def BGD(self, X, Y):
        activations = self.ForwardPropagation(X)
        gradients_w,gradients_b = self.BackPropagation(activations, Y)
        self.UpdateWeightsAndBiases(gradients_w,gradients_b)

    def MBGD(self, X, Y, batch_size):
        m = X.shape[0]
        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            Y_batch = Y[i:i+batch_size]
            activations = self.ForwardPropagation(X_batch)
            gradients_w,gradients_b = self.BackPropagation(activations, Y_batch)
            self.UpdateWeightsAndBiases(gradients_w,gradients_b)

    def GradientCheck(mlp, X, Y, epsilon=1e-9):
        activations = mlp.ForwardPropagation(X)
        A_G_w, A_G_b = mlp.BackPropagation(activations, Y)

        N_G_w = []
        N_G_b = []

        for i in range(len(mlp.weights)):
            num_grad_w = np.zeros(mlp.weights[i].shape)
            for j in range(mlp.weights[i].shape[0]):
                for k in range(mlp.weights[i].shape[1]):
                    W_plus = np.copy(mlp.weights[i])
                    W_minus = np.copy(mlp.weights[i])
                    W_plus[j, k] += epsilon
                    W_minus[j, k] -= epsilon

                    mlp.weights[i] = W_plus
                    loss_plus = mlp.LossFunction(mlp.ForwardPropagation(X)[-1], Y)

                    mlp.weights[i] = W_minus
                    loss_minus = mlp.LossFunction(mlp.ForwardPropagation(X)[-1], Y)

                    num_grad_w[j, k] = (loss_plus - loss_minus) / (2 * epsilon)

            mlp.weights[i] = np.copy(mlp.weights[i])
            N_G_w.append(num_grad_w)

            num_grad_b = np.zeros(mlp.biases[i].shape)
            for j in range(mlp.biases[i].shape[1]):
                B_plus = np.copy(mlp.biases[i])
                B_minus = np.copy(mlp.biases[i])
                B_plus[0, j] += epsilon
                B_minus[0, j] -= epsilon

                mlp.biases[i] = B_plus
                loss_plus = mlp.LossFunction(mlp.ForwardPropagation(X)[-1], Y)

                mlp.biases[i] = B_minus
                loss_minus = mlp.LossFunction(mlp.ForwardPropagation(X)[-1], Y)

                num_grad_b[0, j] = (loss_plus - loss_minus) / (2 * epsilon)

            mlp.biases[i] = np.copy(mlp.biases[i])
            N_G_b.append(num_grad_b)

        for i in range(len(A_G_w)):
            diff_w = np.linalg.norm(A_G_w[i] - N_G_w[i]) / (np.linalg.norm(A_G_w[i]) + np.linalg.norm(N_G_w[i]))
            diff_b = np.linalg.norm(A_G_b[i] - N_G_b[i]) / (np.linalg.norm(A_G_b[i]) + np.linalg.norm(N_G_b[i]))

            print(f"Layer {i+1} - Weight gradient difference: {diff_w}")
            print(f"Layer {i+1} - Bias gradient difference: {diff_b}")

            if diff_w > 1e-7:
                print(f"Gradient Check Failed for Weights at Layer {i+1}")
            else:
                print(f"Gradient Check Passed for Weights at Layer {i+1}")
            
            if diff_b > 1e-7:
                print(f"Gradient Check Failed for Biases at Layer {i+1}")
            else:
                print(f"Gradient Check Passed for Biases at Layer {i+1}")

    def getAllActivations(self,X):
        activations = self.ForwardPropagation(X)
        return activations