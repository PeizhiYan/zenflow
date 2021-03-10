import numpy as np
import zenflow as zf


from sklearn.datasets import make_moons

data = make_moons(n_samples=1000, shuffle=True, noise=0.05, random_state=None) # Gaussian noise std-dev = 0.05
X = data[0]
Y = np.reshape(data[1], [len(data[1]), 1])



model = zf.sequential(loss_function=zf.mean_squared_loss)


print(model)


fc1 = zf.dense_layer(in_dim=2, out_dim=10, activation=zf.relu, initialize='random_uniform')
model.add_layer(fc1)

fc2 = zf.dense_layer(in_dim=10, out_dim=15, activation=zf.relu, initialize='random_uniform')
model.add_layer(fc2)

fc3 = zf.dense_layer(in_dim=15, out_dim=1, activation=zf.sigmoid, initialize='random_uniform')
model.add_layer(fc3)

model.summary()

print(model.model_loss(X, Y))

for epoch in range(30):
    model.auto_grad(X, Y)
    model.update_step(learning_rate=1e-3)
    l = model.model_loss(X, Y)
    if epoch % 1 == 0:
        print('epoch ', epoch, ' loss ', l)


Y_pred = model.predict(X, return_score=False)

print()

def accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()

print("Train accuracy: ", accuracy(Y_pred, Y))

