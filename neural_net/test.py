from random import randint, random
import pandas as pd
from network import Network
import matplotlib.pyplot as plt

num_of_inputs = 4
num_of_outputs = 3
num_of_layers_in_the_network = 4
neurons_in_layer_distribution = [4,4,4,4]
learning_reductor = 0.0001


species_id = {
    'Iris-setosa':0,
    'Iris-versicolor':1,
    'Iris-virginica':2
}




data =  pd.read_csv('iris.data', sep=",",header=None)
net = Network(num_of_inputs,num_of_layers_in_the_network,num_of_outputs,neurons_in_layer_distribution,learning_reductor)

scores=[]

for i in range(150):
    index = randint(0,149)
    input_values = [data[0][index],data[1][index],data[2][index],data[3][index]]
    net.give_initial_values(input_values)
    net.calculate_values()
    if net.network[-1][0].value > 10**14:
        print("exploded")
        break
    expected_values = [0,0,0]
    expected_values[species_id[data[4][index]]] += 1
    score = 1 - net.network[-1][species_id[data[4][index]]].value
    scores.append(score)
    print(f"Value got:{[neuron.value for neuron in net.network[-1]]}  Value expexted: {expected_values} Score:{score} ")

    net.gradient_descent_w_tweaking(expected_values)

plt.plot(scores)
plt.show()


score = 0
for i in range(100):
    index = randint(0,149)
    input_values = [data[0][index],data[1][index],data[2][index],data[3][index]]
    net.give_initial_values(input_values)
    net.calculate_values()
    predictions = [neuron.value for neuron in net.network[-1]]
    if predictions.index(max(predictions)) == species_id[data[4][index]]:
        score +=1
print(score)