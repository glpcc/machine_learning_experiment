
from neuron import Neuron
import random
import sys
import matplotlib.pyplot as plt
import math
# NETWORK CLASS
class Network():
    def __init__(self,num_of_first_neurons,num_of_layers_in_between,num_of_final_neurons,neurons_in_each_layer):
        # Create the first layer
        self.network = [[Neuron([]) for i in range(num_of_first_neurons)]]
        for i in range(num_of_layers_in_between):
            new_layer = [Neuron(self.network[-1]) for i in range(neurons_in_each_layer[i])]
            self.network.append(new_layer)
        new_layer = [Neuron(self.network[-1]) for i in range(num_of_final_neurons)]
        self.network.append(new_layer)



    def give_initial_values(self,values):
        # Clean all values
        for layer in self.network:
            for neuron in layer:
                neuron.value = 0

        # Give the values to the firsts neurons
        for i in range(len(values)):
            self.network[0][i].value = values[i] 

    def calculate_values(self):
        for layer in self.network:
            for neuron in layer:
                neuron.calculate_value()

    def show_final_values(self):
        # print('the values are: \n')
        # for neuron in self.network[-1]:
        #     print(str(neuron.id) + ':' + str(neuron.value))
            
        return [neuron.value for neuron in self.network[-1]]
    
    def show_all_values(self):
        for i in range(len(self.network)):
            print(f"Layer {i} values")
            for neuron in self.network[i]:
                print(str(neuron.id) + ':' + str(neuron.value))

    def change_neuron_wb(self,amount_of_changeW,amount_of_changeB):
        for i in range(len(self.network) - 1):
            for neuron in self.network[i+1]:
                new_weights = [random.uniform(neuron.weights[i]-amount_of_changeW,neuron.weights[i]+amount_of_changeW) for i in range(len(neuron.weights))]
                new_biases = [random.uniform(neuron.biases[i]-amount_of_changeB,neuron.biases[i]+amount_of_changeB) for i in range(len(neuron.biases))]
                neuron.change_weights_and_biases(new_weights,new_biases)
    def set_neuron_wb(self,network_weights,network_biases):
        for layer in range(len(self.network)):
            for neuron in range(len(self.network[layer])):
                self.network[layer][neuron].weights = network_weights[layer][neuron]
                self.network[layer][neuron].biases = network_biases[layer][neuron]

    def get_network_wb(self):
        network_weights = []
        network_biases = []
        for layer in self.network:
            layer_weights = []
            layer_biases = []
            for neuron in layer:
                neuron_weights = neuron.weights
                neuron_biases = neuron.biases
                layer_weights.append(neuron_weights)
                layer_biases.append(neuron_biases)
            network_weights.append(layer_weights)
            network_biases.append(layer_biases)
        return network_weights,network_biases

    def gradient_descent_w_tweaking(self,expected_values):
        # TODO Change activations functions in neurons to relu not tanH
        # PD: first implementation will be for only one output
        # For loop for backpropagation
        # the derivative of the cost with respect of a weight will be:
        #               C'(W_0) = Neuron_0_error
        # The neuron error is calculated by the formula:
        #               Neuron_error = Prev_neuron_error*Connection_weight*/Derivated_activation_function(Neuron_value)/ in this case there is none by now
        # And for the first neuron(last in the network):
        #               Neuron_error = (Expected_value-neuron_value)*/Derivated_activation_function(Neuron_value)/  in this case there is none by now

        # Clean all neurons errors
        for layer in self.network:
            for neuron in layer:
                neuron.error = 0

        #Calculate the error for the last neurons
        for neuron in self.network[-1]:
            prediction_error = (expected_values[self.network[-1].index(neuron)]-neuron.value)#*self.derivated_activation_function(self.network[-1][0].value)
            neuron.error = prediction_error
        
        for i in range(len(self.network)):
            for neuron in self.network[-(i+1)]:
                for j in range(len(neuron.weights)):
                    connection_error = neuron.error*neuron.weights[j]#*self.derivated_activation_function(neuron.value)
                    cost_gradient = neuron.error*neuron.prev_connections[j].value
                    #TODO make sure that in same layer neurons have different weights when learning
                    #print(f"Layer:{i} Neuron:{self.network[-(i+1)].index(neuron)} Weight:{j}\n Neuron Error:{neuron.error} Connection Error:{connection_error} Prev neuron error:{neuron.prev_connections[j].error} \n")
                    neuron.prev_connections[j].error = neuron.prev_connections[j].error + connection_error
                    neuron.weights[j]+=cost_gradient*0.0001
            
                
    def derivated_activation_function(z):
        #in this case im using Relu
        return 1 if z>0 else 0



    def cost_function(expected_value,given_value):
        #it´s derivative is (expected_value-given_value)
        return 0.5*((expected_value-given_value)**2)

sys.path.append(".")
net = Network(2,0,1,[])

def cost_function(expected_value,given_value):
    #it´s derivative is (expected_value-given_value)
    return 0.5*((expected_value-given_value)**2)


scores = []
for i in range(500):
    values = [random.randint(0,100),random.randint(0,100)]
    net.give_initial_values(values)
    net.calculate_values()
    if net.network[-1][0].value > 10**14:
        print("exploded")
        break
    values_expected = [values[0]*5]
    score = [value - net.network[-1][values_expected.index(value)].value for value in values_expected]
    scores.append(score)
    print(f"Value got:{[neuron.value for neuron in net.network[-1]]}  Value expexted: {values_expected} Score:{score} final neuron weights:{net.network[-1][0].weights}")

    net.gradient_descent_w_tweaking(values_expected)

plt.plot([abs(i[0]) for i in scores])
plt.show()