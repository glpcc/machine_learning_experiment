
from neural_net.neuron import Neuron
import random
import sys
import math
# NETWORK CLASS
class Network():
    def __init__(self,num_of_first_neurons,first_neuron_id):
        self.network = [[Neuron([],first_neuron_id) for i in range(num_of_first_neurons)]]
        self.layers = 0

    def add_layer(self,num_of_neurons):
        new_layer = [Neuron(self.network[self.layers],i) for i in range(num_of_neurons)]
        self.network.append(new_layer)
        self.layers += 1


    def give_initial_values(self,values):
        # Clean all values
        for layer in self.network:
            for neuron in layer:
                neuron.value = 0


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
            
        return [math.tanh(neuron.value*0.2) for neuron in self.network[-1]]
    
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
    
sys.path.append(".")