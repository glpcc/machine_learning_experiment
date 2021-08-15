
from neuron import Neuron
import random
import sys
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

    def gradient_descent_w_tweaking(self,expected_value):
        # TODO Change activations functions in neurons to relu not tanH
        # PD: first implementation will be for only one output
        # For loop for backpropagation
        # the derivative of the cost with respect of a weight will be:
        #               C'(W_0) = Neuron_0_error * Neuron_0_input
        # The neuron error is calculated by the formula:
        #               Neuron_error = Prev_neuron_error*Connection_weight*/Derivated_activation_function(Neuron_value)/rethink this if failing
        # And for the first neuron(last in the network):
        #               Neuron_error = (Expected_value-neuron_value)*/Derivated_activation_function(Neuron_value)/rethink this if failing

        # Clean all neurons errors
        for layer in self.network:
            for neuron in layer:
                neuron.error = 0

        prediction_error = (expected_value-self.network[-1][0].value)*self.derivated_activation_function(self.network[-1][0].value)
        self.network[-1][0].error = prediction_error
        for i in range(len(self.network)):
            for neuron in self.network[-(i+1)]:
                for j in range(len(neuron.weights)):
                    connection_error = neuron.error*neuron.weights[j]*self.derivated_activation_function(neuron.value)
                    cost_gradient = connection_error*neuron.prev_connections[j].value
                    neuron.prev_connections[j].error += connection_error
                    neuron.weights[j]-=cost_gradient
            
                
    def derivated_activation_function(z):
        #in this case im using Relu
        return 1 if z>0 else 0


    def cost_function(expected_value,given_value):
        #it´s derivative is (expected_value-given_value)
        return 0.5*((expected_value-given_value)**2)

sys.path.append(".")
