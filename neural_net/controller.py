from typing import ChainMap
from neural_net.network import Network
import math
import sys
class Controller():
    def __init__(self):
        self.networks = []

        
    def create_networks(self,num_of_networks,num_of_layers_in_between,num_neurons_per_layer,num_of_initial_neurons,num_of_final_neurons):

         for i in range(num_of_networks):
            self.networks.append(Network(num_of_initial_neurons,'initial'))
            for j in range(num_of_layers_in_between): 
                self.networks[i].add_layer(num_neurons_per_layer)
            self.networks[i].add_layer(num_of_final_neurons)


    def change_networks_neurons(self,amount_of_changeW,amount_of_changeB):
        for network in self.networks:
            network.change_neuron_wb(amount_of_changeW,amount_of_changeB)

            
    def calculate_values(self,initial_values):
        for network in self.networks:
            network.give_initial_values(initial_values)
            network.calculate_values()

    def set_networks_wb(self,wieghts,biases):
        for network in self.networks:
            network.set_neuron_wb(wieghts,biases)
        
    def show_networks_values(self):
        return [network.show_final_values() for network in self.networks]



sys.path.append(".")







