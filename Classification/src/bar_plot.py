"""
name   : bar_plot
purpose: create bar plots using a number of user defined parameters
name   : Kostas Mammas <mammaskon@gmail.com>
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Bar_Plot:

    def __init__(self):
        self.dataset_object = None
        self.xaxis = None
        self.yaxis = None
        self.xaxisname = None
        self.yaxisname = None
        self.title = None
        self.color = None
        self.group = None
        self.background_color = False
        self.rotate_xaxis = None
        self.text_bars = False
        self.save_path = None
        self.output_name = None

    def gen_bar_plot(self):

        # Generate bar plot
        plt.figure()
        p = sns.barplot(x=self.dataset_object[self.xaxis],
                        y=self.dataset_object[self.yaxis],
                        color = self.color)
        # Define names of axes
        p.set(xlabel= self.xaxisname,
              ylabel= self.yaxisname
        )
        # Set title
        p.set_title(self.title)

        # Rotate x axis labels
        if (self.rotate_xaxis is not None):
            p.set_xticklabels(self.dataset_object[self.xaxis],
                              rotation = self.rotate_xaxis)

        # Add text labels on bars
        if (self.text_bars == True):
            for l in p.patches:
                spend = l.get_height()
                p.text(l.get_x() + l.get_width() / 2., spend + 3, '{:1.0f}'.format(spend), ha="center")

        # Save plot
        if (self.save_path is not None):
            p.figure.savefig(self.save_path + self.output_name)

        return p

    def gen_stacked_bar_plot(self):

        # Generate bar plot
        plt.figure()
        p = sns.barplot(x=self.dataset_object[self.xaxis],
                        y=self.dataset_object[self.yaxis],
                        hue=self.dataset_object[self.group])

        # Define names of axes
        p.set(xlabel= self.xaxisname,
              ylabel= self.yaxisname
        )
        # Set title
        p.set_title(self.title)

        # Rotate x axis labels
        if (self.rotate_xaxis is not None):
            p.set_xticklabels(self.dataset_object[self.xaxis],
                              rotation = self.rotate_xaxis)

        # Save plot
        if (self.save_path is not None):
            p.figure.savefig(self.save_path + self.output_name)

        return p


