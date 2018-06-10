import pandas as pd

class Data_Manipulator:

    def __init__(self, data_object, variable_name):
        self.data_object = data_object
        self.variable_name = variable_name

    def get_map(self):

        # Convert to character
        self.data_object[self.variable_name] = self.data_object[self.variable_name].astype(str)

        # Obtain the unique levels
        unique_levels = self.data_object[self.variable_name].unique()

        # Create a mapping dictionary
        mapping = pd.DataFrame({self.variable_name: unique_levels
                                })

        mapping_wide = pd.get_dummies(mapping)
        mapping = mapping.join(mapping_wide)

        # Convert to string
        return mapping


    def map_level(self):
        """
        :param data_object: input data
        :param variable_name: the name of the feature to be transformed
        :return: dataset with updated features
        """

        # Create column with the integer levels of the feature
        self.data_object = self.data_object.merge(self.get_map(), left_on=self.variable_name,
                                        right_on=self.variable_name,
                                        how='left')

        # Drop original feature
        self.data_object = self.data_object.drop(self.variable_name, 1)

        return self.data_object





