import configparser

class ConfigClassification:

    cfg_file = "/Users/konstantinos.mammas/Documents/examples/classification_examples/main/classification.cfg"

    def parse(self):
        self.parser = configparser.ConfigParser()
        self.parser.read(self.cfg_file)

    def get(self,section,key,defaultValue):
        if not hasattr(self, 'parser'):
            self.parse()
        val = None
        try:
            val = self.parser.get(section,key)
        except:
            print("FAILURE getting prop: "+key+", return default")
        return val if val is not None else defaultValue
