import os

import lasagne

# Available network types
networktypes = ["FullyConnected"]

# Available nonlinearities
nonlinearities = {
    "sigmoid": lasagne.nonlinearities.sigmoid,
    "softmax": lasagne.nonlinearities.softmax
}

class NetworkSettings(object):

    def __init__(self, networktype, layersizes, hlnonlinearity, olnonlinearity, folder, filename, inputshape=(28, 28),
                 output_size=10):

        self.networktype = networktype
        self.layersizes = layersizes
        self.hlnonlinearity = hlnonlinearity
        self.olnonlinearity = olnonlinearity
        self.folder = folder
        self.filename = filename
        self.inputshape = inputshape
        self.output_size = output_size


    def get_hlnonlinearity(self):
        """ Get the actual lasagne nonlinearity as opposed to the string used in saving"""
        return nonlinearities[self.__hlnonlinearity]

    def get_olnonlinearity(self):
        """ Get the actual lasagne nonlinearity as opposed to the string used in saving"""
        return nonlinearities[self.__olnonlinearity]

    @property
    def filename(self):
        return self.__filename

    @filename.setter
    def filename(self, filename):
        assert '.pkl' not in filename, "'.pkl' extension should not be added by the user but is added automatically by the program"
        self.__filename = filename

    @property
    def networktype(self):
        return self.__networktype

    @networktype.setter
    def networktype(self, networktype):
        assert networktype in networktypes, "Networktype {} not supported".format(networktype)
        self.__networktype = networktype

    @property
    def hlnonlinearity(self):
        return self.__hlnonlinearity

    @hlnonlinearity.setter
    def hlnonlinearity(self, hlnonlinearity):
        assert hlnonlinearity in nonlinearities, "hlnonlinearity {} not supported".format(hlnonlinearity)
        self.__hlnonlinearity = hlnonlinearity

    @property
    def olnonlinearity(self):
        return self.__olnonlinearity

    @olnonlinearity.setter
    def olnonlinearity(self, olnonlinearity):
        assert olnonlinearity in nonlinearities, "olnonlinearity {} not supported".format(olnonlinearity)
        self.__olnonlinearity = olnonlinearity

    def get_savename(self):
        """os join the folder and filename"""

        return os.path.join(self.folder, self.filename)

    def __str__(self):
        return "Neural network settings object\n \
            \tnetworktype: {networktype}\n \
            \tlayersizes: {layersizes}\n \
            \thlnonlinearity: {hlnonlinearity}\n \
            \tolnonlinearity: {olnonlinearity}\n \
            \tsave path: {path}\n \
            ----------------------------------".format(
                networktype=self.networktype,
                layersizes=self.layersizes,
                hlnonlinearity=self.hlnonlinearity,
                olnonlinearity=self.olnonlinearity,
                path=self.get_savename()
            )


if __name__ == "__main__":
    print("This should work:")
    settings = NetworkSettings(
        networktype="FullyConnected",
        layersizes=(50,),
        hlnonlinearity="sigmoid",
        olnonlinearity="softmax",
        folder="blaat",
        filename="fiets"
    )
    print(settings)
    print("And this should not work:")
    settings = NetworkSettings(
        networktype="FullydsConnected",
        layersizes=(50,),
        hlnonlinearity="sigmoid",
        olnonlinearity="softmax",
        folder="blaat",
        filename="fiets"
    )
    print(settings)
