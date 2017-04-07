#/usr/bin/python
#weka.py

__author__ = "H. Nathan Rude"


class Weka(object):
    '''
   
    @attribute <attribute-name> <datatype>
    Note - last attribute should be <class> and <possible class types>

    Ex.
    @ATTRIBUTE sepallength  NUMERIC
    @ATTRIBUTE sepalwidth   NUMERIC
    @ATTRIBUTE petallength  NUMERIC
    @ATTRIBUTE petalwidth   NUMERIC
    @ATTRIBUTE class        {Iris-setosa,Iris-versicolor,Iris-virginica}

    @relation <string type>
    Note - relation is the type to be classified

    Ex.
    @RELATION iris

    @data <data 1> <data 2> ... <data n> <optional trainer class type>
    Note - @attribute and @data are list matrices

    Ex.
    @DATA
    5.1,3.5,1.4,0.2,Iris-setosa
    4.9,3.0,1.4,0.2,Iris-setosa
    4.7,3.2,1.3,0.2,Iris-setosa
    4.6,3.1,1.5,0.2,Iris-setosa
    5.0,3.6,1.4,0.2,Iris-setosa
    5.4,3.9,1.7,0.4,Iris-setosa
    4.6,3.4,1.4,0.3,Iris-setosa
    5.0,3.4,1.5,0.2,Iris-setosa
    4.4,2.9,1.4,0.2,Iris-setosa
    4.9,3.1,1.5,0.1,Iris-setosa
    '''

    def __init__(self, relation=""):
        self.dataTag = "@DATA"
        self.relationTag = "@RELATION"
        self.attributeTag = "@ATTRIBUTE"

        #Data type check
        self.attrList = []
        self.relation = relation
        self.data = []

    def write_file(self, filename):
        savefile = open(filename, "w")

        #Write the relation tag and value to the file
        relationLine = self.relationTag + "\t" + self.relation + "\n\n"
        savefile.write(relationLine)

        #Loop through the attributes and print them out
        for attribute in self.attrList:
            attributeLine = self.attributeTag + "\t"

            for attr in attribute:
                attributeLine += attr + '\t'

            savefile.write(attributeLine + '\n')

        #Print out the data tag
        savefile.write('\n' + self.dataTag + '\n')

        #Print out the comma seperated data
        for data in self.data:
            savefile.write(data + '\n')

        savefile.close()

    '''
    append a new list of data to the weka data set
    '''
    def add_data(self, dataSet):
        self.data.append(dataSet)

    def add_attribute(self, name, attr_type):
        attrib = [name, attr_type]
        self.attrList.append(attrib)
        # isEmpty = False
        # popAtr = None

        # if len(self.attrList) == 0:
        #     isEmpty = True

        # attribute = []
        # attribute.append(name)
        # attribute.append(type)

        # if not isEmpty:
        #     popAtr = self.attrList.pop()

        # self.attrList.append(attribute)

        # if not isEmpty:
        #     self.attrList.append(popAtr)

    def add_relation(self, newRelation):
        self.relation = newRelation

#Test out the weka class

if __name__ == "__main__":
    relation = "gender"
    attributes = [["length", "NUMERIC"],
                ["width", "NUMERIC"],
                ["class", "{male, female}"]]
    data = [["5", "4", "male"],
            ["4", "3", "female"],
            ["1", "7", "female"]]
    w = Weka(attributes, ["a", "adf"], data)
    #w.writeARFF("info.arff")

    #w.addAttribute("name", "asdf")
    #w.writeARFF("info1.arff")

    we = Weka()
    we.write_file("info.arff")
    we.add_attribute("name", "asdf")
    we.write_file("info1.arff")
