##Decision Tree starts here...
import math
from sklearn.linear_model import LogisticRegression
import pandas as pd
import sys
features =[]
targets=[]
dataFrame = pd.read_csv(sys.argv[1])
for col in dataFrame.columns[1:]:##feature names extracted.
    features.append(col)
features=features[:-1]
data =[]
for sample in dataFrame.values:
    buffer=[]
    for val in sample[1:]:
        buffer.append(val)
    data.append(buffer)
def findAttributeValues(data):##finding values of attribute.
    featureValues=[]
    for feature in range(len(features)):
        buffer=[]
        buffer.append("For "+features[feature])
        for example in data:
            if not example[feature] in buffer:
                buffer.append(example[feature])
        featureValues.append(buffer.copy())
        buffer.clear()
    return featureValues
def entropyOfAttribute(data,feature,featureValues,header):##finding entropy of attributes for all it's values
    sum = 0
    for val in featureValues[1:]:
        sum=sum+(entropy(data,feature,val,header))
    return sum
def entropy(data,feature,featureValue,header):##classic entropy function with given parameter
    index = header.index(feature)
    pos = 0
    neg = 0
    for sample in data:
        if sample[index] == featureValue:
            if sample[-1] == 0:
                neg=neg+1
            else:
                pos=pos+1
    x=0
    y=0
    if pos !=0:
        x = -(pos/(pos+neg))*math.log(pos/(pos+neg),2)
    if neg !=0:
        y= -(neg/(pos+neg))*math.log(neg/(pos+neg),2)
    return ((pos+neg)/len(data))*(x+y)
def splitData(data,feature,featureValue,header):#split data with given attribute,attributeValuepair.
    splittedData=[]
    index = header.index(feature)
    for sample in data:
        if sample[index]==featureValue:
            splittedData.append(sample)
    return splittedData
featureValues=findAttributeValues(data)
def gain(feature,featureValue,attribute,data,header,featureValues):#classic gain function
    index = header.index(feature)
    splittedData=splitData(data,feature,featureValue,header)
    entropyS= entropy(splittedData,feature,featureValue,header)
    sum = entropyOfAttribute(splittedData,attribute,featureValues[index],header)
    return entropyS-sum
def findRoot(data,features,featureValues,header):##finding root with gain(S,A)
    pos = 0
    neg = 0
    for sample in data:
        if sample[-1]==1:
            pos = pos+1
        else:
            neg = neg +1
    x = 0
    y = 0
    if pos !=0:
        x = - (pos/(pos+neg))*math.log(pos/(pos+neg),2)
    if neg !=0:
        y =- (neg/(pos+neg))*math.log(neg/(pos+neg),2)
    entropyS=x+y
    buffer = []
    for feature in features:
        index = header.index(feature)
        sum = entropyOfAttribute(data,feature,featureValues[index],features)
        buffer.append(entropyS-sum)
    return header[buffer.index(max(buffer))]
class Node(object):##DecisionNodes
    def __init__(self, dat):
        self.data = dat
        self.children = []
        self.branches= []
    def add_branch(self,branch):##adding branches
        self.branches.append(branch)
    def add_child(self,obj):##addingChildNodes
        self.children.append(obj)
    def getData(self):
        return self.data
    def getChildren(self):
        return self.children
    def getBranches(self):
        return self.branches
predictedValues=[]
class logNode(object):#Logistic classifier node .
    def __init__(self):
        self.logisticRegr = LogisticRegression()
        self.branches=[]
    def fit(self,values):#fitting values with given subset
        x_train=[]
        y_train=[]
        for value in values:
            x_train.append(value[:-1])
            y_train.append(value[-1])
        self.logisticRegr.fit(x_train,y_train)
    def predict(self,values):#predictingGivin value
        buffer=[]
        buffer.append(values)
        return self.logisticRegr.predict(buffer)
class leafNode(object):#Node that holds Yes or No value for leaf
    def __init__(self,deger):
        self.data=deger
    def set(self,val):
        self.data=val
    def getData(self):
        return self.data
def allPos(examples):#iff all targets in dataset pos.
    for sample in examples:
        if sample[-1]==0:
            return False
    return True
def allNeg(examples):#if all targets in dataset neg
    for sample in examples:
        if sample[-1]==1:
            return False
    return True
def findCommonTarget(exaples):##If there is no attribute finding most common target value
    pos=0
    for sample in exaples:
        if sample[-1]==1:
            pos=pos+1
    if pos>=len(exaples)/2:
        return 1
    else:
        return 0
def findDataset(branchValues,attributes,data,header):##finding subset of dataset with given attribute value
    buffer=[]
    for sample in data:
        bool = True
        for i in range(len(branchValues)):
            attrIndex=header.index(attributes[i])
            value=branchValues[i]
            if sample[attrIndex]!=value:
                bool=False
        if bool:
            buffer.append(sample)
    return buffer
lol=features.copy()
def ID3(examples,Targetattr,features,depth,flag,branchValues,featureValues,featureValue=None,parents=None,rootAttriubte=None):
    root = Node
    result = all(elem == features[0] for elem in features)
    if allPos(examples):##if splited data's all of targets  are pos return leafNode with value 1 (yes)
        node = leafNode(1)
        return node
    elif allNeg(examples):##if splited data's all of targets  are pos return leafNode with value 0 (no)
        node = leafNode(0)
        return node
    elif result:##    If number of predicting attributes is empty, then Return the single node tree Root,with label = most common value of the target attribute in the examples.
        target = findCommonTarget(examples)
        node = leafNode(target)
        return node
    elif depth==0:##If depth==0 put a logistic regrerssion classifier on that leaf and find values that subset's are branchValues and fit them.
        dataSet=findDataset(branchValues,parents,data,lol)
        node=logNode()
        node.fit(dataSet)
        return node
    else:##continue recursive
        if flag == 0:
            index = features.index(rootAttriubte)
            root=Node(rootAttriubte)
            for val in featureValues[index][1:]:
                features[index]="NONE"
                copy2=branchValues.copy()
                copy2.append(val)
                root.add_branch(val)
                arr=[]
                arr.append(rootAttriubte)
                root.add_child(ID3(splitData(examples,rootAttriubte,val,lol),val,features,depth-1,1,copy2,featureValues,rootAttriubte,arr))
        else:
            buffer=[]
            ##find best gain
            for attr in features:
                if attr!="NONE":##control if attribute that currently checking removed from Attributes (placed on tree)
                    buffer.append(gain(featureValue,Targetattr,attr,data,lol,featureValues))
                else:
                    buffer.append(-1)
            index = buffer.index(max(buffer))
            bestAttr=features[index]
            node=Node(bestAttr)
            for val in featureValues[index][1:]:
                node.add_branch(val)
                copy = branchValues.copy()
                copy.append(val)
                arr=parents.copy()
                arr.append(bestAttr)
                node.add_child(ID3(splitData(examples,bestAttr,val,lol),val,features,depth-1,1,copy,featureValues,bestAttr,arr))
            return node
    return root
def traverseTree(testSample,root,features,featureValues):##recursive traverse function for 1 sample
    if isinstance(root,leafNode):##If root == leafNode (means node has value + or -)
        data = root.getData()
        return data
    elif isinstance(root,logNode): ##If root == logNode (means node has logisticRegressior classifier)
        rgrs=root.logisticRegr
        buffer = []
        buffer.append(testSample[:-1])
        return rgrs.predict(buffer) ##predicting sample with node's classifier
    else:##else continue traverse tree
        nodeBranches=root.getBranches()
        nodeChildren=root.getChildren()
        index = lol.index(root.getData())
        for branch in range(len(nodeBranches)):
            if nodeBranches[branch]==testSample[index]:
               return traverseTree(testSample,nodeChildren[branch],features,featureValues)
def traverseTreeForSamples(testSamples,root,features,featureValues):#testing requires traversing tree.##we need to traverse for making predictions
    predictions = []
    for sample in testSamples:
        prediciton=traverseTree(sample,root,features,featureValues)
        predictions.append(prediciton)
    return predictions

depth=sys.argv[3]
def ComputeMSE(predictions,target):##classic MSE function
    sum = 0
    for i in range(len(target)):
        buffer = math.pow(target[i][-1]-predictions[i],2)
        sum=sum+buffer
    return sum/len(target)
def dataPartition(data):## data folding for 5-foldCV

    foldArray=[]
    buffer=[]
    for sample in (data):
        if len(buffer)==5:
            foldArray.append(buffer.copy())
            buffer.clear()
        buffer.append(sample)
    if len(buffer)!=0:
        foldArray.append(buffer.copy())
    return foldArray
def fiveFoldCv(partitionedData):##fiveFoldCrossValidation
    sum=0
    for fold in partitionedData:
        buffer = []
        for samples in partitionedData:
            if fold!=samples:
                for sample in samples:
                    buffer.append(sample)
        lol = features.copy()
        asd = findRoot(buffer, lol, featureValues, lol)##lol is attribute array (F1,F2,F3...)
        node = ID3(buffer,"",lol,int(depth),0,[],featureValues,rootAttriubte=asd)
        arr=traverseTreeForSamples(fold,node,lol,featureValues)
        sum=sum+ComputeMSE(arr,fold)
    return sum/len(partitionedData)
print("DTLog :"+str(round(fiveFoldCv(dataPartition(data)),2)),end=" ")
##naive Bayes starts here
import pandas as pd
import sys

def test(data,targets,frequencyTable,probYes,probNo):##testing with given data and targets
    testValues = []
    for sample in data:
        sampleProbYes=probYes
        sampleProbNo=probNo
        for index in range(len(sample)):
           prob = calculateProbability(frequencyTable,sample[index],index)
           sampleProbYes=sampleProbYes*prob[0]
           sampleProbNo=sampleProbNo*prob[1]
        if sampleProbYes > sampleProbNo:
            testValues.append(1)
        else:
            testValues.append(0)
    ##mse
    sum = 0
    for index in range(len(targets)):
        sum+=pow(targets[index]-testValues[index],2)
    return sum/len(targets),testValues
def calculateProbability(frequencyTable,featureValue,featureIndex):##calculating probability of feature  freq(prob)table.
     featureValues = frequencyTable[featureIndex]
     for probs in featureValues[1:]:
         if "For "+str(featureValue) == probs[0]:
             return probs[1],probs[2]

def calculateFrequencyOfFeature(datas,feature,featureIndex,valuesForFeature,target):##calculating freq of feature with value from freTable(helper method to above method)
    temp=[]
    temp.append("Probabilities for"+feature)
    totalyes=0
    totalno=0
    for value in target:
        if value ==1:
            totalyes=totalyes+1
        else:
            totalno=totalno+1
    for value in valuesForFeature:
        yes = 0
        no = 0
        arr=[]
        for counter in range(len(datas)):
            if datas[counter][featureIndex] == value:
                if target[counter] == 1 :
                    yes = yes +1
                else:
                    no =no +1
        arr.append("For "+ str(value))
        arr.append(yes/(totalyes))
        arr.append(no/(totalno))
        temp.append(arr.copy())
    return temp
def calculateFrequencyTable(datas,features,valuesForFeatures,target):##calculatin probability table
    frequencytable=[]
    for feature in range(len(features)-1):
        frequencytable.append(calculateFrequencyOfFeature(datas,features[feature],feature,valuesForFeatures[feature],target))
    return frequencytable
def getAllValuesOfFeatures(header,features):##Finding classes of all features
    valueTable=[]
    for i in range(len(header)-1):
        temp=[]
        for j in range(len(features)):
            if(features[j][i] in temp):
                k=0+0
            else:
                temp.append(features[j][i])
        valueTable.append(temp.copy())
    return valueTable
def readCsv(fileName):##reading csv
    data=pd.read_csv(fileName)
    header = data.columns.values
    datas = data.values
    return header,datas
def sepearateFeaturesAndTargets(data):##seperating features and targets
    features =  []
    targets = []
    for i in data :
        temp =[]
        for j in range(len(i)):
            if(j == len(i)-1):
                targets.append(i[j])
            else:
               temp.append(i[j])
        temp.pop(0)
        features.append(temp.copy())
    return features,targets
fileName=sys.argv[1]
def dataPartitionForNaiveBayes(data):## data folding with respect to 5-foldCV
    foldArray=[]
    buffer=[]
    for sample in (data):
        if len(buffer)==5:
            foldArray.append(buffer.copy())
            buffer.clear()
        buffer.append(sample)
    if len(buffer)!=0:
        foldArray.append(buffer.copy())
    return foldArray
def fiveFoldCvForNaiveBayes():##fiveFoldCrossValidation
    csv = readCsv(fileName)
    header = csv[0]
    datas = csv[1]
    temp = []
    for i in header:
        if "Unnamed" in i:
            header = header
        else:
            temp.append(i)
    header = temp.copy()
    data = []
    for sample in datas:
        buffer = []
        for val in sample[1:]:
            buffer.append(val)
        data.append(buffer)
    datas=data
    partitionedData=dataPartitionForNaiveBayes(datas)
    valueTable = getAllValuesOfFeatures(header, datas)
    # partitionedData
    sum=0
    for fold in partitionedData:
        buffer = []
        for samples in partitionedData:
            if fold!=samples:
                for sample in samples:
                    buffer.append(sample)
        targets = sepearateFeaturesAndTargets(buffer)[1]
        freqTable = calculateFrequencyTable(buffer, header, valueTable, targets)
        x = sepearateFeaturesAndTargets(fold)[0]
        y = sepearateFeaturesAndTargets(fold)[1]
        yesProb = 0
        noProb = 0
        for target in y:
            if target == 1:
                yesProb = yesProb + 1
            elif target == 0:
                noProb = noProb + 1
        yesProb = yesProb / len(targets)
        noProb = noProb / len(targets)
        sum = sum+test(x, y, freqTable, yesProb, noProb)[0]
    return sum/len(partitionedData)
print("NB:"+str(round(fiveFoldCvForNaiveBayes(),2))) ##rounded values with 2 prec.

