# My_classifier

## Simple Custom DataSet Trainer Using Pytorch.



1. **About**

   We can train custom dataset using this program with pytorch.

2. **File descripttion**

   | file                 | description                                    |
   | -------------------- | ---------------------------------------------- |
   | ./custom_traindata/* | training image folds. (fold's names are label) |
   | ./custom_testdata/*  | testing image folds. (fold's names are label)  |
   | ./exampleData        | example data for test excuting.                |

3.  **Requirements**

   ```
   python 3.x
   torch
   torchvision
   jupyter
   ```

4. **How to run**

   You can use your own image datasets. 

   Copy your image dataset folders(folder name is image's label) to ''./my_traindata/'' and ''./my_testdata/''

   running example:

   ```
   cd ~/my_classifier
   cd ./exampleData/
   mv my_traindata/ ..
   mv my_testdata/ ..
   jupyter notebook
   
   run the 'my_Classifier.ipynb' in your jupyter environment.
   ```

   output:

   ![./my_classifier_output1.PNG](C:\Users\minq\Desktop\취준\2018_하계인턴\pytorch\my_clasifier\my_classifier_output1.PNG)

![./my_classifier_output2.PNG](C:\Users\minq\Desktop\취준\2018_하계인턴\pytorch\my_clasifier\my_classifier_output2.png)