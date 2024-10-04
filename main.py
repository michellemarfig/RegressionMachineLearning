# Linear Reg5ression MAchine Learning
# Michelle Martinez-Figueroa
# WGU C964
# September 9, 2024

import pandas as pd
from matplotlib import pyplot
from sklearn import linear_model, metrics, model_selection, svm

print('Welcome!')
print('Please wait while data is read ...')
print('')
print('')


file = 'linear_regression_data3.csv'
names = ['Month', 'Year', 'TotalNoOfCalls', 'Average Handle Time in Seconds', 'Attendance Percentage', 'Average Quality Percentage', 'Percentage of Calls Answered Within 3 minutes', 'Profit/Loss']
df = pd.read_csv(file)

mylog_model = linear_model.LogisticRegression(max_iter=1000)
mysvm_model = svm.SVC(max_iter=1000)
y = df.values[:,7]
x = df.values[:, 0:7]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.3)

mylog_model.fit(x_train, y_train)

y_pred_log = mylog_model.predict(x_test)
print('Accuracy Score:')
print(metrics.mean_squared_error(y_test, y_pred_log))

estimate = [[]]

def getPrediction():
    print('Enter the month:')
    month = float(input())
    print('Enter the year:')
    year = float(input())
    print('Enter the Total Calls Received:')
    totalCalls = float(input())
    print('Enter the Average Handle Time in Seconds:')
    handleTime = float(input())
    print('Enter the Attendance Percentage (in decimal):')
    attendance = float(input())
    print('Enter the Average Quality Percentage (in decimal):')
    quality = float(input())
    print('Enter the Percentage of Calls Answered Within 3 Minutes (in decimal):')
    SLA = float(input())

    estimate = [[month, year, totalCalls, handleTime, attendance, quality, SLA]]

    print("The estimated profit/loss for today is: $", mylog_model.predict(estimate))

    mainMenu()
def mainMenu():
    print('Enter 1 to get a profit/loss prediction. Enter 2 to show plots. Or enter anything else to exit.')
    response = input()
    if response == '1':
        getPrediction()
    if response == '2':
        showPlots()
    else:
        print('Exiting...')
        exit()

def showPlots():
    print('Enter A for a line graph. Enter B for a boxplot. Enter C for a histogram. Enter D for the main menu. Or enter anything else to exit.')
    response = input()
    if response == 'A'or response == 'a':
        pyplot.plot(x_train, y_train)
        pyplot.show()
        mainMenu()
    if response == 'B'or response == 'b':
        pyplot.boxplot(x_train)
        pyplot.show()
        mainMenu()
    if response == 'C' or response == 'c':
        pyplot.hist(x_train)
        pyplot.show()
        mainMenu()
    if response == 'D' or response == 'd':
        mainMenu()
    else:

        print('Exiting...')
        exit()

mainMenu()


exit()
