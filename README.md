# Loan-recovery of status-baarred consumer using machine Learning

Step 1 -Loading of file
#Loading Data

dataset = pd.read_excel('Company_x.xlsx')
df = dataset.copy()

Step 2- Drop unwanted columns from file which is not valuable info for the ML model
#drop unwanted columns(EntityID,AccountID) as it is Unqiue numbers

df.drop(['EntityID', 'AccountID'], axis=1, inplace=True)

# Assuming your DataFrame is named df
df.drop(df.columns[df.columns.str.contains('Unnamed', case=False)], axis=1, inplace=True)

Step 3- Check if there are null values then treat those values in terms of Mean, median or mode
#filling missing values with Median

df['PurchasePrice'] = df['PurchasePrice'].fillna(df['PurchasePrice'].median())

df['NumLiableParties'] = df['NumLiableParties'].fillna(df['NumLiableParties'].median())

df['CustomerAge'] = df['CustomerAge'].fillna(df['CustomerAge'].median())

Step 4- Check the frequencies of the  values of the amount, then create a visualization

Step 5 - create a model which is trained by the data
